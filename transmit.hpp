#pragma once

#include <sycl/sycl.hpp>
#include <gen_visa_templates.hpp>

#define divUp(x, m)  \
  ((x + m -1) / m)

#define alignUp(x, c) \
  (divUp(x, c) * c)

template <int SubGroupSize> struct message_type;

template <typename T, int NPeers, int SubGroupSize>
class SimpleTransmit {
  constexpr static int nReg128B = 128 / SubGroupSize / 4;
  constexpr static int firstElem = 0;
  constexpr static int lastElem = nReg128B -1;

  using message_t = sycl::vec<uint32_t, nReg128B>;
  // transaction of 128-byte is not atomic across HBM channel
  constexpr static int nChan8B = 8 / sizeof(message_t);
  constexpr static int lastDataChannel = SubGroupSize -nChan8B;
  constexpr static int firstFlagChannel = SubGroupSize/2 -1;
  constexpr static int lastFlagChannel = SubGroupSize -1;
  constexpr static size_t wireSrcStep = (SubGroupSize-nChan8B)*sizeof(message_t)/sizeof(T);
  constexpr static size_t wireMsgStep = SubGroupSize*sizeof(message_t)/sizeof(T);
public:
  SimpleTransmit(
      sycl::nd_item<1> pos,
      size_t workSize, // should be a array in production code
      uint32_t step,   // Serve as flag for checking
      int rank,
      T* const scatterSink[],
      T* const gatherSink[],
      T* ioBuffer,
      T* const localScatterSink[],
      T* const localGatherSink[]
  ) : ioBuffer(ioBuffer), workSize(workSize), scatterStep(step),
  gatherStep(step), rank(rank), pos(pos) {
#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      this->scatterSink[i] = scatterSink[i];
      this->gatherSink[i] = gatherSink[i];
      this->localScatterSink[i] = localScatterSink[i];
      this->localGatherSink[i] = localGatherSink[i];
    }
  }
  //
  // Process of pack messages
  // 1. multiple load inputs (16 round maximum)
  // 2. Insert flags at the 16th lane
  // 3. Shuffle flag into the middle of second register
  //
  template <int unroll> inline void loadInput(
      message_t (&v)[unroll], T* src, int nElt
  ) {
    auto lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    if (lid < lastDataChannel) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireSrcStep + local_off;
        if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
              v[i], src + off
          );
#else
          (void)off;
#endif
    }}}
  }

  template <int unroll> inline void loadInput(
      message_t (&v)[unroll], T* src
  ) {
    auto lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    if (lid < lastDataChannel) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireSrcStep + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
            v[i], src + off
        );
#else
        (void)off;
#endif
    }}
  }

  template <int unroll>
  inline void insertFlags(
      message_t (& messages)[unroll], uint32_t flag
  ) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    if constexpr (SubGroupSize == 16) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        asm volatile (
            "mov (M1, 1) %0(1, 7)<1> %1(0, 0)<0;1,0>\n"
            : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
            : "rw"(flag)
        );
      }

#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        asm volatile (
            "mov (M1, 1) %0(1, 15)<1> %1(0, 0)<0;1,0>\n"
            : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
            : "rw"(flag)
        );
      }
    } else {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        asm volatile (
            "mov (M1, 1) %0(0, 15)<1> %1(0, 0)<0;1,0>\n"
            : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i])) : "rw"(flag)
        );
      }

#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        asm volatile (
            "mov (M1, 1) %0(0, 31)<1> %1(0, 0)<0;1,0>\n"
            : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i])) : "rw"(flag)
        );
      }
    }
#else
    // Add flags at the middle and tail
    int lid = pos.get_sub_group().get_local_id()[0];
    if (lid == firstFlagChannel || lid == lastFlagChannel) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i)
        messages[i][lastElem] = flag;
    }
#endif
  }

  template <int unroll>
  static inline void shuffleData(message_t (& messages)[unroll]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      if constexpr (SubGroupSize == 16) {
        asm volatile ("\n"
            "mov (M1, 1) %0(0, 15)<1> %0(1, 7)<0;1,0>\n"
            : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
            :
        );
      } else {
        asm volatile ("\n"
            "mov (M1, 1) %0(0, 30)<1> %0(0, 15)<0;1,0>\n"
            : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
            :
        );
      }
    }
#else
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto data = sg.shuffle(messages[i][lastElem], SubGroupSize /2 -1);
      if (sg.get_local_id() == lastDataChannel)
        messages[i][firstElem] = data;
    }
#endif
  }

  template <int unroll>
  static inline void restoreData(message_t (& messages)[unroll]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      if constexpr (SubGroupSize == 16) {
        asm volatile ("\n"
            "mov (M1, 1) %0(1, 7)<1> %0(0, 15)<0;1,0>\n"
            : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
            :
        );
      } else {
        asm volatile ("\n"
            "mov (M1, 1) %0(0, 15)<1> %0(0, 30)<0;1,0>\n"
            : "+rw"(reinterpret_cast<typename message_t::vector_t &>(messages[i]))
            :
        );
      }
    }
#else
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto data = sg.shuffle(messages[i][firstElem], lastDataChannel);
      if (sg.get_local_id() == SubGroupSize / 2 -1)
        messages[i][lastElem] = data;
    }
#endif
  }

  template <int unroll> inline void storeOutput(
      message_t (&v)[unroll], T* dst, int nElt
  ) {
    auto lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);
    if (lid < lastDataChannel) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireSrcStep + local_off;
        if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          lscStore<SubGroupSize, CacheCtrl::L1UC_L3UC>(
              dst + off, v[i]
          );
#endif
    }}}
  }

  // We always push 128-byte packages
  template <int unroll>
  inline void sendMessages(T* ptr, message_t (&messages)[unroll]) {
    int lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscStore<SubGroupSize, CacheCtrl::L1UC_L3UC>(
          ptr + u * wireMsgStep + local_off,
          messages[u]
      );
#else
      (void) lid; (void) local_off;
#endif
    }
  }

  // Scatter local message to peers
  template <int unroll> inline void scatter(size_t offset, size_t nelems) {
    auto offsetInType = offset / sizeof(T);

    constexpr auto eltPerPack = unroll * wireSrcStep;

    bool boundCheck = nelems < eltPerPack;

    //
    // register consumption:
    // 2 x unroll x NPeers;
    //
    // SWSB consumption:
    // unroll * NPeers;
    //
    static_assert(unroll * NPeers * 2 < 64, "Too many registers consumed");
    static_assert(NPeers * 2 < 16, "Too many swsb consumed");
    message_t messages[NPeers][unroll];

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto next = (rank + i) % (NPeers + 1);
      auto peerOffset = next * workSize / sizeof(T);
      auto* ptr = ioBuffer + peerOffset + offsetInType;

      if (boundCheck)
        loadInput(messages[i], ptr, nelems);
      else
        loadInput(messages[i], ptr);
    }

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      shuffleData(messages[i]);
      insertFlags(messages[i], scatterStep);

      auto* dst = scatterSink[i] + offsetInType;
      sendMessages(dst, messages[i]);
    }
  }

  template <int unroll>
  inline void pollRecvReduceGather(
      message_t (&v)[unroll], size_t offset
  ) {
    message_t messages[unroll]; // scraps from remote
    auto offsetInType = offset / sizeof(T);

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    int subgroup_id = sg.get_local_id();

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto flag = scatterStep;
      bool retry;
      do {
        retry = false;
#       pragma unroll
        for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
              messages[u], localScatterSink[i] + offsetInType + u * wireMsgStep);
#else
          (void)offsetInType;
#endif
          retry |=
            (subgroup_id == firstFlagChannel && messages[u][lastElem] != flag)
            || (subgroup_id == lastFlagChannel && messages[u][lastElem] != flag);
        }
      } while(sycl::any_of_group(sg, retry));

      // do we need reload this???
#     pragma unroll
      for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
            messages[u], localScatterSink[i] + offsetInType + u * wireMsgStep);
#endif
      }

      restoreData(messages);

#if 1 //!defined(__SYCL_DEVICE_ONLY__)
      using math_t = sycl::vec<T, sizeof(message_t)/sizeof(T)>;
      auto arith_v = reinterpret_cast<math_t (&)[unroll]>(v);
      auto arith_m = reinterpret_cast<math_t (&)[unroll]>(messages);
#endif

      if (subgroup_id < lastDataChannel) {
#       pragma unroll
        for (int u = 0; u < unroll; ++ u) {
#if 0// defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          v[u] = addAs<T, SubGroupSize>(v[u], messages[u]);
#else
          arith_v[u] = arith_v[u] + arith_m[u];
#endif
        }
      }
    }

    // push to gather sink
#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      shuffleData(v);
      insertFlags(v, gatherStep);

#     pragma unroll
      for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscStore<SubGroupSize, CacheCtrl::L1UC_L3UC>(
            gatherSink[i] + offsetInType + u * wireMsgStep, v[u]);
#endif
      }
    }
  }

private:
  T* scatterSink[NPeers];
  T* gatherSink[NPeers];
  T* ioBuffer;
  T* localScatterSink[NPeers];
  T* localGatherSink[NPeers];

  size_t workSize;
  uint32_t scatterStep;
  uint32_t gatherStep;
  int rank;

  sycl::nd_item<1> pos;
};

template <typename T, int NPeers, int SubGroupSize=16> struct AllReduce {
  constexpr static int nReg128B = 128 / SubGroupSize / 4;
  using message_t = sycl::vec<uint32_t, nReg128B>;

  constexpr static int nChan8B = 8 / sizeof(message_t);
  constexpr static int nDataChannel = SubGroupSize - nChan8B;

  AllReduce(
      T* input, size_t size, int rank, uint32_t step,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[]
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  )
  : ioBuffer(input), rank(rank), step(step)
#if defined(__enable_sycl_stream__)
    , cout(cout)
#endif
  {
    workSize = calcWorkSize(input, size);
    transmitSize = divUp(workSize, 120)*128;

    auto slotShift = [](int rank, int peer) {
      if (rank > peer) return rank -1;
      else return rank;
    };

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      int next = (rank + i) % (NPeers + 1);

      scatterSink[i] = (T *)((uintptr_t)peerBuf0[i]
          + transmitSize * slotShift(rank, next));
      gatherSink[i] = (T *)((uintptr_t)peerBuf1[i]
          + transmitSize * slotShift(rank, next));

      localScatterSink[i] = scatterBuf + slotShift(next, rank) * transmitSize;
      localGatherSink[i] = gatherBuf + slotShift(next, rank) * transmitSize;
    }
  }

  //
  // I found this analogy fascinating:
  //
  // Let correspond sub-group to wire, sequential guaranteed.
  // Bundle sub-groups(wires) into group(cable).
  //
  // Total cables will deliver the full capacity of single loop.
  //
  template <int unroll>
  void scatterTest(sycl::nd_item<1> pos) const {
    constexpr int wireCap = unroll * nDataChannel * sizeof(message_t);
    auto cableCap = wireCap * pos.get_sub_group().get_group_range()[0];
    auto loopSize = pos.get_group_range()[0] * cableCap;

    SimpleTransmit<T, NPeers, SubGroupSize> cable (
        pos, workSize, step, rank, scatterSink, gatherSink,
        ioBuffer, localScatterSink, localGatherSink
    );

    auto groupId = pos.get_group().get_group_id()[0];
    auto subGroupId = pos.get_sub_group().get_group_id()[0];

    // XXX: more cut according to job divide?
    for (size_t gOff = 0; gOff /* * cableCap */ < workSize; gOff += loopSize) {
      auto gPos = gOff + groupId;
      auto cableOff = gPos * cableCap;
      auto wireOff = cableOff + subGroupId * wireCap;
      cable.template scatter<unroll>(wireOff, workSize);
    }
  }

  void operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (
      sycl::nd_item<1> pos
  ) const {
    if constexpr (NPeers < 4)
      scatterTest<4>(pos);
    else
      scatterTest<2>(pos);
  }

private:
  // TODO: buffer plan and start point calc
  static size_t calcWorkSize(T* input, size_t size) {
    // Input must be message size align
    if ((uintptr_t)input % sizeof(message_t) != 0)
      throw std::logic_error("We only support aligned pointer for now");

    auto nChunks = NPeers + 1;
    auto octSize = divUp(size, sizeof(message_t));
    auto chunkSize = divUp(octSize, nChunks);

    if (octSize * sizeof(message_t) != size || chunkSize * sizeof(message_t) * nChunks > size)
      throw std::logic_error("We don't support non-even divide yet");

    // TODO: Production logic needs every rank chunk

    return chunkSize * sizeof(message_t);
  }

  T* scatterSink[NPeers];
  T* gatherSink[NPeers];
  T* localScatterSink[NPeers];
  T* localGatherSink[NPeers];

  T* ioBuffer;

  int rank;
  uint32_t step;
  size_t workSize;
  size_t transmitSize;

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

template <typename T>
sycl::event testSimpleTransmit(
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t size,
    int rank, int world, uint32_t step, sycl::queue queue
);
