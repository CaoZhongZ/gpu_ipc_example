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
  constexpr static int wireSrcStep = (SubGroupSize-nChan8B)*sizeof(message_t)/sizeof(T);
  constexpr static int wireMsgStep = SubGroupSize*sizeof(message_t)/sizeof(T);
public:
  SimpleTransmit(
      sycl::nd_item<1> pos,
      size_t workSize, // should be a array in production code
      uint32_t step,   // Serve as flag for checking
      T* scatterSink[],
      T* gatherSink[],
      T* ioBuffer,
      T* localScatterSink[],
      T* localGatherSink[],
  ) : workSize(workSize), scatterStep(step), gatherStep(step), pos(pos) {
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
  // 1. multiple load inputs (32 round maximum)
  // 2. Insert flags at the 16th lane
  // 3. Shuffle flag into the middle of second register
  //
  template <int unroll> inline void loadInput(
      message_t (&v)[unroll], void* src, int nElt
  ) {
    auto lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    if (lid < lastDataChannel) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireSrcStep + local_off;
        if (off < nElt) {
          lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
              v[i], src + off;
          );
    }}}
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
            : "+rw"(reinterpret_cast<message_t::vector_t &>(messages[i]))
            : "rw"(flag);
        );
      }

#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        asm volatile (
            "mov (M1, 1) %0(1, 15)<1> %1(0, 0)<0;1,0>\n"
            : "+rw"(reinterpret_cast<message_t::vector_t &>(messages[i]))
            : "rw"(flag);
        );
      }
    } else {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        asm volatile (
            "mov (M1, 1) %0(0, 15)<1> %1(0, 0)<0;1,0>\n"
            : "+rw"(reinterpret_cast<message_t::vector_t &>(messages[i])) : "rw"(flag);
        );
      }

#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        asm volatile (
            "mov (M1, 1) %0(0, 31)<1> %1(0, 0)<0;1,0>\n"
            : "+rw"(reinterpret_cast<message_t::vector_t &>(messages[i])) : "rw"(flag);
        );
      }
    }
#else
    // Add flags at the middle and tail
    int lid = pos.get_sub_group().get_local_id()[0];
    if (lid == firstFlag || lid == lastFlag) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i)
        messages[i][lastElem] = flag;
    }
#endif
  }

  static inline void shuffleData(message_t (& messages)[unroll]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      if constexpr (SubGroupSize == 16) {
        asm volatile ("\n"
            "mov (M1, 1) %0(0, 15)<1> %0(1, 7)<0;1,0>\n"
            : "+rw"(reinterpret_cast<message_t::vector_t &>(messages[i]))
            :
        );
      } else {
        asm volatile ("\n"
            "mov (M1, 1) %0(0, 30)<1> %0(0, 15)<0;1,0>\n"
            : "+rw"(reinterpret_cast<message_t::vector_t &>(messages[i]))
            :
        );
      }
    }
#else
    auto sg = this_sub_group();
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto data = sg.shuffle(messages[i][lastElem], SubGroupSize /2 -1);
      if (sg.get_local_id() == lastDataChannel)
        messages[i][firstElem] = data;
    }
#endif
  }

  static inline void restoreData(message_t (& messages)[unroll]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      if constexpr (SubGroupSize == 16) {
        asm volatile ("\n"
            "mov (M1, 1) %0(1, 7)<1> %0(0, 15)<0;1,0>\n"
            : "+rw"(reinterpret_cast<message_t::vector_t &>(messages[i]))
            :
        );
      } else {
        asm volatile ("\n"
            "mov (M1, 1) %0(0, 15)<1> %0(0, 30)<0;1,0>\n"
            : "+rw"(reinterpret_cast<message_t::vector_t &>(messages[i]))
            :
        );
      }
    }
#else
    auto sg = this_sub_group();
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto data = sg.shuffle(messages[i][firstElem], lastDataChannel);
      if (sg.get_local_id() == SubGroupSize / 2 -1)
        messages[i][lastElem] = flag;
    }
#endif
  }

  template <int unroll> inline void storeOutput(
      message_t (&v)[unroll], void* dst, int nElt
  ) {
    auto lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);
    if (lid < lastDatachannel) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireSrcStep + local_off;
        if (off < nElt) {
          lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
              dst + off, v[i];
          );
    }}}
  }

  // We always push 128-byte packages
  template <int unroll>
  inline void sendMessages(T* ptr, message_t (&messages)[unroll]) {
    int lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
      lscStore<SubGroupSize, CacheCtrl::L1UC_L3UC>(
          ptr + u * wireMsgStep + local_off,
          messages[u];
      );
    }
  }

  // Scatter local message to peers
  template <int unroll> scatter(size_t offset, size_t nelems) {
    auto offsetInType = offset / sizeof(T);

    constexpr int eltPerPack = unroll * wireSrcStep;

    int nElt = std::min(nelems, eltPerPack);

    //
    // register consumption:
    // 2 x unroll x NPeers;
    //
    static_assert(unroll * NPeers * 2 < 64, "Too much register consumed");
    message_t messages[NPeers][unroll];

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto next = (rank + i) % (NPeers + 1);
      auto peerOffset = next * workSize / sizeof(T);
      auto* ptr = ioBuffer + peerOffset + offsetInType;

      loadInput(messages[i], ptr, nElt, scatterStep);
    }

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      shuffleData(messages[i], scatterStep);
      insertFlags(messages[i], scatterStep);

      auto* dst = scatterSink[i] + offsetInType;
      sendMessage(messages[i], dst, nElt);
    }
  }

  template <int unroll>
  inline void pollRecvReduceGather(
      message_t (&v)[unroll], size_t offset
  ) {
    message_t messages[unroll]; // scraps from remote
    auto offsetInType = offset / sizeof(T);

#   pragma unroll
    for (i = 0; i < NPeers; ++ i) {
      auto flag = scatterStep;
      bool retry;
      do {
        retry = false;
#       pragma unroll
        for (int u = 0; u < unroll; ++ u) {
          lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
              messages[u], localScatterSink[peer] + offsetInType + u * wireMsgStep);
          retry |=
            subgroup_id == firstFlagChannel && messages[u][lastElem] != flag
            || subgroup_id == lastFlagChannel && messages[u][lastElem] != flag
        }
      } while(sycl::any_of_group(sycl::this_sub_group(), retry));

      // do we need reload this???
#     pragma unroll
      for (int u = 0; u < unroll; ++ u) {
        lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
            messages[u], localScatterSink[peer] + offsetInType + u * wireMsgStep);
      }

      restoreData(messages);

#if !defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      using math_t = sycl::vec<T, sizeof(message_t)/sizeof(T)>;
      arith_v = reinterpret_cast<math_t (&)[unroll]>(v);
      arith_m = reinterpret_cast<math_t (&)[unroll]>(messages);
#endif

      if (sycl::this_sub_group.get_local_id() < lastDataChannel) {
#       pragma unroll
        for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
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
        lscStore<SubGroupSize, CacheCtrl::L1UC_L3UC>(
            gatherSink[i] + offsetInType + u * wireMsgStep, v[u]);
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
  message_t scatterStep;
  message_t gatherStep;

  sycl::nd_item<1> pos;
};

template <typename T, int NPeers, int SubGroupSize=16> struct AllReduce {
  AllReduce(
      T* input, size_t size, int rank,
      T* scatterBuf, T* gatherBuf,
      T* peerBuf0[], T* peerBuf1[], sycl::stream cout
  )
  : ioBuffer(input), rank(rank), cout(cout) {
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
    constexpr int nReg128B = 128 / SubGroupSize / 4;
    constexpr int nDataChannel = SubGroupSize - nReg128B;

    constexpr int wireCap = unroll * nDataChannel * sizeof(message_t);
    auto cableCap = wireCap * pos.get_sub_group().get_group_range()[0];
    auto loopSize = pos.get_group_range() * cableCap;

    SimpleTransmit<T, NPeers, SubGroupSize> cable (
        pos, workSize, step, scatterSink, gatherSink,
        ioBuffer, localScatterSink, localGatherSink
    );

    auto groupId = pos.get_group_id()[0];
    auto subGroupId = pos.get_sub_group().get_group_id()[0];

    // XXX: more cut according to job divide?
    for (size_t gOff = 0; gOff /* * cableCap */ < workSize; gOff += loopSize) {
      auto gPos = gOff + groupId;
      auto cableOff = gPos * cableCap;
      auto wireOff = cableOff + subGroupId * wireCap;
      cable.scatter(wireOff, workSize);
    }
  }

  void operator() (
      sycl::nd_item<1> pos
  ) [[sycl::reqd_sub_group_size(SubGroupSize)]] const {
    scatterTest(pos);
  }

private:
  // TODO: buffer plan and start point calc
  static size_t calcWorkSize(T* input, size_t size) {
  //
  // We only do 8-byte aligned and 8-byte dividable case for now.
  //
    if ((uintptr_t)input % sizeof(message_t) != 0)
      throw std::logic_error("We only support aligned pointer for now");

    auto nChunks = NPeers + 1;
    auto octSize = divUp(size, sizeof(message_t));
    auto chunkSize = divUp(octSize, nChunks);

    if (alignUpSize != size || chunkSize * sizeof(message_t) * nChunks > size)
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
  size_t workSize;
  size_t transmitSize;

  sycl::stream cout;
};

template <typename T>
void testSimpleTransmit(
    sycl::nd_range<1> launchParam,
    T* input, void* ipcbuf0, void* ipcbuf1,
    void* peerbuf0[], void* peerbuf1[], size_t size,
    int rank, int world, sycl::queue queue
);
