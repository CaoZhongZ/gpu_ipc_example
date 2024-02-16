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
      ssize_t workSize, // should be a array in production code
      uint32_t step,   // Serve as flag for checking
      int rank,
      T* const scatterSink[],
      T* const gatherSink[],
      T* ioBuffer,
      T* const localScatterSink[],
      T* const localGatherSink[]
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  ) : ioBuffer(ioBuffer), workSize(workSize), scatterStep(step),
  gatherStep(step + 1), rank(rank), pos(pos)
#if defined(__enable_sycl_stream__)
  , cout(cout)
#endif
  {
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
#if defined(__enable_sycl_stream__)
          // cout<<"["<<rank<<","<<lid<<"]off: "<<off
          //   <<", src "<<src<<":"<<v[i]<<sycl::endl;
#endif
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
      T* dst, message_t (&v)[unroll]
  ) {
    auto lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);
    if (lid < lastDataChannel) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireSrcStep + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscStore<SubGroupSize, CacheCtrl::L1UC_L3UC>(
            dst + off, v[i]
        );
#else
        (void)off; (void)local_off;
#endif
    }}
  }

  template <int unroll> inline void storeOutput(
      T* dst, message_t (&v)[unroll], int nElt
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

  template <int unroll>
  inline void recvMessages(message_t (&messages)[unroll], T* ptr) {
    int lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
          messages[u],
          ptr + u * wireMsgStep + local_off
      );
#else
      (void) lid; (void) local_off;
#endif
    }
  }

  // Scatter local message to peers
  template <int unroll>
  inline void scatter(
      size_t inputOffset, size_t sinkOffset, ssize_t workSize
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workSize / sizeof(T);

    constexpr auto eltPerPack = unroll * wireSrcStep;
    //
    // register consumption:
    // 2 x unroll x NPeers;
    //
    // SWSB consumption:
    // unroll * NPeers;
    //
    static_assert(unroll * NPeers * 2 < 64, "Too many registers consumed");
    static_assert(NPeers * 2 < 16, "Too many swsb consumed");
    if (nelems < eltPerPack) {
      // Slow path. Given we can't lookahead too much, we interleave message
      // load and send
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        message_t messages[unroll];

        auto next = (rank + i + 1) % (NPeers + 1);
        auto peerOffset = next * workSize / sizeof(T);
        auto* ptr = ioBuffer + peerOffset + inputOffInType;
#if defined(__enable_sycl_stream__)
        // auto sg = sycl::ext::oneapi::experimental::this_group<1>();

        // if (sg.get_local_id() == 0)
        //   cout<<"Input offset: "<<inputOffset
        //     <<"workSize: "<<workSize
        //     <<"peerOffset: "<<peerOffset
        //     <<sycl::endl;
#endif
        loadInput(messages, ptr, nelems);

        shuffleData(messages);
        insertFlags(messages, scatterStep);

        auto* dst = scatterSink[i] + sinkOffInType;
        sendMessages(dst, messages);
      }
    } else {
      // Fast path. Batching load and send
      message_t messages[NPeers][unroll];

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        auto next = (rank + i + 1) % (NPeers + 1);
        auto peerOffset = next * workSize / sizeof(T);
        auto* ptr = ioBuffer + peerOffset + inputOffInType;

        loadInput(messages[i], ptr);
      }

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        shuffleData(messages[i]);
        insertFlags(messages[i], scatterStep);

        auto* dst = scatterSink[i] + sinkOffInType;
        sendMessages(dst, messages[i]);
      }
    }
  }

  template <int unroll>
  inline void pollRecvReduceBcast(
      size_t inputOffset, size_t sinkOffset, ssize_t workSize
  ) {
    message_t v[unroll];        // Input
    message_t messages[unroll]; // Scraps from remote

    auto nelems = workSize / sizeof(T);
    auto rankOffset = rank * nelems;
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);

    constexpr auto eltPerPack = unroll * wireSrcStep;
    if (nelems < eltPerPack) {
      loadInput(v, ioBuffer + inputOffInType + rankOffset, nelems);
    } else {
      loadInput(v, ioBuffer + inputOffInType + rankOffset);
    }

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    int lane_id = sg.get_local_id();

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto flag = scatterStep;
      bool retry;
      do {
        retry = false;
#       pragma unroll
        for (int u = 0; u < unroll; ++ u) {
          recvMessages(messages, localScatterSink[i] + sinkOffInType);
          retry |=
            (lane_id == firstFlagChannel && messages[u][lastElem] != flag)
            || (lane_id == lastFlagChannel && messages[u][lastElem] != flag);
        }
      } while(sycl::any_of_group(sg, retry));

      // do we need reload this???
      /*
#     pragma unroll
      for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
            messages[u], localScatterSink[i] + sinkOffInType + u * wireMsgStep);
#else
        (void)sinkOffInType;
#endif
      }*/

      restoreData(messages);

#if 1 //!defined(__SYCL_DEVICE_ONLY__)
      using math_t = sycl::vec<T, sizeof(message_t)/sizeof(T)>;
      auto arith_v = reinterpret_cast<math_t (&)[unroll]>(v);
      auto arith_m = reinterpret_cast<math_t (&)[unroll]>(messages);
#endif

      if (lane_id < lastDataChannel) {
#       pragma unroll
        for (int u = 0; u < unroll; ++ u) {
#if 0// defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          v[u] = addAs<T, SubGroupSize>(v[u], messages[u]);
#else
//           if (lane_id == 0 && u == 0) {
//             cout<<"["<<rank<<"]v:"<<arith_v[0]
//               <<", m:"<<arith_m[0]
//               <<sycl::endl<<sycl::flush;
//           }
          arith_v[u] = arith_v[u] + arith_m[u];
#endif
        }
      }
    }

    // write back locally before shuffle data
    if (nelems < eltPerPack) {
      storeOutput(ioBuffer + inputOffInType + rankOffset, v, nelems);
    } else {
      storeOutput(ioBuffer + inputOffInType + rankOffset, v);
    }

    shuffleData(v);
    insertFlags(v, gatherStep);

    // push to gather sink
#   pragma unroll
    for (int i = 0; i < NPeers; ++ i)
      sendMessages(gatherSink[i] + sinkOffInType, v);
  }

  template <int unroll>
  inline void pollGatherOutputs(
      size_t inputOffset, size_t sinkOffset, ssize_t workSize
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workSize / sizeof(T);

    constexpr auto eltPerPack = unroll * wireSrcStep;
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    int lane_id = sg.get_local_id();

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto flag = gatherStep;
      bool retry;
      message_t messages[unroll];
      do {
        retry = false;
#       pragma unroll
        for (int u = 0; u < unroll; ++ u) {
          recvMessages(messages, localGatherSink[i] + sinkOffInType);
          retry |=
            (lane_id == firstFlagChannel && messages[u][lastElem] != flag)
            || (lane_id == lastFlagChannel && messages[u][lastElem] != flag);
        }
      } while(sycl::any_of_group(sg, retry));

      auto next = (rank + i + 1) % (NPeers + 1);
      auto peerOffset = next * workSize / sizeof(T);
      auto ptr = ioBuffer + peerOffset + inputOffInType;

      restoreData(messages);

      if (nelems < eltPerPack)
        storeOutput(ptr, messages, nelems);
      else
        storeOutput(ptr, messages);
    }
  }

private:
  T* scatterSink[NPeers];
  T* gatherSink[NPeers];
  T* ioBuffer;
  T* localScatterSink[NPeers];
  T* localGatherSink[NPeers];

  ssize_t workSize;
  uint32_t scatterStep;
  uint32_t gatherStep;
  int rank;

  sycl::nd_item<1> pos;
#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

template <typename T,
         int NPeers,
         template <typename, int, int> class Transmit,
         int SubGroupSize = 16>
struct AllReduce {
  constexpr static int nReg128B = 128 / SubGroupSize / 4;
  using message_t = sycl::vec<uint32_t, nReg128B>;

  constexpr static int nChan8B = 8 / sizeof(message_t);
  constexpr static int nDataChannel = SubGroupSize - nChan8B;
  constexpr static int unroll = NPeers < 4 ? 4 : 2;
  constexpr static int wireCapacity = unroll * nDataChannel * sizeof(message_t);
  constexpr static int wireTransSize = unroll * SubGroupSize * sizeof(message_t);

  AllReduce(
      T* input, size_t nelems, int rank, uint32_t step,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[]
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  )
  : ioBuffer(input), rank(rank), step(step),
  workSize(calcWorkSize(input, nelems * sizeof(T))),
  transmitSize(divUp(workSize, wireCapacity) * wireTransSize)
#if defined(__enable_sycl_stream__)
    , cout(cout)
#endif
  {
    auto slotShift = [](int rank, int peer) {
      if (rank > peer) return rank -1;
      else return rank;
    };

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      int next = (rank + i + 1) % (NPeers + 1);

      scatterSink[i] = (T *)((uintptr_t)peerBuf0[next]
          + transmitSize * slotShift(rank, next));
      gatherSink[i] = (T *)((uintptr_t)peerBuf1[next]
          + transmitSize * slotShift(rank, next));

      localScatterSink[i] = (T *)((uintptr_t)scatterBuf
          + slotShift(next, rank) * transmitSize);
      localGatherSink[i] = (T *)((uintptr_t)gatherBuf
          + slotShift(next, rank) * transmitSize);
    }
  }

  // Calculate which slot 'rank' take at 'peer''s sink buffer
  static int slot(int rank, int peer) {
    if (rank > peer) return rank -1;
    else return rank;
  }

  static int scatterVerify(
      uint32_t* host, int rank, uint32_t flag, size_t nWorkElemsInInt
  );
  static int stage2Verify(
      T* host, int rank, uint32_t flag, size_t nWorkElemsInInt
  );
  //
  // Found this analogy fascinating:
  //
  // Let correspond sub-group to wire, sequential guaranteed.
  // Bundle sub-groups(wires) into group(cable).
  //
  // Total cables will deliver the full capacity of single loop.
  //
  inline void scatterTest(sycl::nd_item<1> pos) const {
    auto cableCapacity = wireCapacity * pos.get_sub_group().get_group_range()[0];
    auto cableTSize = wireTransSize * pos.get_sub_group().get_group_range()[0];

    auto loopSize = pos.get_group_range()[0] * cableCapacity;
    auto loopTSize = pos.get_group_range()[0] * cableTSize;

    Transmit<T, NPeers, SubGroupSize> cable (
        pos, workSize, step, rank, scatterSink, gatherSink,
        ioBuffer, localScatterSink, localGatherSink
#if defined(__enable_sycl_stream__)
        , cout
#endif
    );
    auto groupId = pos.get_group().get_group_id()[0];
    auto subGroupId = pos.get_sub_group().get_group_id()[0];

#if defined(__enable_sycl_stream__)
    auto local_id = pos.get_sub_group().get_local_id()[0];
    // if (local_id == 0 && groupId == 0)
    //   cout<<"["<<groupId<<", "<<subGroupId
    //     <<"] scatterTest: ioBuffer:"<<ioBuffer
    //     <<", wireCapacity:"<<wireCapacity
    //     <<", cableCapacity:"<<cableCapacity
    //     <<", wireTransSize:"<<wireTransSize
    //     <<", cableTSize:"<<cableTSize
    //     <<", scatterSink:"<<scatterSink[0]
    //     <<", gatherSink:"<<gatherSink[0]
    //     <<", localScatterSink:"<<localScatterSink[0]
    //     <<", localGatherSink:"<<localGatherSink[0]<<sycl::endl;
#endif

    // XXX: more cut according to job divide?
    for (size_t gOff = 0, tOff = 0;
        gOff /* * cableCapacity */ < workSize;
        gOff += loopSize, tOff += loopTSize) {
      auto wireOff = groupId * cableCapacity + subGroupId * wireCapacity + gOff;
      auto transOff = groupId * cableTSize + subGroupId * wireTransSize + tOff;
#if defined(__enable_sycl_stream__)
      if (local_id == 0 && groupId == 0)
        cout<<"["<<groupId<<", "<<subGroupId
          <<"] loopSize:"<<loopSize
          <<", wireOff:"<<wireOff<<"; "
          <<", transOff:"<<transOff<<"; ";
#endif
      ssize_t workLeft = workSize - wireOff;
      if (workLeft > 0)
        cable.template scatter<unroll>(wireOff, transOff, workSize);
    }
#if defined(__enable_sycl_stream__)
    if (local_id == 0 && groupId == 0)
      cout<<sycl::endl;
#endif
  }

  inline void stage2Test(sycl::nd_item<1> pos) const {
    auto cableCapacity = wireCapacity * pos.get_sub_group().get_group_range()[0];
    auto cableTSize = wireTransSize * pos.get_sub_group().get_group_range()[0];

    auto loopSize = pos.get_group_range()[0] * cableCapacity;
    auto loopTSize = pos.get_group_range()[0] * cableTSize;

    Transmit<T, NPeers, SubGroupSize> cable(
        pos, workSize, step, rank, scatterSink, gatherSink,
        ioBuffer, localScatterSink, localGatherSink
#if defined(__enable_sycl_stream__)
        ,cout
#endif
    );

    auto groupId = pos.get_group().get_group_id()[0];
    auto subGroupId = pos.get_sub_group().get_group_id()[0];

    for (size_t gOff = 0, tOff = 0;
        gOff < workSize; gOff += loopSize, tOff += loopTSize) {
      auto wireOff = groupId * cableCapacity + subGroupId * wireCapacity + gOff;
      auto transOff = groupId * cableTSize + subGroupId * wireTransSize + tOff;
#if defined(__enable_sycl_stream__)
      auto local_id = pos.get_sub_group().get_local_id()[0];
      if (local_id == 0 && groupId == 0)
        cout<<"["<<groupId<<", "<<subGroupId
          <<"] loopSize:"<<loopSize
          <<", wireOff:"<<wireOff<<"; "
          <<", transOff:"<<transOff<<"; "<<sycl::endl;
#endif
      ssize_t workLeft = workSize - wireOff;
      if (workLeft > 0) {
        cable.template scatter<unroll>(wireOff, transOff, workSize);
        cable.template pollRecvReduceBcast<unroll>(wireOff, transOff, workSize);
        cable.template pollGatherOutputs<unroll>(wireOff, transOff, workSize);
      }
    }
  }

  void operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (
      sycl::nd_item<1> pos
  ) const {
    stage2Test(pos);
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
  ssize_t workSize;
  size_t transmitSize;

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

template <typename T, int SubGroupSize = 16>
sycl::event testSimpleTransmit(
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t step, sycl::queue queue
);

template <typename T, int SubGroupSize = 16>
int verifyTransmit(
    T* host, uint32_t step, int rank, int world, size_t nWorkElems
);
