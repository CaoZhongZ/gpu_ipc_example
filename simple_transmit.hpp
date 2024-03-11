#pragma once

template <typename T, int NPeers, int SubGroupSize>
class SimpleTransmit {
  constexpr static int nReg128B = 128 / SubGroupSize / 4;
  constexpr static int firstElem = 0;
  constexpr static int lastElem = nReg128B -1;

protected:
  using message_t = sycl::vec<uint32_t, nReg128B>;
  // transaction of 128-byte is not atomic across HBM channel
  constexpr static int nChan8B = 8 / sizeof(message_t);
  constexpr static int lastDataChannel = SubGroupSize -nChan8B;
  constexpr static int firstFlagChannel = SubGroupSize/2 -1;
  constexpr static int lastFlagChannel = SubGroupSize -1;
  constexpr static size_t wireCapacity = (SubGroupSize-nChan8B) * sizeof(message_t);
  constexpr static size_t wireTransSize = SubGroupSize * sizeof(message_t);
  constexpr static size_t wireCapacityInType = wireCapacity / sizeof(T);
  constexpr static size_t wireTransSizeInType = wireTransSize/ sizeof(T);

public:
  //
  // sectionSize will be renamed later, it represent each temporary buffer
  // section for each rank. configurable, in bytes
  //
  constexpr static size_t sectionSize = 0x100000;
  constexpr static size_t sectionElems = sectionSize / sizeof(T);
  constexpr static size_t scratchSize = alignUp(sectionSize * (NPeers + 1), 0x200000);

public:
  SimpleTransmit(
      T* input,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[],
      ssize_t workSize,
      int rank,
      uint32_t seqNo  // Serve as flag for checking
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  ) : seqNo(seqNo), rank(rank)
#if defined(__enable_sycl_stream__)
  , cout(cout)
#endif
  {
    ioBuffer = (input + rank * workSize / sizeof(T));

    for (int i = 0; i < NPeers; ++ i) {
      int next = (rank + i + 1) % (NPeers + 1);

      scatterSink[i] = (T *)((uintptr_t)peerBuf0[next]
          + sectionSize * rank);
      gatherSink[i] = (T *)((uintptr_t)peerBuf1[next]
          + sectionSize * rank);

      localScatterSink[i] = (T *)((uintptr_t)scatterBuf
          + next * sectionSize);
      localGatherSink[i] = (T *)((uintptr_t)gatherBuf
          + next * sectionSize);

      ioForPeers[i] = input + next * workSize / sizeof(T);
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
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    if (lid < lastDataChannel) { // TODO: diverge
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireCapacityInType + local_off;
        if (off < nElt) {        // TODO: condition branch !
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          lscLoad<SubGroupSize/*, CacheCtrl::L1UC_L3UC*/>(
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
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    if (lid < lastDataChannel) { // XXX: diverge
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireCapacityInType + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscLoad<SubGroupSize/*, CacheCtrl::L1UC_L3UC*/>(
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
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
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
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);
    if (lid < lastDataChannel) { // XXX: Diverge
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireCapacityInType + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscStore<SubGroupSize/*, CacheCtrl::L1UC_L3UC*/>(
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
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);
    if (lid < lastDataChannel) { // XXX: Fixed diverge
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireCapacityInType + local_off;
        if (off < nElt) {        // XXX: runtime condition
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          lscStore<SubGroupSize/*, CacheCtrl::L1UC_L3UC*/>(
              dst + off, v[i]
          );
#endif
    }}}
  }

  // We always push 128-byte packages
  template <int unroll>
  inline void sendMessages(T* ptr, message_t (&messages)[unroll]) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscStore<SubGroupSize, CacheCtrl::L1UC_L3UC>(
          ptr + u * wireTransSizeInType + local_off,
          messages[u]
      );
#else
      (void) lid; (void) local_off;
#endif
    }
  }

  template <int unroll>
  inline bool recvMessages(message_t (&messages)[unroll], T* ptr, uint32_t flag) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    bool retry = false;

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
          messages[u],
          ptr + u * wireTransSizeInType + local_off
      );
#else
      (void) lid; (void) local_off;
#endif
      retry |= (lid == firstFlagChannel && messages[u][lastElem] != flag)
        || (lid == lastFlagChannel && messages[u][lastElem] != flag);
    }
    return retry;
  }

  template <int unroll> inline void accumMessages(
      message_t (&v)[unroll], message_t (&m)[unroll]
  ) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    using math_t = sycl::vec<T, sizeof(message_t)/sizeof(T)>;
#   pragma unroll
    for (int u = 0; u < unroll; ++ u)
      v[u] = sycl::bit_cast<message_t>(
          sycl::bit_cast<math_t>(m[u]) + sycl::bit_cast<math_t>(v[u])
      );
#endif
  }

  // Scatter local message to peers
  template <int unroll>
  inline void scatter(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireCapacityInType;
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

    if (nelems < eltPerPack) {
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        auto* ptr = ioForPeers[i] + inputOffInType;
        loadInput(messages[i], ptr, nelems);
      }
    } else {
      // Fast path. No predicated load
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        auto* ptr = ioForPeers[i] + inputOffInType;
        loadInput(messages[i], ptr);
      }
    }

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      shuffleData(messages[i]);
      insertFlags(messages[i], seqNo);

      auto* dst = scatterSink[i] + sinkOffInType;
      sendMessages(dst, messages[i]);
    }
  }

  template <int unroll>
  inline void pollRecvReduceBcast(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    message_t v[unroll];        // Input
    message_t messages[unroll]; // Scraps from remote

    auto nelems = workLeft / sizeof(T);
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);

    auto inPtr = ioBuffer + inputOffInType;
    constexpr auto eltPerPack = unroll * wireCapacityInType;

    if (nelems < eltPerPack) {
      loadInput(v, inPtr, nelems);
    } else {
      loadInput(v, inPtr);
    }

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto flag = seqNo;

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      bool retry;
      do {
        retry = false;
        retry |= recvMessages(messages, localScatterSink[i] + sinkOffInType, flag);
      } while(sycl::any_of_group(sg, retry));

#if defined(__enable_sycl_stream__)
      if (lane_id == firstFlagChannel || lane_id == lastFlagChannel) {
        cout<<"["<<rank<<","<<lane_id<<"]";
        for (int u = 0; u < unroll; ++ u)
          cout<<sycl::hex<<messages[u]<<"; ";
        cout<<sycl::endl<<sycl::flush;
      }
#endif

      restoreData(messages);
      accumMessages(v, messages);
    }

    // write back locally before shuffle data
    if (nelems < eltPerPack) {
      storeOutput(inPtr, v, nelems);
    } else {
      storeOutput(inPtr, v);
    }

    shuffleData(v);
    insertFlags(v, seqNo);

    // push to gather sinks
#   pragma unroll
    for (int i = 0; i < NPeers; ++ i)
      sendMessages(gatherSink[i] + sinkOffInType, v);
  }

  template <int unroll>
  inline void pollGatherOutputs(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireCapacityInType;
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto flag = seqNo;

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      bool retry;
      message_t messages[unroll];
      do {
        retry = false;
        retry |= recvMessages(messages, localGatherSink[i] + sinkOffInType, flag);
      } while(sycl::any_of_group(sg, retry));

      auto ptr = ioForPeers[i] + inputOffInType;

      restoreData(messages);

      if (nelems < eltPerPack)
        storeOutput(ptr, messages, nelems);
      else
        storeOutput(ptr, messages);
    }
  }

protected:
  T* scatterSink[NPeers];
  T* gatherSink[NPeers];
  T* localScatterSink[NPeers];
  T* localGatherSink[NPeers];
  T* ioBuffer; // point to workload of self
  T* ioForPeers[NPeers]; // point to distributed workload

  uint32_t seqNo;
  int rank;

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};
