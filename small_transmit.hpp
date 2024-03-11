#pragma once

template <typename T, int NPeers, int SubGroupSize>
class SmallTransmit {
  // first row contains data, second row, flags
protected:
  using message_t = sycl::vec<uint32_t, 2>;
  static constexpr int dataElem = 0;
  static constexpr int flagElem = 1;

  constexpr static size_t wireCapacity = SubGroupSize * sizeof(message_t) / 2;
  constexpr static size_t wireTransSize = SubGroupSize * sizeof(message_t);

  constexpr static int wireElems = wireCapacity / sizeof(T);
  constexpr static int wireTransElems = wireTransSize / sizeof(T);

public:
  //
  // sectionSize will be renamed later, it represent each temporary buffer
  // section for each rank. configurable, in bytes
  //
  constexpr static size_t sectionSize = 0x100000;
  constexpr static size_t sectionElems = sectionSize / sizeof(T);
  constexpr static size_t scratchSize = alignUp(sectionSize * (NPeers + 1), 0x200000);

public:
  SmallTransmit(
      T* input, T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[],
      ssize_t workSize,
      int rank,
      uint32_t seqNo   // Serve as flag for checking
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

  // load first row of registers
  template <int unroll> inline void loadInput(
      message_t (&v)[unroll], T* src, int nElt
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireElems + local_off;
      if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16)
          asm volatile ("\n" // Add this partial load to tvisa
              "lsc_load.ugm.df.df (M1, 16) %0:d32 flat[%1]:a64\n"
              : "=rw"(v[i][dataElem]) : "rw"(src + off));
        else
          asm volatile ("\n" // Add this partial load to tvisa
              "lsc_load.ugm.df.df (M1, 32) %0:d32 flat[%1]:a64\n"
              : "=rw"(v[i][dataElem]) : "rw"(src + off));
#else
        v[i][0] = src[off];
#endif
    }}
  }

  template <int unroll> inline void loadInput(
      message_t (&v)[unroll], T* src
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireElems + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      if constexpr (SubGroupSize == 16)
        asm volatile ("\n" // Add this partial load to tvisa
            "lsc_load.ugm.df.df (M1, 16) %0:d32 flat[%1]:a64\n"
            : "=rw"(v[i][dataElem]) : "rw"(src + off));
      else
        asm volatile ("\n" // Add this partial load to tvisa
            "lsc_load.ugm.df.df (M1, 32) %0:d32 flat[%1]:a64\n"
            : "=rw"(v[i][dataElem]) : "rw"(src + off));
#else
      v[i][0] = src[off];
#endif
    }
  }

  //Insert flags to second row
  template <int unroll>
  inline void insertFlags(
      message_t (& messages)[unroll], uint32_t flag
  ) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    if constexpr (SubGroupSize == 16) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        asm volatile (
            "mov (M1, 16) %0(0, 0)<1> %1(0, 0)<0;1,0>\n"
            : "+rw"(messages[i][flagElem])
            : "rw"(flag)
        );
      }
    } else {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        asm volatile (
            "mov (M1, 32) %0(0, 0)<1> %1(0, 0)<0;1,0>\n"
            : "+rw"(messages[i][flagElem]) : "rw"(flag)
        );
      }
    }
#else
#   pragma unroll
    for (int i = 0; i < unroll; ++ i)
      messages[i][flagElem] = flag;
#endif
  }

  template <int unroll> inline void storeOutput(
      T* dst, message_t (&v)[unroll]
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireElems + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      if constexpr (SubGroupSize == 16)
        asm volatile ("\n"
            "lsc_store.ugm.df.df (M1, 16) flat[%0]:a64 %1:d32\n"
            :: "rw"(dst + off), "rw"(v[i][dataElem]));
      else
        asm volatile ("\n"
            "lsc_store.ugm.df.df (M1, 32) flat[%0]:a64 %1:d32\n"
            :: "rw"(dst + off), "rw"(v[i][dataElem]));
#else
      dst[off] = v[i][0];
#endif
    }
  }

  template <int unroll> inline void storeOutput(
      T* dst, message_t (&v)[unroll], int nElt
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireElems + local_off;
      if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16)
          asm volatile ("\n"
              "lsc_store.ugm.df.df (M1, 16) flat[%0]:a64 %1:d32\n"
              :: "rw"(dst + off), "rw"(v[i][dataElem]));
        else
          asm volatile ("\n"
              "lsc_store.ugm.df.df (M1, 32) flat[%0]:a64 %1:d32\n"
              :: "rw"(dst + off), "rw"(v[i][dataElem]));
#else
      dst[off] = v[i][dataElem];
#endif
    }}
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
          ptr + u * wireTransElems + local_off,
          messages[u]
      );
#else
      (void) lid; (void) local_off;
#endif
    }
  }

  template <int unroll>
  inline bool recvMessages(message_t (&messages)[unroll], T* ptr, int flag) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    bool retry = false;

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
          messages[u], ptr + u * wireTransElems + local_off
      );
#else
      (void) lid; (void) local_off;
#endif
      retry |= messages[u][flagElem] != flag;
    }

    return retry;
  }

  // Scatter local message to peers
  template <int unroll>
  inline void scatter(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireElems;
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
      // Slow path. Given we can't lookahead too much, we interleave message
      // load and send
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        auto* ptr = ioForPeers[i] + inputOffInType;
        loadInput(messages[i], ptr, nelems);
        insertFlags(messages[i], seqNo);

        auto* dst = scatterSink[i] + sinkOffInType;
        sendMessages(dst, messages[i]);
      }
    } else {
      // Fast path. Batching load and send
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        auto* ptr = ioForPeers[i] + inputOffInType;
        loadInput(messages[i], ptr);
      }

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        insertFlags(messages[i], seqNo);

        auto* dst = scatterSink[i] + sinkOffInType;
        sendMessages(dst, messages[i]);
      }
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

    constexpr auto eltPerPack = unroll * wireElems;
    if (nelems < eltPerPack) {
      loadInput(v, ioBuffer + inputOffInType, nelems);
    } else {
      loadInput(v, ioBuffer + inputOffInType);
    }

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto flag = seqNo;
      bool retry;
      do {
        retry = false;
        retry |= recvMessages(messages, localScatterSink[i] + sinkOffInType, flag);
      } while(sycl::any_of_group(sg, retry));

#     pragma unroll
      for (int u = 0; u < unroll; ++ u) {
#if 0// defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          v[u] = addAs<T, SubGroupSize>(v[u], messages[u]);
#else
        auto& v_math = reinterpret_cast<
          sycl::vec<T, sizeof(uint32_t)/sizeof(T)> &>(v[u][0]);
        auto& m_math = reinterpret_cast<
          sycl::vec<T, sizeof(uint32_t)/sizeof(T)> &>(messages[u][0]);
        v_math += m_math;
#endif
      }
    }

    // write back locally before shuffle data
    if (nelems < eltPerPack) {
      storeOutput(ioBuffer + inputOffInType, v, nelems);
    } else {
      storeOutput(ioBuffer + inputOffInType, v);
    }

    insertFlags(v, seqNo);

    // push to gather sink
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

    constexpr auto eltPerPack = unroll * wireElems;

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto flag = seqNo;
      bool retry;
      message_t messages[unroll];
      do {
        retry = false;
        retry |= recvMessages(messages, localGatherSink[i] + sinkOffInType, flag);
      } while(sycl::any_of_group(sg, retry));

      auto* ptr = ioForPeers[i] + inputOffInType;

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


