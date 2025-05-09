#pragma once

template <typename T, int NRanks, int SubGroupSize>
class BisectPTransmit {
  constexpr static int BiNRanks = NRanks / 2;
  constexpr static int NPeers = BiNRanks -1;
  constexpr static int nReg128B = 128 / SubGroupSize / 4;
  constexpr static int firstElem = 0;
  constexpr static int lastElem = nReg128B -1;

  // Configurations
  constexpr static auto CommReadCacheCtrl = CacheCtrl::L1UC_L3C;
  constexpr static auto CommWriteCacheCtrl = CacheCtrl::L1UC_L3WB;
  constexpr static auto PrefetchCacheCtrl = CacheCtrl::DEFAULT;

protected:
  using message_t = sycl::vec<uint32_t, nReg128B>;
  // transaction of 128-byte is not atomic across HBM channel
  constexpr static int nChan8B = 8 / sizeof(message_t);
  constexpr static int lastDataChannel = SubGroupSize -nChan8B;
  constexpr static int firstFlagChannel = SubGroupSize/2 -1;
  constexpr static int lastFlagChannel = SubGroupSize -1;

  constexpr static int wireCapacity = (SubGroupSize-nChan8B) * sizeof(message_t);
  constexpr static int wireTransSize = SubGroupSize * sizeof(message_t);

  constexpr static size_t wireElems = wireCapacity /sizeof(T);
  constexpr static size_t wireTransElems = wireTransSize /sizeof(T);

public:
  //
  // sectionSize represents each temporary buffer section for each rank.
  // configurable, in bytes.
  //
  constexpr static size_t nSlot = 8;
  constexpr static size_t ringSize = wireTransSize * nSlot;
  constexpr static size_t maxLaunch = 64 * 64; // 64 wire, 64 SS

  typedef T (* ringPtr)[nSlot][wireTransElems];

private:
  // factor basic communication into another class
  template <int unroll> static inline void loadInput(
      message_t (&v)[unroll], T* src, int nElt
  ) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    if (lid < lastDataChannel) { // TODO: diverge
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireElems + local_off;
        if (off < nElt) {        // TODO: condition branch !
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          lscLoad<SubGroupSize>(v[i], src + off);
#else
          (void)off;
#endif
    }}}
  }

  template <int unroll> static inline void preload(T *ptr) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireTransSize + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscPrefetch<T, sizeof(message_t)/sizeof(T),
        SubGroupSize, PrefetchCacheCtrl>(ptr + off);
#else
      (void)off;
#endif
    }
  }

  template <int unroll> static inline void loadInput(
      message_t (&v)[unroll], T* src
  ) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    if (lid < lastDataChannel) { // XXX: diverge
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireElems + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscLoad<SubGroupSize>(v[i], src + off);
#else
        (void)off;
#endif
    }}
  }

  template <int unroll>
  static inline void insertFlags(
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
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
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
#endif
  }

  template <int unroll> static inline void storeOutput(
      T* dst, message_t (&v)[unroll]
  ) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);
    if (lid < lastDataChannel) { // XXX: Diverge
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireElems + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscStore<SubGroupSize>(dst + off, v[i]);
#else
        (void)off; (void)local_off;
#endif
    }}
  }

  template <int unroll> static inline void storeOutput(
      T* dst, message_t (&v)[unroll], int nElt
  ) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);
    if (lid < lastDataChannel) { // XXX: Fixed diverge
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * wireElems + local_off;
        if (off < nElt) {        // XXX: runtime condition
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          lscStore<SubGroupSize>(dst + off, v[i]);
#endif
    }}}
  }

  // We always push 128-byte packages
  template <int unroll>
  static inline void sendMessages(T* ptr, message_t (&messages)[unroll]) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscStore<SubGroupSize, CommWriteCacheCtrl>(
          ptr + u * wireTransElems + local_off,
          messages[u]
      );
#else
      (void) lid; (void) local_off;
#endif
    }
  }

  template <int unroll>
  static inline bool recvMessages(message_t (&messages)[unroll], T* ptr, uint32_t flag) {
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    bool retry = false;
#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscLoad<SubGroupSize, CommReadCacheCtrl>(
          messages[u],
          ptr + u * wireTransElems + local_off
      );
#else
      (void) lid; (void) local_off;
#endif
      retry |= (lid == firstFlagChannel && messages[u][lastElem] != flag)
        || (lid == lastFlagChannel && messages[u][lastElem] != flag);
    }
    return retry;
  }

  template <int unroll> static inline void accumMessages(
      message_t (&v)[unroll], message_t (&m)[unroll]
  ) {
#if defined(__SYCL_DEVICE_ONLY__)
    using math_t = sycl::vec<T, sizeof(message_t)/sizeof(T)>;
#   pragma unroll
    for (int u = 0; u < unroll; ++ u)
      v[u] = sycl::bit_cast<message_t>(
          sycl::bit_cast<math_t>(m[u]) + sycl::bit_cast<math_t>(v[u])
      );
#endif
  }
public:
  template <int unroll> inline void scatterFar(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireElems;

    auto wireId = sycl::ext::oneapi::this_work_item::
      get_nd_item<1>().get_global_id(0) / SubGroupSize;
    auto y_id = wireId % BiNRanks;
    auto *ptr = ioForFar + y_id * workElems + inputOffInType;

    message_t messages[unroll];

    if (nelems < eltPerPack) {
      loadInput(messages, ptr, nelems);
    } else {
      loadInput(messages, ptr);
    }

    shuffleData(messages);
    insertFlags(messages, flag);
    auto* dst = farScatterSink[wireId][tStep % nSlot];
    sendMessages(dst, messages);
  }

  template <int unroll> inline void pollFarGatherOutput(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto nelems = workLeft / sizeof(T);

    auto wireId = sycl::ext::oneapi::this_work_item::
      get_nd_item<1>().get_global_id(0) / SubGroupSize;
    auto y_id = wireId % BiNRanks;

    constexpr auto eltPerPack = unroll * wireElems;

    message_t messages[unroll];
    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localFarGatherSink[wireId][tStep % nSlot], flag
      );
    } while(sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

    // if (sg.get_local_id()[0] == 15)
    //   cout<<messages[0]<<sycl::endl<<sycl::flush;

    restoreData(messages);

    if (nelems < eltPerPack)
      storeOutput(ioForFar + y_id * workElems + inputOffInType, messages, nelems);
    else
      storeOutput(ioForFar + y_id * workElems + inputOffInType, messages);
  }

  template <int unroll> inline void closeUnifiedPollReduceScatterGather(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    auto wireId = sycl::ext::oneapi::this_work_item::
      get_nd_item<1>().get_global_id(0) / SubGroupSize;
    auto y_id = wireId % BiNRanks;
    auto wireId_g = wireId / BiNRanks * BiNRanks;

    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto nelems = workLeft / sizeof(T);
    constexpr auto eltPerPack = unroll * wireElems;

    message_t v[unroll];

    auto* ioPtr = ioForPeers + y_id * workElems + inputOffInType;
    if (nelems < eltPerPack)
      loadInput(v, ioPtr, nelems);
    else
      loadInput(v, ioPtr);

    message_t messages[unroll];

    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localFarScatterSink[wireId][tStep%nSlot], flag
      );
    } while(sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry));

    shuffleData(v);
    accumMessages(v, messages);

    //------------------------- sub-group diverge 3:1 -------------------
    if (y_id != l_rank) {
      insertFlags(v, flag);
      sendMessages(scatterSink[y_id][wireId_g][tStep%nSlot], v); // 1. xNPeers <scatter>

      bool retry;
      do {
        retry = false;
        retry |= recvMessages(v, localGatherSink[wireId][tStep%nSlot], flag);
      } while(sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), retry)
        );                                          // 4. xNPeers waits for <gather>
    }

    if (y_id == l_rank) {
#     pragma unroll
      for (int i =0; i < NPeers; ++ i) {
        bool retry;
        do {
          retry = false;
          retry |= recvMessages(
              messages, localScatterSink[i][wireId_g][tStep%nSlot], flag);
        } while (sycl::any_of_group(
              sycl::ext::oneapi::this_work_item::get_sub_group(), retry)
          );                                        // 2. wait for <scatter> xNPeers
        accumMessages(v, messages);
      }

      insertFlags(v, flag);

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i)             // 3. signal <gather>
        sendMessages(gatherSink[i][wireId_g][tStep%nSlot], v);
    }
    //-------------------------group converge-------------------

    sendMessages(farGatherSink[wireId][tStep%nSlot], v);
    restoreData(v);

    if (nelems < eltPerPack)
      storeOutput(ioPtr, v, nelems);
    else
      storeOutput(ioPtr, v);
  }

protected:
  BisectPTransmit(
      T* input,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[],
      size_t workSize, int rank, uint32_t seqNo
  ) : workElems(workSize/sizeof(T)), l_rank(rank/2), seqNo(seqNo)
  {
    auto pairRank = rank ^ 1;

    auto* pairBuf0 = peerBuf0[pairRank];
    auto* pairBuf1 = peerBuf1[pairRank];

    auto ioClosePart = [&](T* p) {
      return (rank & 1) ? (T *)((uintptr_t)p + workSize * BiNRanks) : p;
    };
    auto ioFarPart = [&](T* p) {
      return (rank & 1) ? p : (T *)((uintptr_t)p + workSize * BiNRanks);
    };
    auto ipcClosePart = [&](T *p) {
      return p;
    };
    auto ipcFarPart = [&](T* p) {
      return (T *)((uintptr_t)p + ringSize * maxLaunch);
    };

    ioForPeers = ioClosePart(input);
    ioForFar = ioFarPart(input);

    farScatterSink = reinterpret_cast<ringPtr>(ipcFarPart(pairBuf0));
    farGatherSink = reinterpret_cast<ringPtr>(ipcFarPart(pairBuf1));
    localFarScatterSink = reinterpret_cast<ringPtr>(ipcFarPart(scatterBuf));
    localFarGatherSink = reinterpret_cast<ringPtr>(ipcFarPart(gatherBuf));
    localGatherSink = reinterpret_cast<ringPtr>(ipcClosePart(gatherBuf));

    // Indicated by y-id
    for (int i = 0; i < BiNRanks; ++ i) {
      // even: 0, 2, 4, 6
      // odd:  1, 3, 5, 7
      auto r_index = 2 * i + (rank & 1);

      scatterSink[i] = reinterpret_cast<ringPtr>(
          (uintptr_t)ipcClosePart(peerBuf0[r_index]) + l_rank * ringSize);
    }

    // Indicated by next?
    for (int i = 0; i < NPeers; ++ i) {
      int l_next = (l_rank + i + 1) % BiNRanks;
      int next = (rank + i * 2 + 2) % (2 * BiNRanks);

      localScatterSink[i] = reinterpret_cast<ringPtr>(
          (uintptr_t)ipcClosePart(scatterBuf) + l_next * ringSize);

      gatherSink[i] = reinterpret_cast<ringPtr>(
          (uintptr_t)ipcClosePart(peerBuf1[next]) + l_rank * ringSize);
    }
  }

  // --------------------Input/Output buffer-------------------
  // Input partitions
  T* ioForPeers;
  T* ioForFar;
  ssize_t workElems;

  // ---------------------IPC buffers-------------------------
  ringPtr farScatterSink;
  ringPtr farGatherSink;

  // ----------------Sinks, written by remotes----------------
  ringPtr localFarScatterSink;
  ringPtr localFarGatherSink;

  int l_rank;
  uint32_t seqNo;

  ringPtr scatterSink[BiNRanks];
  ringPtr gatherSink[NPeers];

  ringPtr localScatterSink[NPeers];
  ringPtr localGatherSink;
};

template <typename T, int NRanks,
         template <typename, int> class Proto, int SubGroupSize = 16>
class BisectTransmit {
protected:
  using ProtoT = Proto<T, SubGroupSize>;

  using typename ProtoT::message_t;
  using ProtoT::wireCapacityInType;

  using ProtoT::wireTransSize;
  using ProtoT::wireTransElems;

  using ProtoT::loadInput;
  using ProtoT::shuffleData;
  using ProtoT::insertFlags;
  using ProtoT::sendMessages;
  using ProtoT::recvMessages;
  using ProtoT::accumMessages;
  using ProtoT::restoreData;
  using ProtoT::storeOutput;
  using ProtoT::wireCapacity;

  constexpr static int BiNRanks = NRanks / 2;
  constexpr static int NPeers = BiNRanks -1;
  static constexpr int parallel_sg = BiNRanks;

  constexpr static size_t nSlot = 8;
  constexpr static size_t maxLaunch = 64 * 64; // 64 wire, 64 bundle

  //for one set: T ring[BiNRanks][nSlot][maxLaunch/BiNRanks][wireTransElems];
  typedef T (* ringPtr)[nSlot][maxLaunch/BiNRanks][wireTransElems];
  constexpr static size_t ringSize = nSlot * maxLaunch * wireTransSize;

protected:
  BisectTransmit(
      T* input,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[],
      size_t workSize, int rank, uint32_t seqNo
  ) :workElems(workSize / sizeof(T)), l_rank(rank/2), seqNo(seqNo) {
    auto pairRank = rank ^ 1;
    auto* pairBuf0 = peerBuf0[pairRank];
    auto* pairBuf1 = peerBuf1[pairRank];

    auto ioClosePart = [&](T* p) {
      return (rank & 1) ? (T *)((uintptr_t)p + workSize * BiNRanks) : p;
    };
    auto ioFarPart = [&](T* p) {
      return (rank & 1) ? p : (T *)((uintptr_t)p + workSize * BiNRanks);
    };
    auto ipcClosePart = [&](T *p) {
      return p;
    };
    auto ipcFarPart = [&](T* p) {
      return (T *)((uintptr_t)p + ringSize);
    };

    ioForPeers = ioClosePart(input);
    ioForFar = ioFarPart(input);
  }

public:
  inline void scatterFar(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto nelems = workLeft / sizeof(T);

    auto wireId = sycl::ext::oneapi::this_work_item::get_nd_item<1>()
      .get_global_id(0) / SubGroupSize;
    auto rank_id = wireId % BiNRanks;
    auto *ptr = ioForFar + rank_id * workElems + inputOffInType;

    message_t messages;

    // could loop certain steps
    loadInput(messages, ptr, nelems);

    shuffleData(messages);
    insertFlags(messages, flag);

    sendMessages(farScatterSink[tStep % nSlot][wireId], messages);
  }

  inline void pollFarGatherOutput(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto nelems = workLeft / sizeof(T);

    auto wireId = sycl::ext::oneapi::this_work_item::get_nd_item<1>().
      get_global_id(0) / SubGroupSize;
    auto rank_id = wireId % BiNRanks;

    message_t messages;
    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localFarGatherSink[tStep % nSlot][wireId], flag
      );
    } while(sycl::any_of_group(
          sycl::ext::oneapi::this_work_item::get_sub_group(), retry)
      );

    restoreData(messages);
    storeOutput(ioForFar + rank_id * workElems + inputOffInType, messages, nelems);
  }

  inline void closeUnifiedPollReduceScatterGather(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    auto wireId = sycl::ext::oneapi::this_work_item::get_nd_item<1>()
      .get_global_id(0) / SubGroupSize;
    auto rank_id = wireId % BiNRanks;
    auto wireId_g = wireId / BiNRanks * BiNRanks;

    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto nelems = workLeft / sizeof(T);

    message_t v; // more than one in future?
    auto* ioPtr = ioForPeers + rank_id * workElems + inputOffInType;
    loadInput(v, ioPtr, nelems);

    message_t messages;

    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localFarScatterSink[tStep%nSlot][wireId], flag
      );
    } while(
        sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_sub_group(), retry)
    );

    shuffleData(v);
    accumMessages(v, messages);

    // ------------------ diverge N:1 -----------------------
    if (rank_id != l_rank) {
      insertFlags(v, flag);
      sendMessages(scatterSink[rank_id][tStep%nSlot][wireId_g], v);

      bool retry;
      do {
        retry = false;
        retry |= recvMessages(v, localGatherSink[tStep%nSlot][wireId], flag);
      } while(sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), retry)
        );
    } else {
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        bool retry;
        do {
          retry = false;
          retry |= recvMessages(
              messages, localScatterSink[i][tStep%nSlot][wireId_g], flag);
        } while (sycl::any_of_group(
              sycl::ext::oneapi::this_work_item::get_sub_group(), retry)
          );
        accumMessages(v, messages);
      }
      insertFlags(v, flag);
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i)
        sendMessages(gatherSink[i][tStep%nSlot][wireId_g], v);
    } // -------------------- converge -------------------

    sendMessages(farGatherSink[tStep%nSlot][wireId], v);
    restoreData(v);
    storeOutput(ioPtr, v, nelems);
  }

protected:
  // --------------------Input/Output buffer-------------------
  // Input partitions
  T* ioForPeers;
  T* ioForFar;
  ssize_t workElems;

  // ---------------------IPC buffers-------------------------
  ringPtr farScatterSink;
  ringPtr farGatherSink;

  // ----------------Sinks, written by remotes----------------
  ringPtr localFarScatterSink;
  ringPtr localFarGatherSink;

  int l_rank;
  uint32_t seqNo;

  ringPtr scatterSink[BiNRanks];
  ringPtr gatherSink[NPeers];

  ringPtr localScatterSink[NPeers];
  ringPtr localGatherSink;
};
