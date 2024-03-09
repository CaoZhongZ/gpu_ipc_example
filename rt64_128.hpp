#pragma once

template <typename T, int SubGroupSize>
class rt64_128 {
protected:
  constexpr static auto CommReadCacheCtrl = CacheCtrl::L1UC_L3C;
  constexpr static auto CommWriteCacheCtrl = CacheCtrl::L1UC_L3WB;
  constexpr static auto PrefetchCacheCtrl = CacheCtrl::DEFAULT;

  constexpr static int nReg128B = 128 / SubGroupSize / 4;
  using message_t = sycl::vec<uint32_t, nReg128B>;
  constexpr static int firstElem = 0;
  constexpr static int lastElem = nReg128B -1;

  constexpr static int nChan8B = 8 / sizeof(message_t);
  constexpr static int lastDataChannel = SubGroupSize -nChan8B;
  constexpr static int firstFlagChannel = SubGroupSize/2 -1;
  constexpr static int lastFlagChannel = SubGroupSize -1;
  constexpr static int wireCapacity = (SubGroupSize-nChan8B) * sizeof(message_t);
  constexpr static int wireTransSize = SubGroupSize * sizeof(message_t);

  constexpr static size_t wireElems = wireCapacity /sizeof(T);
  constexpr static size_t wireTransElems = wireTransSize /sizeof(T);

  template <int unroll> inline void loadInput(
      message_t (&v)[unroll], T* src, int nElt
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
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

  template <int unroll> inline void loadInput(
      message_t (&v)[unroll], T* src
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
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

  template <int unroll> inline void preload(T *ptr) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
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
        auto off = i * wireElems + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        lscStore<SubGroupSize>(dst + off, v[i]);
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
        auto off = i * wireElems + local_off;
        if (off < nElt) {        // XXX: runtime condition
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          lscStore<SubGroupSize>(dst + off, v[i]);
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
  inline bool recvMessages(message_t (&messages)[unroll], T* ptr, uint32_t flag) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
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

  template <int unroll> inline void accumMessages(
      message_t (&v)[unroll], message_t (&m)[unroll]
  ) {
    using math_t = sycl::vec<T, sizeof(message_t)/sizeof(T)>;
#   pragma unroll
    for (int u = 0; u < unroll; ++ u)
      arith_v[u] = sycl::bit_cast<message_t>(
          sycl::bit_cast<math_t>(arith_m[u])
          + sycl::bit_cast<math_t>(arith_v[u])
      );
  }

  inline void recvMessage(message_t &message, T* ptr) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
        message, ptr + local_off
    );
#else
    (void) lid; (void) local_off;
#endif
  }
};