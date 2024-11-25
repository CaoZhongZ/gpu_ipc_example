#pragma once

template <typename T, int SubGroupSize> struct Rt64_PCIE {
  using message_t = sycl::vec<uint32_t, 4>;
#if defined(__SYCL_DEVICE_ONLY__)
  using inner_t = message_t::vector_t;
#endif

  constexpr static size_t wireCapacity = SubGroupSize * sizeof(message_t) / 2;
  constexpr static size_t wireTransSize = SubGroupSize * sizeof(message_t);

  constexpr static int wireCapacityInType = wireCapacity / sizeof(T);
  constexpr static int wireTransElems = wireTransSize / sizeof(T);

  constexpr static auto CommReadCacheCtrl = CacheCtrl::L1UC_L3C;
  constexpr static auto CommWriteCacheCtrl = CacheCtrl::L1UC_L3WB;

  // load first row of registers
  template <int unroll> static inline void loadInput(
      message_t (&v)[unroll], T* src, int nElt
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(uint32_t) / sizeof(T);

#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireCapacityInType + local_off;
      if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16)
          asm volatile ("\n" // Add this partial load to tvisa
              "lsc_load.ugm.df.df (M1, 16) %0:d32x2 flat[%1]:a64\n"
              : "=rw"(reinterpret_cast<inner_t &>(v[i])) : "rw"(src + off));
        if constexpr (SubGroupSize == 32)
          asm volatile ("\n" // Add this partial load to tvisa
              "lsc_load.ugm.df.df (M1, 32) %0:d32x2 flat[%1]:a64\n"
              : "=rw"(reinterpret_cast<inner_t &>(v[i])) : "rw"(src + off));
#endif
    }}
  }

  template <int unroll> static inline void loadInput(
      message_t (&v)[unroll], T* src
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(uint32_t) / sizeof(T);

#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireCapacityInType + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      if constexpr (SubGroupSize == 16)
        asm volatile ("\n" // Add this partial load to tvisa
            "lsc_load.ugm.df.df (M1, 16) %0:d32x2 flat[%1]:a64\n"
            : "=rw"(reinterpret_cast<inner_t &>(v[i])) : "rw"(src + off));
      if constexpr (SubGroupSize == 32)
        asm volatile ("\n" // Add this partial load to tvisa
            "lsc_load.ugm.df.df (M1, 32) %0:d32x2 flat[%1]:a64\n"
            : "=rw"(reinterpret_cast<inner_t &>(v[i])) : "rw"(src + off));
#else
      (void)off;
#endif
    }
  }

  template <int unroll>
  static inline void shuffleData(message_t (& messages)[unroll]) {}
  template <int unroll>
  static inline void restoreData(message_t (& messages)[unroll]) {}

  template <int unroll> inline void accumMessages(
      message_t (&v)[unroll], message_t (&m)[unroll]
  ) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    using math_t = sycl::vec<T, sizeof(uint32_t)/sizeof(T)>;
#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
      v[u][0] = sycl::bit_cast<uint32_t>(
          sycl::bit_cast<math_t>(m[u][0]) + sycl::bit_cast<math_t>(v[u][0])
      );
      v[u][1] = sycl::bit_cast<uint32_t>(
          sycl::bit_cast<math_t>(m[u][1]) + sycl::bit_cast<math_t>(v[u][1])
      );
    }
#endif
  }

  //Insert flags to second row
  template <int unroll>
  static inline void insertFlags(
      message_t (& messages)[unroll], uint32_t flag
  ) {
#   pragma unroll
    for (int i = 0; i < unroll; ++ i)
      messages[i][2] = messages[i][3] = flag;
  }

  template <int unroll> inline void storeOutput(
      T* dst, message_t (&v)[unroll]
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(uint32_t) / sizeof(T);
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireCapacityInType + local_off;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      if constexpr (SubGroupSize == 16)
        asm volatile ("\n"
            "lsc_store.ugm.df.df (M1, 16) flat[%0]:a64 %1:d32x2\n"
            :: "rw"(dst + off), "rw"(reinterpret_cast<inner_t &>(v[i])));
      if constexpr (SubGroupSize == 32)
        asm volatile ("\n"
            "lsc_store.ugm.df.df (M1, 32) flat[%0]:a64 %1:d32x2\n"
            :: "rw"(dst + off), "rw"(reinterpret_cast<inner_t &>(v[i])));
#else
      (void)off;
#endif
    }
  }

  template <int unroll> static inline void storeOutput(
      T* dst, message_t (&v)[unroll], int nElt
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(uint32_t) / sizeof(T);
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireCapacityInType + local_off;
      if (off < nElt) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        if constexpr (SubGroupSize == 16)
          asm volatile ("\n"
              "lsc_store.ugm.df.df (M1, 16) flat[%0]:a64 %1:d32x2\n"
              :: "rw"(dst + off), "rw"(reinterpret_cast<inner_t &>(v[i])));
        if constexpr (SubGroupSize == 32)
          asm volatile ("\n"
              "lsc_store.ugm.df.df (M1, 32) flat[%0]:a64 %1:d32x2\n"
              :: "rw"(dst + off), "rw"(reinterpret_cast<inner_t &>(v[i])));
#endif
    }}
  }

  // We always push 128-byte packages
  template <int unroll>
  static inline void sendMessages(T* ptr, message_t (&messages)[unroll]) {
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
      (void)local_off;
#endif
    }
  }

  template <int unroll>
  static inline bool recvMessages(message_t (&messages)[unroll], T* ptr, int flag) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    bool retry = false;

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscLoad<SubGroupSize, CommReadCacheCtrl>(
          messages[u], ptr + u * wireTransElems + local_off
      );
#else
      (void)local_off;
#endif
      retry |= (messages[u][2] != flag) || (messages[u][3] != flag);
    }

    return retry;
  }
};

template <typename T, int SubGroupSize> struct Rt64 {
  using message_t = sycl::vec<uint32_t, 2>;
  static constexpr int dataElem = 0;
  static constexpr int flagElem = 1;

  constexpr static size_t wireCapacity = SubGroupSize * sizeof(message_t) / 2;
  constexpr static size_t wireTransSize = SubGroupSize * sizeof(message_t);

  constexpr static int wireCapacityInType = wireCapacity / sizeof(T);
  constexpr static int wireTransElems = wireTransSize / sizeof(T);

  constexpr static auto CommReadCacheCtrl = CacheCtrl::L1UC_L3C;
  constexpr static auto CommWriteCacheCtrl = CacheCtrl::L1UC_L3WB;

  // load first row of registers
  template <int unroll> static inline void loadInput(
      message_t (&v)[unroll], T* src, int nElt
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(uint32_t) / sizeof(T);

#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireCapacityInType + local_off;
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
        v[i][dataElem] = src[off];
#endif
    }}
  }

  template <int unroll> static inline void loadInput(
      message_t (&v)[unroll], T* src
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(uint32_t) / sizeof(T);

#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireCapacityInType + local_off;
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
      v[i][dataElem] = src[off];
#endif
    }
  }

  template <int unroll>
  static inline void shuffleData(message_t (& messages)[unroll]) {}
  template <int unroll>
  static inline void restoreData(message_t (& messages)[unroll]) {}

  template <int unroll> inline void accumMessages(
      message_t (&v)[unroll], message_t (&m)[unroll]
  ) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    using math_t = sycl::vec<T, sizeof(uint32_t)/sizeof(T)>;
#   pragma unroll
    for (int u = 0; u < unroll; ++ u)
      v[u][0] = sycl::bit_cast<uint32_t>(
          sycl::bit_cast<math_t>(m[u][0]) + sycl::bit_cast<math_t>(v[u][0])
      );
#endif
  }

  //Insert flags to second row
  template <int unroll>
  static inline void insertFlags(
      message_t (& messages)[unroll], uint32_t flag
  ) {
#   pragma unroll
    for (int i = 0; i < unroll; ++ i)
      messages[i][flagElem] = flag;
  }

  template <int unroll> inline void storeOutput(
      T* dst, message_t (&v)[unroll]
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(uint32_t) / sizeof(T);
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireCapacityInType + local_off;
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

  template <int unroll> static inline void storeOutput(
      T* dst, message_t (&v)[unroll], int nElt
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(uint32_t) / sizeof(T);
#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireCapacityInType + local_off;
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
  static inline void sendMessages(T* ptr, message_t (&messages)[unroll]) {
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
  static inline bool recvMessages(message_t (&messages)[unroll], T* ptr, int flag) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

    bool retry = false;

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscLoad<SubGroupSize, CommReadCacheCtrl>(
          messages[u], ptr + u * wireTransElems + local_off
      );
#else
      (void) lid; (void) local_off;
#endif
      retry |= (messages[u][flagElem] != flag);
    }

    return retry;
  }
};
