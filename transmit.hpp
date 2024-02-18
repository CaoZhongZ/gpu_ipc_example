#pragma once

#include <sycl/sycl.hpp>
#include <gen_visa_templates.hpp>

#define divUp(x, m)  \
  ((x + m -1) / m)

#define alignUp(x, c) \
  (divUp(x, c) * c)

template <typename T, int NPeers, int SubGroupSize>
class smallTransmit {
  // first row contains data, second row, flags
protected:
  using message_t = sycl::vec<uint32_t, 2>;
  static constexpr int dataElem = 0;
  static constexpr int flagElem = 1;

  constexpr static int wireCapacity = SubGroupSize * sizeof(message_t) / 2 / sizeof(T);
  constexpr static int wireTransSize = SubGroupSize * sizeof(message_t) / sizeof(T);

public:
  smallTransmit(
      int rank,
      uint32_t seqNo   // Serve as flag for checking
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  ) : scatterStep(seqNo), gatherStep(seqNo + 1), rank(rank)
#if defined(__enable_sycl_stream__)
  , cout(cout)
#endif
  {}

  // load first row of registers
  template <int unroll> inline void loadInput(
      message_t (&v)[unroll], T* const src, int nElt
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int i = 0; i < unroll; ++ i) {
      auto off = i * wireCapacity + local_off;
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
      auto off = i * wireCapacity + local_off;
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
      auto off = i * wireCapacity + local_off;
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
      auto off = i * wireCapacity + local_off;
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
          ptr + u * wireTransSize + local_off,
          messages[u]
      );
#else
      (void) lid; (void) local_off;
#endif
    }
  }

  template <int unroll>
  inline void recvMessages(message_t (&messages)[unroll], T* ptr) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
          messages[u],
          ptr + u * wireTransSize + local_off
      );
#else
      (void) lid; (void) local_off;
#endif
    }
  }

  // Scatter local message to peers
  template <int unroll>
  inline void scatter(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireCapacity;
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
        insertFlags(messages[i], scatterStep);

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
        insertFlags(messages[i], scatterStep);

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

    constexpr auto eltPerPack = unroll * wireCapacity;
    if (nelems < eltPerPack) {
      loadInput(v, ioBuffer + inputOffInType, nelems);
    } else {
      loadInput(v, ioBuffer + inputOffInType);
    }

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto flag = scatterStep;
      bool retry;
      do {
        retry = false;
#       pragma unroll
        for (int u = 0; u < unroll; ++ u) {
          recvMessages(messages, localScatterSink[i] + sinkOffInType);
          retry |= (messages[u][flagElem] != flag);
        }
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

    insertFlags(v, gatherStep);

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

    constexpr auto eltPerPack = unroll * wireCapacity;

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
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
          retry |= messages[u][flagElem] != flag;
        }
      } while(sycl::any_of_group(sg, retry));

#if defined(__enable_sycl_stream__)
      if (sg.get_local_id() == 0)
        cout<<"["<<rank<<"] Message: "<<sycl::hex<<messages[0][1]
          <<sycl::endl<<sycl::flush;
#endif
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

  uint32_t scatterStep;
  uint32_t gatherStep;
  int rank;

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

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
  constexpr static size_t wireCapacity = (SubGroupSize-nChan8B)*sizeof(message_t)/sizeof(T);
  constexpr static size_t wireTransSize = SubGroupSize*sizeof(message_t)/sizeof(T);

public:
  SimpleTransmit(
      int rank,
      uint32_t seqNo  // Serve as flag for checking
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  ) : scatterStep(seqNo), gatherStep(seqNo + 1), rank(rank)
#if defined(__enable_sycl_stream__)
  , cout(cout)
#endif
  {}

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
        auto off = i * wireCapacity + local_off;
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
        auto off = i * wireCapacity + local_off;
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
        auto off = i * wireCapacity + local_off;
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
        auto off = i * wireCapacity + local_off;
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
          ptr + u * wireTransSize + local_off,
          messages[u]
      );
#else
      (void) lid; (void) local_off;
#endif
    }
  }

  template <int unroll>
  inline void recvMessages(message_t (&messages)[unroll], T* ptr) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto lid = sg.get_local_id()[0];
    int local_off = lid * sizeof(message_t) / sizeof(T);

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
      lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
          messages[u],
          ptr + u * wireTransSize + local_off
      );
#else
      (void) lid; (void) local_off;
#endif
    }
  }

  // Scatter local message to peers
  template <int unroll>
  inline void scatter(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireCapacity;
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

        auto* ptr = ioForPeers[i] + inputOffInType;
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
        auto* ptr = ioForPeers[i] + inputOffInType;
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
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    message_t v[unroll];        // Input
    message_t messages[unroll]; // Scraps from remote

    auto nelems = workLeft / sizeof(T);
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);

    auto inPtr = ioBuffer + inputOffInType;
    constexpr auto eltPerPack = unroll * wireCapacity;

    if (nelems < eltPerPack) {
      loadInput(v, inPtr, nelems);
    } else {
      loadInput(v, inPtr);
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

#if defined(__enable_sycl_stream__)
      if (lane_id == firstFlagChannel || lane_id == lastFlagChannel) {
        cout<<"["<<rank<<","<<lane_id<<"]";
        for (int u = 0; u < unroll; ++ u)
          cout<<sycl::hex<<messages[u]<<"; ";
        cout<<sycl::endl<<sycl::flush;
      }
#endif

      restoreData(messages);

#if 1 //!defined(__SYCL_DEVICE_ONLY__)
      using math_t = sycl::vec<T, sizeof(message_t)/sizeof(T)>;
      auto arith_v = reinterpret_cast<math_t (&)[unroll]>(v);
      auto arith_m = reinterpret_cast<math_t (&)[unroll]>(messages);
#endif

      if (lane_id < lastDataChannel) {  // XXX: Fixed diverge
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
          arith_v[u] += arith_m[u];
#endif
        }
      }
    }

    // write back locally before shuffle data
    if (nelems < eltPerPack) {
      storeOutput(inPtr, v, nelems);
    } else {
      storeOutput(inPtr, v);
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
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireCapacity;
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

  uint32_t scatterStep;
  uint32_t gatherStep;
  int rank;

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};
