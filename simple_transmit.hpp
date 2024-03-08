#pragma once

#include "rt64_128.hpp"

template <typename T, int NPeers, int SubGroupSize>
class SimpleTransmit : rt64_128<T, SubGroupSize> {
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
      insertFlags(messages[i], scatterStep);

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
    constexpr auto eltPerPack = unroll * wireElems;

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
          recvMessage(messages[u], localScatterSink[i] + sinkOffInType);
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

    constexpr auto eltPerPack = unroll * wireElems;
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
          recvMessage(messages[u], localGatherSink[i] + sinkOffInType);
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
