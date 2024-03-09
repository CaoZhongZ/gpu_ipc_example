#pragma once

#include "rt64_128.hpp"

template <typename T, int NPeers, int SubGroupSize>
class SimpleTransmit : public rt64_128<T, SubGroupSize> {
protected:
  using super = rt64_128<T, SubGroupSize>;
  using super::wireElems;
  using super::loadInput;
  using super::storeOutput;
  using super::shuffleData;
  using super::restoreData;
  using super::insertFlags;
  using super::sendMessages;
  using super::recvMessages;
  using super::accumMessages;

  using message_t = typename rt64_128<T, SubGroupSize>::message_t;
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

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto flag = scatterStep;
      bool retry;
      do {
        retry = false;
        retry |= recvMessages(
            messages, localScatterSink[i] + sinkOffInType, flag);
      } while(sycl::any_of_group(sg, retry));

#if defined(__enable_sycl_stream__)
      int lane_id = sg.get_local_id();
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

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto flag = gatherStep;
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

  uint32_t scatterStep;
  uint32_t gatherStep;
  int rank;

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};
