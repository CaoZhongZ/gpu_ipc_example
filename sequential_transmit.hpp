#pragma once

//
// For those do not have sub-group level independent forward progress
// and PCIE connection without switch
//

template <typename T, int NRanks,
         template <typename, int> class Proto, int SubGroupSize = 16>
class SequentialTransmit : public Proto<T, SubGroupSize> {
protected:
  static constexpr int NPeers = NRanks -1;
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

public:
  constexpr static size_t nSlot = 8;
  constexpr static size_t ringSize = wireTransSize * nSlot;
  constexpr static size_t maxLaunch = 64 * 64;

  typedef T (* ringPtr)[nSlot][wireTransElems];

public:
  SequentialTransmit(
      T* input, T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[],
      ssize_t workSize,
      int rank,
      uint32_t seqNo   // Serve as flag for checking
  ) : workElems(workSize/sizeof(T)), rank(rank), seqNo(seqNo) {
    ioBuffer = input + rank * workElems;

    for (int i = 0; i < NPeers; ++ i) {
      int next = (rank + i + 1) % NRanks;

      scatterSink[i] = reinterpret_cast<ringPtr>(
          (uintptr_t)peerBuf0[next] + rank * ringSize);
      gatherSink[i] = reinterpret_cast<ringPtr>(
          (uintptr_t)peerBuf1[next] + rank * ringSize);

      localScatterSink[i] = reinterpret_cast<ringPtr>(
          (uintptr_t)scatterBuf + next * ringSize);
      localGatherSink[i] = reinterpret_cast<ringPtr>(
          (uintptr_t)gatherBuf + next * ringSize);

      ioForPeers[i] = input + next * workElems;
    }
  }

  template <int unroll>
  inline void run(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    auto wireId = sycl::ext::oneapi::experimental::
      this_nd_item<1>().get_global_id(0) / SubGroupSize;

    auto wireId_xrank = wireId;
    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireCapacityInType;

    // scatter
    message_t v[NPeers][unroll];
    if (nelems < eltPerPack) {
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i)
        loadInput(v[i], ioForPeers[i] + inputOffInType, nelems);
    } else {
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i)
        loadInput(v[i], ioForPeers[i] + inputOffInType);
    }

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      shuffleData(v[i]);
      insertFlags(v[i], flag);
      sendMessages(scatterSink[i][wireId_xrank][tStep%nSlot], v[i]);
    }

    // gather, reduce and broadcast
    message_t messages[unroll];
    message_t in[unroll];

    if (nelems < eltPerPack)
      loadInput(in, ioBuffer + inputOffInType, nelems);
    else
      loadInput(in, ioBuffer + inputOffInType);

    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localScatterSink[i][wireId_xrank][tStep%nSlot], flag);
    } while (sycl::any_of_group(
          sycl::ext::oneapi::experimental::this_sub_group(), retry)
      );
#if defined(__enable_device_verbose__)
    if (sycl::ext::oneapi::experimental::
        this_nd_item<1>().get_global_id(0) % SubGroupSize == (SubGroupSize -1))
      sycl::ext::oneapi::experimental::printf("%#x,%#x\n", messages[0][0], messages[0][1]);
    else
      sycl::ext::oneapi::experimental::printf("%#x,%#x; ", messages[0][0], messages[0][1]);
#endif

    shuffleData(in);
    accumMessages(in, messages);

#if defined(__enable_device_verbose__)
    if (sycl::ext::oneapi::experimental::this_nd_item<1>().get_global_id(0)
         % SubGroupSize == (SubGroupSize -1))
      sycl::ext::oneapi::experimental::printf("%#x,%#x\n", in[0][0], in[0][1]);
    else
      sycl::ext::oneapi::experimental::printf("%#x,%#x; ", in[0][0], in[0][1]);
#endif

#   pragma unroll
    for (int i = 1; i < NPeers; ++ i) {
      do {
        retry = false;
        retry |= recvMessages(
            messages, localScatterSink[i][wireId_xrank][tStep%nSlot], flag);
      } while (sycl::any_of_group(
            sycl::ext::oneapi::experimental::this_sub_group(), retry)
        );
#if defined(__enable_device_verbose__)
      if (sycl::ext::oneapi::experimental::
          this_nd_item<1>().get_global_id(0) % SubGroupSize == (SubGroupSize -1))
        sycl::ext::oneapi::experimental::printf("%#x,%#x\n", messages[0][0], messages[0][1]);
      else
        sycl::ext::oneapi::experimental::printf("%#x,%#x; ", messages[0][0], messages[0][1]);
#endif
      accumMessages(in, messages);
    }

    insertFlags(in, flag);

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i)
      sendMessages(gatherSink[i][wireId_xrank][tStep%nSlot], in);

    restoreData(in);

    if (nelems < eltPerPack)
      storeOutput(ioBuffer + inputOffInType, in, nelems);
    else
      storeOutput(ioBuffer + inputOffInType, in);

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      bool retry;
      message_t in[unroll];

      do {
        retry = false;
        retry |= recvMessages(in, localGatherSink[i][wireId_xrank][tStep%nSlot], flag);
      } while(sycl::any_of_group(
            sycl::ext::oneapi::experimental::this_sub_group(), retry)
        );
      auto ptr = ioForPeers[i] + inputOffInType;
      restoreData(in);

      if (nelems < eltPerPack)
        storeOutput(ptr, in, nelems);
      else
        storeOutput(ptr, in);
    }
  }

protected:
  T* ioBuffer;
  T* ioForPeers[NPeers];

  ssize_t workElems;

  int rank;
  uint32_t seqNo;

  ringPtr scatterSink[NPeers];
  ringPtr gatherSink[NPeers];

  ringPtr localScatterSink[NPeers];
  ringPtr localGatherSink[NPeers];
};
