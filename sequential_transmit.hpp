#pragma once

//
// For those do not have sub-group level independent forward progress
// or PCIE connection without switch (remote polling).
//
//
// When split barrier is not supported, signal become null,
// wait will be both signal and wait.
static inline void sbarrier_signal_compat() {
#if defined(XE_PLUS) && defined(__SYCL_DEVICE_ONLY__) \
  && defined(__SPIR__) && defined(ATOB_SUPPORT)
  sbarrier_signal();
#endif
}

static inline void sbarrier_wait_compat() {
#if defined(XE_PLUS) && defined(__SYCL_DEVICE_ONLY__) \
  && defined(__SPIR__) && defined(ATOB_SUPPORT)
  sbarrier_wait();
#endif
}

template <typename T, int NRanks,
         template <typename, int> class Proto, int SubGroupSize = 16>
class SequentialTransmit : public Proto<T, SubGroupSize> {
protected:
  static constexpr int NPeers = NRanks -1;
  static constexpr int parallel_sg = 1;
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
  constexpr static size_t nSlot = 4;
  constexpr static size_t maxLaunch = 64 * 64;
  constexpr static size_t ringSize = maxLaunch * wireTransSize * nSlot;

  static_assert(ringSize <= 4 * 1024 * 1024ull * SubGroupSize/16);

  typedef T (* ringPtr)[maxLaunch][wireTransElems];

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
    if (workLeft <= 0) return;

    auto wireId = sycl::ext::oneapi::this_work_item::
      get_nd_item<1>().get_global_id(0) / SubGroupSize;

    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireCapacityInType;

    message_t v[NPeers][unroll];
    message_t in[unroll];
    message_t messages[unroll];

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
      sendMessages(scatterSink[i][tStep%nSlot][wireId], v[i]);
    }
#if defined(__enable_device_verbose__)
    if (sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0)
         % SubGroupSize == (SubGroupSize -1))
      sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x\n", v[0][0][0], v[0][0][1], v[0][0][2], v[0][0][3]);
    else
      sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x; ", v[0][0][0], v[0][0][1], v[0][0][2], v[0][0][3]);
#endif

    sbarrier_signal_compat();

    if (nelems < eltPerPack)
      loadInput(in, ioBuffer + inputOffInType, nelems);
    else
      loadInput(in, ioBuffer + inputOffInType);

    sbarrier_wait_compat();

    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localScatterSink[0][tStep%nSlot][wireId], flag);
    } while (sycl::any_of_group(
          sycl::ext::oneapi::this_work_item::get_sub_group(), retry)
      );
#if defined(__enable_device_verbose__)
    if (sycl::ext::oneapi::this_work_item::
        get_nd_item<1>().get_global_id(0) % SubGroupSize == (SubGroupSize -1))
      sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x\n", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
    else
      sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x; ", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
#endif

    shuffleData(in);
    accumMessages(in, messages);

#   pragma unroll
    for (int i = 1; i < NPeers; ++ i) {
      do {
        retry = false;
        retry |= recvMessages(
            messages, localScatterSink[i][tStep%nSlot][wireId], flag);
      } while (sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), retry)
        );
#if defined(__enable_device_verbose__)
      if (sycl::ext::oneapi::this_work_item::
          get_nd_item<1>().get_global_id(0) % SubGroupSize == (SubGroupSize -1))
        sycl::ext::oneapi::experimental::printf("%#x,%#x,%#x,%#x\n", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
      else
        sycl::ext::oneapi::experimental::printf("%#x,%#x,%#x,%#x; ", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
#endif
      accumMessages(in, messages);
    }

    insertFlags(in, flag);

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i)
      sendMessages(gatherSink[i][tStep%nSlot][wireId], in);

    sbarrier_signal_compat();

    restoreData(in);

    if (nelems < eltPerPack)
      storeOutput(ioBuffer + inputOffInType, in, nelems);
    else
      storeOutput(ioBuffer + inputOffInType, in);

    sbarrier_wait_compat();

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      bool retry;
      do {
        retry = false;
        retry |= recvMessages(in, localGatherSink[i][tStep%nSlot][wireId], flag);
      } while(sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), retry)
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

template <typename T, int NRanks,
         template <typename, int> class Proto, int SubGroupSize = 16>
class RingTransmit : public Proto<T, SubGroupSize> {
protected:
  static constexpr int parallel_sg = 1;
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

public:
  constexpr static size_t nSlot = 4;
#if defined(BMG)
  constexpr static size_t maxLaunch = 64 * 20;
#else
  constexpr static size_t maxLaunch = 64 * 64;
#endif
  constexpr static size_t ringSize = maxLaunch * wireTransSize * nSlot;
  static_assert(ringSize <= 4 * 1024 * 1024ull * SubGroupSize/16);

  typedef T (* ringPtr)[nSlot][maxLaunch][wireTransElems];

public:
  RingTransmit(
      T* input, T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[],
      ssize_t workSize,
      int rank,
      uint32_t seqNo   // Serve as flag for checking
  ) : workElems(workSize/sizeof(T)), rank(rank), seqNo(seqNo) {
    auto next = (rank + 1) % NRanks;
    ingress = input;
    egress = input;

    scatterSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf0[next]);
    gatherSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf1[next]);

    localScatterSink = reinterpret_cast<ringPtr>((uintptr_t)scatterBuf);
    localGatherSink = reinterpret_cast<ringPtr>((uintptr_t)gatherBuf);
  }

  RingTransmit(
      T* input, T* output, T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[],
      ssize_t workSize,
      int rank,
      uint32_t seqNo   // Serve as flag for checking
  ) : workElems(workSize/sizeof(T)), rank(rank), seqNo(seqNo) {
    auto next = (rank + 1) % NRanks;
    ingress = input;
    egress = output;

    scatterSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf0[next]);
    gatherSink = reinterpret_cast<ringPtr>((uintptr_t)peerBuf1[next]);

    localScatterSink = reinterpret_cast<ringPtr>((uintptr_t)scatterBuf);
    localGatherSink = reinterpret_cast<ringPtr>((uintptr_t)gatherBuf);
  }

  template <int __dummy__> inline void run(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    runAllreduce(inputOffset, tStep, workLeft);
  }

  static inline void pollMessages(message_t& messages, T* adrs, uint32_t flag) {
#if defined(__enable_device_verbose__)
    uint32_t count = 100000;
#endif
    bool retry;
    do {
      retry = false;
      retry |= recvMessages(messages, adrs, flag);
    } while (sycl::any_of_group(
          sycl::ext::oneapi::this_work_item::get_sub_group(), retry)
#if defined(__enable_device_verbose__)
        && count -- != 0
#endif
        );
#if defined(__enable_device_verbose__)
    auto lane = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    if (lane.get_local_id(0) == 0)
      sycl::ext::oneapi::experimental::printf("Count remain: %d\n", count);

    if (lane.get_global_id(0) % SubGroupSize == SubGroupSize -1)
      sycl::ext::oneapi::experimental::printf(
          "%x,%x,%x,%x\n",
          messages[0], messages[1], messages[2], messages[3]);
    else
      sycl::ext::oneapi::experimental::printf(
          "%x,%x,%x,%x; ",
          messages[0], messages[1], messages[2], messages[3]);
#endif
  }

  inline void send(
      int wireId, int peer, size_t offset,
      uint32_t flag, uint32_t slot, ssize_t nelems
  ) {
    message_t v;
    auto* ptr = ingress + peer * workElems + offset;
    loadInput(v, ptr, nelems);

    shuffleData(v);
    insertFlags(v, flag);

    sendMessages(scatterSink[peer][slot][wireId], v);
    sbarrier_signal_compat();
  }

  inline void loadRecvReduceSend(
      int wireId, int peer, size_t offset,
      uint32_t flag, uint32_t slot, ssize_t nelems
  ) {
    message_t v;
    message_t messages;

    auto* ptr = ingress + peer * workElems + offset; 
    loadInput(v, ptr, nelems);

    sbarrier_wait_compat();
    pollMessages(messages, localScatterSink[peer][slot][wireId], flag);

    shuffleData(v);
    accumMessages(v, messages);
    insertFlags(v, flag);

    sendMessages(scatterSink[peer][slot][wireId], v);
    sbarrier_signal_compat();
  }

  inline void loadRecvReduceSendWrtback(
      int wireId, int peer, size_t offset,
      uint32_t flag, uint32_t slot, ssize_t nelems
  ) {
    message_t v;
    message_t messages;

    auto* ptr = ingress + peer * workElems + offset;
    loadInput(v, ptr, nelems);

    sbarrier_wait_compat();
    pollMessages(messages, localScatterSink[peer][slot][wireId], flag);

    shuffleData(v);
    accumMessages(v, messages);

    insertFlags(v, flag);
    sendMessages(gatherSink[peer][slot][wireId], v);
    sbarrier_signal_compat();
 
    restoreData(v);

    ptr = egress + peer * workElems + offset;
    storeOutput(ptr, v, nelems);
  }

  inline void recvSendWrtback(
      int wireId, int peer, size_t offset,
      uint32_t flag, uint32_t slot, ssize_t nelems
  ) {
    message_t v;

    sbarrier_wait_compat();
    pollMessages(v, localGatherSink[peer][slot][wireId], flag);

    insertFlags(v, flag);
    sendMessages(gatherSink[peer][slot][wireId], v);
    sbarrier_signal_compat();

    restoreData(v);

    auto* ptr = egress + peer * workElems + offset;
    storeOutput(ptr, v, nelems);
  }

  inline void recvWrtback(
      int wireId, int peer, size_t offset,
      uint32_t flag, uint32_t slot, ssize_t nelems
  ) {
    message_t v;

    sbarrier_wait_compat();
    pollMessages(v, localGatherSink[peer][slot][wireId], flag);
    restoreData(v);

    auto* ptr = egress + peer * workElems + offset;
    storeOutput(ptr, v, nelems);
  }

  inline void runAllreduce(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    if (workLeft <= 0) return;

    auto wireId = sycl::ext::oneapi::this_work_item::
      get_nd_item<1>().get_global_id(0) / SubGroupSize;

    auto offset = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto slot = (seqNo + tStep) % nSlot;
    auto nelems = workLeft / sizeof(T);

    uint32_t p_idx = 0;
    int peer = (rank + p_idx) % NRanks;

    // Step 0
    send(wireId, peer, offset, flag, slot, nelems);

    // Step 1 to N-1
#   pragma unroll
    for (int i = 1; i < NRanks -1; ++i) {
      p_idx = (p_idx -1) % NRanks;
      peer = (rank + p_idx) % NRanks;
      loadRecvReduceSend(wireId, peer, offset, flag, slot, nelems);
    }

    // Step N
    p_idx = (p_idx -1) % NRanks;
    peer = (rank + p_idx) % NRanks;
    loadRecvReduceSendWrtback(wireId, peer, offset, flag, slot, nelems);

    // write back
#   pragma unroll
    for (uint32_t i = 1; i < NRanks -1; ++i) {
      p_idx = (p_idx -1) % NRanks; // 0
      peer = (rank + p_idx) % NRanks;
      recvSendWrtback(wireId, peer, offset, flag, slot, nelems);
    }

    p_idx = (p_idx -1) % NRanks;
    peer = (rank + p_idx) % NRanks;
    recvWrtback(wireId, peer, offset, flag, slot, nelems);
  }

  inline void runAllgather(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    if (workLeft <= 0) return;

    auto wireId = sycl::ext::oneapi::this_work_item::
      get_nd_item<1>().get_global_id(0) / SubGroupSize;

    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto slot = (seqNo + tStep) % nSlot;
    auto nelems = workLeft / sizeof(T);

    message_t v;

    uint32_t p_idx = 0;
    int peer = (rank + p_idx) % NRanks;

    auto* ptr = ingress + inputOffInType;
    auto* o_ptr = egress + peer * workElems + inputOffInType;
    loadInput(v, ptr, nelems);

    if (ptr != o_ptr)
      storeOutput(o_ptr, v, nelems);

    shuffleData(v);
    insertFlags(v, flag);
    sendMessages(gatherSink[peer][slot][wireId], v);

    sbarrier_signal_compat();

#   pragma unroll
    for (uint32_t i = 1; i < NRanks -1; ++i) {
      p_idx = (p_idx -1) % NRanks; // 0
      peer = (rank + p_idx) % NRanks;
      recvSendWrtback(wireId, peer, inputOffInType, flag, slot, nelems);
    }

    p_idx = (p_idx -1) % NRanks;
    peer = (rank + p_idx) % NRanks;

    recvWrtback(wireId, peer, inputOffInType, flag, slot, nelems);
  }

protected:
  T* ingress;
  T* egress;

  ssize_t workElems;
  int rank;
  uint32_t seqNo;

  ringPtr scatterSink;
  ringPtr gatherSink;

  ringPtr localScatterSink;
  ringPtr localGatherSink;
};
