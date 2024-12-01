#pragma once

//
// For those do not have sub-group level independent forward progress
// or PCIE connection without switch (remote polling).
//

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

#if defined(ATOB_SUPPORT)
  template <int unroll>
  inline void run_xe2_plus(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    if (workLeft <= 0) return;

    auto wireId = sycl::ext::oneapi::experimental::
      this_nd_item<1>().get_global_id(0) / SubGroupSize;

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

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    sbarrier_signal();
#endif

#if defined(__enable_device_verbose__)
    if (sycl::ext::oneapi::experimental::this_nd_item<1>().get_global_id(0)
         % SubGroupSize == (SubGroupSize -1))
      sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x\n", v[0][0][0], v[0][0][1], v[0][0][2], v[0][0][3]);
    else
      sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x; ", v[0][0][0], v[0][0][1], v[0][0][2], v[0][0][3]);
#endif

    if (nelems < eltPerPack)
      loadInput(in, ioBuffer + inputOffInType, nelems);
    else
      loadInput(in, ioBuffer + inputOffInType);

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    sbarrier_wait();
#endif

    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localScatterSink[0][tStep%nSlot][wireId], flag);
    } while (sycl::any_of_group(
          sycl::ext::oneapi::experimental::this_sub_group(), retry)
      );
#if defined(__enable_device_verbose__)
    if (sycl::ext::oneapi::experimental::
        this_nd_item<1>().get_global_id(0) % SubGroupSize == (SubGroupSize -1))
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
            sycl::ext::oneapi::experimental::this_sub_group(), retry)
        );
#if defined(__enable_device_verbose__)
      if (sycl::ext::oneapi::experimental::
          this_nd_item<1>().get_global_id(0) % SubGroupSize == (SubGroupSize -1))
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

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    sbarrier_signal();
#endif

    restoreData(in);

    if (nelems < eltPerPack)
      storeOutput(ioBuffer + inputOffInType, in, nelems);
    else
      storeOutput(ioBuffer + inputOffInType, in);

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    sbarrier_wait();
#endif

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      bool retry;
      do {
        retry = false;
        retry |= recvMessages(in, localGatherSink[i][tStep%nSlot][wireId], flag);
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
#elif defined(XE_PLUS)
  template <int unroll>
  inline void run_sbarrier(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    auto wireId = sycl::ext::oneapi::experimental::
      this_nd_item<1>().get_global_id(0) / SubGroupSize;

    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireCapacityInType;

    message_t v[NPeers][unroll];
    message_t in[unroll];
    message_t messages[unroll];

    if (workLeft > 0) {
      if (nelems < eltPerPack) {
#       pragma unroll
        for (int i = 0; i < NPeers; ++ i)
          loadInput(v[i], ioForPeers[i] + inputOffInType, nelems);
      } else {
#       pragma unroll
        for (int i = 0; i < NPeers; ++ i)
          loadInput(v[i], ioForPeers[i] + inputOffInType);
      }

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        shuffleData(v[i]);
        insertFlags(v[i], flag);
        sendMessages(scatterSink[i][tStep%nSlot][wireId], v[i]);
      }
    }

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    sbarrier_signal();
#endif

    if (workLeft > 0) {
#if defined(__enable_device_verbose__)
      if (sycl::ext::oneapi::experimental::this_nd_item<1>().get_global_id(0)
           % SubGroupSize == (SubGroupSize -1))
        sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x\n", v[0][0][0], v[0][0][1], v[0][0][2], v[0][0][3]);
      else
        sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x; ", v[0][0][0], v[0][0][1], v[0][0][2], v[0][0][3]);
#endif

      if (nelems < eltPerPack)
        loadInput(in, ioBuffer + inputOffInType, nelems);
      else
        loadInput(in, ioBuffer + inputOffInType);
    }

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    sbarrier_wait();
#endif

    if (workLeft > 0) {
      bool retry;
      do {
        retry = false;
        retry |= recvMessages(
            messages, localScatterSink[0][tStep%nSlot][wireId], flag);
      } while (sycl::any_of_group(
            sycl::ext::oneapi::experimental::this_sub_group(), retry)
        );
#if   defined(__enable_device_verbose__)
      if (sycl::ext::oneapi::experimental::
          this_nd_item<1>().get_global_id(0) % SubGroupSize == (SubGroupSize -1))
        sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x\n", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
      else
        sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x; ", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
#endif

      shuffleData(in);
      accumMessages(in, messages);

#     pragma unroll
      for (int i = 1; i < NPeers; ++ i) {
        do {
          retry = false;
          retry |= recvMessages(
              messages, localScatterSink[i][tStep%nSlot][wireId], flag);
        } while (sycl::any_of_group(
              sycl::ext::oneapi::experimental::this_sub_group(), retry)
          );
#if   defined(__enable_device_verbose__)
        if (sycl::ext::oneapi::experimental::
            this_nd_item<1>().get_global_id(0) % SubGroupSize == (SubGroupSize -1))
          sycl::ext::oneapi::experimental::printf("%#x,%#x,%#x,%#x\n", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
        else
          sycl::ext::oneapi::experimental::printf("%#x,%#x,%#x,%#x; ", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
#endif
        accumMessages(in, messages);
      }

      insertFlags(in, flag);

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i)
        sendMessages(gatherSink[i][tStep%nSlot][wireId], in);
    }

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    sbarrier_signal();
#endif

    if (workLeft > 0) {

      restoreData(in);

      if (nelems < eltPerPack)
        storeOutput(ioBuffer + inputOffInType, in, nelems);
      else
        storeOutput(ioBuffer + inputOffInType, in);
    }

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    sbarrier_wait();
#endif

    if (workLeft > 0) {
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        bool retry;
        do {
          retry = false;
          retry |= recvMessages(in, localGatherSink[i][tStep%nSlot][wireId], flag);
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
  }
#else
  template <int unroll>
  inline void run_pre_xe(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
    auto wireId = sycl::ext::oneapi::experimental::
      this_nd_item<1>().get_global_id(0) / SubGroupSize;

    auto inputOffInType = inputOffset / sizeof(T);
    auto flag = seqNo + tStep / nSlot;
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireCapacityInType;

    message_t v[NPeers][unroll];
    message_t in[unroll];
    message_t messages[unroll];

    if (workLeft > 0) {
      if (nelems < eltPerPack) {
#       pragma unroll
        for (int i = 0; i < NPeers; ++ i)
          loadInput(v[i], ioForPeers[i] + inputOffInType, nelems);
      } else {
#       pragma unroll
        for (int i = 0; i < NPeers; ++ i)
          loadInput(v[i], ioForPeers[i] + inputOffInType);
      }

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        shuffleData(v[i]);
        insertFlags(v[i], flag);
        sendMessages(scatterSink[i][tStep%nSlot][wireId], v[i]);
      }
#if defined(__enable_device_verbose__)
      if (sycl::ext::oneapi::experimental::this_nd_item<1>().get_global_id(0)
           % SubGroupSize == (SubGroupSize -1))
        sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x\n", v[0][0][0], v[0][0][1], v[0][0][2], v[0][0][3]);
      else
        sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x; ", v[0][0][0], v[0][0][1], v[0][0][2], v[0][0][3]);
#endif

      if (nelems < eltPerPack)
        loadInput(in, ioBuffer + inputOffInType, nelems);
      else
        loadInput(in, ioBuffer + inputOffInType);
    }

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    barrier();
#endif

    if (workLeft > 0) {
      bool retry;
      do {
        retry = false;
        retry |= recvMessages(
            messages, localScatterSink[0][tStep%nSlot][wireId], flag);
      } while (sycl::any_of_group(
            sycl::ext::oneapi::experimental::this_sub_group(), retry)
        );
#if   defined(__enable_device_verbose__)
      if (sycl::ext::oneapi::experimental::
          this_nd_item<1>().get_global_id(0) % SubGroupSize == (SubGroupSize -1))
        sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x\n", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
      else
        sycl::ext::oneapi::experimental::printf("%x,%x,%x,%x; ", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
#endif

      shuffleData(in);
      accumMessages(in, messages);

#     pragma unroll
      for (int i = 1; i < NPeers; ++ i) {
        do {
          retry = false;
          retry |= recvMessages(
              messages, localScatterSink[i][tStep%nSlot][wireId], flag);
        } while (sycl::any_of_group(
              sycl::ext::oneapi::experimental::this_sub_group(), retry)
          );
#if   defined(__enable_device_verbose__)
        if (sycl::ext::oneapi::experimental::
            this_nd_item<1>().get_global_id(0) % SubGroupSize == (SubGroupSize -1))
          sycl::ext::oneapi::experimental::printf("%#x,%#x,%#x,%#x\n", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
        else
          sycl::ext::oneapi::experimental::printf("%#x,%#x,%#x,%#x; ", messages[0][0], messages[0][1], messages[0][2], messages[0][3]);
#endif
        accumMessages(in, messages);
      }

      insertFlags(in, flag);

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i)
        sendMessages(gatherSink[i][tStep%nSlot][wireId], in);

      restoreData(in);

      if (nelems < eltPerPack)
        storeOutput(ioBuffer + inputOffInType, in, nelems);
      else
        storeOutput(ioBuffer + inputOffInType, in);
    }

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    barrier();
#endif

    if (workLeft > 0) {
#     pragma unroll
      for (int i = 0; i < NPeers; ++ i) {
        bool retry;
        do {
          retry = false;
          retry |= recvMessages(in, localGatherSink[i][tStep%nSlot][wireId], flag);
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
  }
#endif

  template <int unroll>
  inline void run(
      size_t inputOffset, size_t tStep, ssize_t workLeft
  ) {
#if defined(ATOB_SUPPORT)
    run_xe2_plus(inputOffset, tStep, workLeft);
#elif defined(XE_PLUS)
    run_sbarrier(inputOffset, tStep, workLeft);
#else
    run_pre_xe(inputOffset, tStep, workLeft);
#endif
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
