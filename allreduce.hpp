#pragma once

#include <sycl/sycl.hpp>
#include <gen_visa_templates.hpp>

#include "transmit.hpp"

template <typename T,
         int NRanks,
         template <typename, int> class Proto,
         template <typename, int, template <typename, int> class, int> class Transmit,
         int SubGroupSize = 16>
struct AllReduce : public Transmit<T, NRanks, Proto, SubGroupSize> {
  using Super = Transmit<T, NRanks, Proto, SubGroupSize>;
  using message_t = typename Super::message_t;
  constexpr static int wireCapacity = Super::wireCapacity;

  AllReduce(
      T* input, size_t nelems, int rank, uint32_t seqNo,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[]
  )
  : Transmit<T, NRanks, Proto, SubGroupSize>(
      input, scatterBuf, gatherBuf, peerBuf0, peerBuf1,
      calcWorkSize(input, nelems * sizeof(T)),
      rank, seqNo
  ), workSize(calcWorkSize(input, nelems * sizeof(T)))
  {}

  static int scatterVerify(
      uint32_t* host, int rank, uint32_t flag, size_t nWorkElemsInInt
  );
  static int stage2Verify(
      T* host, int rank, uint32_t flag, size_t nWorkElemsInInt
  );

  sycl::nd_range<1> getLaunchParam(uint32_t& updateSeqNo) const {
    constexpr uint32_t nThreads = 64; /* TODO: get EU/thread config */
#if defined(PVC)
    constexpr size_t maxSS = 64;
#elif defined(BMG)
    constexpr size_t maxSS = 20;
#elif defined(DG2)
    constexpr size_t maxSS = 32;
#endif
    int w = Super::parallel_sg;
    size_t wirePerSS = nThreads / w;
    size_t nWire = divUp(workSize, wireCapacity);
    size_t nSS = divUp(nWire, wirePerSS);
    auto actualSS = std::min(nSS, maxSS);
    auto nSteps = divUp(nWire, actualSS * wirePerSS);
    updateSeqNo += nSteps;
    //
    // XXX: we over updated sequence number. Should be nSteps / nSlot
    // No harm, but not nice.
    //

    return sycl::nd_range<1>(
      actualSS * wirePerSS * w * SubGroupSize,
      nThreads * SubGroupSize
    );
  }

  static void launch(
      T* input, T* ipcbuf0, T* ipcbuf1, T* const peerbuf0[], T* const peerbuf1[],
      size_t nelems, int rank, uint32_t& step, sycl::queue queue) {
    AllReduce offload(input, nelems, rank, step, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1);

    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
          offload.getLaunchParam(step), offload
        );
    });
  }
  //
  // Found this analogy fascinating:
  //
  // Let correspond sub-group to wire, sequential guaranteed.
  // Bundle sub-groups(wires) into group(cable).
  //
  // Total cables will deliver the full capacity of single loop.
  //
  void operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (
      sycl::nd_item<1> pos
  ) const {
    auto nWires = pos.get_global_range(0) / SubGroupSize;
    auto wireId_x = pos.get_global_id(0) / SubGroupSize / Super::parallel_sg;

    auto loopSize = nWires / Super::parallel_sg * wireCapacity;

    for (size_t gOff = 0, tOff = 0;
        gOff < workSize; gOff += loopSize, ++ tOff) {
      auto wireOff = wireId_x * wireCapacity + gOff;

      ssize_t workLeft = workSize - wireOff;
#if defined(__enable_device_verbose__)
      auto local_id = pos.get_sub_group().get_local_id()[0];
      if (local_id == 0)
        sycl::ext::oneapi::experimental::printf(
            "wireOff %d, workLeft %ld, wireId %d\n", wireOff, workLeft, wireId_x);
#endif
      const_cast<AllReduce *>(this)-> template run<1>(wireOff, tOff, workLeft);
    }
  }

private:
  // TODO: buffer plan and start point calc
  static size_t calcWorkSize(T* input, size_t size) {
    // Input must be message size align
    if ((uintptr_t)input % sizeof(message_t) != 0)
      throw std::logic_error("We only support aligned pointer for now");

    auto nChunks = NRanks;
    auto msgSize = divUp(size, sizeof(message_t));
    auto chunkSize = divUp(msgSize, nChunks);

    if (msgSize * sizeof(message_t) != size || chunkSize * sizeof(message_t) * nChunks > size)
      throw std::logic_error("We don't support non-even divide yet");

    // TODO: Production logic needs every rank chunk
    return chunkSize * sizeof(message_t);
  }

  ssize_t workSize;
};

template <typename T,
         int NRanks,
         template <typename, int, int> class Transmit,
         int SubGroupSize = 16>
struct bisectAllReduce : public Transmit<T, NRanks, SubGroupSize> {
  using Super = Transmit<T, NRanks, SubGroupSize>;
  using message_t = typename Super::message_t;
  constexpr static int unroll = 1 /*NPeers < 4 ? 4 : 2*/;
  constexpr static int wireCapacity = Super::wireCapacity;
  constexpr static int wireTransSize = Super::wireTransSize;

  bisectAllReduce(
      T* input, size_t nelems, int rank, uint32_t seqNo,
      T* scatterBuf, T* gatherBuf, T* const peerBuf0[], T* const peerBuf1[]
  ) :
  Super(input, scatterBuf, gatherBuf, peerBuf0, peerBuf1,
      calcWorkSize(input, nelems * sizeof(T)),
      divUp(calcWorkSize(input, nelems * sizeof(T)), wireCapacity)
      * wireTransSize, rank, seqNo
    ), workSize(calcWorkSize(input, nelems * sizeof(T)))
  {}

  static int stage1Verify(T* host, int rank, uint32_t flag, size_t nelems);
  static int stage2Verify(T* host, int rank, uint32_t flag, size_t nelems);
  static int stage3Verify(T* host, int rank, uint32_t flag, size_t nelems);
  static int stage4Verify(T* host, int rank, uint32_t flag, size_t nelems);
  static int stage5Verify(T* host, int rank, uint32_t flag, size_t nelems);

  void operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (
      sycl::nd_item<1> pos
  ) const {
    auto groupRange = pos.get_group_range()[0];
    int subGroupRange = pos.get_sub_group().get_group_range()[0];

    auto cableCapacity = wireCapacity * subGroupRange;
    auto cableTSize = wireTransSize * subGroupRange;

    auto groupId = pos.get_group().get_group_id()[0];
    auto subGroupId = pos.get_sub_group().get_group_id()[0];

    auto loopSize = groupRange * cableCapacity;
    auto loopTSize = groupRange * cableTSize;

    for (size_t gOff = 0, tOff = 0;
        gOff < workSize; gOff += loopSize, tOff += loopTSize) {
      auto wireOff = groupId * cableCapacity + subGroupId * wireCapacity + gOff;
      auto transOff = groupId * cableTSize + subGroupId * wireTransSize + tOff;
      ssize_t workLeft = workSize - wireOff;
      if (workLeft > 0) {
        const_cast<bisectAllReduce *>(this)->
          template scatterFar<unroll>(wireOff, transOff, workLeft);
        const_cast<bisectAllReduce *>(this)->
          template closePollReduceScatter<unroll>(wireOff, transOff, workLeft);
        const_cast<bisectAllReduce *>(this)->
          template closePollRecvReduceBcast<unroll>(wireOff, transOff, workLeft);
        const_cast<bisectAllReduce *>(this)->
          template pollGatherOutputs<unroll>(wireOff, transOff, workLeft);
        const_cast<bisectAllReduce *>(this)->
          template pollFarGatherOutput<unroll>(wireOff, transOff, workLeft);
      }
    }
  }

private:
  // TODO: buffer plan and start point calc
  static size_t calcWorkSize(T* input, size_t size) {
    // Input must be message size align
    if ((uintptr_t)input % sizeof(message_t) != 0)
      throw std::logic_error("We only support aligned pointer for now");

    auto nChunks = NRanks;
    auto octSize = divUp(size, sizeof(message_t));
    auto chunkSize = divUp(octSize, nChunks);

    if (octSize * sizeof(message_t) != size || chunkSize * sizeof(message_t) * nChunks > size)
      throw std::logic_error("We don't support non-even divide yet");

    // TODO: Production logic needs every rank chunk
    return chunkSize * sizeof(message_t);
  }

  ssize_t workSize;
};

// Best candidate for throughput
template <typename T,
         int NRanks,
         template <typename, int, int> class Transmit = BisectPTransmit,
         int SubGroupSize = 16>
struct bisectPAllReduce : public Transmit<T, NRanks, SubGroupSize> {
  using Super = Transmit<T, NRanks, SubGroupSize>;
  using message_t = typename Super::message_t;
  constexpr static int unroll = 1 /*NPeers < 4 ? 4 : 2*/;
  constexpr static int wireCapacity = Super::wireCapacity;
  constexpr static int wireTransSize = Super::wireTransSize;
  constexpr static int BiNRanks = NRanks/2;

  bisectPAllReduce(
      T* input, size_t nelems, int rank, uint32_t seqNo,
      T* scatterBuf, T* gatherBuf, T* const peerBuf0[], T* const peerBuf1[]
  ) :
  Super(input, scatterBuf, gatherBuf, peerBuf0, peerBuf1,
      calcWorkSize(input, nelems * sizeof(T)), rank, seqNo
    ), workSize(calcWorkSize(input, nelems * sizeof(T)))
  {}

  static int stage1Verify(T* host, int rank, uint32_t flag, size_t nelems);
  static int stage2Verify(T* host, int rank, uint32_t flag, size_t nelems);
  static int stage3Verify(T* host, int rank, uint32_t flag, size_t nelems);
  static int stage4Verify(T* host, int rank, uint32_t flag, size_t nelems);

  sycl::nd_range<1> getLaunchParam(uint32_t& updateSeqNo) const {
    constexpr uint32_t nThreads = 64; /* TODO: get EU/thread config */
    constexpr size_t maxSS = 64;
    int w = NRanks;
    size_t wirePerSS = nThreads  / w;
    size_t nWire = divUp(workSize, wireCapacity);
    size_t nSS = divUp(nWire, wirePerSS);
    auto actualSS = std::min(nSS, maxSS);
    auto nSteps = divUp(nWire, actualSS * wirePerSS);
    updateSeqNo += nSteps;
    //
    // XXX: we over updated sequence number. Should be nSteps / nSlot
    // No harm, but not nice.
    //

    return sycl::nd_range<1>(
      actualSS * wirePerSS * w * SubGroupSize,
      wirePerSS * w * SubGroupSize
    );
  }

  static void launch(
      T* input, T* ipcbuf0, T* ipcbuf1, T* const peerbuf0[], T* const peerbuf1[],
      size_t nelems, int rank, uint32_t& step, sycl::queue queue) {
    bisectPAllReduce offload(input, nelems, rank, step, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1);

    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
          offload.getLaunchParam(step), offload
        );
    });
  }

  //
  // We use linear group configuration because subgroup is linear only
  // And we want to control coalescing as much as possible
  //
  void operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (
      sycl::nd_item<1> pos
  ) const {
    auto nWires = pos.get_global_range(0)/SubGroupSize;
    auto loopSize = nWires / BiNRanks * wireCapacity;
    auto wireId_x = pos.get_global_id(0)/SubGroupSize/BiNRanks;

    for (size_t gOff = 0, tOff = 0;
        gOff < workSize; gOff += loopSize, ++ tOff) {
      auto wireOff = wireId_x * wireCapacity + gOff;
      ssize_t workLeft = workSize - wireOff;

      if (workLeft > 0) { // Y parallel equals bisect Ranks
        const_cast<bisectPAllReduce *>(this)->
          template scatterFar<unroll>(wireOff, tOff, workLeft);
        const_cast<bisectPAllReduce *>(this)->
          template closeUnifiedPollReduceScatterGather<unroll>(wireOff, tOff, workLeft);
        const_cast<bisectPAllReduce *>(this)->
          template pollFarGatherOutput<unroll>(wireOff, tOff, workLeft);
      }
    }
  }

private:
  // TODO: buffer plan and start point calc
  static size_t calcWorkSize(T* input, size_t size) {
    // Input must be message size align
    if ((uintptr_t)input % sizeof(message_t) != 0)
      throw std::logic_error("We only support aligned pointer for now");

    auto nChunks = NRanks;
    auto octSize = divUp(size, sizeof(message_t));
    auto chunkSize = divUp(octSize, nChunks);

    if (octSize * sizeof(message_t) != size || chunkSize * sizeof(message_t) * nChunks > size)
      throw std::logic_error("We don't support non-even divide yet");

    // TODO: Production logic needs every rank chunk
    return chunkSize * sizeof(message_t);
  }

  ssize_t workSize;
};

template <typename T,
         template <typename, int> class Proto,
         template <typename, int, template<typename, int> class, int> class Transmit>
sycl::event testTransmit(
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t step, uint32_t simd, sycl::queue queue
);

template <typename T, template <typename, int, int> class Transmit>
sycl::event testBisectTransmit(
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t step, uint32_t simd, sycl::queue queue
);

template <typename T>
sycl::event testTransmit(
    std::string transmitType,
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t step, uint32_t subgroup, sycl::queue queue);

template<typename T> int verifyTransmit(
    T* host, T* host2,
    uint32_t step, int rank, int world, uint32_t simd, size_t nelems
);
