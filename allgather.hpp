#pragma once

#include <sycl/sycl.hpp>
#include <gen_visa_templates.hpp>

#include "transmit.hpp"

template <typename T,
         int NRanks,
         template <typename, int> class Proto,
         template <typename, int, template <typename, int> class, int> class Transmit,
         int SubGroupSize = 16>
struct AllGather : public Transmit<T, NRanks, Proto, SubGroupSize> {
  using Super = Transmit<T, NRanks, Proto, SubGroupSize>;
  using message_t = typename Super::message_t;
  constexpr static int wireCapacity = Super::wireCapacity;
  using Super::runAllgather;

  AllGather(
      T* input, T* output, size_t nelems, int rank, uint32_t seqNo,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[]
  )
  : Transmit<T, NRanks, Proto, SubGroupSize>(
      input, output, scatterBuf, gatherBuf, peerBuf0, peerBuf1,
      calcWorkSize(input, output, nelems * sizeof(T)),
      rank, seqNo
  ), workSize(calcWorkSize(input, output, nelems * sizeof(T)))
  {}

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
      T* input, T* output, T* ipcbuf0, T* ipcbuf1,
      T* const peerbuf0[], T* const peerbuf1[],
      size_t nelems, int rank, uint32_t& step, sycl::queue queue) {
    AllGather offload(input, output, nelems, rank, step,
        ipcbuf0, ipcbuf1, peerbuf0, peerbuf1);

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
      const_cast<AllGather *>(this)->runAllgather(wireOff, tOff, workLeft);
    }
  }

private:
  // TODO: buffer plan and start point calc
  static size_t calcWorkSize(T* input, T* output, size_t input_sz) {
    // Input must be message size align
    if ((uintptr_t)input % sizeof(message_t) != 0
        || (uintptr_t)output % sizeof(message_t) != 0)
      throw std::logic_error("We only support aligned pointer for now");

    auto octSize = divUp(input_sz, sizeof(message_t));

    // predicates granularity
    if (octSize * sizeof(message_t) != input_sz)
      throw std::logic_error("We don't support non-even divide yet");

    return octSize * sizeof(message_t);
  }

  ssize_t workSize;
};

template <typename T>
sycl::event testAllgather(
    std::string transmitType,
    sycl::nd_range<1> launchParam,
    T* input, T* output, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t step, uint32_t subgroup, sycl::queue queue);

template <typename T> int verifyAllgather(
    T* host, int rank, int world, size_t nelems
);
