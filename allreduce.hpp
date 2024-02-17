#pragma once

#include <sycl/sycl.hpp>
#include <gen_visa_templates.hpp>

#include "transmit.hpp"

template <typename T,
         int NPeers,
         template <typename, int, int> class Transmit,
         int SubGroupSize = 16>
struct AllReduce {
  constexpr static int nReg128B = 128 / SubGroupSize / 4;
  using message_t = sycl::vec<uint32_t, nReg128B>;

  constexpr static int nChan8B = 8 / sizeof(message_t);
  constexpr static int nDataChannel = SubGroupSize - nChan8B;
  constexpr static int unroll = 1 /*NPeers < 4 ? 4 : 2*/;
  constexpr static int wireCapacity = unroll * nDataChannel * sizeof(message_t);
  constexpr static int wireTransSize = unroll * SubGroupSize * sizeof(message_t);

  AllReduce(
      T* input, size_t nelems, int rank, uint32_t step,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[]
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  )
  : workSize(calcWorkSize(input, nelems * sizeof(T))),
  transmitSize(divUp(workSize, wireCapacity) * wireTransSize),
  rank(rank), step(step),
  ioBuffer(input + rank * workSize / sizeof(T))
#if defined(__enable_sycl_stream__)
    , cout(cout)
#endif
  {
    auto slotShift = [](int rank, int peer) {
      if (rank > peer) return rank -1;
      else return rank;
    };

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      int next = (rank + i + 1) % (NPeers + 1);

      scatterSink[i] = (T *)((uintptr_t)peerBuf0[next]
          + transmitSize * slotShift(rank, next));
      gatherSink[i] = (T *)((uintptr_t)peerBuf1[next]
          + transmitSize * slotShift(rank, next));

      localScatterSink[i] = (T *)((uintptr_t)scatterBuf
          + slotShift(next, rank) * transmitSize);
      localGatherSink[i] = (T *)((uintptr_t)gatherBuf
          + slotShift(next, rank) * transmitSize);

      ioForPeers[i] = input + next * workSize / sizeof(T);
    }
  }

  // Calculate which slot 'rank' take at 'peer''s sink buffer
  static int slot(int rank, int peer) {
    if (rank > peer) return rank -1;
    else return rank;
  }

  static int scatterVerify(
      uint32_t* host, int rank, uint32_t flag, size_t nWorkElemsInInt
  );
  static int stage2Verify(
      T* host, int rank, uint32_t flag, size_t nWorkElemsInInt
  );
  //
  // Found this analogy fascinating:
  //
  // Let correspond sub-group to wire, sequential guaranteed.
  // Bundle sub-groups(wires) into group(cable).
  //
  // Total cables will deliver the full capacity of single loop.
  //
  inline void stage2Test(sycl::nd_item<1> pos) const {
    Transmit<T, NPeers, SubGroupSize> cable(
        pos, scatterSink, gatherSink, localScatterSink, localGatherSink,
        ioBuffer, ioForPeers, step, rank
#if defined(__enable_sycl_stream__)
        ,cout
#endif
    );

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

#if defined(__enable_sycl_stream__)
      auto local_id = pos.get_sub_group().get_local_id()[0];
      if (local_id == 0 && groupId == 0)
        cout<<"["<<groupId<<", "<<subGroupId
          <<"] loopSize:"<<loopSize
          <<", wireOff:"<<wireOff<<"; "
          <<", transOff:"<<transOff<<"; "<<sycl::endl;
#endif
      ssize_t workLeft = workSize - wireOff;
      if (workLeft > 0) {
        cable.template scatter<unroll>(wireOff, transOff, workLeft);
        cable.template pollRecvReduceBcast<unroll>(wireOff, transOff, workLeft);
        cable.template pollGatherOutputs<unroll>(wireOff, transOff, workLeft);
      }
    }
  }

  void operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (
      sycl::nd_item<1> pos
  ) const {
    stage2Test(pos);
  }

private:
  // TODO: buffer plan and start point calc
  static size_t calcWorkSize(T* input, size_t size) {
    // Input must be message size align
    if ((uintptr_t)input % sizeof(message_t) != 0)
      throw std::logic_error("We only support aligned pointer for now");

    auto nChunks = NPeers + 1;
    auto octSize = divUp(size, sizeof(message_t));
    auto chunkSize = divUp(octSize, nChunks);

    if (octSize * sizeof(message_t) != size || chunkSize * sizeof(message_t) * nChunks > size)
      throw std::logic_error("We don't support non-even divide yet");

    // TODO: Production logic needs every rank chunk

    return chunkSize * sizeof(message_t);
  }
  ssize_t workSize;
  size_t transmitSize;
  int rank;
  uint32_t step;

  T* scatterSink[NPeers];
  T* gatherSink[NPeers];
  T* localScatterSink[NPeers];
  T* localGatherSink[NPeers];

  T* ioBuffer;
  T* ioForPeers[NPeers];

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

template <typename T>
sycl::event testSimpleTransmit(
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t step, uint32_t simd, sycl::queue queue
);

template <typename T>
int verifyTransmit(
    T* host, uint32_t step, int rank, int world, uint32_t simd, size_t nWorkElems
);
