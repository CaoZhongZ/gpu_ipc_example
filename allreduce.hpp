#pragma once

#include <sycl/sycl.hpp>
#include <gen_visa_templates.hpp>

#include "transmit.hpp"

template <typename T,
         int NPeers,
         template <typename, int, int> class Transmit,
         int SubGroupSize = 16>
struct AllReduce : public Transmit<T, NPeers, SubGroupSize> {
  using Super = Transmit<T, NPeers, SubGroupSize>;
  using message_t = typename Super::message_t;
  constexpr static int unroll = 1 /*NPeers < 4 ? 4 : 2*/;
  constexpr static int wireCapacity = Super::wireCapacity;
  constexpr static int wireTransSize = Super::wireTransSize;

  AllReduce(
      T* input, size_t nelems, int rank, uint32_t seqNo,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[]
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  )
  : Transmit<T, NPeers, SubGroupSize>(rank, seqNo
#if defined(__enable_sycl_stream__)
      , cout
#endif
  ), workSize(calcWorkSize(input, nelems * sizeof(T))),
  transmitSize(divUp(workSize, wireCapacity) * wireTransSize)
#if defined(__enable_sycl_stream__)
    , cout(cout)
#endif
  {
    auto slotShift = [](int rank, int peer) {
      if (rank > peer) return rank -1;
      else return rank;
    };

    Super::ioBuffer = (input + rank * workSize / sizeof(T));

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      int next = (rank + i + 1) % (NPeers + 1);

      Super::scatterSink[i] = (T *)((uintptr_t)peerBuf0[next]
          + transmitSize * slotShift(rank, next));
      Super::gatherSink[i] = (T *)((uintptr_t)peerBuf1[next]
          + transmitSize * slotShift(rank, next));

      Super::localScatterSink[i] = (T *)((uintptr_t)scatterBuf
          + slotShift(next, rank) * transmitSize);
      Super::localGatherSink[i] = (T *)((uintptr_t)gatherBuf
          + slotShift(next, rank) * transmitSize);

      Super::ioForPeers[i] = input + next * workSize / sizeof(T);
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
        const_cast<AllReduce *>(this)->
          template scatter<unroll>(wireOff, transOff, workLeft);
        const_cast<AllReduce *>(this)->
          template pollRecvReduceBcast<unroll>(wireOff, transOff, workLeft);
        const_cast<AllReduce *>(this)->
          template pollGatherOutputs<unroll>(wireOff, transOff, workLeft);
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

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

// New AllReduce API will be here
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
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  ) :
  Super(input, scatterBuf, gatherBuf, peerBuf0, peerBuf1,
      calcWorkSize(input, nelems * sizeof(T)),
      divUp(calcWorkSize(input, nelems * sizeof(T)), wireCapacity)
      * wireTransSize, rank, seqNo
#if defined(__enable_sycl_stream__)
      , cout
#endif
    ), workSize(calcWorkSize(input, nelems * sizeof(T)))
#if defined(__enable_sycl_stream__)
  , cout(cout)
#endif
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
#if defined(__enable_sycl_stream__)
      auto local_id = pos.get_sub_group().get_local_id()[0];
      if (local_id == 0 && groupId == 0)
        cout<<"["<<groupId<<", "<<subGroupId
          <<"] loopSize:"<<loopSize
          <<", wireOff:"<<wireOff
          <<", transOff:"<<transOff
          <<", workLeft:"<<workLeft<<sycl::endl;
#endif
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
#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

template <typename T,
         int NRanks,
         template <typename, int, int> class Transmit = bisectPTransmit,
         int SubGroupSize = 16>
struct bisectPAllReduce : public Transmit<T, NRanks, SubGroupSize> {
  using Super = Transmit<T, NRanks, SubGroupSize>;
  using typename Super::message_t;
  constexpr static int unroll = 1 /*NPeers < 4 ? 4 : 2*/;
  constexpr static int wireCapacity = Super::wireCapacity;
  constexpr static int wireTransSize = Super::wireTransSize;
  constexpr static int BiNRanks = NRanks/2;

  bisectPAllReduce(
      T* input, size_t nelems, int rank, uint32_t seqNo,
      T* scatterBuf, T* gatherBuf, T* const peerBuf0[], T* const peerBuf1[]
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  ) :
  Super(input, scatterBuf, gatherBuf, peerBuf0, peerBuf1,
      calcWorkSize(input, nelems * sizeof(T)),
      divUp(calcWorkSize(input, nelems * sizeof(T)), wireCapacity)
      * wireTransSize, rank, seqNo
#if defined(__enable_sycl_stream__)
      , cout
#endif
    ), workSize(calcWorkSize(input, nelems * sizeof(T)))
#if defined(__enable_sycl_stream__)
  , cout(cout)
#endif
  {}

  static int stage1Verify(T* host, int rank, uint32_t flag, size_t nelems);
  static int stage2Verify(T* host, int rank, uint32_t flag, size_t nelems);
  static int stage3Verify(T* host, int rank, uint32_t flag, size_t nelems);
  static int stage4Verify(T* host, int rank, uint32_t flag, size_t nelems);

  //
  // We use linear group configuration because subgroup is linear only
  // And we want to control coalescing as much as possible
  //
  void operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (
      sycl::nd_item<1> pos
  ) const {
    auto groupRange = pos.get_group_range()[0];
    int subGroupXRange = pos.get_sub_group().get_group_range()[0]/BiNRanks;

    auto cableCapacity = wireCapacity * subGroupXRange;
    auto cableTSize = wireTransSize * subGroupXRange;

    // SubGroup 0, 1, 2, 3 is stacked up to the same groupXId
    // Hence work on same offsets
    auto subGroupXId = pos.get_sub_group().get_group_id()[0] /BiNRanks;
    auto groupId = pos.get_group().get_group_id()[0];

    auto loopSize = groupRange * cableCapacity;
    auto loopTSize = groupRange * cableTSize;

    for (size_t gOff = 0, tOff = 0;
        gOff < workSize; gOff += loopSize, tOff += loopTSize) {
      auto wireOff = groupId * cableCapacity + subGroupXId * wireCapacity + gOff;
      auto transOff = groupId * cableTSize + subGroupXId * wireTransSize + tOff;
      ssize_t workLeft = workSize - wireOff;

      auto nextOff = wireOff + gOff;
      auto nextLeft = workSize - nextOff;
#if defined(__enable_sycl_stream__)
      auto localId = pos.get_sub_group().get_local_id()[0];
      if (localId == 0 && groupId == 0)
        cout<<"["<<groupId<<", "<<subGroupXId
          <<"] loopSize:"<<loopSize
          <<", wireOff:"<<wireOff
          <<", transOff:"<<transOff
          <<", workLeft:"<<workLeft<<sycl::endl;
#endif
      if (workLeft > 0) { // Y parallel equals bisect Ranks
        if (nextLeft > cableTSize)
          const_cast<bisectPAllReduce *>(this)-> template PreloadNext<unroll>(nextOff);
        const_cast<bisectPAllReduce *>(this)->
          template scatterFar<unroll>(wireOff, transOff, workLeft);
        const_cast<bisectPAllReduce *>(this)->
          template closeUnifiedPollReduceScatterGather<unroll>(wireOff, transOff, workLeft);
        const_cast<bisectPAllReduce *>(this)->
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
#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

template <typename T,
         int NRanks,
         template <typename, int, int> class Transmit = bisectPPTransmit,
         int SubGroupSize = 16>
struct bisectPPAllReduce : public Transmit<T, NRanks, SubGroupSize> {
  using Super = Transmit<T, NRanks, SubGroupSize>;
  using message_t = typename Super::message_t;
  constexpr static int unroll = 1 /*NPeers < 4 ? 4 : 2*/;
  constexpr static int wireCapacity = Super::wireCapacity;
  constexpr static int wireTransSize = Super::wireTransSize;
  constexpr static int BiNRanks = NRanks/2;

  bisectPPAllReduce(
      T* input, size_t nelems, int rank, uint32_t seqNo,
      T* scatterBuf, T* gatherBuf, T* const peerBuf0[], T* const peerBuf1[]
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  ) :
  Super(input, scatterBuf, gatherBuf, peerBuf0, peerBuf1,
      calcWorkSize(input, nelems * sizeof(T)),
      divUp(calcWorkSize(input, nelems * sizeof(T)), wireCapacity)
      * wireTransSize, rank, seqNo
#if defined(__enable_sycl_stream__)
      , cout
#endif
    ), workSize(calcWorkSize(input, nelems * sizeof(T)))
#if defined(__enable_sycl_stream__)
  , cout(cout)
#endif
  {}

  static int stage4Verify(T* host, int rank, uint32_t flag, size_t nelems);

  //
  // We use linear group configuration because subgroup is linear only
  // And we want to control coalescing as much as possible
  //
  void operator() [[sycl::reqd_sub_group_size(SubGroupSize)]] (
      sycl::nd_item<1> pos
  ) const {
    auto groupRange = pos.get_group_range()[0];
    int subGroupXRange = pos.get_sub_group().get_group_range()[0]/BiNRanks/2;

    auto cableCapacity = wireCapacity * subGroupXRange;
    auto cableTSize = wireTransSize * subGroupXRange;

    // 0, 1, 2, 3, 4, 5, 6, 7 work on same offset
    auto subGroupXId = pos.get_sub_group().get_group_id()[0] /(BiNRanks *2);
    auto subGroupYId = pos.get_sub_group().get_group_id()[0] %(BiNRanks *2);
    auto groupId = pos.get_group().get_group_id()[0];

    auto loopSize = groupRange * cableCapacity;
    auto loopTSize = groupRange * cableTSize;

    for (size_t gOff = 0, tOff = 0;
        gOff < workSize; gOff += loopSize, tOff += loopTSize) {
      auto wireOff = groupId * cableCapacity + subGroupXId * wireCapacity + gOff;
      auto transOff = groupId * cableTSize + subGroupXId * wireTransSize + tOff;
      ssize_t workLeft = workSize - wireOff;

#if defined(__enable_sycl_stream__)
      auto localId = pos.get_sub_group().get_local_id()[0];
      if (localId == 0 && groupId == 0)
        cout<<"["<<groupId<<", "<<subGroupXId
          <<"] loopSize:"<<loopSize
          <<", wireOff:"<<wireOff
          <<", transOff:"<<transOff
          <<", workLeft:"<<workLeft<<sycl::endl;
#endif
      if (workLeft > 0) { // Y parallel equals bisect Ranks
        if (subGroupYId < 2) {
          const_cast<bisectPPAllReduce *>(this)->
            template scatterFar<unroll>(wireOff, transOff, workLeft);
        } else if (subGroupYId < 4) {
          const_cast<bisectPPAllReduce *>(this)->
            template pollFarGatherOutput<unroll>(wireOff, transOff, workLeft);
        } else if (subGroupYId < 8) {
          const_cast<bisectPPAllReduce *>(this)->
           template closeUnifiedPollReduceScatterGather<unroll>(wireOff, transOff, workLeft);
        }
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
#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

template <typename T, template <typename, int, int> class Transmit>
sycl::event testTransmit(
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t step, uint32_t simd, sycl::queue queue
);

template <typename T, template <typename, int, int> class Transmit>
int verifyTransmit(
    T* host, T* host2,
    uint32_t step, int rank, int world, uint32_t simd, size_t nWorkElems
);
