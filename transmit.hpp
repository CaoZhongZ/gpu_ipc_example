#pragma once

#include <sycl/sycl.hpp>
#include <gen_visa_templates.hpp>

#define divUp(x, m)  \
  ((x + m -1) / m)

#define alignUp(x, c) \
  (divUp(x, c) * c)

//
// XXX: group level class object.
//
template <typename T, int NPeers, int SubGroupSize=16>
class SimpleTransmit {
  // using message_t sycl::vec<uint32_t, 2>; // 128 bytes
  using message_t uint64_t; // layout problem???

public:
  SimpleTransmit(
      sycl::nd_item<1> pos,
      size_t workSize, // should be a array in production code
      uint64_t step,   // Serve as flag for checking
      T* scatterSink[],
      T* gatherSink[],
      T* ioBuffer,
      T* localScatterSink[],
      T* localGatherSink[],
  ) : workSize(workSize), scatterStep(step), gatherStep(step), pos(pos) {
#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      this->scatterSink[i] = scatterSink[i];
      this->gatherSink[i] = gatherSink[i];
      this->localScatterSink[i] = localScatterSink[i];
      this->localGatherSink[i] = localGatherSink[i];
    }
  }

  static constexpr int inputStep = (SubGroupSize -1) * sizeof(uint64_t) / sizeof(T);
  static constexpr int messageStep = SubGroupSize * sizeof(uint64_t) / sizeof(T);

  template <int unroll> inline void loadInput(
      uint64_t (&v)[unroll], void* src, int nElt
  ) {
    auto lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(uint64_t) / sizeof(T);

    if (lid != SubGroupSize -1) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * inputStep + local_off;
        if (off < nElt) {
          lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
              v[i], src + off;
          );
    }}}
  }

  template <int unroll> inline void storeOutput(
      uint64_t (&v)[unroll], void* dst, int nElt
  ) {
    auto lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(uint64_t) / sizeof(T);
    if (lid != SubGroupSize -1) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i) {
        auto off = i * inputStep + local_off;
        if (off < nElt) {
          lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
              dst + off, v[i];
          );
    }}}
  }

  // sub-group level description
  template <int unroll>
  inline void packMessage(
      message_t (& messages)[unroll], T* src, int nElt, uint64_t flag
  ) {
    int lid = pos.get_sub_group().get_local_id()[0];
    // Add flags at the tail lane
    if (lid == SubGroupSize -1) {
#     pragma unroll
      for (int i = 0; i < unroll; ++ i)
        messages[i] = flag;
    }

    //
    // contiguous load 15 lanes of data (120 bytes) multiple times
    //
    // For Subgroup 16 lane configuration, 128B message delivered
    // possibly in two chunk or one Last lane are flags to occupy 8-byte
    // and offset address always in 8-byte aligned fashion
    //
    loadInput(messages, src, nElt);
  }

  // We always push 128-byte packages
  template <int unroll>
  inline void sendMessages(T* ptr, uint64_t (&messages)[unroll]) {
    int lid = pos.get_sub_group().get_local_id()[0];
    int local_off = lid * sizeof(uint64_t) / sizeof(T);

#   pragma unroll
    for (int u = 0; u < unroll; ++ u) {
      lscStore<SubGroupSize, CacheCtrl::L1UC_L3UC>(
          ptr + u * messageStep + local_off,
          messages[u];
      );
    }
  }

  // Scatter local message to peers
  template <int unroll> scatter(size_t offset, size_t nelems) {
    auto offsetInType = offset / sizeof(T);

    constexpr int eltPerPack =
      unroll * (SubGroupSize -1) * sizeof(uint64_t) / sizeof(T);

    int nElt = std::min(nelems, eltPerPack);

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto next = (rank + i) % (NPeers + 1);
      auto peerOffset = next * workSize / sizeof(T);
      auto* ptr = ioBuffer + peerOffset + offsetInType;
      auto* dst = scatterSink[i] + offsetInType;

      message_t messages[unroll];
      packMessage(messages, ptr, nElt, scatterStep);
      sendMessage(messages, dst, nElt);
    }
  }

  template <int unroll>
  inline void pollRecvReduceGather(
      uint64_t (&v)[unroll], size_t offset
  ) {
    uint64_t messages[unroll]; // scraps from remote
    auto offsetInType = offset / sizeof(T);

#   pragma unroll
    for (i = 0; i < NPeers; ++ i) {
      auto flag = scatterStep;
      bool retry;
      do {
        retry = false;
#       pragma unroll
        for (int u = 0; u < unroll; ++ u) {
          lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
              message[u], localScatterSink[peer] + offsetInType + u * messageStep);
          retry |=
            subgroup_id == (SubGroupSize -1)
            && message[u] != flag;
        }
      } while(sycl::any_of_group(sg, poll));
#     pragma unroll
      for (int u = 0; u < unroll; ++ u) {
        lscLoad<SubGroupSize, CacheCtrl::L1UC_L3UC>(
            message[u], localScatterSink[peer] + offsetInType + u * messageStep);
        v[u] = addAs<T, SubGroupSize>(v[u], message[u]);
      }
    }

    // push to gather sink
#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      auto flag = gatherStep;
      int lid = pos.get_sub_group().get_local_id()[0];

#     pragma unroll
      for (int u = 0; u < unroll; ++ u) {
        if (lid == SubGroupSize -1)
          v[u] = flag;
      }

#     pragma unroll
      for (int u = 0; u < unroll; ++ u) {
        lscStore<SubGroupSize>, CacheCtrl::L1UC_L3UC>(
            gatherSink[i] + offsetInType + u * messageStep, v[u]);
      }
    }
  }

private:
  T* scatterSink[NPeers];
  T* gatherSink[NPeers];
  T* ioBuffer;
  T* localScatterSink[NPeers];
  T* localGatherSink[NPeers];

  size_t workSize;
  uint64_t scatterStep;
  uint64_t gatherStep;

  sycl::nd_item<1> pos;
};

template <typename T, int NPeers, int SubGroupSize=16> struct AllReduce {
  AllReduce(
      T* input, size_t size, int rank,
      T* scatterBuf, T* gatherBuf,
      T* peerBuf0[], T* peerBuf1[], sycl::stream cout
  )
  : ioBuffer(input), rank(rank), cout(cout) {
    workSize = calcWorkSize(input, size);
    transmitSize = divUp(workSize, 120)*128;

    auto slotShift = [](int rank, int peer) {
      if (rank > peer) return rank -1;
      else return rank;
    };

#   pragma unroll
    for (int i = 0; i < NPeers; ++ i) {
      int next = (rank + i) % (NPeers + 1);

      scatterSink[i] = (T *)((uintptr_t)peerBuf0[i]
          + transmitSize * slotShift(rank, next));
      gatherSink[i] = (T *)((uintptr_t)peerBuf1[i]
          + transmitSize * slotShift(rank, next));

      localScatterSink[i] = scatterBuf + slotShift(next, rank) * transmitSize;
      localGatherSink[i] = gatherBuf + slotShift(next, rank) * transmitSize;
    }
  }

  //
  // I found this analogy fascinating:
  //
  // Let correspond sub-group to wire, sequential guaranteed.
  // Bundle sub-groups(wires) into group(cable).
  //
  // Total cables will deliver the full capacity of single loop.
  //
  template <int unroll>
  void scatterTest(sycl::nd_item<1> pos) const {
    constexpr int wireCap = unroll * (SubGroupSize -1) * sizeof(uint64_t);
    auto cableCap = wireCap * pos.get_sub_group().get_group_range()[0];
    auto loopSize = pos.get_group_range() * cableCap;

    SimpleTransmit<T, NPeers, SubGroupSize> cable (
        pos, workSize, step, scatterSink, gatherSink,
        ioBuffer, localScatterSink, localGatherSink
    );

    auto groupId = pos.get_group_id()[0];
    auto subGroupId = pos.get_sub_group().get_group_id()[0];

    for (size_t gOff = 0; gOff < workSize; gOff += loopSize) {
      auto gPos = gOff + groupId;
      auto cableOff = gPos * cableCap;
      auto wireOff = cableOff + subGroupId * wireCap;
      cable.scatter(wireOff, workSize);
    }
  }

  void operator() (
      sycl::nd_item<1> pos
  ) [[sycl::reqd_sub_group_size(SubGroupSize)]] const {
    scatterTest(pos);
  }

private:
  // TODO: buffer plan and start point calc
  static size_t calcWorkSize(T* input, size_t size) {
  //
  // We only do 8-byte aligned and 8-byte dividable case for now.
  //
    if ((uintptr_t)input % sizeof(uint64_t) != 0)
      throw std::logic_error("We only support aligned pointer for now");

    auto nChunks = NPeers + 1;
    auto octSize = divUp(size, sizeof(uint64_t));
    auto chunkSize = divUp(octSize, nChunks);

    if (alignUpSize != size || chunkSize * sizeof(uint64_t) * nChunks > size)
      throw std::logic_error("We don't support non-even divide yet");

    // TODO: Production logic needs every rank chunk

    return chunkSize * sizeof(uint64_t);
  }

  T* scatterSink[NPeers];
  T* gatherSink[NPeers];
  T* localScatterSink[NPeers];
  T* localGatherSink[NPeers];

  T* ioBuffer;

  int rank;
  size_t workSize;
  size_t transmitSize;

  sycl::stream cout;
};

template <typename T>
void testSimpleTransmit(
    sycl::nd_range<1> launchParam,
    T* input, void* ipcbuf0, void* ipcbuf1,
    void* peerbuf0[], void* peerbuf1[], size_t size,
    int rank, int world, sycl::queue queue
);
