#pragma once

#include "rt64_128.hpp"

//
// Requires multiple dimension launch with Y dimention equal to 'BiNRanks'
//
template <typename T, int NRanks, int SubGroupSize>
class bisectPTransmit : public rt64_128<T, SubGroupSize> {
  constexpr static int BiNRanks = NRanks / 2;
  constexpr static int NPeers = BiNRanks -1;

public:
  using rt64_128<T, SubGroupSize>::preload;
  using rt64_128<T, SubGroupSize>::wireElems;
  using typename rt64_128<T, SubGroupSize>::message_t;

  template <int unroll> inline void PreloadNext(
      size_t inputOffset
  ) {
    // given ioForPeers vs. ioForFar is stride with multiple of 1024
    // Presume loss of L1 for accessing each other
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % BiNRanks;
    preload<unroll>(ioForPeers[y_id] + inputOffset/sizeof(T));
    preload<unroll>(ioForFar[y_id] + inputOffset/sizeof(T));
  }

  template <int unroll> inline void scatterFar(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireElems;

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % BiNRanks;
    auto *ptr = ioForFar[y_id] + inputOffInType;

    message_t messages[unroll];

    if (nelems < eltPerPack) {
      loadInput(messages, ptr, nelems);
    } else {
      loadInput(messages, ptr);
    }

    shuffleData(messages);
    insertFlags(messages, seqNo);
    auto* dst = farScatterSink[y_id] + sinkOffInType;
    sendMessages(dst, messages);
  }

  template <int unroll> inline void pollFarGatherOutput(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % BiNRanks;

    constexpr auto eltPerPack = unroll * wireElems;

    message_t messages[unroll];
    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localFarGatherSink[y_id] + sinkOffInType, seqNo);
    } while(sycl::any_of_group(sg, retry));

    // if (sg.get_local_id()[0] == 15)
    //   cout<<messages[0]<<sycl::endl<<sycl::flush;

    restoreData(messages);

    if (nelems < eltPerPack)
      storeOutput(ioForFar[y_id] + inputOffInType, messages, nelems);
    else
      storeOutput(ioForFar[y_id] + inputOffInType, messages);
  }

  template <int unroll> inline void closeUnifiedPollReduceScatterGather(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % BiNRanks;

    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);
    constexpr auto eltPerPack = unroll * wireElems;

    message_t v[unroll];

    auto* ioPtr = ioForPeers[y_id] + inputOffInType;
    if (nelems < eltPerPack)
      loadInput(v, ioPtr, nelems);
    else
      loadInput(v, ioPtr);

    message_t messages[unroll];

    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localFarScatterSink[y_id] + sinkOffInType, seqNo);
    } while(sycl::any_of_group(sg, retry));

    shuffleData(v);
    accumMessages(v, messages);

    //------------------------- sub-group diverge 3:1 -------------------
    if (y_id != l_rank) {
      insertFlags(v, seqNo);
      sendMessages(scatterSink[y_id] + sinkOffInType, v); // 1. xNPeers <scatter>

      bool retry;
      do {
        retry = false;
        retry |= recvMessages(
            v, localGatherSink[y_id] + sinkOffInType, seqNo);
      } while(sycl::any_of_group(sg, retry));             // 4. xNPeers waits for <gather>
    } else {
#     pragma unroll
      for (int i =0; i < NPeers; ++ i) {
        bool retry;
        do {
          retry = false;
          retry |= recvMessages(
              messages, localScatterSink[i] + sinkOffInType, seqNo);
        } while (sycl::any_of_group(sg, retry));          // 2. wait for <scatter> xNPeers
        accumMessages(v, messages);
      }

      insertFlags(v, seqNo);

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i)                   // 3. signal <gather>
        sendMessages(gatherSink[i] + sinkOffInType, v);
    }
    //-------------------------group converge-------------------

    sendMessages(farGatherSink[y_id] + sinkOffInType, v);
    restoreData(v);

    if (nelems < eltPerPack)
      storeOutput(ioPtr, v, nelems);
    else
      storeOutput(ioPtr, v);
  }

protected:
  bisectPTransmit(
      T* input,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[],
      size_t workSize, size_t transmitSize,
      int rank, uint32_t seqNo
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  ) : l_rank(rank/2), seqNo(seqNo)
#if defined(__enable_sycl_stream__)
      , cout(cout)
#endif
  {
    auto pairRank = rank ^ 1;

    auto* pairBuf0 = peerBuf0[pairRank];
    auto* pairBuf1 = peerBuf1[pairRank];

    auto ioClosePart = [&](T* p) {
      return (rank & 1) ? (T *)((uintptr_t)p + workSize * BiNRanks) : p;
    };
    auto ioFarPart = [&](T* p) {
      return (rank & 1) ? p : (T *)((uintptr_t)p + workSize * BiNRanks);
    };
    auto ipcClosePart = [&](T *p) {
      return p;
    };
    auto ipcFarPart = [&](T* p) {
      return (T *)((uintptr_t)p + transmitSize * BiNRanks);
    };

    // Indicated by y-id
    for (int i = 0; i < BiNRanks; ++ i) {
      ioForPeers[i] = (T *)((uintptr_t)ioClosePart(input) + i * workSize);
      ioForFar[i] = (T *)((uintptr_t)ioFarPart(input) + i * workSize);

      farScatterSink[i] = (T *)((uintptr_t)ipcFarPart(pairBuf0)
          + i * transmitSize);
      farGatherSink[i] = (T *)((uintptr_t)ipcFarPart(pairBuf1)
          + i * transmitSize);

      localFarScatterSink[i] = (T *)((uintptr_t)ipcFarPart(scatterBuf)
          + i * transmitSize);
      localFarGatherSink[i] = (T *)((uintptr_t)ipcFarPart(gatherBuf)
          + i * transmitSize);

      // Will jump over rank slot, XXX: be careful
      if (l_rank != i) {
        // even: 0, 2, 4, 6
        // odd:  1, 3, 5, 7
        auto r_index = 2 * i + (rank & 1);
        scatterSink[i] = (T *)((uintptr_t)ipcClosePart(peerBuf0[r_index])
            + l_rank * transmitSize);
        localGatherSink[i] = (T *)((uintptr_t)ipcClosePart(gatherBuf)
            + i * transmitSize);
      }
    }

    // Indicated by next?
    for (int i = 0; i < NPeers; ++ i) {
      int l_next = (l_rank + i + 1) % BiNRanks;
      int next = (rank + i * 2 + 2) % (2 * BiNRanks);

      localScatterSink[i] = (T *)((uintptr_t)ipcClosePart(scatterBuf)
          + l_next * transmitSize);
      gatherSink[i] = (T *)((uintptr_t)ipcClosePart(peerBuf1[next])
          + l_rank * transmitSize);
    }
  }

  // --------------------Input/Output buffer-------------------
  // Input partitions
  T* ioForPeers[BiNRanks];
  T* ioForFar[BiNRanks];

  // ---------------------IPC buffers-------------------------
  T* farScatterSink[BiNRanks];
  T* farGatherSink[BiNRanks];

  T* scatterSink[BiNRanks];
  T* gatherSink[NPeers];

  // ----------------Sinks, written by remotes----------------
  T* localFarScatterSink[BiNRanks];
  T* localFarGatherSink[BiNRanks];

  T* localScatterSink[NPeers];
  T* localGatherSink[BiNRanks];

  int l_rank;
  uint32_t seqNo;

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

template <typename T, int NRanks, int SubGroupSize>
class bisectPTransmitOpt : public rt64_128<T, SubGroupSize> {
protected:
  constexpr static int BiNRanks = NRanks / 2;
  constexpr static int NPeers = BiNRanks -1;
  using super=rt64_128<T, SubGroupSize>;

  using super::wireElems;
  using super::loadInput;
  using super::storeOutput;
  using super::shuffleData;
  using super::restoreData;
  using super::insertFlags;
  using super::sendMessages;
  using super::recvMessages;
  using super::accumMessages;
  using typename super::message_t;

public:
  template <int unroll> inline void PreloadNext(
      size_t inputOffset
  ) {
    // given ioForPeers vs. ioForFar is stride with multiple of 1024
    // Presume loss of L1 for accessing each other
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % BiNRanks;
    auto y_off = y_id * workElems;

    super::template preload<unroll>(ioForPeers + y_off + inputOffset/sizeof(T));
    super::template preload<unroll>(ioForFar + y_off + inputOffset/sizeof(T));
  }

  template <int unroll> inline void scatterFar(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireElems;

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % BiNRanks;
    auto y_off = y_id * workElems;
    auto *ptr = ioForFar + y_off + inputOffInType;

    message_t messages[unroll];

    if (nelems < eltPerPack) {
      loadInput(messages, ptr, nelems);
    } else {
      loadInput(messages, ptr);
    }

    shuffleData(messages);
    insertFlags(messages, seqNo);
    auto* dst = farScatterSink[y_id] + sinkOffInType;
    sendMessages(dst, messages);
  }

  template <int unroll> inline void pollFarGatherOutput(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % BiNRanks;
    auto y_off = y_id * workElems;

    constexpr auto eltPerPack = unroll * wireElems;

    message_t messages[unroll];
    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localFarGatherSink[y_id] + sinkOffInType, seqNo);
    } while(sycl::any_of_group(sg, retry));

    // if (sg.get_local_id()[0] == 15)
    //   cout<<messages[0]<<sycl::endl<<sycl::flush;

    restoreData(messages);

    if (nelems < eltPerPack)
      storeOutput(ioForFar + y_off + inputOffInType, messages, nelems);
    else
      storeOutput(ioForFar + y_off + inputOffInType, messages);
  }

  template <int unroll> inline void closeUnifiedPollReduceScatterGather(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % BiNRanks;
    auto y_off = y_id * workElems;

    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);
    constexpr auto eltPerPack = unroll * wireElems;

    message_t v[unroll];

    auto* ioPtr = ioForPeers + y_off + inputOffInType;
    if (nelems < eltPerPack)
      loadInput(v, ioPtr, nelems);
    else
      loadInput(v, ioPtr);

    message_t messages[unroll];

    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localFarScatterSink[y_id] + sinkOffInType, seqNo);
    } while(sycl::any_of_group(sg, retry));

    shuffleData(v);
    accumMessages(v, messages);

    //------------------------- sub-group diverge 3:1 -------------------
    if (y_id != l_rank) {
      insertFlags(v, seqNo);
      sendMessages(scatterSink[y_id] + sinkOffInType, v); // 1. xNPeers <scatter>

      bool retry;
      do {
        retry = false;
        retry |= recvMessages(
            v, localGatherSink[y_id] + sinkOffInType, seqNo);
      } while(sycl::any_of_group(sg, retry));             // 4. xNPeers waits for <gather>
    } else {
#     pragma unroll
      for (int i =0; i < NPeers; ++ i) {
        bool retry;
        do {
          retry = false;
          retry |= recvMessages(
              messages, localScatterSink[i] + sinkOffInType, seqNo);
        } while (sycl::any_of_group(sg, retry));          // 2. wait for <scatter> xNPeers
        accumMessages(v, messages);
      }

      insertFlags(v, seqNo);

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i)                   // 3. signal <gather>
        sendMessages(gatherSink[i] + sinkOffInType, v);
    }
    //-------------------------group converge-------------------

    sendMessages(farGatherSink[y_id] + sinkOffInType, v);
    restoreData(v);

    if (nelems < eltPerPack)
      storeOutput(ioPtr, v, nelems);
    else
      storeOutput(ioPtr, v);
  }

protected:
  bisectPTransmitOpt(
      T* input,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[],
      size_t workSize, size_t transmitSize,
      int rank, uint32_t seqNo
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  ) : l_rank(rank/2), seqNo(seqNo), workElems(workSize/sizeof(T))
#if defined(__enable_sycl_stream__)
      , cout(cout)
#endif
  {
    auto pairRank = rank ^ 1;

    auto* pairBuf0 = peerBuf0[pairRank];
    auto* pairBuf1 = peerBuf1[pairRank];

    auto ioClosePart = [&](T* p) {
      return (rank & 1) ? (T *)((uintptr_t)p + workSize * BiNRanks) : p;
    };
    auto ioFarPart = [&](T* p) {
      return (rank & 1) ? p : (T *)((uintptr_t)p + workSize * BiNRanks);
    };
    auto ipcClosePart = [&](T *p) {
      return p;
    };
    auto ipcFarPart = [&](T* p) {
      return (T *)((uintptr_t)p + transmitSize * BiNRanks);
    };

    ioForPeers = ioClosePart(input);
    ioForFar = ioFarPart(input);

    // Indicated by y-id
    for (int i = 0; i < BiNRanks; ++ i) {
      farScatterSink[i] = (T *)((uintptr_t)ipcFarPart(pairBuf0)
          + i * transmitSize);
      farGatherSink[i] = (T *)((uintptr_t)ipcFarPart(pairBuf1)
          + i * transmitSize);

      localFarScatterSink[i] = (T *)((uintptr_t)ipcFarPart(scatterBuf)
          + i * transmitSize);
      localFarGatherSink[i] = (T *)((uintptr_t)ipcFarPart(gatherBuf)
          + i * transmitSize);

      // Will jump over rank slot, XXX: be careful
      if (l_rank != i) {
        // even: 0, 2, 4, 6
        // odd:  1, 3, 5, 7
        auto r_index = 2 * i + (rank & 1);
        scatterSink[i] = (T *)((uintptr_t)ipcClosePart(peerBuf0[r_index])
            + l_rank * transmitSize);
        localGatherSink[i] = (T *)((uintptr_t)ipcClosePart(gatherBuf)
            + i * transmitSize);
      }
    }

    // Indicated by next?
    for (int i = 0; i < NPeers; ++ i) {
      int l_next = (l_rank + i + 1) % BiNRanks;
      int next = (rank + i * 2 + 2) % (2 * BiNRanks);

      localScatterSink[i] = (T *)((uintptr_t)ipcClosePart(scatterBuf)
          + l_next * transmitSize);
      gatherSink[i] = (T *)((uintptr_t)ipcClosePart(peerBuf1[next])
          + l_rank * transmitSize);
    }
  }

  // --------------------Input/Output buffer-------------------
  // Input partitions
  T* ioForPeers;
  T* ioForFar;

  // ---------------------IPC buffers-------------------------
  T* farScatterSink[BiNRanks];
  T* farGatherSink[BiNRanks];

  T* scatterSink[BiNRanks];
  T* gatherSink[NPeers];

  // ----------------Sinks, written by remotes----------------
  T* localFarScatterSink[BiNRanks];
  T* localFarGatherSink[BiNRanks];

  T* localScatterSink[NPeers];
  T* localGatherSink[BiNRanks];

  int l_rank;
  uint32_t seqNo;
  size_t workElems;

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};

template <typename T, int NRanks, int SubGroupSize>
class bisectPPTransmit : public rt64_128<T, SubGroupSize> {
public:
  constexpr static int BiNRanks = NRanks / 2;
  constexpr static int NPeers = BiNRanks -1;

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

  using typename rt64_128<T, SubGroupSize>::message_t;

  // [0, 1] -> [[0, 1], [2, 3]]
  template <int unroll> inline void scatterFar(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    constexpr auto eltPerPack = unroll * wireElems;
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % 2;
    auto *ptr0 = ioForFar[y_id * 2] + inputOffInType;
    auto *ptr1 = ioForFar[y_id * 2 + 1] + inputOffInType;

    message_t messages0[unroll];
    message_t messages1[unroll];

    if (nelems < eltPerPack) {
      loadInput(messages0, ptr0, nelems);
      loadInput(messages1, ptr1, nelems);
    } else {
      loadInput(messages0, ptr0);
      loadInput(messages1, ptr1);
    }

    shuffleData(messages0);
    shuffleData(messages1);

    insertFlags(messages0, seqNo);
    insertFlags(messages1, seqNo);
    auto* dst0 = farScatterSink[y_id * 2] + sinkOffInType;
    auto* dst1 = farScatterSink[y_id * 2 + 1] + sinkOffInType;
    sendMessages(dst0, messages0);
    sendMessages(dst1, messages1);
  }

  // [0, 1] -> [[0, 1], [2, 3]]
  template <int unroll> inline void pollFarGatherOutput(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);

    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % 2;

    constexpr auto eltPerPack = unroll * wireElems;

    message_t messages0[unroll];
    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages0, localFarGatherSink[y_id * 2] + sinkOffInType, seqNo);
    } while(sycl::any_of_group(sg, retry));

    message_t messages1[unroll];
    do {
      retry = false;
      retry |= recvMessages(
          messages1, localFarGatherSink[y_id * 2 + 1] + sinkOffInType, seqNo);
    } while(sycl::any_of_group(sg, retry));

    // if (sg.get_local_id()[0] == 15)
    //   cout<<messages[0]<<sycl::endl<<sycl::flush;

    restoreData(messages0);
    restoreData(messages1);

    if (nelems < eltPerPack) {
      storeOutput(ioForFar[y_id * 2] + inputOffInType, messages0, nelems);
      storeOutput(ioForFar[y_id * 2 + 1] + inputOffInType, messages1, nelems);
    } else {
      storeOutput(ioForFar[y_id * 2] + inputOffInType, messages0);
      storeOutput(ioForFar[y_id * 2 + 1] + inputOffInType, messages1);
    }
  }

  // [0, 1, 2, 3]
  template <int unroll> inline void closeUnifiedPollReduceScatterGather(
      size_t inputOffset, size_t sinkOffset, ssize_t workLeft
  ) {
    auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    auto y_id = sg.get_group_id()[0] % BiNRanks;

    auto inputOffInType = inputOffset / sizeof(T);
    auto sinkOffInType = sinkOffset / sizeof(T);
    auto nelems = workLeft / sizeof(T);
    constexpr auto eltPerPack = unroll * wireElems;

    message_t v[unroll];

    auto* ioPtr = ioForPeers[y_id] + inputOffInType;
    if (nelems < eltPerPack)
      loadInput(v, ioPtr, nelems);
    else
      loadInput(v, ioPtr);

    message_t messages[unroll];

    bool retry;
    do {
      retry = false;
      retry |= recvMessages(
          messages, localFarScatterSink[y_id] + sinkOffInType, seqNo);
    } while(sycl::any_of_group(sg, retry));

    shuffleData(v);
    accumMessages(v, messages);

    //------------------------- group diverge 3:1 -------------------
    if (y_id != l_rank) {
      insertFlags(v, seqNo);
      sendMessages(scatterSink[y_id] + sinkOffInType, v); // 1. xNPeers <scatter>

      bool retry;
      do {
        retry = false;
        retry |= recvMessages(
            v, localGatherSink[y_id] + sinkOffInType, seqNo);
      } while(sycl::any_of_group(sg, retry));             // 4. xNPeers waits for <gather>
    } else {
#     pragma unroll
      for (int i =0; i < NPeers; ++ i) {
        bool retry;
        do {
          retry = false;
          retry |= recvMessages(
              messages, localScatterSink[i] + sinkOffInType, seqNo);
        } while (sycl::any_of_group(sg, retry));          // 2. wait for <scatter> xNPeers
        accumMessages(v, messages);
      }

      insertFlags(v, seqNo);

#     pragma unroll
      for (int i = 0; i < NPeers; ++ i)                   // 3. signal <gather>
        sendMessages(gatherSink[i] + sinkOffInType, v);
    }
    //-------------------------group converge-------------------

    sendMessages(farGatherSink[y_id] + sinkOffInType, v);
    restoreData(v);

    if (nelems < eltPerPack)
      storeOutput(ioPtr, v, nelems);
    else
      storeOutput(ioPtr, v);
  }

protected:
  bisectPPTransmit(
      T* input,
      T* scatterBuf, T* gatherBuf,
      T* const peerBuf0[], T* const peerBuf1[],
      size_t workSize, size_t transmitSize,
      int rank, uint32_t seqNo
#if defined(__enable_sycl_stream__)
      , sycl::stream cout
#endif
  ) : l_rank(rank/2), seqNo(seqNo)
#if defined(__enable_sycl_stream__)
      , cout(cout)
#endif
  {
    auto pairRank = rank ^ 1;

    auto* pairBuf0 = peerBuf0[pairRank];
    auto* pairBuf1 = peerBuf1[pairRank];

    auto ioClosePart = [&](T* p) {
      return (rank & 1) ? (T *)((uintptr_t)p + workSize * BiNRanks) : p;
    };
    auto ioFarPart = [&](T* p) {
      return (rank & 1) ? p : (T *)((uintptr_t)p + workSize * BiNRanks);
    };
    auto ipcClosePart = [&](T *p) {
      return p;
    };
    auto ipcFarPart = [&](T* p) {
      return (T *)((uintptr_t)p + transmitSize * BiNRanks);
    };

    // Indicated by y-id
    for (int i = 0; i < BiNRanks; ++ i) {
      ioForPeers[i] = (T *)((uintptr_t)ioClosePart(input) + i * workSize);
      ioForFar[i] = (T *)((uintptr_t)ioFarPart(input) + i * workSize);

      farScatterSink[i] = (T *)((uintptr_t)ipcFarPart(pairBuf0)
          + i * transmitSize);
      farGatherSink[i] = (T *)((uintptr_t)ipcFarPart(pairBuf1)
          + i * transmitSize);

      localFarScatterSink[i] = (T *)((uintptr_t)ipcFarPart(scatterBuf)
          + i * transmitSize);
      localFarGatherSink[i] = (T *)((uintptr_t)ipcFarPart(gatherBuf)
          + i * transmitSize);

      // Will jump over rank slot, XXX: be careful
      if (l_rank != i) {
        // even: 0, 2, 4, 6
        // odd:  1, 3, 5, 7
        auto r_index = 2 * i + (rank & 1);
        scatterSink[i] = (T *)((uintptr_t)ipcClosePart(peerBuf0[r_index])
            + l_rank * transmitSize);
        localGatherSink[i] = (T *)((uintptr_t)ipcClosePart(gatherBuf)
            + i * transmitSize);
      }
    }

    // Indicated by next?
    for (int i = 0; i < NPeers; ++ i) {
      int l_next = (l_rank + i + 1) % BiNRanks;
      int next = (rank + i * 2 + 2) % (2 * BiNRanks);

      localScatterSink[i] = (T *)((uintptr_t)ipcClosePart(scatterBuf)
          + l_next * transmitSize);
      gatherSink[i] = (T *)((uintptr_t)ipcClosePart(peerBuf1[next])
          + l_rank * transmitSize);
    }
  }

  // --------------------Input/Output buffer-------------------
  // Input partitions
  T* ioForPeers[BiNRanks];
  T* ioForFar[BiNRanks];

  // ---------------------IPC buffers-------------------------
  T* farScatterSink[BiNRanks];
  T* farGatherSink[BiNRanks];

  T* scatterSink[BiNRanks];
  T* gatherSink[NPeers];

  // ----------------Sinks, written by remotes----------------
  T* localFarScatterSink[BiNRanks];
  T* localFarGatherSink[BiNRanks];

  T* localScatterSink[NPeers];
  T* localGatherSink[BiNRanks];

  int l_rank;
  uint32_t seqNo;

#if defined(__enable_sycl_stream__)
  sycl::stream cout;
#endif
};
