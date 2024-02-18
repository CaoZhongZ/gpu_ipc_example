#include "utils.hpp"
#include "allreduce.hpp"

template <typename T,
         int NPeers,
         template <typename, int, int> class Transmit,
         int SubGroupSize>
int AllReduce<T, NPeers, Transmit, SubGroupSize>::scatterVerify(
    uint32_t* host, int rank, uint32_t flag, size_t nWorkElemsInInt
){
  constexpr auto n120B = 120 / 4;
  constexpr auto wireCapInInt = wireCapacity / sizeof(uint32_t);
  constexpr auto wireTransInInt = wireTransSize / sizeof(uint32_t);

  auto nTransmitElemsInInt
    = divUp(nWorkElemsInInt, wireCapInInt) * wireTransInInt;

  for (int i = 0; i < NPeers; ++ i) {
    int next = (rank + i + 1) % (NPeers + 1);
    auto* peer_ptr = host + nTransmitElemsInInt * slot(next, rank);
    size_t contentOff = rank * nWorkElemsInInt;

    // we are expecting pattern = (scale | next)
    size_t nChunks = divUp(nWorkElemsInInt, wireCapInInt);
    for (int chunk = 0; chunk < nChunks; ++ chunk) {
      uint32_t temp[32];
      uint32_t scrat[32];

      for (size_t b = 0, j = 0; b < wireCapInInt; ++ b, ++ j) {
        if (b + chunk * wireCapInInt < nWorkElemsInInt)
          temp[b % n120B] = (b + chunk * wireCapInInt + contentOff) % 32 | next << 16;
        else
          temp[b % n120B] = 0xffffffff;
        scrat[j % 32] = peer_ptr[j + chunk * wireTransInInt];

        // wireCapInInt will be divided by n120B.
        if (b % n120B == n120B -1) {
          temp[30] = temp[15]; temp[15] = flag; temp[31] = flag;
          scrat[30] = peer_ptr[++j + chunk * wireTransInInt];
          scrat[31] = peer_ptr[++j + chunk * wireTransInInt];

          for (auto k = 0; k < 32; ++ k) {
            if (temp[k] != scrat[k] && temp[k] != 0xffffffff) {
              std::cout<<"["<<rank<<"] Verify failed @"<<i<<", "<<k
                <<", expect:"<<temp[k]<<", but get:"<<scrat[k]<<std::endl;
              return -1;
    }}}}}
  }

  return 0;
}

template <typename T>
static void allreduce(T* allRanks[], int nRanks, size_t nelems) {
  for (int i = 0; i < nelems; ++ i) {
    T sum = 0.0;
    for (int r = 0; r < nRanks; ++ r)
      sum += allRanks[r][i];

    for (int r = 0; r < nRanks; ++ r)
      allRanks[r][i] = sum;
  }
}

template <typename T,
         int NPeers,
         template <typename, int, int> class Transmit,
         int SubGroupSize>
int AllReduce<T, NPeers, Transmit, SubGroupSize>::stage2Verify(
    T* host, int rank, uint32_t flag, size_t nelems
){
  constexpr auto n120B = 120 / sizeof(T);
  constexpr auto n128B = 128 / sizeof(T);
  constexpr auto wireCapInType = wireCapacity / sizeof(T);
  constexpr auto wireTransInType = wireTransSize / sizeof(T);
  constexpr auto NRanks = NPeers + 1;

  auto nWorkElems = nelems / NRanks;
  size_t nChunks = divUp(nWorkElems, wireCapInType);
  auto nTransmitElems = nChunks * wireTransInType;

  T* allRanks[NRanks];

  for (int i = 0; i < NRanks; ++ i)
    allRanks[i] = (T *)malloc(sizeof(T) * nWorkElems * NRanks);

  __scope_guard free_pointers([&] {
    for (int i = 0; i < NRanks; ++ i)
      free(allRanks[i]);
  });

  for (int i = 0; i < NRanks; ++ i)
    fill_pattern(allRanks[i], i, nelems);

  // simulate an allreduce
  allreduce(allRanks, NRanks, nelems);

  // Check each gather buffer
  for (int i = 0; i < NPeers; ++ i) {
    int next = (rank + i + 1) % (NPeers + 1);
    auto* peer_ptr = host + nTransmitElems * slot(next, rank);
    auto* local_ptr = allRanks[0] + nWorkElems * next;

    // we are expecting pattern = (scale | next)
    for (int chunk = 0; chunk < nChunks; ++ chunk) {
      T temp[n128B];
      T scrat[n128B];

      for (size_t b = 0, j = 0; b < wireCapInType; ++ b, ++ j) {
        if (b + chunk * wireCapInType < nWorkElems) {
          temp[b % n120B] = local_ptr[b + chunk * wireCapInType];
        } else
          temp[b % n120B] = -1.;
        scrat[j % n128B] = peer_ptr[j + chunk * wireTransInType];

        // wireCapInInt will be divided by n120B.
        if (b % n120B == n120B -1) {
          temp[60] = temp[30]; temp[61] = temp[31];

          *(uint32_t *)&temp[30] = flag;
          *(uint32_t *)&temp[62] = flag;

          scrat[60] = peer_ptr[++j + chunk * wireTransInType];
          scrat[61] = peer_ptr[++j + chunk * wireTransInType];

          // flag
          scrat[62] = peer_ptr[++j + chunk * wireTransInType];
          scrat[63] = peer_ptr[++j + chunk * wireTransInType];

          for (auto k = 0; k < n128B; ++ k) {
            if (temp[k] - scrat[k] > 1e-5 && temp[k] != -1.) {
              std::cout<<"["<<rank<<"] Verify failed @"
                <<i<<","<<k<<","<<b<<","<<chunk
                <<", expect:"<<temp[k]<<", but get:"<<scrat[k]<<std::endl;
              return -1;
    }}}}}
  }

  return 0;
}

template<typename T, template <typename, int, int> class Transmit>
int verifyTransmit(
    T* host, uint32_t step, int rank, int world, uint32_t simd, size_t nelems
) {
  if (simd == 16) {
    constexpr int SubGroupSize = 16;
    switch(world) {
    case 2:
      return AllReduce<T, 2 -1, Transmit, SubGroupSize>::stage2Verify(
          host, rank, step, nelems
      );
      break;
    case 4:
      return AllReduce<T, 4 -1, Transmit, SubGroupSize>::stage2Verify(
          host, rank, step, nelems
      );
      break;
    case 8:
      return AllReduce<T, 8 -1, Transmit, SubGroupSize>::stage2Verify(
          host, rank, step, nelems
      );
      break;
    default:
      throw std::logic_error("Not supported communication pattern");
    }
  } else {
    constexpr int SubGroupSize = 32;
    switch(world) {
    case 2:
      return AllReduce<T, 2 -1, Transmit, SubGroupSize>::stage2Verify(
          host, rank, step, nelems
      );
      break;
    case 4:
      return AllReduce<T, 4 -1, Transmit, SubGroupSize>::stage2Verify(
          host, rank, step, nelems
      );
      break;
    case 8:
      return AllReduce<T, 8 -1, Transmit, SubGroupSize>::stage2Verify(
          host, rank, step, nelems
      );
      break;
    default:
      throw std::logic_error("Not supported communication pattern");
    }
  }
}

//
// We will remove sycl::event return in real API call.
// It's for test only.
//
template <typename T, template <typename, int, int> class Transmit>
sycl::event testTransmit(
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t step, uint32_t subgroup, sycl::queue queue) {
  if (subgroup == 16) {
    constexpr int SubGroupSize = 16;
  switch(world) {
  case 2:
    return queue.submit([&](sycl::handler &cgh) {
#if defined(__enable_sycl_stream__)
      sycl::stream cout(1024 * 1024, 16 * 1024, cgh);
#endif
        cgh.parallel_for(
          launchParam,
          AllReduce<T, 2 -1, Transmit, SubGroupSize>(
            input, nelems, rank, step,
            ipcbuf0, ipcbuf1, peerbuf0, peerbuf1
#if defined(__enable_sycl_stream__)
            , cout
#endif
            )
        );
    });
  case 4:
    return queue.submit([&](sycl::handler &cgh) {
#if defined(__enable_sycl_stream__)
      sycl::stream cout(1024 * 1024, 16 * 1024, cgh);
#endif
        cgh.parallel_for(
          launchParam,
          AllReduce<T, 4 -1, Transmit, SubGroupSize>(
            input, nelems, rank, step,
            ipcbuf0, ipcbuf1, peerbuf0, peerbuf1
#if defined(__enable_sycl_stream__)
            , cout
#endif
            )
        );
    });
  case 8:
    return queue.submit([&](sycl::handler &cgh) {
#if defined(__enable_sycl_stream__)
      sycl::stream cout(1024 * 1024, 16 * 1024, cgh);
#endif
        cgh.parallel_for(
          launchParam,
          AllReduce<T, 8 -1, Transmit, SubGroupSize>(
            input, nelems, rank, step,
            ipcbuf0, ipcbuf1, peerbuf0, peerbuf1
#if defined(__enable_sycl_stream__)
            , cout
#endif
            )
        );
    });
  default:
    throw std::logic_error("Unsupported communication topology");
  }
  } else {
    constexpr int SubGroupSize = 32;
  switch(world) {
  case 2:
    return queue.submit([&](sycl::handler &cgh) {
#if defined(__enable_sycl_stream__)
      sycl::stream cout(1024 * 1024, 16 * 1024, cgh);
#endif
        cgh.parallel_for(
          launchParam,
          AllReduce<T, 2 -1, Transmit, SubGroupSize>(
            input, nelems, rank, step,
            ipcbuf0, ipcbuf1, peerbuf0, peerbuf1
#if defined(__enable_sycl_stream__)
            , cout
#endif
            )
        );
    });
  case 4:
    return queue.submit([&](sycl::handler &cgh) {
#if defined(__enable_sycl_stream__)
      sycl::stream cout(1024 * 1024, 16 * 1024, cgh);
#endif
        cgh.parallel_for(
          launchParam,
          AllReduce<T, 4 -1, Transmit, SubGroupSize>(
            input, nelems, rank, step,
            ipcbuf0, ipcbuf1, peerbuf0, peerbuf1
#if defined(__enable_sycl_stream__)
            , cout
#endif
            )
        );
    });
  case 8:
    return queue.submit([&](sycl::handler &cgh) {
#if defined(__enable_sycl_stream__)
      sycl::stream cout(1024 * 1024, 16 * 1024, cgh);
#endif
        cgh.parallel_for(
          launchParam,
          AllReduce<T, 8 -1, Transmit, SubGroupSize>(
            input, nelems, rank, step,
            ipcbuf0, ipcbuf1, peerbuf0, peerbuf1
#if defined(__enable_sycl_stream__)
            , cout
#endif
            )
        );
    });
  default:
    throw std::logic_error("Unsupported communication topology");
  }

  }
}

template sycl::event testTransmit<sycl::half, SimpleTransmit>(
    sycl::nd_range<1> launchParam,
    sycl::half* input, sycl::half* ipcbuf0, sycl::half* ipcbuf1,
    sycl::half* const peerbuf0[], sycl::half* const peerbuf1[], size_t size,
    int rank, int world, uint32_t step, uint32_t simd, sycl::queue queue);

template
int verifyTransmit<sycl::half, SimpleTransmit>(
    sycl::half* host, uint32_t step, int rank, int world, uint32_t simd, size_t nWorkElems
);
/* disabled temporarily for saving compile time
template sycl::event testTransmit<float, SimpleTransmit>(
    sycl::nd_range<1> launchParam,
    float* input, float* ipcbuf0, float* ipcbuf1,
    float* const peerbuf0[], float* const peerbuf1[], size_t size,
    int rank, int world, uint32_t step, uint32_t simd, sycl::queue queue);

template
int verifyTransmit<float, SimpleTransmit>(
    float* host, uint32_t step, int rank, int world, uint32_t simd, size_t nWorkElems
);
*/
