#include "transmit.hpp"

template <typename T,
         int NPeers,
         template <typename, int, int> class Transmit,
         int SubGroupSize>
ine AllReduce<T, NPeers, Transmit, SubGroupSize>::scatterVerify(
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

template <typename T,
         int NPeers,
         template <typename, int, int> class Transmit,
         int SubGroupSize>
ine AllReduce<T, NPeers, Transmit, SubGroupSize>::stage2Verify(
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
        if (b + chunk * wireCapInInt < nWorkElemsInInt) {
          temp[b % n120B] = (b + chunk * wireCapInInt + contentOff) % 32 | next << 16;
        } else
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

template<typename T, int SubGroupSize>
int verifyTransmit(
    uint32_t* host, uint32_t step, int rank, int world, size_t nWorkElems
) {
  constexpr auto nElemPerInt = sizeof(uint32_t) / sizeof(T);
  switch(world) {
  case 2:
    return AllReduce<T, 2 -1, SimpleTransmit, SubGroupSize>::stage2Verify(
        host, rank, step, nWorkElems/nElemPerInt
    );
    break;
  case 4:
    return AllReduce<T, 4 -1, SimpleTransmit, SubGroupSize>::stage2Verify(
        host, rank, step, nWorkElems/nElemPerInt
    );
    break;
  case 8:
    return AllReduce<T, 8 -1, SimpleTransmit, SubGroupSize>::stage2Verify(
        host, rank, step, nWorkElems/nElemPerInt
    );
    break;
  default:
    throw std::logic_error("Not supported communication pattern");
  }
}

//
// We will remove sycl::event return in real API call.
// It's for test only.
//
template <typename T, int SubGroupSize> sycl::event testSimpleTransmit(
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t step, sycl::queue queue) {
  switch(world) {
  case 2:
    return queue.submit([&](sycl::handler &cgh) {
#if defined(__enable_sycl_stream__)
      sycl::stream cout(1024 * 1024, 16 * 1024, cgh);
#endif
        cgh.parallel_for(
          launchParam,
          AllReduce<T, 2 -1, SimpleTransmit, SubGroupSize>(
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
          AllReduce<T, 4 -1, SimpleTransmit, SubGroupSize>(
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
          AllReduce<T, 8 -1, SimpleTransmit, SubGroupSize>(
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

template sycl::event testSimpleTransmit<sycl::half, 16>(
    sycl::nd_range<1> launchParam,
    sycl::half* input, sycl::half* ipcbuf0, sycl::half* ipcbuf1,
    sycl::half* const peerbuf0[], sycl::half* const peerbuf1[], size_t size,
    int rank, int world, uint32_t step, sycl::queue queue);

template
int verifyTransmit<sycl::half, 16>(
    uint32_t* host, uint32_t step, int rank, int world, size_t nWorkElems
);
/* disabled temporarily for saving compile time
template sycl::event testSimpleTransmit<float, 16>(
    sycl::nd_range<1> launchParam,
    float* input, float* ipcbuf0, float* ipcbuf1,
    float* const peerbuf0[], float* const peerbuf1[], size_t size,
    int rank, int world, uint32_t step, sycl::queue queue);

template sycl::event testSimpleTransmit<sycl::half, 32>(
    sycl::nd_range<1> launchParam,
    sycl::half* input, sycl::half* ipcbuf0, sycl::half* ipcbuf1,
    sycl::half* const peerbuf0[], sycl::half* const peerbuf1[], size_t size,
    int rank, int world, uint32_t step, sycl::queue queue);

template sycl::event testSimpleTransmit<float, 32>(
    sycl::nd_range<1> launchParam,
    float* input, float* ipcbuf0, float* ipcbuf1,
    float* const peerbuf0[], float* const peerbuf1[], size_t size,
    int rank, int world, uint32_t step, sycl::queue queue);
*/
