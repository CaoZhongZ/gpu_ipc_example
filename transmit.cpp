#include "transmit.hpp"

template<typename T, int SubGroupSize>
void verifyTransmit(
    uint32_t* input, uint32_t* host,
    uint32_t step, int rank, int world, size_t nWorkElems
) {
  constexpr auto nElemPerInt = sizeof(uint32_t) / sizeof(T);
  switch(world) {
  case 2:
    AllReduce<T, 2 -1, SimpleTransmit, SubGroupSize>::scatterVerify(
        input, host, rank, step, nWorkElems/nElemPerInt
    );
    break;
  case 4:
    AllReduce<T, 4 -1, SimpleTransmit, SubGroupSize>::scatterVerify(
        input, host, rank, step, nWorkElems/nElemPerInt
    );
    break;
  case 8:
    AllReduce<T, 8 -1, SimpleTransmit, SubGroupSize>::scatterVerify(
        input, host, rank, step, nWorkElems/nElemPerInt
    );
    break;
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
void verifyTransmit<sycl::half, 16>(
    uint32_t* input, uint32_t* host, uint32_t step, size_t nWorkElems, int world
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
