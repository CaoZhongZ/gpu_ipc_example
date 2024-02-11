#include "transmit.hpp"

//
// We will remove sycl::event return in real API call.
// It's for test only.
//
template <typename T> sycl::event testSimpleTransmit(
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t size,
    int rank, int world, uint32_t step, sycl::queue queue) {
  return queue.submit([&](sycl::handler &cgh) {
      sycl::stream cout(16 * 1024, 1024 * 1024, cgh);
      switch(world) {
      case 2:
        cgh.parallel_for(
          launchParam,
          AllReduce<T, 2 -1, 16>(input, size, rank, step, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1, cout)
        );
      case 4:
        cgh.parallel_for(
          launchParam,
          AllReduce<T, 4 -1, 16>(input, size, rank, step, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1, cout)
        );
      case 8:
        cgh.parallel_for(
          launchParam,
          AllReduce<T, 8 -1, 16>(input, size, rank, step, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1, cout)
        );
      default:
        throw std::logic_error("Unsupported communication pattern!");
      }
  });
}

template sycl::event testSimpleTransmit<sycl::half>(
    sycl::nd_range<1> launchParam,
    sycl::half* input, sycl::half* ipcbuf0, sycl::half* ipcbuf1,
    sycl::half* const peerbuf0[], sycl::half* const peerbuf1[], size_t size,
    int rank, int world, uint32_t step, sycl::queue queue);
