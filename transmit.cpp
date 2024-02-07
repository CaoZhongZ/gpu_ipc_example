#include "transmit.hpp"

template <typename T> makeAllReduce(T *input, size_t size, int rank, int world,
    T* scatterBuf, T* gatherBuf, T* peerBuf0[], T* peerBuf1[], sycl::stream cout) {
  switch(world) {
  case 2:
    return AllReduce<T, 2, 16>(
        input, size, rank,
        scatterBuf, gatherbuf, peerBuf0, peerBuf1, cout);
  case 4:
    return AllReduce<T, 4, 16>(
        input, size, rank,
        scatterBuf, gatherbuf, peerBuf0, peerBuf1, cout);
  case 8:
    return AllReduce<T, 8, 16>(
        input, size, rank,
        scatterBuf, gatherbuf, peerBuf0, peerBuf1, cout);
  default:
    throw std::logic_error("Not supported ranks");
  }
}

//
// We will remove sycl::event return in real API call.
// It's for test only.
//
template <typename T> sycl::event testSimpleTransmit(
    sycl::nd_range<1> launchParam,
    T* input, void* ipcbuf0, void* ipcbuf1,
    void* peerbuf0[], void* peerbuf1[], size_t size,
    int rank, int world, sycl::queue queue) {
  return queue.submit([&](sycl::handler &cgh) {
      stream cout(16 * 1024, 1024 * 1024, cgh);
      auto kernel = makeAllReduce(
          input, size, rank, world,
          scatterBuf, gatherBuf, peerBuf0, peerBuf1, cout
      );
      cgh.parallel_for(
        launchParam, sycl::range<1>(16)), kernel
      );
  });
}
