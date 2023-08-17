#include <sycl/sycl.cpp>
#include "sycl_misc.hpp"

int main(int argc, char *argv[]) {
  auto queue = currentQueue(0, 0);

  struct coord {
    size_t y, x;
  };

  coord* buffer = sycl::malloc_shared(2048 * sizeof(coord), queue);

  queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<2>({2, 128}, {2, 64}),
          [=](sycl::nd_item<2> pos) {
            buffer[pos.get_global_linear_id()] = {pos.get_local_id(0), pos.get_local_id(1)};
          });
  });
}
