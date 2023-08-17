#include <sycl/sycl.hpp>
#include "sycl_misc.hpp"

int main(int argc, char *argv[]) {
  auto queue = currentQueue(0, 0);

  struct coord {
    size_t y;
    size_t x;
  };

  coord* buf = (coord *)sycl::malloc_shared(2048 * sizeof(coord), queue);
  queue.memset(buf, 1, 2048 * sizeof(coord));

  auto e = queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<2>({2, 128}, {2, 64}),
          [=](sycl::nd_item<2> pos) {
            buf[pos.get_global_linear_id()] = {pos.get_local_id(0), pos.get_local_id(1)};
          });
  });

  e.wait();

  for (int i = 0; i < 256; ++ i)
    printf("(%ld, %ld) ", buf[i].y, buf[i].x);
  printf("\n");
}
