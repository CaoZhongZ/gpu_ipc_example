#include <iostream>
#include <sycl/sycl.hpp>

#include "cxxopts.hpp"
#include "sycl_misc.hpp"

size_t parse_nelems(const std::string& nelems_string) {
  size_t base = 1;
  size_t pos = nelems_string.rfind("K");
  if (pos != std::string::npos) {
    base = 1024ull;
  } else {
    pos = nelems_string.rfind("M");
    if (pos != std::string::npos)
      base = 1024 * 1024ull;
    else {
      pos = nelems_string.rfind("G");
      if (pos != std::string::npos)
        base = 1024 * 1024 * 1024ull;
    }
  }

  return stoull(nelems_string) * base;
}

void fill_sequential(uint64_t *p, int rank, size_t nelems) {
  for (size_t i = 0; i < nelems; ++ i) {
    p[i] = i + rank;
  }
}

int main(int argc, char *argv[]) {
  cxxopts::Options opts("Copy", "Copy baseline for performance");
  opts.allow_unrecognised_options();
  opts.add_options()
    ("n,nelems", "Number of elements", cxxopts::value<std::string>()->default_value("16MB"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems_string = parsed_opts["nelems"].as<std::string>();

  auto nelems = parse_nelems(nelems_string);
  using test_type = uint64_t;
  size_t alloc_size = nelems * sizeof(test_type);

  auto queue = currentQueue(0, 0);

  auto* src = (test_type *)sycl::malloc_device(alloc_size, queue);
  auto* dst = (sycl::vec<uint32_t, 2> *)sycl::malloc_device(alloc_size, queue);
  auto* b_host = sycl::malloc_host(alloc_size, queue);
  auto* b_check = sycl::malloc_host(alloc_size, queue);

  release_guard __guard([&]{
    sycl::free(src, queue);
    sycl::free(dst, queue);
    sycl::free(b_host, queue);
    sycl::free(b_check, queue);
  });

  fill_sequential((uint64_t *)b_host, 1, nelems);

  union merge {
    uint64_t whole;
    sycl::vec<uint32_t, 2> split;
  };

  queue.memcpy(src, b_host, alloc_size);
  queue.wait();

  queue.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::nd_range<1> {nelems, 32}, [=](sycl::nd_item<1> pos) {
        static constexpr int simd_array[32] = {
          16, 77, 102, 154, 816, 0xffff, 0xffff, 0xffff,
          0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff,
          16, 77, 102, 154, 816, 0xffff, 0xffff, 0xffff,
          0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff
        };

        int i = simd_array[pos.get_local_id()];
        dst[pos.get_global_id(0)] = i;
      });
  });

  queue.memcpy(b_check, dst, alloc_size);
  queue.wait();
}
