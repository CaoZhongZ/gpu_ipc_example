#include <iostream>
#include <sycl/sycl.hpp>

#include "cxxopts.hpp"
#include "sycl_misc.hpp"

template <typename T, int lane_v>
struct copy_persist {
  using v_T = sycl::vec<T, lane_v/sizeof(T)>;

  void operator() [[sycl::reqd_sub_group_size(16)]] (
      sycl::nd_item<1> pos) const {
    for (size_t off = pos.get_global_id(0);
        off < v_elems; off += pos.get_global_range(0)) {
      dst[off] = src[off];
    }
  }

  copy_persist(T* dst, const T* src, size_t elems) :
    src(reinterpret_cast<const v_T *>(src)),
    dst(reinterpret_cast<v_T *>(dst)),
    v_elems(elems/v_T::size()) {}

  static sycl::event launch(T* dst, const T* src, size_t nelems) {
    constexpr size_t max_group = 512;
    constexpr size_t local_size = 128;

    size_t group_size = (nelems/v_T::size() + local_size -1)/local_size;
    group_size = std::min(group_size, max_group);
    size_t global_size = group_size * local_size;

    auto queue = currentQueue(0, 0);
    auto e = queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}),
              copy_persist(dst, src, nelems));
    });
    return e;
  }

  const v_T* src;
  v_T* dst;
  size_t v_elems;
};

size_t parse_nelems(const std::string& nelems_string) {
  size_t base = 1;
  size_t pos = nelems_string.rfind("M");
  if (pos != std::string::npos) {
    base = 1024 * 1024ull;
  } else {
    pos = nelems_string.rfind("G");
    if (pos != std::string::npos)
      base = 1024 * 1024 * 1024ull;
  }

  return stoull(nelems_string) * base;
}

template <typename T>
void fill_sequential(void *p, int rank, size_t size) {
  auto typed_sz = size / sizeof(T);
  auto *p_t = reinterpret_cast<T *>(p);

  for (size_t i = 0; i < typed_sz; ++ i) {
    p_t[i] = i + rank;
  }
}

template <typename T>
double bandwidth_from_event(sycl::event e, size_t nelems) {
  e.wait();
  auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  // since timestamp is in the unit of ns, then the bandwidth is GB/s in unit
  return (double)(nelems * sizeof(T)) / (double)(end - start);
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
  using test_type = sycl::half;
  constexpr uint32_t v_lane = 16;
  size_t alloc_size = nelems * sizeof(test_type);

  auto queue = currentQueue(0, 0);

  auto* src = (test_type *)sycl::malloc_device(alloc_size, queue);
  auto* dst = (test_type *)sycl::malloc_device(alloc_size, queue);
  auto* b_host = sycl::malloc_device(alloc_size, queue);
  fill_sequential<test_type>(b_host, 0, alloc_size);

  queue.memcpy(src, b_host, alloc_size);
  queue.wait();

  auto e = copy_persist<test_type, v_lane>::launch(dst, src, nelems);
  auto bandwidth = bandwidth_from_event<test_type>(e, nelems);
  printf("Copy persistent kernel bandwidth: %fGB/s\n", bandwidth);
}
