#include <iostream>
#include <sycl/sycl.hpp>

#include "cxxopts.hpp"
#include "sycl_misc.hpp"

template <typename T, int Unroll>
struct seq_copy {
  static inline void run(sycl::nd_item<1> pos, T* dst, const T* src, size_t elems) {
    for (size_t off = pos.get_global_id(0);
        off < elems; off += pos.get_global_range(0)) {
#     pragma unroll
      for (int i = 0; i < Unroll; ++ i) {
        auto i_off = Unroll * off + i;
        dst[i_off] = src[i_off];
      }
    }
  }
};

template <typename T, int Unroll>
struct jump_copy {
  static inline void run(sycl::nd_item<1> pos, T* dst, const T* src, size_t elems) {
    for (size_t off = pos.get_global_id(0);
        off  < elems * Unroll; off += pos.get_global_range(0)* Unroll) {
#     pragma unroll
      for (int i = 0; i < Unroll; ++ i) {
        auto i_off = off + pos.get_global_range(0) * i;
        dst[i_off] = src[i_off];
      }
    }
  }
};

template <typename T, int lane_v, int Unroll, template <typename, int> class copy_policy>
struct copy_persist {
  using v_T = sycl::vec<T, lane_v/sizeof(T)>;

  void operator() [[sycl::reqd_sub_group_size(16)]] (sycl::nd_item<1> pos) const {
    copy_policy<v_T, Unroll>::run(pos, dst, src, vu_nelems);
  }

  copy_persist(T* dst, const T* src, size_t elems) :
    src(reinterpret_cast<const v_T *>(src)),
    dst(reinterpret_cast<v_T *>(dst)),
    vu_nelems(elems/v_T::size()/Unroll) {}

  static sycl::event launch(sycl::queue queue, T* dst, const T* src, size_t nelems,
      size_t max_group =64, size_t local_size = 1024, uint32_t repeat = 1) {
    if (nelems < v_T::size() || nelems % v_T::size() != 0)
      throw std::logic_error("Vectorize can't be satisfied");

    auto v_nelems = nelems / v_T::size();

    if (v_nelems % Unroll != 0)
      throw std::logic_error("Unroll can't be satisfied");

    auto vu_nelems = v_nelems / Unroll;

    size_t required_groups = (vu_nelems + local_size -1)/local_size;
    auto group_num = std::min(required_groups, max_group);
    size_t global_size = group_num * local_size;

    printf("Launch copy_persist (%zu, %zu) with unroll %d\n", group_num, local_size, Unroll);

    auto e = queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}),
              copy_persist(dst, src, nelems));
    });

    for (int i = 1; i < repeat; ++ i) {
      e = queue.submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}),
                copy_persist(dst, src, nelems));
      });
    }
    return e;
  }

  const v_T* src;
  v_T* dst;
  size_t vu_nelems;
};

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
  return (double)(nelems * sizeof(T) * 2) / (double)(end - start);
}

double time_from_event(sycl::event e) {
  e.wait();
  auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  return (double)(end -start);
}

int main(int argc, char *argv[]) {
  cxxopts::Options opts("Copy", "Copy baseline for performance");
  opts.allow_unrecognised_options();
  opts.add_options()
    ("n,nelems", "Number of elements", cxxopts::value<std::string>()->default_value("16MB"))
    ("u,unroll", "Unroll request", cxxopts::value<uint32_t>()->default_value("1"))
    ("g,groups", "Max Group Size", cxxopts::value<size_t>()->default_value("64"))
    ("l,local", "Local size", cxxopts::value<size_t>()->default_value("512"))
    ("s,sequential", "Sequential Unroll", cxxopts::value<bool>()->default_value("false"))
    ("t,tile", "On which tile to deploy the test", cxxopts::value<uint32_t>()->default_value("0"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems_string = parsed_opts["nelems"].as<std::string>();
  auto unroll =  parsed_opts["unroll"].as<uint32_t>();
  auto local = parsed_opts["local"].as<size_t>();
  auto max_groups = parsed_opts["groups"].as<size_t>();
  auto seq = parsed_opts["sequential"].as<bool>();
  auto tile = parsed_opts["tile"].as<uint32_t>();

  auto nelems = parse_nelems(nelems_string);
  using test_type = sycl::half;
  constexpr uint32_t v_lane = 16;
  size_t alloc_size = nelems * sizeof(test_type);

  auto queue = currentQueue(tile >> 1, tile & 1);

  auto* src = (test_type *)sycl::malloc_device(alloc_size, queue);
  auto* dst = (test_type *)sycl::malloc_device(alloc_size, queue);
  auto* b_host = sycl::malloc_host(alloc_size, queue);
  fill_sequential<test_type>(b_host, 0, alloc_size);

  queue.memcpy(src, b_host, alloc_size);
  queue.wait();

  sycl::event e;

  if (seq) {
    switch (unroll) {
      case 1:
        e = copy_persist<test_type, v_lane, 1, seq_copy>::launch(queue, dst, src, nelems, max_groups, local);
        break;
      case 2:
        e = copy_persist<test_type, v_lane, 2, seq_copy>::launch(queue, dst, src, nelems, max_groups, local);
        break;
      case 4:
        e = copy_persist<test_type, v_lane, 4, seq_copy>::launch(queue, dst, src, nelems, max_groups, local);
        break;
      case 8:
        e = copy_persist<test_type, v_lane, 8, seq_copy>::launch(queue, dst, src, nelems, max_groups, local);
        break;
      default:
        throw std::logic_error("Unroll request not supported");
    }
  } else {
    switch (unroll) {
      case 1:
        e = copy_persist<test_type, v_lane, 1, jump_copy>::launch(queue, dst, src, nelems, max_groups, local);
        break;
      case 2:
        e = copy_persist<test_type, v_lane, 2, jump_copy>::launch(queue, dst, src, nelems, max_groups, local);
        break;
      case 4:
        e = copy_persist<test_type, v_lane, 4, jump_copy>::launch(queue, dst, src, nelems, max_groups, local);
        break;
      case 8:
        e = copy_persist<test_type, v_lane, 8, jump_copy>::launch(queue, dst, src, nelems, max_groups, local);
        break;
      default:
        throw std::logic_error("Unroll request not supported");
    }
  }
  // auto e = queue.memcpy(dst, src, alloc_size);
  auto bandwidth = bandwidth_from_event<test_type>(e, nelems);
  auto time = time_from_event(e);
  printf("Copy %zu half in %fns, bandwidth: %fGB/s\n", alloc_size, time, bandwidth);

  auto* b_check = sycl::malloc_host(alloc_size, queue);
  queue.memcpy(b_check, dst, alloc_size);
  queue.wait();

  if (memcmp(b_check, b_host, alloc_size) == 0)
    printf("Verified\n");
}
