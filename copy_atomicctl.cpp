#include <iostream>
#include <sycl/sycl.hpp>

#include "cxxopts.hpp"
#include "sycl_misc.hpp"
#include "ipc_exchange.h"

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

template <typename T, int Unroll>
struct group_copy {
  static inline void run(sycl::nd_item<1> pos, T* dst, const T* src, size_t elems) {
    auto grp = pos.get_group();
    auto n_grps = grp.get_group_linear_range();
    auto grp_sz = grp.get_local_linear_range();

    auto grp_id = grp.get_group_id(0);
    auto loc_id = grp.get_local_id(0);
    auto slice = elems * Unroll / n_grps;
    auto base_off = grp_id * slice;

    for (auto off = base_off + loc_id; off < base_off + slice; off += grp_sz * Unroll) {
#     pragma unroll
      for (int i = 0; i < Unroll; ++ i) {
        dst[off + i * grp_sz] = src[off + i * grp_sz];
      }
    }
  }
};

template <typename T>
class atomic_ref: public sycl::atomic_ref<T,
                      sycl::memory_order::relaxed,
                      sycl::memory_scope::device,
                      sycl::access::address_space::global_space> {
public:
  atomic_ref(T& r) : sycl::atomic_ref<T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space> (r)
  {}
};

template <typename T>
class slm_atomic_ref : public sycl::atomic_ref<T,
                      sycl::memory_order::relaxed,
                      sycl::memory_scope::device,
                      sycl::access::address_space::local_space> {
public:
  slm_atomic_ref(T& r) : sycl::atomic_ref<T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::local_space> (r)
  {}
};

class dummy_sync {
public:
  // do nothing
  static inline void init_slm_flags(
      sycl::nd_item<1>,
      sycl::local_ptr<uint32_t>,
      sycl::local_ptr<bool>,
      size_t
  ) {}

  static inline void wait_on(
      sycl::nd_item<1>, uint32_t*, uint32_t
      sycl::local_ptr<uint32_t>,
      sycl::local_ptr<uint32_t>
  ) {}

  static inline void finish(
      sycl::nd_item<1>, uint32_t, uint32_t*, uint32_t*
  ) {};

  static inline size_t get_target(
      sycl::nd_item<1>
  ) {
    return 0;
  }
};

class flat_sync {
public:
  // do nothing
  static inline void init_slm_flags(
      sycl::nd_item<1> pos,
      sycl::local_ptr<uint32_t> counter_array,
      sycl::local_ptr<bool> wait_array,
      size_t n_slot
  ) {}

  static inline void wait_on(
      sycl::nd_item<1> pos, uint32_t* flag, uint32_t value
      sycl::local_ptr<uint32_t>,
      sycl::local_ptr<uint32_t>
  ) {
    atomic_ref g_flag(*flag);

    if (pos.get_sub_group().leader()) {
      while(g_flag.load() != value);
    }

    // remove it if not needed
    sycl::group_barrier(g.get_sub_group());
  }

  static inline void finish(
      sycl::nd_item<1> pos, uint32_t target, uint32_t* flag, uint32_t *remote
  ) {
    atomic_ref g_flag(*flag);
    atomic_ref r_flag(*remote);

    if (pos.get_sub_group().leader()) {
      uint32_t count = g_flag++;

      if (count == target - 1 || count == 2*target -1)
        r_flag += g_flag.load();
    }
    // remove it if not needed
    sycl::group_barrier(g.get_sub_group());
  };

  static inline size_t get_target(
      sycl::nd_item<1> pos
  ) {
    return pos.get_sub_group().get_group_range(0);
  }
};

class hierarchy_sync {
  static inline bool group_leader(
      sycl::nd_item<1> pos,
      sycl::local_ptr<uint32_t> local_counter
  ) {
    slm_atomic_ref l_c(*local_counter);
    return (l_c ++ == 0);
  }

  static inline bool group_tail(
      sycl::nd_item<1> pos,
      sycl::local_ptr<uint32_t> local_counter
  ) {
    slm_atomic_ref l_c(*local_counter);
    return (l_c ++ == 2*pos.get_sub_group().get_group_range() -1);
  }

public:
  // slow, group barrier inside
  static inline void init_slm_flags(
      sycl::nd_item<1> pos,
      sycl::local_ptr<uint32_t> counter_array,
      sycl::local_ptr<bool> wait_array,
      size_t n_slot
  ) {
    auto off = pos.get_local_size();
    if (off < n_slot) {
      counter_array[off] = 0;
      wait_array[off] = true;
    }

    sycl::group_barrier(pos.get_group());
  }

  static inline void wait_on(
      sycl::nd_item<1> pos,
      uint32_t* flag, uint32_t value,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {
    slm_atomic_ref l_wait(*local_wait);
    slm_atomic_ref l_c(*local_counter);

    atomic_ref g_flag(*flag);

    if (pos.get_sub_group().leader()) {
      if (group_leader(pos, local_counter)) {
        while(g_flag.load() != value);
        l_wait.store(false);
      } else {
        while(l_wait.load());
      }
    }

    // remove it if not needed
    sycl::group_barrier(g.get_sub_group());
  }

  static inline void finish(
      sycl::nd_item<1> pos, uint32_t target, uint32_t* flag, uint32_t *remote
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {
    atomic_ref g_flag(*flag);
    atomic_ref r_flag(*remote);
    slm_atomic_ref l_c(*local_counter);
    slm_atomic_ref l_w(*local_wait);

    if (pos.get_sub_group().leader()) {
      if (group_tail(pos, local_counter)) {
        uint32_t count = g_flag++;

        if (count == target - 1 || count == 2*target -1)
          r_flag += g_flag.load();

        l_c.store(0);
        l_w.store(true);
      }
    }
    // remove it if not needed
    sycl::group_barrier(g.get_sub_group());
  };

  static inline size_t get_target(
      sycl::nd_item<1> pos
  ) {
    return pos.get_group_range(0);
  }
};

//
// Next: Copy in chunks of certain size for maximized group occupation
//
// building block for 4 , 8, 16, 32, 64 elems copy
//
template <typename T, size_t NElems>
struct chunk_copy {
  static constexpr size_t v_lane = NElems * sizeof(T) < 16 ? 8 : 16;
  static constexpr size_t v_T = sycl::vec<T, v_lane/sizeof(T)>;
  static_assert(NElems % v_T::size() == 0);
  static constexpr size_t n_loop = NElems / v_T::size();

  static inline size_t chunk_size(sycl::nd_item<1> pos) {
    return NElems * pos.get_global_size(0) * n_loop;
  }

  static inline size_t item_size() {
    return NElems;
  }

  static inline void run(
      sycl::nd_item<1> pos,
      T* dst, const T* src,
      size_t start, size_t nelems
  ) {
    auto* v_dst = reinterpret_cast<v_T *>(dst);
    auto* v_src = reinterpret_cast<v_T *>(src);
    auto bound = nelems / v_T::size();
    auto off = pos.get_global_id(0) + start;
    auto step = pos.get_global_range(0);

#   pragma unroll
    for (int n = 0; n < n_loop; ++ n) {
      if (off < bound)
        v_dst[off] = v_src[off];
      off += step;
    }
  }
}

//
// If we launching full GPU occupancy, we have 256K ~ 4M elems of half type
//
template <typename T, class SyncProto, template <typename, size_t> CopyPolicy>
struct copy_persist {
  static constexpr int sync_size = 16;
  static constexpr int slice = 32; // 2M granularity

  using copy_type = CopyPolicy<T, slice>;
  using v_T = copy_type::v_T;

  copy_persist(T* dst, const T* src, uint32_t *sync, uint32_t* remote, size_t nelems)
    : src(src), dst(dst), sync(sync), remote(remote), nelems(nelems) {}

  void operator() [[sycl::reqd_sub_group_size(16)]] (sycl::nd_item<1> pos) const {
    auto* local_sync = __shared__<uint32_t[sync_size]>(pos.get_group());
    auto* local_wait = __shared__<bool[sync_size]>(pos.get_group());
    sync_protocol::init_slm_flags(pos, local_sync, local_wait, sync_size);

    uint32_t progress = 0;
    uint32_t start = 0;

    while (start < nelems) {
      SyncProto::wait_on(
          pos, sync + progress,
          local_sync + progress % sync_size, local_wait + progress % sync_size);

      copy_type::run(pos, dst, src, start, nelems);

      SyncProto::finish(
          pos, sync_protocol::get_target(),
          sync + progress, remote + progress,
          local_sync + progress % sync_size, local_wait + progress % sync_size);

      ++ progress;
      start += copy_type<T, slice>::chunk_size(pos);
    }
  }

  static sycl::event launch(
      sycl::queue queue, T* dst, const T* src,
      uint32_t *sync, uint32_t* remote, size_t nelems,
      size_t max_group =64, size_t local_size = 1024, uint32_t repeat = 1
  ) {
    if (nelems < v_T::size() || nelems % v_T::size() != 0)
      throw std::logic_error("Vectorize can't be satisfied");

    size_t required_items = (nelems + chunk_copy::item_size() -1)
                                    / chunk_copy::item_size();
    size_t required_groups = (required_items + local_size -1)
                                    / local_size;

    size_t actual_groups = std::min(required_groups, max_group);
    auto global_size = actual_groups * local_size;

    printf("Launch copy_persist (%zu, %zu)", actual_groups, local_size);

    auto e = queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}),
              copy_persist(dst, src, nelems));
    });

    // for performance evaluation
    for (int i = 1; i < repeat; ++ i) {
      e = queue.submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}),
                copy_persist(dst, src, nelems));
      });
    }
    return e;
  }

private:
  const T* src;
  T* dst;
  uint32_t *sync;
  uint32_t *remote;
  size_t nelems;
};

template <template <typename, class, class> class Copy,
         typename T, class SyncProto, template <typename, int> class CopyPolicy,
         typename ... Args>
static sycl::event launch(sycl::queue queue, Args&& ... args) {
  return Copy<T, SyncProto, CopyPolicy>::launch(queue, std::forward<Args>(args)...);
}

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
    ("s,sync_mode", "Synchronous mode", cxxopts::value<int>()->default_value("0"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems_string = parsed_opts["nelems"].as<std::string>();
  auto unroll =  parsed_opts["unroll"].as<uint32_t>();
  auto local = parsed_opts["local"].as<size_t>();
  auto max_groups = parsed_opts["groups"].as<size_t>();
  auto sync = parsed_opts["sync_mode"].as<int>();

  // init section
  auto ret = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    std::cout<<"MPI init error"<<std::endl;
    return -1;
  }

  auto nelems = parse_nelems(nelems_string);
  using test_type = sycl::half;
  size_t data_size = nelems * sizeof(test_type);
  size_t gpu_pagesz = 2 * 1024 * 1024ull;
  size_t sync_grain = 256 * 1024ull;

  auto sync_size = (alloc_size + sync_grain -1) / sync_grain;
  auto alloc_size = (data_size + sync_size + gpu_pagesz -1) / gpu_pagesz * gpu_pagesz;

  auto queue = currentQueue(rank >> 1, rank & 1);

  auto* src = (test_type *)sycl::malloc_device(data_size, queue);
  auto* dst = (test_type *)sycl::malloc_device(alloc_size, queue);

  // initialize data + sync (init)
  auto* b_host = sycl::malloc_host(alloc_size, queue);
  // check data + sync (write)
  auto* b_check = sycl::malloc_host(alloc_size, queue);

  size_t sync_off = (data_size + 128 -1) / 128 * 128;
  auto* s_host = b_host + sync_off;
  auto* sync = dst + sync_off;

  release_guard __guard([&]{
    sycl::free(src, queue);
    sycl::free(dst, queue);
    sycl::free(b_host, queue);
    sycl::free(b_check, queue);
  });

  release_guard __mpifinalize([&] {MPI_Finalize();});

  zeCheck(zeInit(0));
  int rank, world;

  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  fill_sequential<test_type>(b_host, 0, data_size);
  memset(s_host, 0, sync_size);

  queue.memcpy(src, b_host, data_size);
  queue.memcpy(sync, s_host, sync_size);
  queue.wait();

  void* peer_bases[world];
  size_t offsets[world];
  auto ipc_handle = open_peer_ipc_mems(dst, rank, world, peer_bases, offsets);

  void *peer_ptrs[world];
  std::transform(peer_bases, peer_bases + world, offsets, peer_ptrs,
      [](void *p, size_t off) {return (char *)p + off;});

  sycl::event e;
  auto *remote = peer_ptrs[rank ^ 1] + sync_off;

  switch(sync) {
  case 0:
    e = launch<copy_persist, test_type, dummy_sync, chunk_copy>(
        queue, dst, src, sync, remote, nelems, max_groups, local);
    break;
  case 1:
    e = launch<copy_persist, test_type, flat_sync, chunk_copy>(
        queue, dst, src, sync, remote, nelems, max_groups, local);
    break;
  case 2:
    e = launch<copy_persist, test_type, hierarchy_sync, chunk_copy>(
        queue, dst, src, sync, remote, nelems, max_groups, local);
    break;
  default:
    throw std::logic_error("Unsupported synchronous mode.");
  }

  // auto e = queue.memcpy(dst, src, alloc_size);
  auto bandwidth = bandwidth_from_event<test_type>(e, nelems);
  auto time = time_from_event(e);
  printf("Copy %zu half in %fns, bandwidth: %fGB/s\n", alloc_size, time, bandwidth);

  queue.memcpy(b_check, dst, alloc_size);
  queue.wait();

  if (memcmp(b_check, b_host, alloc_size) == 0)
    printf("Verified\n");
}
