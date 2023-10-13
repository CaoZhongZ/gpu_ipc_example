#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>

#include "cxxopts.hpp"
#include "sycl_misc.hpp"
#include "ze_exception.hpp"
#include "ipc_exchange.h"
#include "copy_policy.hpp"
#include "sync_policy.hpp"

//
// If we launching full GPU occupancy, we have 256K ~ 4M elems of half type
//
template <typename T, int N_Peers,
         class SyncProto, template <typename, size_t> class CopyPolicy>
struct copy_persist {
  static constexpr int slice = 32; // 2M granularity

  using copy_type = CopyPolicy<T, slice>;
  using v_T = typename copy_type::v_T;

  static constexpr size_t n_loop = copy_type::n_loop;
  static constexpr int sync_size = SyncProto::local_count;

  copy_persist(
      const T* input,
      T* peer_ptrs[], uint32_t* peer_syncs[],
      T* scratch_ptrs[], uint32_t* semaphores[],
      size_t nelems, int rank, size_t group_limit_start, size_t group_limit_size
  ) : input(input), bank(peer_ptrs[rank]), lock_self(peer_syncs[rank]),
  far_bank(peer_ptrs[rank ^ 1]), lock_remote(peer_syncs[rank ^ 1]),
  nelems(nelems), rank(rank), group_limit_start(group_limit_start),
  group_limit_size(group_limit_size) {

    for (int i = 0; i < N_Peers; ++ i) {
      this->scratches[i] = scratch_ptrs[2*i + (rank & 1)];
      this->semaphores[i] = semaphores[2*i + (rank & 1)];
    }
  }

  static inline bool inactive(
      sycl::nd_item<1> pos, size_t start_group, size_t group_sz
  ) {
    auto group_id = pos.get_group(0);
    return (group_id < start_group || group_id >= start_group + group_sz);
  }

  static inline void original_copy(
      sycl::nd_item<1> pos,
      T* dst, const T* src, uint32_t *sync, uint32_t* remote,
      sycl::local_ptr<uint32_t> local_sync, sycl::local_ptr<uint32_t> local_wait,
      size_t nelems
  ) {
    size_t progress = 0;
    auto step = pos.get_global_range(0);

    for (auto off = pos.get_global_id()[0];
        off < nelems/v_T::size(); off += step * n_loop) {
      size_t local_off = 0;
      if constexpr (sync_size != 0)
        local_off = progress % sync_size;

      SyncProto::block_if_eq(
          pos, sync + progress, SyncProto::get_target(pos),
          local_sync + local_off, local_wait + local_off);

      copy_type::run(dst, src, off, step, nelems);

      SyncProto::finish(
          pos, SyncProto::get_target(pos),
          sync + progress, remote + progress,
          local_sync + local_off, local_wait + local_off);

      ++ progress;
    }
  }

  static inline void group_copy(
      sycl::nd_item<1> pos, size_t start_group, size_t group_sz,
      T* dst, const T* src, uint32_t *sync, uint32_t* remote,
      sycl::local_ptr<uint32_t> local_sync, sycl::local_ptr<uint32_t> local_wait,
      size_t nelems
  ) {
    if (inactive(pos, start_group, group_sz))
      return;

    auto start = start_group * pos.get_local_range(0);
    auto stride = group_sz * pos.get_local_range(0);
    size_t progress = 0;

    for (auto off = pos.get_global_id(0) - start;
        off < nelems/v_T::size(); off += stride * n_loop) {
      size_t local_off = 0;
      if constexpr (sync_size != 0)
        local_off = progress % sync_size;

      SyncProto::block_if_eq(
          pos, sync + progress, group_sz,
          local_sync + local_off, local_wait + local_off);

      copy_type::run(dst, src, off, stride, nelems);

      SyncProto::finish(
          pos, group_sz,
          sync + progress, remote + progress,
          local_sync + local_off, local_wait + local_off);

      ++ progress;
    }
  }

  // target 150GB/s for 4-peers
  static inline void group_reduce_scatter(
      sycl::nd_item<1> pos, size_t start_group, size_t group_sz,
      uint32_t signal, T* const dsts[], const T* src0, const T* src1,
      uint32_t *sync, uint32_t* const semaphores[], int rank, size_t nelems,
      sycl::local_ptr<uint32_t> local_sync,
      sycl::local_ptr<uint32_t> local_wait
  ) {
    if (inactive(pos, start_group, group_sz))
      return;

    // jump over next block
    auto start = start_group * pos.get_local_range(0);
    auto stride = group_sz * pos.get_local_range(0);
    auto start_off = (rank & 1) * (stride * n_loop);

    size_t progress = 0;

    auto group_id = pos.get_group(0) - start_group;

    auto* dst = dsts[group_id % N_Peers];
    auto* event = semaphores[group_id % N_Peers];

    int rank_off = (group_id % N_Peers) + rank/2 < N_Peers
                        ? rank/2 : (rank/2 - N_Peers);

    constexpr int comm_set = 2;

    for (auto off = pos.get_global_id(0) - start + start_off;
        off < nelems/v_T::size(); off += stride * n_loop * comm_set) {
      size_t local_off = 0;
      if constexpr (sync_size != 0)
        local_off = progress % sync_size;

      SyncProto::block_if_not(
          pos, sync + progress, signal,
          local_sync + local_off, local_wait + local_off);

      auto dst_off = off + rank_off * pos.get_local_range(0);

      copy_type::reduce(dst, src0, src1, dst_off, off, stride, nelems);

      SyncProto::signal(
          pos, event + progress, 1,
          local_sync + local_off, local_wait + local_off);

      progress += comm_set;
    }
  }

  inline void test_group_copy(sycl::nd_item<1> pos) const {
    uint32_t *local_sync = nullptr;
    uint32_t *local_wait = nullptr;

    if constexpr (sync_size != 0) {
      local_sync = __shared__<uint32_t [sync_size]>(pos.get_group());
      local_wait = __shared__<uint32_t [sync_size]>(pos.get_group());
      SyncProto::init_slm_flags(pos.get_local_id(0), local_sync, local_wait, sync_size);
    }

    auto* src = input;
    auto* dst = bank;
    auto* sync = lock_self;
    auto* remote = lock_remote;

    group_copy(pos, group_limit_start, group_limit_size,
        dst, src, sync, remote, local_sync, local_wait, nelems);
  }

  inline void test_reduce_scatter(sycl::nd_item<1> pos, uint32_t signal) const {
    uint32_t *local_sync = nullptr;
    uint32_t *local_wait = nullptr;

    if constexpr (sync_size != 0) {
      local_sync = __shared__<uint32_t [sync_size]>(pos.get_group());
      local_wait = __shared__<uint32_t [sync_size]>(pos.get_group());
      SyncProto::init_slm_flags(pos.get_local_id(0), local_sync, local_wait, sync_size);
    }

    auto* src0 = bank;
    auto* src1 = far_bank;

    group_reduce_scatter(pos, group_limit_start, group_limit_size, signal,
        scratches, src0, src1, lock_self, semaphores, rank, nelems, local_sync, local_wait);
  }

  void operator() [[sycl::reqd_sub_group_size(16)]] (sycl::nd_item<1> pos) const {
    test_reduce_scatter(pos, 0);
  }

  static sycl::event launch(
      sycl::queue queue,
      const T* input, T* interns[], T* scratches[], uint32_t* semaphores[],
      size_t nelems, int rank, size_t limit_start, size_t limit_size, uint32_t repeat = 1
  ) {
    if (nelems < v_T::size() || nelems % v_T::size() != 0)
      throw std::logic_error("Vectorize can't be satisfied");

    constexpr size_t local_size = 1024;
    constexpr size_t max_group = 64;

    size_t required_items = (nelems + copy_type::item_size() -1)
                                    / copy_type::item_size();
    size_t required_groups = (required_items + local_size -1)
                                    / local_size;

    size_t actual_groups = std::min(required_groups, max_group);
    auto global_size = actual_groups * local_size;

    printf("Launch copy_persist (%zu, %zu)\n", actual_groups, local_size);

    copy_persist offload(
        input, interns, scratches, semaphores, nelems, rank, limit_start, limit_size);

    auto e = queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}), offload);
    });

    // for performance evaluation
    for (int i = 1; i < repeat; ++ i) {
      e = queue.submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}), offload);
      });
    }
    return e;
  }

private:
  const T* input;

  // for cross MDFI
  T* bank;
  uint32_t* lock_self;

  T* far_bank;
  uint32_t* lock_remote;

  // for cross XeLink
  T* scratches[N_Peers];
  uint32_t* semaphores[N_Peers];

  size_t nelems;
  int rank;

  size_t group_limit_start;
  size_t group_limit_size;
};

template <template <typename, int, class,
         template <typename, size_t> class> class Copy,
         typename T, class SyncProto,
         template <typename, size_t> class CopyPolicy,
         typename ... Args>
static sycl::event launch(sycl::queue queue, int world, Args&& ... args) {
  switch (world) {
    case 2:
      return Copy<T, 1, SyncProto, CopyPolicy>::launch(
          queue, std::forward<Args>(args)...);
    case 4:
      return Copy<T, 2, SyncProto, CopyPolicy>::launch(
          queue, std::forward<Args>(args)...);
    case 8:
      return Copy<T, 4, SyncProto, CopyPolicy>::launch(
          queue, std::forward<Args>(args)...);
    default:
      throw std::logic_error("Unsupported world size!");
  }
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
void fill_sequential(void *p, int rank, size_t nelems) {
  auto *p_t = reinterpret_cast<T *>(p);

  for (size_t i = 0; i < nelems; ++ i) {
    p_t[i] = (float)(i + rank) / 1000.;
  }
}

template <typename T>
void fill_constant(void *p, T c, size_t nelems) {
  auto *p_t = reinterpret_cast<T *>(p);

  for (size_t i = 0; i < nelems; ++ i) {
    p_t[i] = c;
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
    ("s,sync_mode", "Synchronous mode", cxxopts::value<int>()->default_value("0"))
    ("b,begin", "Group Begin", cxxopts::value<size_t>()->default_value("0"))
    ("e,end", "Group End", cxxopts::value<size_t>()->default_value("64"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems_string = parsed_opts["nelems"].as<std::string>();
  auto sync_mode = parsed_opts["sync_mode"].as<int>();
  auto begin = parsed_opts["begin"].as<size_t>();
  auto end = parsed_opts["end"].as<size_t>();

  auto group_limit_start = begin;
  auto group_limit_size = end - begin;

  zeCheck(zeInit(0));

  // init section
  auto ret = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    std::cout<<"MPI init error"<<std::endl;
    return -1;
  }

  release_guard __mpifinalize([&] {MPI_Finalize();});

  int rank, world;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto nelems = parse_nelems(nelems_string);
  using test_type = sycl::half;
  size_t data_size = nelems * sizeof(test_type);
  size_t gpu_pagesz = 2 * 1024 * 1024ull;

  // sync every 256K elems
  size_t sync_grain = 256 * 1024ull;

  auto sync_elems = (nelems + sync_grain -1) / sync_grain;
  auto sync_size = (sync_elems * sizeof(uint32_t) + gpu_pagesz -1)
                  / gpu_pagesz * gpu_pagesz;

  auto queue = currentQueue(rank >> 1, rank & 1);

  auto* src = (test_type *)sycl::malloc_device(data_size, queue);

  auto* dst = (test_type *)sycl::malloc_device(data_size, queue);
  auto* dst_sync = sycl::malloc_device(sync_size, queue);

  auto* scratch = (test_type *)sycl::malloc_device(data_size, queue);
  auto* semaphore = sycl::malloc_device(sync_size, queue);

  auto* b_host = sycl::malloc_host(data_size, queue);
  auto* b_check = sycl::malloc_host(data_size, queue);
  auto* b_sync = sycl::malloc_host(sync_size, queue);

  release_guard __guard([&]{
    sycl::free(src, queue);
    sycl::free(dst, queue);
    sycl::free(dst_sync, queue);
    sycl::free(scratch, queue);
    sycl::free(semaphore, queue);
    sycl::free(b_host, queue);
    sycl::free(b_check, queue);
    sycl::free(b_sync, queue);
  });

  fill_constant<test_type>(b_host, rank, nelems);
  memset(b_sync, 0, sync_size);

  queue.memcpy(src, b_host, data_size);
  queue.memcpy(dst, b_host, data_size);
  queue.memcpy(dst_sync, b_sync, sync_size);
  queue.memset(scratch, 0, data_size);
  queue.memcpy(semaphore, b_sync, sync_size);
  queue.wait();

  void* peer_bases[world];
  size_t offsets[world];

  union ipc_handle_t {
    ze_ipc_mem_handle_t ipc_handle;
    int fd;
  } handle_fd, lock_fd, scratch_fd, sync_fd;

#define exchange_buffers(handle, ptr, bases, offs, ptrs) \
  handle.ipc_handle = open_peer_ipc_mems(ptr, rank, world, bases, offs); \
  release_guard __close_ipc_handles_ ## handle ([&] { \
      auto l0_ctx = sycl::get_native< \
          sycl::backend::ext_oneapi_level_zero>(queue.get_context()); \
      for (int i = 0; i < world; ++ i) \
        if (i != rank) \
          zeCheck(zeMemCloseIpcHandle(l0_ctx, peer_bases[i])); \
      /* zeCheck(zeMemPutIpcHandle(l0_ctx, handle.ipc_handle)); */ \
  }); \
  std::transform(bases, bases + world, offs, ptrs, \
      [](void *p, size_t off) {return (char *)p + off;});

  void* peer_ptrs[world];
  exchange_buffers(handle_fd, dst, peer_bases, offsets, peer_ptrs);

  void* sync_bases[world];
  void* sync_ptrs[world];
  exchange_buffers(lock_fd, dst_sync, sync_bases, offsets, sync_ptrs);

  void* scratch_bases[world];
  void* scratches[world];
  exchange_buffers(scratch_fd, scratch, scratch_bases, offsets, scratches);

  void* sema_bases[world];
  void* semaphores[world];
  exchange_buffers(sync_fd, semaphore, sema_bases, offsets, semaphores);

  // for debugger purpose
  auto* monitor = mmap_host(data_size, handle_fd.fd);
  auto* mon_scratch = mmap_host(data_size, scratch_fd.fd);
  uint32_t* mon_lock = (uint32_t *)mmap_host(sync_size, lock_fd.fd);
  uint32_t* mon_sync = (uint32_t *)mmap_host(sync_size, sync_fd.fd);
  (void)monitor; (void)mon_scratch; (void)mon_lock;

  sycl::event e;
  switch(sync_mode) {
  case 0:
    e = launch<copy_persist, test_type, dummy_sync, chunk_copy>(
        queue, world, (test_type *)src,
        (test_type **)peer_ptrs, (uint32_t **)sync_ptrs,
        (test_type **)scratches, (uint32_t **)semaphores,
        nelems, rank, group_limit_start, group_limit_size);
    break;
  case 1:
    e = launch<copy_persist, test_type, hierarchy_sync, chunk_copy>(
        queue, world, (test_type *)src,
        (test_type **)peer_ptrs, (uint32_t **)sync_ptrs,
        (test_type **)scratches, (uint32_t **)semaphores,
        nelems, rank, group_limit_start, group_limit_size);
    break;
  default:
    throw std::logic_error("Unsupported synchronous mode.");
  }

  // auto e = queue.memcpy(dst, src, alloc_size);
  auto bandwidth = bandwidth_from_event<test_type>(e, nelems/2);
  auto time = time_from_event(e);
  printf("[%d]Copy %zu half in %fns, bandwidth: %fGB/s\n",
      rank, data_size, time, bandwidth);

  queue.memcpy(b_check, dst, data_size);
  queue.wait();

  int pos = memcmp(b_check, b_host, data_size);
  if ( pos == 0)
    printf("Verified\n");
  else
    printf("Error at %d\n", pos);

  printf("Sync elems %#x, %#x, ..., %#x\n",
      mon_sync[0], mon_sync[1], mon_sync[sync_elems -1]);
}
