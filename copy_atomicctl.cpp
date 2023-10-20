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
      T* input, uint32_t* team_sync,
      T* peer_ptrs[], uint32_t* peer_syncs[],
      T* temp_ptrs[], uint32_t* temp_syncs[],
      size_t nelems, int rank, size_t group_limit_start, size_t group_limit_size,
      sycl::stream cout
  ) : input(input), team_sync(team_sync),
  bank(peer_ptrs[rank]), lock_self(peer_syncs[rank]),
  far_bank(peer_ptrs[rank ^ 1]), lock_remote(peer_syncs[rank ^ 1]),
  nelems(nelems), rank(rank), group_limit_start(group_limit_start),
  group_limit_size(group_limit_size), cout(cout) {
    for (int i = 0; i < N_Peers; ++ i) {
      this->peer_ptrs[i] = peer_ptrs[2*i + (rank & 1)];
      this->peer_syncs[i] = peer_syncs[2*i + (rank & 1)];
      this->r_peer_syncs[i] = peer_syncs[2*i + 1 - (rank & 1)];

      this->temp_ptrs[i] = temp_ptrs[2*i + (rank & 1)];
      this->temp_syncs[i] = temp_syncs[2*i + (rank & 1)];

      this->r_temp_ptrs[i] = temp_ptrs[2*i + 1 - (rank & 1)];
      this->r_temp_syncs[i] = temp_syncs[2*i + 1 - (rank & 1)];
    }
  }

  static inline bool inactive(
      sycl::nd_item<1> pos, size_t start_group, size_t group_sz
  ) {
    auto group_id = pos.get_group(0);
    return (group_id < start_group || group_id >= start_group + group_sz);
  }

  inline void test_group_copy(sycl::nd_item<1> pos, uint32_t) const {
    uint32_t *local_sync = nullptr;
    uint32_t *local_wait = nullptr;

    if constexpr (sync_size != 0) {
      local_sync = __shared__<uint32_t [sync_size]>(pos.get_group());
      local_wait = __shared__<uint32_t [sync_size]>(pos.get_group());
      SyncProto::init_slm_flags(pos.get_local_id(0), local_sync, local_wait, sync_size);
      sycl::group_barrier(pos.get_group());
    }

    auto* src = input;
    auto* dst = bank;
    auto* sync = lock_self;
    auto* remote = lock_remote;

    fill_temp(pos, group_limit_start, group_limit_size,
        dst, src, sync, remote, nelems, rank, local_sync, local_wait);
  }

  // target peak bandwidth
  static inline void fill_temp(
      sycl::nd_item<1> pos, size_t start_group, size_t group_sz,
      T* dst, const T* src, uint32_t *sync, uint32_t* r_sync, size_t nelems, int rank,
      sycl::local_ptr<uint32_t> local_sync, sycl::local_ptr<uint32_t> local_wait
  ) {
    if (inactive(pos, start_group, group_sz))
      return;

    auto start = start_group * pos.get_local_range(0);
    auto g_range = group_sz * pos.get_local_range(0);

    auto src_start = (1 - (rank & 1)) * copy_type::cover(g_range);

    size_t progress = (rank & 1);

    for (auto off = pos.get_global_id(0) - start + src_start;
        off < nelems/v_T::size(); off += copy_type::cover(g_range) * 2) {
      size_t local_off = 0;
      if constexpr (sync_size != 0)
        local_off = progress % sync_size;

      SyncProto::block_if_eq(
          pos, sync + progress, group_sz,
          local_sync + local_off, local_wait + local_off);

      copy_type::run(dst, src, off, g_range, nelems);

      SyncProto::finish(
          pos, group_sz,
          sync + progress, r_sync + progress,
          local_sync + local_off, local_wait + local_off);

      progress += 2;
    }
  }

  inline void test_reduce_scatter(sycl::nd_item<1> pos, uint32_t signal) const {
    uint32_t *local_sync = nullptr;
    uint32_t *local_wait = nullptr;

    if constexpr (sync_size != 0) {
      local_sync = __shared__<uint32_t [sync_size]>(pos.get_group());
      local_wait = __shared__<uint32_t [sync_size]>(pos.get_group());
      SyncProto::init_slm_flags(pos, local_sync, local_wait, sync_size);
    }

    auto* src0 = input;
    auto* src1 = far_bank;

    group_reduce_scatter(pos, group_limit_start, group_limit_size, signal,
        temp_ptrs, src0, src1, lock_self, temp_syncs,
        rank, nelems, local_sync, local_wait);
  }

  static inline void group_reduce_scatter(
      sycl::nd_item<1> pos, size_t start_group, size_t group_sz,
      uint32_t signal, T* const dsts[], const T* src0, const T* src1,
      uint32_t *sync, uint32_t* const temp_syncs[], int rank, size_t nelems,
      sycl::local_ptr<uint32_t> local_sync,
      sycl::local_ptr<uint32_t> local_wait
  ) {
    if (inactive(pos, start_group, group_sz))
      return;

    auto start = start_group  * pos.get_local_range(0);
    auto stride = group_sz * pos.get_local_range(0);
    auto bisect_off = (rank & 1) * pos.get_local_range(0);

    size_t progress = 0;

    auto group_id = pos.get_group(0) - start_group;

    auto* dst = dsts[group_id % N_Peers];
    auto* event = temp_syncs[group_id % N_Peers];

    int rank_off = (group_id % N_Peers) + rank/2 < N_Peers
                        ? rank/2 * pos.get_local_range(0)
                        : (rank/2 - N_Peers) * pos.get_local_range(0);

    for (auto off = pos.get_global_id(0) - start;
        off < nelems/v_T::size(); off += stride * n_loop * 2) {
      size_t local_off = 0;
      if constexpr (sync_size != 0)
        local_off = progress % sync_size;

      SyncProto::block_if_not(
          pos, sync + progress, signal,
          local_sync + local_off, local_wait + local_off);

      auto src_off = off + bisect_off;
      auto dst_off = off + rank_off;

      copy_type::reduce(dst, src0, src1, dst_off, src_off, stride, nelems);

      SyncProto::signal(
          pos, event + progress, 1,
          local_sync + local_off, local_wait + local_off);

      progress += 2;
    }
  }

  inline void test_reduce_gather(sycl::nd_item<1> pos ) const {
    uint32_t *local_sync = nullptr;
    uint32_t *local_wait = nullptr;

    if constexpr (sync_size != 0) {
      local_sync = __shared__<uint32_t [sync_size]>(pos.get_group());
      local_wait = __shared__<uint32_t [sync_size]>(pos.get_group());
      SyncProto::init_slm_flags(pos, local_sync, local_wait, sync_size);
    }

    group_reduce_gather(pos, group_limit_start, group_limit_size,
        temp_ptrs, temp_syncs, r_temp_syncs, rank, nelems, local_sync, local_wait/*, cout*/);
  }

  static inline void group_reduce_gather(
      sycl::nd_item<1> pos,
      size_t start_group, size_t group_sz,
      T* const temp_ptrs[], uint32_t* const temp_syncs[],
      uint32_t* const remote_temp_syncs[], int rank, size_t nelems,
      sycl::local_ptr<uint32_t> local_sync, sycl::local_ptr<uint32_t> local_wait
      /*, sycl::stream cout*/
  ) {
    if (inactive(pos, start_group, group_sz))
      return;

    auto src_start = 0;
    auto dst_start = N_Peers * group_sz * pos.get_local_range(0);

    auto rank_off = rank/2 * pos.get_local_range(0);

    auto* src = temp_ptrs[rank/2];

    size_t progress = 0;

    auto bound = (nelems/v_T::size() + N_Peers * pos.get_local_range(0) - 1)
                  / (N_Peers * pos.get_local_range(0));
    auto scramble = pos.get_sub_group().get_group_id()[0]
                  / (pos.get_sub_group().get_group_range()[0] / N_Peers);

    auto* sync = temp_syncs[rank/2];

    for (auto gid = pos.get_group(0) - start_group;
        gid < bound; gid += group_sz * 2) {
      size_t local_off = 0;
      if constexpr (sync_size != 0)
        local_off = (progress / 2) % sync_size;

      SyncProto::block_if_not(
          pos, sync + progress, group_sz,
          local_sync + local_off, local_wait + local_off);

      auto g_off = N_Peers * gid * pos.get_local_range(0) + pos.get_local_id(0);

      auto src_off = src_start + g_off;
      auto dst_off = dst_start + rank_off + g_off;

      copy_type::template reduce_gather<N_Peers>(
          temp_ptrs, src, dst_off, src_off,
          pos.get_local_range(0), nelems, scramble);

      SyncProto::template signal<N_Peers>(
          pos, sync + progress + 1, temp_syncs, remote_temp_syncs,
          progress + 1, group_sz,
          local_sync + local_off, local_wait + local_off);

      progress += 2;
    }
  }

  inline void test_copy_back (sycl::nd_item<1> pos, uint32_t signal) const {
    uint32_t *local_sync = nullptr;
    uint32_t *local_wait = nullptr;

    if constexpr (sync_size != 0) {
      local_sync = __shared__<uint32_t [sync_size]>(pos.get_group());
      local_wait = __shared__<uint32_t [sync_size]>(pos.get_group());
      SyncProto::init_slm_flags(pos, local_sync, local_wait, sync_size);
    }

    auto* sync = temp_syncs[rank/2];

    auto* src0 = (rank & 1) ? r_temp_ptrs[rank/2] : temp_ptrs[rank/2];
    auto* src1 = (rank & 1) ? temp_ptrs[rank/2] : r_temp_ptrs[rank/2];

    copy_back(pos, group_limit_start, group_limit_size, signal,
        input, sync, src0, src1, rank, nelems, local_sync, local_wait);
  }

  static inline void copy_back(
      sycl::nd_item<1> pos,
      size_t start_group, size_t group_sz, uint32_t signal,
      T* dst, uint32_t* sync, const T* src_left, const T* src_right,
      int rank, size_t nelems,
      sycl::local_ptr<uint32_t> local_sync,
      sycl::local_ptr<uint32_t> local_wait
  ) {
    if (inactive(pos, start_group, group_sz))
      return;

    auto start = start_group * pos.get_local_range(0);
    auto stride = group_sz * pos.get_local_range(0);

    size_t progress = 0;

    for (auto off = pos.get_global_id(0) - start;
        off < nelems/v_T::size(); off += stride * n_loop * 2) {
      size_t local_off = 0;
      if constexpr (sync_size != 0)
        local_off = progress % sync_size;

      SyncProto::block_if_not(
          pos, sync + progress + 1, signal,
          local_sync + local_off, local_wait + local_off);

      copy_type::merge(dst, src_left, src_right,
          off, off + stride * n_loop, stride, nelems);

      SyncProto::reset(
          pos, sync + progress + 1,
          local_sync + local_off + 1, local_wait + local_off + 1);

      progress += 2;
    }
  }

  void operator() [[sycl::reqd_sub_group_size(16)]] (sycl::nd_item<1> pos) const {
    // test_reduce_scatter(pos, 0x10);
    test_copy_back(pos, 0x10);
  }

  static sycl::event launch(
      sycl::queue queue,
      T* input, uint32_t* team_sync, T* interns[], uint32_t* peer_syncs[],
      T* temp_ptrs[], uint32_t* temp_syncs[],
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

    auto e = queue.submit([&](sycl::handler &cgh) {
        sycl::stream out(1024 * 1024, 1024, cgh);
        cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}),
            copy_persist(input, team_sync, interns, peer_syncs, temp_ptrs,
              temp_syncs, nelems, rank, limit_start, limit_size, out));
    });

    // for performance evaluation
    // for (int i = 1; i < repeat; ++ i) {
    //   e = queue.submit([&](sycl::handler &cgh) {
    //       cgh.parallel_for(sycl::nd_range<1>({global_size}, {local_size}), offload);
    //   });
    // }
    return e;
  }

private:
  T* input;
  uint32_t* team_sync;

  // for cross MDFI
  T* bank;
  uint32_t* lock_self;

  T* far_bank;
  uint32_t* lock_remote;

  T* peer_ptrs[N_Peers];
  uint32_t* peer_syncs[N_Peers];
  uint32_t* r_peer_syncs[N_Peers];

  // for cross XeLink
  T* temp_ptrs[N_Peers];
  uint32_t* temp_syncs[N_Peers];

  T* r_temp_ptrs[N_Peers];
  uint32_t* r_temp_syncs[N_Peers];

  size_t nelems;
  int rank;

  size_t group_limit_start;
  size_t group_limit_size;

  sycl::stream cout;
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
bool verify_reduce_scatter(T* input, T* targets[], int rank, int world,
    size_t bisect_size, size_t stripe, size_t nelems) {

  auto n_peers = world / 2;

  for (size_t i = 0; i < nelems; ++ i) {
    auto bisect_no = i / bisect_size;
    auto stripe_no = i / stripe;

    if ( bisect_no % 2 == 0 ) {
      auto src0 = targets[2 * ((stripe_no + rank/2) % n_peers)];
      auto src1 = targets[2 * ((stripe_no + rank/2) % n_peers) + 1];

      int rank_off = 0;

      if ( (stripe_no % n_peers) + rank/2 < n_peers )
        rank_off = rank / 2;
      else
        rank_off = rank / 2 - n_peers;

      auto bisect_off = (rank & 1) ? bisect_size : 0;
      auto reduce = src0[i + rank_off * stripe + bisect_off]
                  + src1[i + rank_off * stripe + bisect_off];

      if (input[i] != reduce) {
        std::cout<<"Error occurred @"<<i<<"; expect "
          <<reduce<<" vs. "<<input[i]<<std::endl;
        return false;
      }
    }
  }

  return true;
}

template <typename T>
bool check_group_reduce_scatter(T* result, T* targets[],
    int rank, int world, size_t nelems) {
  return verify_reduce_scatter<T>(
      result, targets, rank, world, 512 * 1024, 8 * 1024, nelems);
}

template <typename T>
bool verify_reduce_gather(T* input, T* targets[], int rank, int world,
    size_t bisect_size, size_t stripe, size_t nelems) {

  auto n_peers = world / 2;

  for (size_t i = 0; i < nelems; ++ i) {
    auto bisect_no = i / bisect_size;
    auto stripe_no = i / stripe;

    if ( bisect_no % 2 == 1 ) {
      auto peer_no = 2 * (stripe_no % n_peers) + (rank & 1);
      auto target = targets[peer_no];
      auto rank_off = peer_no/2 * stripe;

      T reduce = 0.f;

      for (int p = 0; p < n_peers; ++ p) {
        reduce += target[i + p * stripe - rank_off - bisect_size];
      }

      if ( input[i] != reduce ) {
        std::cout<<"Error occurred @"<<i<<"; expect "
          <<reduce<<" vs. "<<input[i]<<std::endl;
        return false;
      }
    }
  }

  return true;
}

template <typename T>
bool check_group_reduce_gather(T* result, T* targets[],
    int rank, int world, size_t nelems) {
  return verify_reduce_gather<T>(
      result, targets, rank, world, 512 * 1024, 8 * 1024, nelems);
}

template <typename T>
bool verify_merge(T* input, T* targets[], int rank, int world,
    size_t bisect_size, size_t nelems) {

  for (size_t i = 0; i < nelems; ++ i) {
    auto bisect_no = i / bisect_size;
    auto* src0 = (rank & 1) ? targets[rank ^ 1] : targets[rank];
    auto* src1 = (rank & 1) ? targets[rank] : targets[rank ^ 1];
    if (bisect_no % 2 == 1) {
      if (input[i] != src1[i]) {
        std::cout<<"Error occurred @"<<i<<"; expect "
          <<src1[i]<<" vs. "<<input[i]<<std::endl;
        return false;
      }
    } else {
      if (input[i] != src0[i]) {
        std::cout<<"Error occurred @"<<i<<"; expect "
          <<src0[i]<<" vs. "<<input[i]<<std::endl;
        return false;
      }
    }
  }

  return true;
}

template <typename T>
bool check_copy_back(T* src, T* targets[], int rank, int world, size_t nelems) {
  return verify_merge(src, targets, rank, world, 512 * 1024, nelems);
}

template <typename T>
double bandwidth_from_event(sycl::event e, size_t nelems) {
  e.wait();
  auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  // since timestamp is in the unit of ns, then the bandwidth is GB/s in unit
  return (double)(nelems * sizeof(T)) / (double)(end - start);
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
    ("i,init", "Sync contents init", cxxopts::value<uint32_t>()->default_value("0"))
    ("t,temp", "Temp sync init", cxxopts::value<uint32_t>()->default_value("0"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems_string = parsed_opts["nelems"].as<std::string>();
  auto sync_mode = parsed_opts["sync_mode"].as<int>();
  auto begin = parsed_opts["begin"].as<size_t>();
  auto end = parsed_opts["end"].as<size_t>();
  auto sync_init = parsed_opts["init"].as<uint32_t>();
  auto sync2_init = parsed_opts["temp"].as<uint32_t>();

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
  constexpr size_t t_sync_sz = 64 * 1024;

  // sync every 256K elems
  size_t sync_grain = 256 * 1024ull;

  auto sync_elems = (nelems + sync_grain -1) / sync_grain;
  auto sync_size = (sync_elems * sizeof(uint32_t) + gpu_pagesz -1)
                  / gpu_pagesz * gpu_pagesz;

  auto queue = currentQueue(rank >> 1, rank & 1);

  auto* src = (test_type *)sycl::malloc_device(data_size, queue);
  auto* team_sync = (uint32_t *)sycl::malloc_device(t_sync_sz, queue);

  auto* dst = (test_type *)sycl::malloc_device(data_size, queue);
  auto* dst_sync = sycl::malloc_device(sync_size, queue);

  auto* scratch = (test_type *)sycl::malloc_device(data_size, queue);
  auto* temp_sync = sycl::malloc_device(sync_size, queue);

  auto* h_sync = (uint32_t *)sycl::malloc_host(sync_size, queue);
  auto* h_temp = (uint32_t *)sycl::malloc_host(sync_size, queue);

  test_type* h_check = (test_type *)sycl::malloc_host(data_size, queue);
  test_type* h_check_peers[world];

  for (int i = 0; i < world; ++ i) {
    h_check_peers[i] = (test_type *)sycl::malloc_host(data_size, queue);
  }

  test_type* h_src = h_check_peers[rank];

  release_guard __guard([&]{
    sycl::free(src, queue);
    sycl::free(team_sync, queue);
    sycl::free(dst, queue);
    sycl::free(dst_sync, queue);
    sycl::free(scratch, queue);
    sycl::free(temp_sync, queue);
    sycl::free(h_sync, queue);
    sycl::free(h_temp, queue);
    sycl::free(h_check, queue);

    for (int i = 0; i < world; ++ i) {
      sycl::free(h_check_peers[i], queue);
    }
  });

  for (int i = 0; i < world; ++ i) {
    fill_constant<test_type>(h_check_peers[i], i, nelems);
  }

  memset(h_check, 0, data_size);

  std::fill(h_sync, h_sync + sync_elems, sync_init);

  for (int i = 0; i < sync_elems; i+=2) {
    h_temp[i] = sync2_init;
  }

  queue.memset(team_sync, 0, t_sync_sz);

  queue.memset(src, 0, data_size);
  queue.memcpy(dst, h_src, data_size);
  queue.memcpy(scratch, h_src, data_size);

  queue.memcpy(dst_sync, h_sync, sync_size);
  queue.memcpy(temp_sync, h_temp, sync_size);
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
          zeCheck(zeMemCloseIpcHandle(l0_ctx, bases[i])); \
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
  void* temp_ptrs[world];
  exchange_buffers(scratch_fd, scratch, scratch_bases, offsets, temp_ptrs);

  void* sema_bases[world];
  void* temp_syncs[world];
  exchange_buffers(sync_fd, temp_sync, sema_bases, offsets, temp_syncs);

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
        queue, world, (test_type *)src, team_sync,
        (test_type **)peer_ptrs, (uint32_t **)sync_ptrs,
        (test_type **)temp_ptrs, (uint32_t **)temp_syncs,
        nelems, rank, group_limit_start, group_limit_size);
    break;
  case 1:
    e = launch<copy_persist, test_type, hierarchy_sync, chunk_copy>(
        queue, world, (test_type *)src, team_sync,
        (test_type **)peer_ptrs, (uint32_t **)sync_ptrs,
        (test_type **)temp_ptrs, (uint32_t **)temp_syncs,
        nelems, rank, group_limit_start, group_limit_size);
    break;
  default:
    throw std::logic_error("Unsupported synchronous mode.");
  }

  // auto e = queue.memcpy(dst, src, alloc_size);
  auto bandwidth = bandwidth_from_event<test_type>(e, nelems/2);
  auto time = time_from_event(e);
  printf("[%d]Copy %zu half in %fns, nominal bandwidth: %fGB/s\n",
      rank, data_size, time, bandwidth);

  queue.memcpy(h_check, src, data_size);
  queue.wait();

  /*
  bool pos = check_group_reduce_scatter(
      h_check, h_check_peers, rank, world, nelems);
  */
  bool pos = check_copy_back (
      h_check, h_check_peers, rank, world, nelems);

  if ( pos )
    printf("Verified\n");
  else
    printf("Error at %d\n", pos);

  printf("[%d] Lock elems %#x, %#x, %#x, %#x, ..., %#x\n", rank,
      mon_lock[0], mon_lock[1], mon_lock[2], mon_lock[3], mon_lock[sync_elems -1]);

  printf("[%d] Sync elems %#x, %#x, %#x, %#x, ..., %#x\n", rank,
      mon_sync[0], mon_sync[1], mon_sync[2], mon_sync[3], mon_sync[sync_elems -1]);
}
