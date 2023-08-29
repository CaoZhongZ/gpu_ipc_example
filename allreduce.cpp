#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <system_error>
#include <stdarg.h>

#include <initializer_list>
#include <string>
#include <random>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include "cxxopts.hpp"
#include "ze_exception.hpp"
#include "sycl_misc.hpp"

#include "protocol.h"

static constexpr int msg_len = 2048;

struct exchange_contents {
  // first 4-byte is file descriptor for drmbuf or gem object
  union {
    ze_ipc_mem_handle_t ipc_handle;
    int fd = -1;
  };
  size_t offset = 0;
  int pid = -1;
};

#define sysCheck(x) \
  if (x == -1) {  \
    throw std::system_error(  \
        std::make_error_code(std::errc(errno)));  \
  }

template <typename T>
class atomic_ref : public sycl::atomic_ref<T,
                      sycl::memory_order::relaxed,
                      sycl::memory_scope::device,
                      sycl::access::address_space::global_space> {
public:
  atomic_ref(T& r) : sycl::atomic_ref<T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space> (r)
  {}

  static inline void wait_ne(T& r, T value) {
    atomic_ref<T> flag(r);
    while(flag != value);
  }

  static inline void wait_on(T& r, T value) {
    wait_ne(r, value);
  }

  static inline void wait_lt(T& r, T value) {
    atomic_ref<T> flag(r);
    while(flag < value);
  }

  static inline void wait_gt(T& r, T value) {
    atomic_ref<T> flag(r);
    while(flag > value);
  }

  static inline void wait_le(T& r, T value) {
    atomic_ref<T> flag(r);
    while(flag <= value);
  }

  static inline void wait_ge(T& r, T value) {
    atomic_ref<T> flag(r);
    while(flag >= value);
  }

  static inline void store(T& r, T value) {
    atomic_ref<T> flag(r);
    flag.store(value);
  }

  static inline void incre(T& r) {
    atomic_ref<T> flag(r);
    flag += 1;
  }

  static inline void clear(T& r) {
    atomic_ref<T> flag(r);
    flag.store(0);
  }

  static inline void mask_wait(T& r, T value, T mask) {
    atomic_ref<T> flag(r);

    while (flag.load() & mask != value);
  }

  static inline void or_(T& r, T value) {
    atomic_ref<T> flag(r);
    flag |= value;
  }
};

ze_ipc_mem_handle_t open_peer_ipc_mems(
    void* ptr, int rank, int world, void *peer_bases[], size_t offsets[]) {
  // Step 1: Get base address of the pointer
  sycl::queue queue = currentQueue(rank / 2, rank & 1);
  sycl::context ctx = queue.get_context();
  auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  void *base_addr;
  size_t base_size;
  zeCheck(zeMemGetAddressRange(l0_ctx, ptr, &base_addr, &base_size));

  // Step 2: Get IPC mem handle from base address
  alignas(64) exchange_contents send_buf;
  alignas(64) exchange_contents recv_buf[world];

  // fill in the exchange info
  zeCheck(zeMemGetIpcHandle(l0_ctx, base_addr, &send_buf.ipc_handle));
  send_buf.offset = (char*)ptr - (char*)base_addr;
  send_buf.pid = getpid();

  // Step 3: Exchange the handles and offsets
  memset(recv_buf, 0, sizeof(recv_buf));
  // Overkill if we don't really needs all peer's handles
  MPI_Allgather(
      &send_buf, sizeof(send_buf), MPI_BYTE, recv_buf, sizeof(send_buf), MPI_BYTE, MPI_COMM_WORLD);

  for (int i = 0; i < world; ++ i) {
    if (i == rank) {
      peer_bases[i] = ptr;
      offsets[i] = 0;
    } else {
      auto* peer = recv_buf + i;
      auto pid_fd = syscall(__NR_pidfd_open, peer->pid, 0);
      sysCheck(pid_fd);

      //
      // Step 5: Duplicate GEM object handle to local process
      // and overwrite original file descriptor number
      //
      peer->fd = syscall(__NR_pidfd_getfd, pid_fd, peer->fd, 0);
      sysCheck(peer->fd);

      // Step 6: Open IPC handle of remote peer
      auto l0_device
        = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
      void* peer_base;

      zeCheck(zeMemOpenIpcHandle(
            l0_ctx, l0_device, peer->ipc_handle, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &peer_base));

      peer_bases[i] = peer_base;
      offsets[i] = peer->offset;
    }
  }
  return send_buf.ipc_handle;
}

std::tuple<void*, size_t, ze_ipc_mem_handle_t> open_peer_ipc_mem(
    void* ptr, int rank, int world) {
  // Step 1: Get base address of the pointer
  sycl::queue queue = currentQueue(rank / 2, rank & 1);
  sycl::context ctx = queue.get_context();
  auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  void *base_addr;
  size_t base_size;
  zeCheck(zeMemGetAddressRange(l0_ctx, ptr, &base_addr, &base_size));

  // Step 2: Get IPC mem handle from base address
  alignas(64) exchange_contents send_buf;
  alignas(64) exchange_contents recv_buf[world];

  // fill in the exchange info
  zeCheck(zeMemGetIpcHandle(l0_ctx, base_addr, &send_buf.ipc_handle));
  send_buf.offset = (char*)ptr - (char*)base_addr;
  send_buf.pid = getpid();

  // Step 3: Exchange the handles and offsets
  memset(recv_buf, 0, sizeof(recv_buf));
  // Overkill if we don't really needs all peer's handles
  MPI_Allgather(
      &send_buf, sizeof(send_buf), MPI_BYTE, recv_buf, sizeof(send_buf), MPI_BYTE, MPI_COMM_WORLD);

  // Step 4: Prepare pid file descriptor of next process
  int next_peer = rank + 1;
  if (next_peer >= world) next_peer = next_peer - world;

  auto* peer = recv_buf + next_peer;
  auto pid_fd = syscall(__NR_pidfd_open, peer->pid, 0);
  sysCheck(pid_fd);

  //
  // Step 5: Duplicate GEM object handle to local process
  // and overwrite original file descriptor number
  //
  peer->fd = syscall(__NR_pidfd_getfd, pid_fd, peer->fd, 0);
  sysCheck(peer->fd);

  // Step 6: Open IPC handle of remote peer
  auto l0_device
    = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
  void* peer_base;

  zeCheck(zeMemOpenIpcHandle(
        l0_ctx, l0_device, peer->ipc_handle, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &peer_base));

  return std::make_tuple(
      (char*)peer_base + peer->offset, peer->offset, send_buf.ipc_handle);
}

static size_t align_up(size_t size, size_t align_sz) {
  return ((size + align_sz -1) / align_sz) * align_sz;
}

void *mmap_host(size_t map_size, int dma_buf_fd) {
  auto page_size = getpagesize();
  map_size = align_up(map_size, page_size);
  return mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, dma_buf_fd, 0);
}

template <typename T>
bool checkResults(T *ptr, T c, size_t count) {
  for (int i = 0; i < count; ++ i) {
    if (ptr[i] != c) {
      std::cout<<"Expect: "<<c<<" but get: "<<ptr[i]<<std::endl;
      return false;
    }
  }
  return true;
}

template <int F> struct remote_info {
  static constexpr int n_peers = F;
  std::array<int, F> peers;

  remote_info(const std::array<int, F>& peer_list) : peers(peer_list) {}
};

template <typename T, typename groupTrait, int nPersistGroups>
struct allreduce_4stages {
  static constexpr stage = 2;
  static constexpr n_buf = 8;

  using v_T = sycl::vec<T, groupTrait::laneWidth/sizeof(T)>;

  using copyBuffer = Cell<T, groupTrait> [n_buf * 2][nPersistGroups];
  using stepBuffer = Cell<T, groupTrait> [n_buf][nPersistGroups][2];

  static inline void dispatch_group(
      sycl::nd_item<2> pos, int rank, v_T* input,
      stepBuffer* evens[], stepBuffer* odds[],
      size_t v_nelems, size_t n_step) {
    auto group_role = pos.get_group(1);

    if (group_role == 0) {
      if (rank & 1) {
        for (int i = 0; i < n_step; ++ i)
          copy_to_temp(
              pos, (copyBuffer*) odds[rank/2], (copyBuffer*) evens[rank/2],
              input, v_nelems, i, 0);
      } else {
        for (int i = 0; i < n_step; ++ i)
          copy_to_temp(
              pos, (copyBuffer*) evens[rank/2], (copyBuffer*) odds[rank/2],
              input, v_nelems, i, 0);
      }
    }

    else if (group_role == 1) {
      if (rank & 1) {
        for (int i = 0; i < n_step; ++ i)
          reduce_scatter<1>(pos, rank, odds, odds[rank/2], evens[rank/2], i, 0);
      } else {
        for (int i = 0; i < n_step; ++ i)
          reduce_scatter<0>(pos, rank, evens, evens[rank/2], odds[rank/2], i, 0);
      }
    }

    else if (group_role == 2) {
      if (rank & 1) {
        for (int i = 0; i < n_step; ++ i)
          reduce_bcast<1>(pos, rank, odds, evens, odds[rank/2], i, 0);
      } else {
        for (int i = 0; i < n_step; ++ i)
          reduce_bcast<0>(pos, rank, evens, odds, evens[rank/2], i, 0);
      }
    }

    else if (group_role == 3) {
      if ( rank & 1 ) {
        for (int i = 0; i < n_step; ++ i)
          temp_to_input<1>(pos, input, evens[rank/2], odds[rank/2], v_nelems, i, 0);
      } else {
        for (int i = 0; i < n_step; ++ i)
          temp_to_input<0>(pos, input, evens[rank/2], odds[rank/2], v_nelems, i, 0);
      }
    }
  }

  // role 0, copy input to local and signal both local and pair
  static inline void copy_to_temp(
      sycl::nd_item<2> pos, copyBuffer* local, copyBuffer* pair,
      v_T* input, size_t v_nelems, size_t step, int seq_no) {
    auto group_position = pos.get_group(1) / stage;
    auto local_y = pos.get_local_id(0);
    auto local_x = pos.get_local_id(1);
    auto global_stride = pos.get_global_range().size();

    auto g_off = local_x + local_y * pos.get_local_range(1)
      + group_position * pos.get_local_range().size() + step * global_stride;

    // we treat buffer in linear instead of dual-set
    auto b_idx = step % (n_buf * 2);

    constexpr int stage = 0;

    auto& slot = local[stage][b_idx][group_position];

    if (local_x == 0 && local_y == 0)
      atomic_ref<uint32_t>::mask_wait(slot.atomics[0], 0, 0xffff);

    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);

    if (g_off < v_nelems)
      slot.data[local_y][local_x] = input[g_off];

    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);

    if (local_x == 0 && local_y == 0)
      atomic_ref<uint32_t>::incre(slot.atomics[0]);

    if (local_x == 0 && local_y == 2)
      atomic_ref<uint32_t>::or_(
          pair[stage][b_idx][group_position].atomics[0], 1 << 16);
  }

  // role 1
  template <int eo>
  static inline void reduce_scatter(
      sycl::nd_item<2> pos, int rank,
      stepBuffer* const peers[], stepBuffer* local, stepBuffer* pair,
      size_t step, int seq_no) {
    auto group_position = pos.get_group(1) / stage;
    auto local_y = pos.get_local_id(0);
    auto local_x = pos.get_local_id(1);
    auto r_off = rank / 2;
    constexpr int last_stage = 0;
    constexpr int stage = 1;
    auto b_idx = step % n_buf;

    auto& self = local[last_stage][b_idx][group_position][eo];
    auto& remote = pair[last_stage][b_idx][group_position][eo];

    if (pos.get_local_linear_id() == 0) {
      atomic_ref<uint32_t>::wait_on(self.atomics[0], 0x10001);
    }

    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);

    auto& peer = peers[local_y] [stage][b_idx][group_position][0];
    peer.data[r_off][local_x]
      = remote.data[local_y][local_x] + self.data[local_y][local_x];

    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);

    if (local_x == 0)
      atomic_ref<uint32_t>::incre(peer.atomics[0]);
  }

  // role 2
  template <int eo>
  static inline void reduce_bcast(
      sycl::nd_item<2> pos, int rank,
      stepBuffer* const peers[], stepBuffer* const pairs[],
      stepBuffer* local, size_t step, int seq_no) {
    auto local_y = pos.get_local_id(0);
    auto group_position = pos.get_group(1) / stage;
    auto local_x = pos.get_local_id(1);
    auto r_off = rank / 2;
    auto b_idx = step % n_buf;

    constexpr int last_stage = 1;
    constexpr int stage = 1;

    auto& slot = local[last_stage][b_idx][group_position][0];

    if (pos.get_local_linear_id() == 0) {
      atomic_ref<uint32_t>::wait_on (slot.atomics[0], seq_no + 4);
    }
    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);

    if (local_y == 0) {
      v_T sum {};

#     pragma unroll
      for (int i = 0; i < groupTrait::groupY; ++ i)
        sum += slot.data[i][local_x];

#     pragma unroll
      for (int i = 0; i < groupTrait::groupY; ++ i)
        peers[i][stage][b_idx][group_position][1].data[r_off][local_x] = sum;
    }

    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);

    if (pos.get_local_linear_id() == 0) {
      atomic_ref<uint32_t>::store(slot.atomics[0], seq_no);
    }

    if (local_x == 0 && local_y == 0) {
      auto peer = peers[local_y][stage][b_idx][group_position][1];
      auto pair = pairs[local_y][stage][b_idx][group_position][1];

      atomic_ref<uint32_t>::incre(peer.atomics[0]);
      atomic_ref<uint32_t>::incre(pair.atomics[0]);
    }
  }

  // group 3
  template <int eo>
  static inline void temp_to_input(
      sycl::nd_item<2> pos, v_T* input,
      stepBuffer *first, stepBuffer* second,
      size_t v_nelems, size_t step, int seq_no) {
    auto group_position = pos.get_group(1) / stage;
    auto local_y = pos.get_local_id(0);
    auto local_x = pos.get_local_id(1);
    auto global_stride = pos.get_global_range().size() * 2;
    auto b_idx = step % n_buf;
    constexpr int stage = 1;

    auto even_off = local_x + local_y * pos.get_local_range(1)
      + group_position * pos.get_local_range().size() + step * global_stride;
    auto odd_off = even_off + pos.get_local_range().size();

    auto& left = first[stage][b_idx][group_position][1];
    auto& right = second[stage][b_idx][group_position][1];

    if constexpr (eo)
      if (pos.get_local_linear_id() == 0)
        atomic_ref<uint32_t>::wait_on(right.atomics[0], seq_no + 8);
    else
      if (pos.get_local_linear_id() == 0)
        atomic_ref<uint32_t>::wait_on(left.atmoics[0], seq_no + 8);

    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);

    if (even_off < v_nelems)
      input[even_off] = left.data[local_y][local_x];

    if (odd_off < v_nelems)
      input[odd_off] = right.data[local_y][local_x];

    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);

    if constexpr (eo)
      if (pos.get_local_linear_id() == 0)
        atomic_ref<uint32_t>::store(right.atomics[0], seq_no);
    else
      if (pos.get_local_linear_id() == 0)
        atomic_ref<uint32_t>::store(left.atmoics[0], seq_no);

    // signal copy buffer
    if (local_x == 0 && local_y == 1)
      atomic_ref<uint32_t>::store(
          first[stage -1][b_idx][group_position][eo].atomics[0], seq_no);

    if (local_x == 0 && local_y == 2)
      atomic_ref<uint32_t>::store(
          second[stage -1][b_idx][group_position][eo].atomics[0], seq_no);
  }
};

template <typename T, int lane_v, typename R,
         template <typename, int> class allreduce_policy>
struct xelink_allreduce {
  using v_T = sycl::vec<T, sizeof(float)/sizeof(T) * lane_v>;
  static constexpr int n_peers = R::n_peers;
  static constexpr size_t hw_groups = 512;
  static constexpr size_t sub_group_size = 16;

public:
  xelink_route(void *input, void *peer_ptrs[], R&& remote_info,
      int src, int root, int world, size_t nelems)
    : root(root), input(reinterpret_cast<v_T *>(input)),
    local(reinterpret_cast<v_T *>(peer_ptrs[root])),
    source(reinterpret_cast<v_T *>(peer_ptrs[src])),
    nelems(nelems)/*, cout(s)*/ {

    std::sort(remote_info.peers.begin(), remote_info.peers.end());
    auto it = std::find_if(
        remote_info.peers.begin(), remote_info.peers.end(),
        [&](int peer) {return peer > root;});
    peers[it - remote_info.peers.begin()] = reinterpret_cast<v_T *>(peer_ptrs[root]);

    for (int f = 0; f < n_peers; ++f) {
      auto remote = remote_info.peers[f];
      peers[f + (remote > root)] = reinterpret_cast<v_T *>(peer_ptrs[remote]);
    }
  }

  void operator() [[sycl::reqd_sub_group_size(sub_group_size)]] (sycl::nd_item<2> pos) const {
    allreduce_policy<v_T, n_peers>::run(
        pos, source, root/2, local, peers, nelems/v_T::size());
  }

  static bool valid_peers(int src, int root, const std::array<int, R::n_peers>& peers) {
    auto it = std::find(peers.begin(), peers.end(), root);
    if (it == peers.end()) {
      it = std::find(peers.begin(), peers.end(), src);
      if (it == peers.end())
        return root != src;
    }

    return false;
  }

  static sycl::event launch(void* peers[], R&& remote_info,
      int src, int root, int world, size_t nelems, char *msg = nullptr) {
    if (!valid_peers(src, root, remote_info.peers))
      throw std::logic_error("Invalid transmit pattern!");
    size_t local_y = n_peers + 1;
    size_t local_x = 2 * sub_group_size;
    size_t local_sz = local_y * local_x;

    size_t data_groups = (nelems/v_T::size() + local_sz - 1) / local_sz;
    size_t group_size = std::min(data_groups, hw_groups);
    size_t global_x = group_size * local_x;
    size_t global_y = 1 * local_y;

    if (msg != nullptr) {
      snprintf(msg, 2048,
          "Launch route from %d on rank %d: (%ld, %ld)x(%ld, %ld)\n",
          src, root, global_y, global_x, local_y, local_x);
    }

    auto queue = currentQueue(root/2, root & 1);
    auto e = queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<2>({global_y, global_x}, {local_y, local_x}),
          xelink_route(peers, std::move(remote_info), src, root, world, nelems));
    });

    return e;
  }

private:
  v_T *peers[n_peers + 1]; // including the root
  int root;
  v_T *input;
  v_T *local;
  v_T *source;

  size_t nelems;
  // sycl::stream cout;
};

// Calc bandwidth from event;
template <typename T>
double bandwidth_from_event(sycl::event e, size_t nelems) {
  e.wait();
  auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  // since timestamp is in the unit of ns, then the bandwidth is GB/s in unit
  return (double)(nelems * sizeof(T)) / (double)(end - start);
}

template <template <typename, int, typename, template <typename, int> class> class coll,
         typename T, int lane_v, template <typename, int> class fan_policy,
         typename ... Args>
static sycl::event launch(
    void* peers_ptr[], const std::vector<int>& remotes, Args&& ... args) {
#define CASE(n, ...)  \
  case n: \
    { \
      remote_info<n> i_remote ({__VA_ARGS__}); \
      return coll<T, lane_v, decltype(i_remote), fan_policy>::launch( \
          peers_ptr, std::move(i_remote), std::forward<Args>(args)...); \
    } \
    break;

  switch (remotes.size()) {
  CASE(1, remotes[0]);
  CASE(2, remotes[0], remotes[1]);
  CASE(3, remotes[0], remotes[1], remotes[2]);
  CASE(6, remotes[0], remotes[1], remotes[2], remotes[3], remotes[4], remotes[5]);
  default:
    throw std::length_error("Unsupported broadcast pattern.");
    break;
  }
#undef CASE
}

std::vector<int> commalist_to_vector(const std::string& str) {
  std::vector<int> list;
  char* pos = const_cast<char *>(str.c_str());
  do {
    list.push_back(std::strtol(pos, &pos, 10));
    // expect deliminator or termination
    if (*pos == ',')
      pos ++;
    else if ( *pos != 0 )
      throw std::logic_error("Unexpected deliminator");
  } while (*pos);

  return list;
}

// collective call, don't miss ranks!
void r_print(const char *check_msg, int rank, int world) {
  char check_msgs[world][msg_len];
  MPI_Gather(check_msg, msg_len, MPI_BYTE, check_msgs, msg_len, MPI_BYTE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < world; ++ i) {
      if (*check_msgs[i] != '\0')
        printf("%s", check_msgs[i]);
    }
  }
}

// collective call, don't miss ranks!
void r_printf(int rank, int world, const char* fmt, ...) {
  char check_msg[msg_len];

  va_list args;
  va_start(args, fmt);
  snprintf(check_msg, sizeof(check_msg), fmt, args);
  va_end(args);

  r_print(check_msg, rank, world);
}

void peek_buffer(char *check_msg, uint32_t* host_buf, size_t alloc_size, int rank, int world) {
  auto offset = snprintf(check_msg, 2048,
      "\nRank %d Peek: %#x, ..., %#x, %#x, ..., %#x, %#x, ..., %#x, %#x, ..., %#x\n",
      rank,
      host_buf[0], host_buf[alloc_size / sizeof(uint32_t) -1],
      host_buf[alloc_size / sizeof(uint32_t)], host_buf[alloc_size * 2/ sizeof(uint32_t) -1],
      host_buf[alloc_size * 2/ sizeof(uint32_t)], host_buf[alloc_size * 3/ sizeof(uint32_t) -1],
      host_buf[alloc_size * 3 / sizeof(uint32_t)], host_buf[alloc_size * 4 / sizeof(uint32_t) -1]);
  snprintf(check_msg + offset, 2048 - offset,
      "Rank %d Peek: %#x, ..., %#x, %#x, ..., %#x, %#x, ..., %#x, %#x, ..., %#x\n\n",
      rank,
      host_buf[alloc_size * 4 / sizeof(uint32_t)], host_buf[alloc_size * 5/ sizeof(uint32_t) -1],
      host_buf[alloc_size * 5 / sizeof(uint32_t)], host_buf[alloc_size * 6/ sizeof(uint32_t) -1],
      host_buf[alloc_size * 6 / sizeof(uint32_t)], host_buf[alloc_size * 7/ sizeof(uint32_t) -1],
      host_buf[alloc_size * 7 / sizeof(uint32_t)], host_buf[alloc_size * 8/ sizeof(uint32_t) -1]);
}

template <typename T>
void peek_slice(char *check_msg, T* host_buf, size_t slice_size, int rank, int world) {
  auto offset = snprintf(check_msg, 2048,
      "\nRank %d Peek: %.2f, %.2f, ..., %.2f, %.2f",
      rank,
      (float)host_buf[0], (float)host_buf[1],
      (float)host_buf[slice_size -2], (float)host_buf[slice_size -1]);
  snprintf(check_msg + offset, 2048 - offset,
      "; %.2f, %.2f, ..., %.2f, %.2f\n",
      (float)host_buf[slice_size], (float)host_buf[slice_size + 1],
      (float)host_buf[2 * slice_size-2], (float)host_buf[2*slice_size-1]);
}

template <typename T>
void fill_sequential(void *p, int rank, size_t size) {
  auto typed_sz = size / sizeof(T);
  auto *p_t = reinterpret_cast<T *>(p);

  for (size_t i = 0; i < typed_sz; ++ i) {
    p_t[i] = i + rank;
  }
}

void fill_random(void *p, int rank, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(1, 0x4000);

  auto sz_int = size / sizeof(int);

  for (size_t i = 0; i < sz_int; ++ i) {
    ((uint32_t *)p)[i] = distrib(gen) + rank;
  }
}

std::vector<int> bisect_ranks(int rank, int world) {
  std::vector<int> even_ranks {0, 2, 4, 6};
  std::vector<int> odd_ranks {1, 3, 5, 7};
  std::vector<int> peer_ranks;

  auto it = std::find(even_ranks.begin(), even_ranks.end(), rank);
  if (it != even_ranks.end()) {
    even_ranks.erase(it);
    peer_ranks = even_ranks;
  } else {
    it = std::find(odd_ranks.begin(), odd_ranks.end(), rank);
    odd_ranks.erase(it);
    peer_ranks = odd_ranks;
  }

  return peer_ranks;
}

template <typename T>
void adjust_bisect_pointers(
    void *new_ptrs[], void *peer_ptrs[], int rank, int world, size_t nelems) {
  for (int i = 0; i < world; ++ i) {
    auto *peer_ptr = reinterpret_cast<T *>(peer_ptrs[i]);
    if ( (rank % 2) != 0 )
      new_ptrs[i] = (void *)(peer_ptr + nelems/2);
    else
      new_ptrs[i] = (void *)peer_ptr;
  }
}

template <typename T, int lane_v>
void test_reduce_scatter(void *peer_ptrs[], int rank, int world, size_t nelems, int repeat) {
  auto peer_ranks = bisect_ranks(rank, world);

  void *new_ptrs[world];

  adjust_bisect_pointers<T>(new_ptrs, peer_ptrs, rank, world, nelems);
  char check_msg[2048];
  snprintf(check_msg, sizeof(check_msg), "%p, %p, %p, %p, %p, %p, %p, %p\n",
      new_ptrs[0], new_ptrs[1], new_ptrs[2], new_ptrs[3],
      new_ptrs[4], new_ptrs[5], new_ptrs[6], new_ptrs[7]);
  r_print(check_msg, rank, world);

  auto source = rank ^ 0x1; // rank in same pair

  auto e = launch<xelink_route, T, lane_v, route_alternate_reduce_scatter>(
      new_ptrs, peer_ranks, source, rank, world, nelems/2, check_msg);

  r_print(check_msg, rank, world);

  double b = 0.0;

  for (int i = 0; i < repeat; ++ i) {
    auto e = launch<xelink_route, T, lane_v, route_alternate_reduce_scatter>(
        new_ptrs, peer_ranks, source, rank, world, nelems/2);
    b = bandwidth_from_event<T>(e, nelems/2);
  }
  snprintf(check_msg, sizeof(check_msg), "Rank %d scatter bandwidth: %fGB/s\n", rank, b);
  r_print(check_msg, rank, world);
}

int main(int argc, char* argv[]) {
  // parse command line options
  cxxopts::Options opts(
      "Fill remote GPU memory",
      "Exchange IPC handle to next rank (wrap around), and fill received buffer");

  opts.allow_unrecognised_options();
  opts.add_options()
    ("n,nelems", "Number of elements", cxxopts::value<std::string>()->default_value("16MB"))
    ("i,repeat", "Repeat times", cxxopts::value<uint32_t>()->default_value("16"))
    ("s,source", "Data source", cxxopts::value<std::string>()->default_value("1"))
    ("r,root", "Root of send", cxxopts::value<std::string>()->default_value("0"))
    ("d,dst", "Destinatino of send", cxxopts::value<std::string>()->default_value("1"))
    ("m,split", "Split position of transfer", cxxopts::value<size_t>()->default_value("128"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems_string = parsed_opts["nelems"].as<std::string>();
  auto repeat = parsed_opts["repeat"].as<uint32_t>();
  auto sources = commalist_to_vector(parsed_opts["source"].as<std::string>());
  auto roots = commalist_to_vector(parsed_opts["root"].as<std::string>());
  auto dst_ranks = commalist_to_vector(parsed_opts["dst"].as<std::string>());
  auto split = parsed_opts["split"].as<size_t>();

  // init section
  auto ret = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    std::cout<<"MPI init error"<<std::endl;
    return -1;
  }

  struct scopeCall {
    ~scopeCall() { MPI_Finalize(); }
  } scopeGuard;

  zeCheck(zeInit(0));
  int rank, world;

  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (std::any_of(roots.begin(), roots.end(),
        [&](int r) {return r >= world;}) ||
      std::any_of(dst_ranks.begin(), dst_ranks.end(),
        [&](int d) {return d >= world;})) {
    std::cout
      <<"Configuration error, root or destination must be inside the comm size"<<std::endl;
    return -1;
  }

  // rank 0, device 0, subdevice 0
  // rank 1, device 0, subdevice 1
  // rank 2, device 1, subdevice 0
  // ...
  auto queue = currentQueue(rank / 2, rank & 1);
  // a GPU page

  size_t base = 1;
  size_t pos = nelems_string.rfind("M");
  if (pos != std::string::npos) {
    base = 1024 * 1024ull;
  } else {
    pos = nelems_string.rfind("G");
    if (pos != std::string::npos)
      base = 1024 * 1024 * 1024ull;
  }

  size_t nelems;
  nelems = stoull(nelems_string) * base;

  using test_type = sycl::half;
  constexpr uint32_t v_lane = 4;
  // allocate more for gather case
  size_t alloc_size = nelems * sizeof(test_type);

  void* buffer = sycl::malloc_device(alloc_size * world, queue);
  void* b_host = sycl::malloc_host(alloc_size * world, queue);

  char check_msg[2048];
  // auto it = std::find(roots.begin(), roots.end(), rank);
  fill_sequential<test_type>(b_host, rank, alloc_size);

  peek_slice<test_type>(check_msg, (test_type *)b_host, split, rank, world);
  r_print(check_msg, rank, world);

  queue.memcpy(buffer, b_host, alloc_size * world);
  queue.wait();

  void *peer_bases[world];
  size_t offsets[world];
  auto ipc_handle = open_peer_ipc_mems(buffer, rank, world, peer_bases, offsets);

  void *peer_ptrs[world];

  for (int i = 0; i < world; ++ i) {
    peer_ptrs[i] = (char *)peer_bases[i] + offsets[i];
  }

  // avoid race condition
  MPI_Barrier(MPI_COMM_WORLD);

  // Or we map the device to host
  int dma_buf = 0;
  memcpy(&dma_buf, &ipc_handle, sizeof(int));
  auto *host_buf = (test_type *)mmap_host(alloc_size * world, dma_buf);

  MPI_Barrier(MPI_COMM_WORLD);

  test_reduce_scatter<test_type, v_lane>(peer_ptrs, rank, world, nelems, repeat);
  MPI_Barrier(MPI_COMM_WORLD);

  // Clean up, close/put ipc handles, free memory, etc.
  auto l0_ctx = sycl::get_native<
    sycl::backend::ext_oneapi_level_zero>(queue.get_context());

  munmap(host_buf, alloc_size);

  for (int i = 0; i < world; ++ i)
    if (i != rank)
      zeCheck(zeMemCloseIpcHandle(l0_ctx, peer_bases[i]));

  // zeCheck(zeMemPutIpcHandle(l0_ctx, ipc_handle)); /* the API is added after v1.6 */
  sycl::free(buffer, queue);
  return 0;
}
