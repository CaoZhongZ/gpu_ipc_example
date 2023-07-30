#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <system_error>

#include <initializer_list>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include "cxxopts.hpp"
#include "ze_exception.hpp"
#include "sycl_misc.hpp"

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

template <typename T, int lane_v, int fanout=1>
struct xelink_send {
  using v_T = sycl::vec<T, sizeof(float)/sizeof(T) * lane_v>;
  // 64 SS * 8 threads, each with 16(sub-groups) * 8(eu) SIMD lanes
  static constexpr size_t hw_groups = 512;
  static constexpr size_t local_size = 128;

public:
  xelink_send(
      void *peers[], int remote, int rank, int world, size_t nelems/*, sycl::stream s*/)
    : peer(reinterpret_cast<v_T *>(peers[remote]))
      , local(reinterpret_cast<v_T *>(peers[rank]))
      , nelems(nelems)/*, cout(s)*/ {}

  void operator() (sycl::nd_item<1> pos) const {
    for (size_t i = pos.get_global_id(0); i < nelems/v_T::size(); i += pos.get_global_range(0))
      peer[i] = local[i];
  }

  static void launch(void* peer_ptrs[], int remote, int root, int world, size_t nelems,
      bool profiling = false) {
    size_t data_groups = (nelems/v_T::size() + local_size - 1) / local_size;
    size_t group_size = std::min(data_groups, hw_groups);

    auto global_size = group_size * local_size;
    auto queue = currentQueue(root/2, root & 1);
    std::cout<<"Launching send from "<<root<<" to "<<remote
      <<" ("<<group_size<<", "<<local_size<<")"<<std::endl;
    auto e = queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>({global_size}, {local_size}),
          xelink_send(peer_ptrs, remote, root, world, nelems));
    });

    if (profiling) {
      e.wait();
      auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
      auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();

      // since timestamp is in the unit of ns, then the bandwidth is GB/s in unit
      auto bandwidth = (double)(nelems * sizeof(T)) / (double)(end - start);
      std::cout<<"Copy bandwidth is "<<bandwidth<<"GB/s"<<std::endl;
    }
  }

private:
  v_T *peer;
  v_T *local;
  size_t nelems;
  // sycl::stream cout;
};

template <int F> struct remotes {
  static constexpr int fanout = F;
  std::array<int, F> peers;

  remotes(const std::array<int, F> &peer_list) : peers(peer_list) {}
};

template <typename T, int fanout>
struct fanout_in_thread {
  static inline void run(sycl::nd_item<1> pos, T *peers[], T value, size_t off) {
#   pragma unroll (fanout)
    for (int f = 0; f < fanout; ++ f) {
      peers[f][off] = value;
    }
  }
};

// strategy 1, fanout in thread
template <typename T, int lane_v, typename R, template <typename, int> class fanout_policy>
struct xelink_bcast {
  using v_T = sycl::vec<T, sizeof(float)/sizeof(T) * lane_v>;
  static constexpr int fanout = R::fanout;
  // 64 SS * 8 threads, each with 16(sub-groups) * 8(eu) SIMD lanes
  static constexpr size_t hw_groups = 512;
  static constexpr size_t local_size = 128;

public:
  xelink_bcast(void *peer_ptrs[], R&& remote_info,
      int root, int world, size_t nelems/*, sycl::stream s*/)
    : local(reinterpret_cast<v_T *>(peer_ptrs[root])), nelems(nelems)/*, cout(s)*/ {
#   pragma unroll (fanout)
    for (int f = 0; f < fanout; ++ f) {
      auto remote = remote_info.peers[f]; // can't be root
      peers[f] = reinterpret_cast<v_T *>(peer_ptrs[remote]);
    }
  }

  void operator() (sycl::nd_item<1> pos) const {
    for (size_t off = pos.get_global_id(0);
        off < nelems/v_T::size(); off += pos.get_global_range(0)) {
      fanout_policy<v_T, fanout>::run(pos, peers, local[off], off);
    }
  }

  static void launch(void* peers[], R&& remote_info,
      int root, int world, size_t nelems, bool profiling = false) {
    size_t data_groups = (nelems/v_T::size() + local_size - 1) / local_size;
    size_t group_size = std::min(data_groups, hw_groups);
    size_t global_size = group_size * local_size;

    auto queue = currentQueue(root/2, root & 1);
    auto e = queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>({global_size}, {local_size}),
          xelink_bcast(peers, remote_info, root, world, nelems));
    });

    if (profiling) {
      e.wait();
      auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
      auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();

      // since timestamp is in the unit of ns, then the bandwidth is GB/s in unit
      auto bandwidth = (double)(nelems * sizeof(T)) / (double)(end - start);
      std::cout<<"Copy bandwidth is "<<bandwidth<<"GB/s"<<std::endl;
    }
  }

private:
  v_T *peers[fanout];
  v_T *local;
  size_t nelems;
  // sycl::stream cout;
};

int main(int argc, char* argv[]) {
  // parse command line options
  cxxopts::Options opts(
      "Fill remote GPU memory",
      "Exchange IPC handle to next rank (wrap around), and fill received buffer");

  opts.allow_unrecognised_options();
  opts.add_options()
    ("n,nelems", "Number of elements", cxxopts::value<size_t>()->default_value("8192"))
    ("i,repeat", "Repeat times", cxxopts::value<uint32_t>()->default_value("16"))
    ("r,root", "Root of send", cxxopts::value<int>()->default_value("0"))
    ("d,dst", "Destinatino of send", cxxopts::value<int>()->default_value("1"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems = parsed_opts["nelems"].as<size_t>();
  auto repeat = parsed_opts["repeat"].as<uint32_t>();
  auto dst_rank = parsed_opts["dst"].as<int>();
  auto root = parsed_opts["root"].as<int>();

  if (root == dst_rank) {
    std::cout<<"Root and Destination can't be the same"<<std::endl;
    return -1;
  }

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

  if (root >= world || dst_rank >= world) {
    std::cout
      <<"Configuration error, root or destination must be inside the comm"<<std::endl;
    return -1;
  }

  // rank 0, device 0, subdevice 0
  // rank 1, device 0, subdevice 1
  // rank 2, device 1, subdevice 0
  // ...
  auto queue = currentQueue(rank / 2, rank & 1);
  // a GPU page
  size_t alloc_size = nelems * sizeof(sycl::half);
  void* buffer = sycl::malloc_device(alloc_size, queue);
  queue.memset(buffer, rank + 42, alloc_size);

  void *peer_bases[world];
  size_t offsets[world];
  auto ipc_handle = open_peer_ipc_mems(buffer, rank, world, peer_bases, offsets);

  void *peer_ptrs[world];

  for (int i = 0; i < world; ++ i) {
    peer_ptrs[i] = (char *)peer_bases[i] + offsets[i];
  }

  if ( rank == root ) {
    std::cout<<"Warmup run"<<std::endl;
    xelink_send<sycl::half, 1, 1>::launch(
        peer_ptrs, dst_rank, rank, world, nelems, true);

    std::cout<<"Repeat run"<<std::endl;
    for (int i = 0; i < repeat; ++ i)
      xelink_send<sycl::half, 1, 1>::launch(
          peer_ptrs, dst_rank, rank, world, nelems, true);
  }

  // avoid race condition
  queue.wait();
  MPI_Barrier(MPI_COMM_WORLD);

  // Or we map the device to host
  int dma_buf = 0;
  memcpy(&dma_buf, &ipc_handle, sizeof(int));
  uint32_t *host_buf = (uint32_t *)mmap_host(alloc_size, dma_buf);

  MPI_Barrier(MPI_COMM_WORLD);

  std::cout<<std::hex<<"Peek: "<<host_buf[0]<<", "<<host_buf[1]<<", "<<host_buf[2]
    <<", ..., "<<host_buf[alloc_size / sizeof(uint32_t) -1]
    <<", "<<host_buf[alloc_size / sizeof(uint32_t) -2]<<std::endl;

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
