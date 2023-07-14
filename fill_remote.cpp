#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <system_error>

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

struct atomic_baseline {
  constexpr static int max_peers = 16;
  template <typename T>
  using atomic_ref = sycl::atomic_ref<T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>;
public:
  atomic_baseline(void *peer_ptrs[], int rank, int world, sycl::stream s)
    : rank(rank), world(world), cout(s) {
    for (int i = 0; i < world; ++ i)
      ptrs[i] = (uint32_t *)peer_ptrs[i];
  }

  // create contension
  void operator() (sycl::nd_item<1> pos) const {
    auto *ptr = ptrs[rank];
    auto local_id = pos.get_local_id();
    atomic_ref<uint32_t> atomic_slot(ptr[local_id]);

    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);
    // if (local_id == 0) {
      // cout<<"["<<pos.get_group(0)<<"] ";
      atomic_slot++;
    // }
    // cout<<sycl::endl;

    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);
  }

private:
  uint32_t *ptrs[max_peers];
  int rank;
  int world;
  sycl::stream cout;
};

struct atomic_stresser {
  constexpr static int max_peers = 16;
  template <typename T>
  using atomic_ref = sycl::atomic_ref<T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>;
public:
  atomic_stresser(void *peer_ptrs[], int rank, int world, sycl::stream s)
    : rank(rank), world(world), cout(s) {
    for (int i = 0; i < world; ++ i)
      ptrs[i] = (uint32_t *)peer_ptrs[i];
  }

  // create contension
  void operator() (sycl::nd_item<1> pos) const {
    auto local_id = pos.get_local_id();

    for (int peer = 0; peer < world; ++ peer) {
      auto* peer_ptr = ptrs[peer];
      atomic_ref<uint32_t> atomic_slot(peer_ptr[local_id]);
      atomic_slot.fetch_add(1);
    }
  }

private:
  uint32_t *ptrs[max_peers];
  int rank;
  int world;
  sycl::stream cout;
};

void stress_test(
    int rank, int world,
    void *peer_bases[], size_t offsets[],
    size_t global_sz, size_t group_sz) {
  void* peer_ptrs[world];

  for (int i = 0; i < world; ++ i)
    peer_ptrs[i] = reinterpret_cast<char *>(peer_bases[i]) + offsets[i];

  auto queue = currentQueue(rank/2, rank & 1);
  std::cout<<"start run with ["<<global_sz<<", "<<group_sz<<"]"<<std::endl;

  queue.submit([&](sycl::handler &cgh) {
    sycl::stream s(4096, 32, cgh);
    cgh.parallel_for(sycl::nd_range<1>({global_sz}, {group_sz}),
      // atomic_baseline(peer_ptrs, rank, world, s));
      atomic_stresser(peer_ptrs, rank, world, s));
  });
}

int main(int argc, char* argv[]) {
  // parse command line options
  cxxopts::Options opts(
      "Fill remote GPU memory",
      "Exchange IPC handle to next rank (wrap around), and fill received buffer");

  opts.allow_unrecognised_options();
  opts.add_options()
    ("g,global_size", "Launching global size", cxxopts::value<size_t>()->default_value("8192"))
    ("l,group_size", "Launching group size", cxxopts::value<size_t>()->default_value("16"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto global_sz = parsed_opts["global_size"].as<size_t>();
  auto group_sz = parsed_opts["group_size"].as<size_t>();

  // init section
  auto ret = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    std::cout<<"MPI init error"<<std::endl;
    return -1;
  }

  struct scopeCall {
    ~scopeCall() { MPI_Finalize(); }
  }scopeGuard;

  zeCheck(zeInit(0));
  int rank, world;

  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // rank 0, device 0, subdevice 0
  // rank 1, device 0, subdevice 1
  // rank 2, device 1, subdevice 0
  // ...
  auto queue = currentQueue(rank / 2, rank & 1);
  // a GPU page
  size_t alloc_size = 64 * 1024;
  void* buffer = sycl::malloc_device(alloc_size, queue);

  void *peer_bases[world];
  size_t offsets[world];
  auto ipc_handle = open_peer_ipc_mems(buffer, rank, world, peer_bases, offsets);

  stress_test(
      rank, world, peer_bases, offsets, global_sz, group_sz);

  // avoid race condition
  queue.wait();
  MPI_Barrier(MPI_COMM_WORLD);

  // Or we map the device to host
  int dma_buf = 0;
  memcpy(&dma_buf, &ipc_handle, sizeof(int));
  uint32_t *host_buf = (uint32_t *)mmap_host(alloc_size, dma_buf);
  std::cout<<"Peak: "<<host_buf[0]<<", "<<host_buf[1]<<", "<<host_buf[2]<<", ..."<<std::endl;

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
