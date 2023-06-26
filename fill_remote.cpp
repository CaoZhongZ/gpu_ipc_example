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
    ze_ipc_event_pool_handle_t ipc_pool;
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

static constexpr ze_event_pool_desc_t default_pool_desc {
  .stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  .pNext = nullptr;
  .flags = ZE_EVENT_POOL_FLAG_IPC;
  .count = 32;
};

ze_event_pool_handle_t create_event_pool(int rank, int world) {
  sycl::queue queue = currentQueue(rank /2, rank & 1);
  sycl::context ctx = queue.get_context();

  auto l0_ctx = sycl::get_native<
    sycl::backend::ext_oneapi_level_zero>(ctx);
  auto l0_dev = sycl::get_native<
    sycl::backend::ext_oneapi_level_zero>(queue.get_device());

  ze_event_pool_handle_t ret;
  ze_event_pool_desc_t pool_desc = default_pool_desc;

  zeCheck(zeEventPoolCreate(l0_ctx, &pool_desc, 1, &l0_dev, &ret));
  return ret;
}

std::tuple<ze_event_pool_handle_t, ze_event_pool_handle_t> open_peer_ipc_pool(
    ze_event_pool_handle_t handle, int rank, int world) {
  // Get IPC Pool handle out of local IPC handle
  sycl::queue queue = currentQueue(rank / 2, rank & 1);
  sycl::context ctx = queue.get_context();
  auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  alignas(64) exchange_contents send_buf;
  alignas(64) exchange_contents recv_buf[world];

  // fill in the exchange info
  zeCheck(zeEventPoolGetIpcHandle(l0_ctx, ptr, &send_buf.ipc_pool));
  send_buf.offset = 0;
  send_buf.pid = getpid();

  // Step 3: Exchange the handles and offsets
  memset(recv_buf, 0, sizeof(recv_buf));
  // Overkill if we don't really needs all peer's handles
  MPI_Allgather(
      &send_buf, sizeof(send_buf), MPI_BYTE,
      recv_buf, sizeof(send_buf), MPI_BYTE, MPI_COMM_WORLD);

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
  ze_event_pool_handle_t peer_handle;

  zeCheck(zeEventPoolOpenIpcHandle(l0_ctx, peer->ipc_pool, &peer_handle));
  return std::make_pair(peer_handle, send_buf.ipc_pool);
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

template <typename T>
bool checkResults(T *ptr, T c, size_t count) {
  for (int i = 0; i < count; ++ i) {
    if (*ptr != c) return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
  // parse command line options
  cxxopts::Options opts(
      "Fill remote GPU memory",
      "Exchange IPC handle to next rank (wrap around), and fill received buffer");

  opts.allow_unrecognised_options();
  opts.add_options()
    ("c,count", "Data content count", cxxopts::value<size_t>()->default_value("8192"))
    ("t,type", "Data content type", cxxopts::value<std::string>()->default_value("fp16"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto count = parsed_opts["count"].as<size_t>();
  auto dtype = parsed_opts["type"].as<std::string>();

  size_t alloc_size = 0;

  if (dtype == "fp16")
    alloc_size = count * sizeof(sycl::half);
  else if (dtype == "float")
    alloc_size = count * sizeof(float);

  // init section
  auto ret = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    std::cout<<"MPI init error"<<std::endl;
    return -1;
  }

  zeCheck(zeInit(0));
  int rank, world;

  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // rank 0, device 0, subdevice 0
  // rank 1, device 0, subdevice 1
  // rank 2, device 1, subdevice 0
  // ...
  auto h_event_pool = create_event_pool(rank, world);
  auto [remote_ipc_pool, local_ipc_pool] = open_peer_ipc_pool(h_event, rank, world);

  MPI_Barrier(MPI_COMM_WORLD);

  std::cout<<"Successfully get remote handle"<<std::endl;

  // Clean up, close/put ipc handles, free memory, etc.
  zeCheck(zeEventPoolCloseIpcHandle(remote_ipc_pool));
  MPI_Barrier(MPI_COMM_WORLD);

  // zeCheck(zeEventPoolPutIpcHandle(l0_ctx, local_ipc_pool)); /* the API is added after v1.6 */
  zeCheck(zeEventDestroy(h_event_pool));
}
