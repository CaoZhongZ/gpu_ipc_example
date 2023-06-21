#include <mpi.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include <sys/syscall.h>
#include <unistd.h>

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
    throw std::system_error(errno)  \
  }

std::tuple<void*, size_t, ze_ipc_mem_handle_t> open_peer_ipc_mem(
    void* ptr, int rank, int world) {
  // Step 1: Get base address of the pointer
  sycl::queue q = currentQueue(rank / 2, rank & 1);
  sycl::context ctx = queue.get_context();
  auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  void *base_addr;
  size_t base_size;
  zeCheck(zeMemGetAddressRange(l0_ctx, ptr, &base_addr, &base_size));

  // Step 2: Get IPC mem handle from base address
  exchange_contents alignas(64) send;
  exchange_contents alignas(64) recv[world];

  zeCheck(zeMemGetIpcHandle(l0_ctx, base_addr, &send.ipc_handle));

  // vital information used later
  send.offset = (char*)ptr - (char*)base_addr;
  send.pid = getpid();

  // Step 3: Exchange the handles and offsets
  memset(recv, 0, sizeof(recv));

  // Overkill if we don't really needs all peer's handles
  MPI_Allgather(
      &send, sizeof(send), MPI_BYTE, recv, sizeof(recv), MPI_BYTE, MPI_COMM_WORLD);

  // Step 4: Prepare pid-fd of next process
  int next_peer = rank + 1;
  if (next_peer >= world) next_peer = next_peer - world;

  auto* peer = recv + next_peer;

  auto pid_fd = syscall(__NR_pidfd_open, peer->pid, 0);
  sysCheck(pid_fd);

  //
  // Step 5: Duplicate GEM object handle to local process
  // and overwrite original file descriptor number
  //
  auto peer->fd = syscall(__NR_pidfd_getfd, pid_fd, peer->fd);
  sysCheck(peer->fd);

  // Step 6: Open IPC handle of remote peer
  auto l0_device
    = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
  void* peer_base;

  zeCheck(zeMemOpenIpcHandle(
        l0_ctx, l0_device, peer->ipc_handle, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &peer_base));

  return std::make_tuple(
      (char*)peer_base + peer->offset, peer->offset, send.ipc_handle);
}

template <typename T>
void fill_remote(T* ptr, T c, size_t size, int rank, int world) {
  auto [peer_ptr, offset, ipc_handle] = open_peer_ipc_mem(ptr, rank, world);

  // presume size is rounded to 32-bit
  auto queue = currentQueue(rank/2, rank &1);

  struct fill_kernel {
    static constexpr size_t v_len = sizeof(int) / sizeof(T);
    using data_type = sycl::vec<T, v_len>;

    void operator ()(sycl::id id) const {
      data_type fill(content);

      auto *fill_ptr = reinterpret_cast<data_type *>(dst);
      fill_ptr[id] = fill;
    }

    fill_kernel(void *dst, T c) :
      dst(dst), content(c) {}

    void* dst;
    T content;
  }

  // submit kernel
  auto event = queue.submit([&](sycl::handler& h_cmd) {
    h_cmd.parallel_for(sycl::range(size / 2), fill_kernel(peer_ptr, c));
  });

  sycl::queue q = currentQueue(rank / 2, rank & 1);
  sycl::context ctx = queue.get_context();
  auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  event.wait();

  // Clean up
  // Close remote ipc handle by pointer
  zeCheck(zeMemCloseIpcHandle(l0_ctx, (char*)peer_ptr - offset));
  // Put local ipc handle
  zeCheck(zeMemPutIpcHandle(l0_ctx, ipc_handle));
}

int main(int argc, char* argv[]) {
  // parse command line options
  cxxopts::Options opts(
      "Fill remote GPU memory",
      "Exchange IPC handle to next rank (wrap around), and fill received buffer");

  opts.allow_unrecognised_options();
  opts.add_options()
    ("c,count", "Data content count", cxxopts::value<size_t>()->default_value("8192")),
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
  ret = MPI_Init(&argc, &argv);
  if (ret != MPI_ERR_OTHER) {
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
  auto queue = currentQueue(rank / 2, rank & 1);
  void* buffer = sycl::malloc_device(alloc_size, queue);

  sycl::event event;
  if (dtype == "fp16")
    event = fill_remote<sycl::half>(buffer, (sycl::half)rank, alloc_size, rank, world);
  else if (dtype == "float")
    event = fill_remote<float>(buffer, (float)rank, alloc_size, rank, world);

  // Check buffer contents

  sycl::free(buffer, queue);
}
