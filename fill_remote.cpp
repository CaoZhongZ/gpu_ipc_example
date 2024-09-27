#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <system_error>

#include <iostream>
#include <chrono>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <libdrm/i915_drm.h>

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
std::string get_device_path(int devNo, int subNo) {
  sycl::device dev = currentSubDevice(devNo, subNo);
  auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);

  ze_pci_ext_properties_t pcie_prop {.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
  zeCheck(zeDevicePciGetPropertiesExt(l0_device, &pcie_prop));

  std::string device_path ("/dev/dri/by-path/");
  std::string device_suffix ("-render");

  char device_bfd[32];
  std::snprintf(device_bfd, 32, "pci-0000:%02x:%02x.%x", pcie_prop.address.bus, pcie_prop.address.device, pcie_prop.address.function);

  auto device_file = device_path + std::string(device_bfd) + device_suffix;

  return device_file;
}

int open_drmfd(int rank) {
  auto device_file = get_device_path(rank/2, rank &1);

  int devfd = open(device_file.c_str(), O_RDWR);
  sysCheck(devfd);

  return devfd;
}

int get_gem_handle(int dmabuf_fd, int devfd) {
  struct drm_prime_handle req = {0, 0, 0};
  req.fd = dmabuf_fd;

  int ret = ioctl(devfd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &req);
  sysCheck(ret);

  return req.handle;
}

int get_dmabuf_fd(int gem_handle, int devfd) {
  struct drm_prime_handle req = { 0, 0, 0 };
  req.flags = DRM_CLOEXEC | DRM_RDWR;
  req.handle = gem_handle;
  int ret = ioctl(devfd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &req);
  sysCheck(ret);
  return req.fd;
}

std::tuple<void*, size_t, ze_ipc_mem_handle_t> open_peer_ipc_mem_drm(
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
  union {
    ze_ipc_mem_handle_t ipc_handle;
    int dmabuf_fd;
  } uni_handle;

  zeCheck(zeMemGetIpcHandle(l0_ctx, base_addr, &uni_handle.ipc_handle));

  auto dev_fd = open_drmfd(rank);
  auto gem_handle = get_gem_handle(uni_handle.dmabuf_fd, dev_fd);

  send_buf.ipc_handle = uni_handle.ipc_handle;
  send_buf.fd = gem_handle;
  send_buf.offset = (char*)ptr - (char*)base_addr;
  send_buf.pid = rank; /* should be device number */

  // Step 3: Exchange the handles and offsets
  memset(recv_buf, 0, sizeof(recv_buf));
  // Overkill if we don't really needs all peer's handles
  MPI_Allgather(
      &send_buf, sizeof(send_buf), MPI_BYTE, recv_buf, sizeof(send_buf), MPI_BYTE, MPI_COMM_WORLD);

  // Step 4: Prepare pid file descriptor of next process
  int next_peer = rank + 1;
  if (next_peer >= world) next_peer = next_peer - world;

  auto* peer = recv_buf + next_peer;
  auto remote_dev = (rank / 2 == peer->pid / 2) ? dev_fd : open_drmfd(peer->pid);

  peer->fd = get_dmabuf_fd(peer->fd, remote_dev);

  // Step 6: Open IPC handle of remote peer
  auto l0_device
    = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
  void* peer_base;

  zeCheck(zeMemOpenIpcHandle(
        l0_ctx, l0_device, peer->ipc_handle, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &peer_base));

  if (rank / 2 != peer->pid/2)
    sysCheck(close(remote_dev));

  sysCheck(close(dev_fd));

  return std::make_tuple(
      (char*)peer_base + peer->offset, peer->offset, uni_handle.ipc_handle);
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

  open_drmfd(rank);

  // rank 0, device 0, subdevice 0
  // rank 1, device 0, subdevice 1
  // rank 2, device 1, subdevice 0
  // ...
  auto queue = currentQueue(rank / 2, rank & 1);
  void* buffer = sycl::malloc_device(alloc_size, queue);
  void* host_buf = sycl::malloc_host(alloc_size, queue);

   MPI_Barrier(MPI_COMM_WORLD);

 // XXX: gain access to remote pointers

  auto start = std::chrono::high_resolution_clock::now();
  auto [peer_ptr, offset, ipc_handle] = open_peer_ipc_mem_drm(buffer, rank, world);
  auto delta = std::chrono::high_resolution_clock::now() - start;
  auto elapse = std::chrono::duration_cast<std::chrono::microseconds>(delta).count();
  printf("Exchange time is %ldus\n", elapse);

  // run fill kernel to fill remote GPU memory
  if (dtype == "fp16")
    queue.fill<sycl::half>((sycl::half *)peer_ptr, (sycl::half)rank, count);
  else if (dtype == "float")
    queue.fill<float>((float *)peer_ptr, (float)rank, count);

  // avoid race condition
  queue.wait();
  MPI_Barrier(MPI_COMM_WORLD);

  // Check buffer contents
  queue.memcpy(host_buf, buffer, alloc_size);
  queue.wait();

  bool check = false;

  int next_rank = rank + 1;
  if (next_rank >= world) next_rank -= world;

  if (dtype == "fp16")
    check = checkResults((sycl::half *)host_buf, (sycl::half)next_rank, count);
  else
    check = checkResults((float*)host_buf, (float)next_rank, count);

  if (check)
    std::cout<<"Successfully fill remote buffer"<<std::endl;
  else
    std::cout<<"Error occured when fill remote buffer"<<std::endl;

  // Clean up, close/put ipc handles, free memory, etc.
  auto l0_ctx = sycl::get_native<
    sycl::backend::ext_oneapi_level_zero>(queue.get_context());

  zeCheck(zeMemCloseIpcHandle(l0_ctx, (char*)peer_ptr - offset));
  // zeCheck(zeMemPutIpcHandle(l0_ctx, ipc_handle)); /* the API is added after v1.6 */
  sycl::free(buffer, queue);
  sycl::free(host_buf, queue);
}
