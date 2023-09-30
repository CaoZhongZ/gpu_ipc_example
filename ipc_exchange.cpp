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
#include "ipc_exchange.h"

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

void fill_random(void *p, int rank, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(1, 0x4000);

  auto sz_int = size / sizeof(int);

  for (size_t i = 0; i < sz_int; ++ i) {
    ((uint32_t *)p)[i] = distrib(gen) + rank;
  }
}
