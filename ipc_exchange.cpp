#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/ioctl.h>
#include <stddef.h>
#include <unistd.h>
#include <poll.h>
#include <system_error>
#include <stdarg.h>

#include <initializer_list>
#include <string>
#include <random>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include "ze_exception.hpp"
#include "ipc_exchange.h"
#include "utils.hpp"

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

// We can't inherit it from cmsghdr because flexible array member
struct exchange_fd {
  char obscure[CMSG_LEN(sizeof(int)) - sizeof(int)];
  int fd;

  exchange_fd(int cmsg_level, int cmsg_type, int fd)
    : fd(fd) {
    auto* cmsg = reinterpret_cast<cmsghdr *>(obscure);
    cmsg->cmsg_len = sizeof(exchange_fd);
    cmsg->cmsg_level = cmsg_level;
    cmsg->cmsg_type = cmsg_type;
  }

  exchange_fd() : fd(-1) {
    memset(obscure, 0, sizeof(obscure));
  };
};

void un_send_fd(int sock, int fd, int rank, size_t offset) {
  iovec iov[1];
  msghdr msg;
  auto rank_offset = std::make_pair(rank, offset);

  iov[0].iov_base = &rank_offset;
  iov[0].iov_len = sizeof(rank_offset);
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_name = nullptr;
  msg.msg_namelen = 0;

  exchange_fd cmsg (SOL_SOCKET, SCM_RIGHTS, fd);

  msg.msg_control = &cmsg;
  msg.msg_controllen = sizeof(exchange_fd);
  sysCheck(sendmsg(sock, &msg, 0));
}

std::tuple<int, int, size_t> un_recv_fd(int sock) {
  iovec iov[1];
  msghdr msg;
  std::pair<int, size_t> rank_offset;

  iov[0].iov_base = &rank_offset;
  iov[0].iov_len = sizeof(rank_offset);
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_name = nullptr;
  msg.msg_namelen = 0;

  exchange_fd cmsg;
  msg.msg_control = &cmsg;
  msg.msg_controllen = sizeof(exchange_fd);
  int n_recv = recvmsg(sock, &msg, 0);
  sysCheck(n_recv);
  // assert(n_recv == sizeof(int));

  return std::make_tuple(cmsg.fd, rank_offset.first, rank_offset.second);
}

int prepare_socket(const char *sockname) {
  sockaddr_un un;
  memset(&un, 0, sizeof(un));
  un.sun_family = AF_UNIX;
  strcpy(un.sun_path, sockname);

  auto sock = socket(AF_UNIX, SOCK_STREAM, 0);
  sysCheck(sock);

  int on = 1;
  sysCheck(ioctl(sock, FIONBIO, &on));

  auto size = offsetof(sockaddr_un, sun_path) + strlen(un.sun_path);
  sysCheck(bind(sock, (sockaddr *)&un, size));

  return sock;
}

int server_listen(const char *sockname) {
  // unlink(sockname);
  auto sock = prepare_socket(sockname);
  sysCheck(listen(sock, 10));

  return sock;
}

int serv_accept(int listen_sock) {
  sockaddr_un  un;

  socklen_t len = sizeof(un);
  auto accept_sock = accept(listen_sock, (sockaddr *)&un, &len);
  sysCheck(accept_sock);

  return accept_sock;
}

int client_connect(const char *server, const char *client) {
  auto sock = prepare_socket(client);
  sockaddr_un sun;
  memset(&sun, 0, sizeof(sun));
  sun.sun_family = AF_UNIX;
  strcpy(sun.sun_path, server);
  auto len = offsetof(sockaddr_un, sun_path) + strlen(server);
  sysCheck(connect(sock, (sockaddr *)&sun, len));
  return sock;
}

void un_allgather(exchange_contents* send_buf, exchange_contents recv_buf[], int rank, int world) {
  const char* servername_prefix = "open-peer-ipc-mem-server-rank_";
  const char* clientname_prefix = "open-peer-ipc-mem-client-rank_";
  char server_name[64];
  snprintf(server_name, sizeof(server_name), "%s%d", servername_prefix, rank);
  unlink(server_name);
  auto s_listen = server_listen(server_name);

  MPI_Barrier(MPI_COMM_WORLD);

  pollfd fdarray[world];
  int recv_socks[world-1];

  for (auto& pollfd : fdarray) pollfd.fd = -1;
  std::fill(recv_socks, recv_socks + world -1, -1);

  __scope_guard free_fd([&]() {
    for (int i = 0, j = 0; i < world; ++ i) {
      if ( i != rank && recv_socks[j] != -1)
        sysCheck(close(recv_socks[j++]));
      if ( fdarray[i].fd != -1 )
        sysCheck(close(fdarray[i].fd));
    }
  });

  // connect to all ranks
  for (int i = 0; i < world; ++ i) {
    if (rank == i) {
      fdarray[i].fd = s_listen;
      fdarray[i].events = POLLIN;
      fdarray[i].revents = 0;
    } else {
      char peer_name[64];
      char client_name[64];

      snprintf(client_name, sizeof(client_name), "%s%d-%d", clientname_prefix, rank, i);
      unlink(client_name);

      snprintf(peer_name, sizeof(peer_name), "%s%d", servername_prefix, i);
      fdarray[i].fd = client_connect(peer_name, client_name);
      fdarray[i].events = POLLOUT;
      fdarray[i].revents = 0;
    }
  }

  // std::future<std::tuple<int, int, size_t>> future_fds[world -1];
  int slot = 0;
  uint32_t send_progress = 1<<rank;

  while (slot < world-1 || send_progress != (1<<world) -1) {
    sysCheck(ppoll(fdarray, world, nullptr, nullptr));

    for (int i = 0; i < world; ++ i) {
      if (i == rank && (fdarray[i].revents & POLLIN)) {
        // auto accept_sock = serv_accept(fdarray[i].fd);
        // future_fds[slot ++] = std::async(
        //     std::launch::async, [=]() {
        //     struct sock_guard{
        //       int sock;
        //       sock_guard(int sock) : sock(sock) {}
        //       ~guard_sock() {sysCheck(close(sock));}
        //     } release(accept_sock);
        //     auto ret = un_recv_fd(accept_sock);
        //     return ret;});
        recv_socks[slot ++] = serv_accept(fdarray[i].fd);
      } else if ((send_progress & (1<<i)) == 0 && fdarray[i].revents & POLLOUT) {
        un_send_fd(fdarray[i].fd, send_buf->fd, rank, send_buf->offset);
        send_progress |= 1<<i;
      }
    }
  }

  for (int i = 0; i < world -1; ++i) {
    // future_fds[i].wait();
    // auto [fd, peer, offset] = future_fds[i].get();
    auto [fd, peer, offset] = un_recv_fd(recv_socks[i]);
    recv_buf[peer].fd = fd;
    recv_buf[peer].offset = offset;
  }

  recv_buf[rank] = *send_buf;
}

ze_ipc_mem_handle_t open_all_ipc_mems(
    sycl::queue queue, void* ptr, int rank, int world, void *peer_bases[], size_t offsets[]
) {
  // Step 1: Get base address of the pointer
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

  un_allgather(&send_buf, recv_buf, rank, world);

  for (int i = 0; i < world; ++ i) {
    if (i == rank) {
      peer_bases[i] = ptr;
      offsets[i] = 0;
    } else {
      auto* peer = recv_buf + i;

      auto l0_device
        = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
      void* peer_base;

      zeCheck(zeMemOpenIpcHandle(
          l0_ctx, l0_device, peer->ipc_handle, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &peer_base
      ));

      peer_bases[i] = peer_base;
      offsets[i] = peer->offset;
    }
  }
  return send_buf.ipc_handle;
}

std::tuple<void*, size_t, ze_ipc_mem_handle_t> open_peer_ipc_mem(
    sycl::queue queue, void* ptr, int rank, int world
) {
  // Step 1: Get base address of the pointer
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

void *mmap_host(size_t map_size, ze_ipc_mem_handle_t ipc_handle) {
  auto page_size = getpagesize();
  map_size = align_up(map_size, page_size);

  union {
    ze_ipc_mem_handle_t ipc_handle;
    int fd = -1;
  } uhandle {ipc_handle};

  return mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, uhandle.fd, 0);
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
