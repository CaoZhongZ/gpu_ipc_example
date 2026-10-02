#include "bulk_fifo_ipc.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace bulk_fifo {

void zeCheck(ze_result_t result, const char* operation) {
  if (result == ZE_RESULT_SUCCESS) return;
  std::ostringstream message;
  message << operation << " failed: Level Zero result 0x" << std::hex << result;
  throw std::runtime_error(message.str());
}

namespace {

class Fd {
public:
  explicit Fd(int value = -1) : value_(value) {}
  ~Fd() { if (value_ >= 0) ::close(value_); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& other) noexcept : value_(other.value_) { other.value_ = -1; }
  Fd& operator=(Fd&& other) noexcept {
    if (value_ >= 0) ::close(value_);
    value_ = other.value_;
    other.value_ = -1;
    return *this;
  }
  int get() const { return value_; }
  int release() {
    const int value = value_;
    value_ = -1;
    return value;
  }
private:
  int value_;
};

void socketCheck(bool success, const char* operation) {
  if (!success)
    throw std::runtime_error(std::string(operation) + ": " +
                             std::strerror(errno));
}

Fd newSocket() {
  Fd socket(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
  socketCheck(socket.get() >= 0, "socket");
  const timeval timeout{30, 0};
  socketCheck(::setsockopt(socket.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout,
                          sizeof(timeout)) == 0, "socket receive timeout");
  socketCheck(::setsockopt(socket.get(), SOL_SOCKET, SO_SNDTIMEO, &timeout,
                          sizeof(timeout)) == 0, "socket send timeout");
  return socket;
}

struct Metadata {
  std::uint32_t owner;
  RangeKind kind;
  std::uint64_t offset;
  std::uint64_t bytes;
  ze_ipc_mem_handle_t handle;
};

void sendRange(int socket, const Metadata& metadata) {
  int fd;
  std::memcpy(&fd, &metadata.handle, sizeof(fd));
  iovec payload{const_cast<Metadata*>(&metadata), sizeof(metadata)};
  alignas(cmsghdr) unsigned char ancillary[CMSG_SPACE(sizeof(int))]{};
  msghdr message{};
  message.msg_iov = &payload;
  message.msg_iovlen = 1;
  message.msg_control = ancillary;
  message.msg_controllen = sizeof(ancillary);
  auto* control = CMSG_FIRSTHDR(&message);
  control->cmsg_level = SOL_SOCKET;
  control->cmsg_type = SCM_RIGHTS;
  control->cmsg_len = CMSG_LEN(sizeof(fd));
  std::memcpy(CMSG_DATA(control), &fd, sizeof(fd));
  ssize_t result;
  do { result = ::sendmsg(socket, &message, MSG_NOSIGNAL); }
  while (result < 0 && errno == EINTR);
  socketCheck(result == sizeof(metadata), "send IPC range");
}

struct Received {
  Metadata metadata{};
  Fd fd;
};

Received receiveRange(int socket) {
  Metadata metadata{};
  iovec payload{&metadata, sizeof(metadata)};
  alignas(cmsghdr) unsigned char ancillary[CMSG_SPACE(sizeof(int))]{};
  msghdr message{};
  message.msg_iov = &payload;
  message.msg_iovlen = 1;
  message.msg_control = ancillary;
  message.msg_controllen = sizeof(ancillary);
  ssize_t result;
  do { result = ::recvmsg(socket, &message, MSG_CMSG_CLOEXEC); }
  while (result < 0 && errno == EINTR);
  socketCheck(result == sizeof(metadata), "receive IPC range");
  auto* control = CMSG_FIRSTHDR(&message);
  if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) || !control ||
      control->cmsg_level != SOL_SOCKET || control->cmsg_type != SCM_RIGHTS ||
      control->cmsg_len != CMSG_LEN(sizeof(int)))
    throw std::runtime_error("invalid IPC descriptor ancillary data");
  int fd;
  std::memcpy(&fd, CMSG_DATA(control), sizeof(fd));
  std::memcpy(&metadata.handle, &fd, sizeof(fd));
  return Received{metadata, Fd(fd)};
}

struct SocketPath {
  char directory[64]{};
  std::string socket;
  ~SocketPath() {
    if (!socket.empty()) ::unlink(socket.c_str());
    if (directory[0]) ::rmdir(directory);
  }
};

} // namespace

RingIpc::RingIpc(sycl::queue& queue, MPI_Comm communicator, int rank, int world,
                 const std::array<void*, 3>& localRanges,
                 const std::array<std::size_t, 3>& rangeBytes)
    : context_(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          queue.get_context())) {
  const auto device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
  std::array<Metadata, 3> local{};
  for (unsigned kind = 0; kind < 3; ++kind) {
    void* base = nullptr;
    std::size_t allocationBytes = 0;
    zeCheck(zeMemGetAddressRange(context_, localRanges[kind], &base,
                                &allocationBytes), "zeMemGetAddressRange");
    const auto offset = reinterpret_cast<std::uintptr_t>(localRanges[kind]) -
                        reinterpret_cast<std::uintptr_t>(base);
    if (offset > allocationBytes || rangeBytes[kind] > allocationBytes - offset)
      throw std::runtime_error("logical USM range exceeds exported allocation");
    zeCheck(zeMemGetIpcHandle(context_, base, &exports_[kind]),
            "zeMemGetIpcHandle");
    exported_[kind] = true;
    local[kind] = Metadata{static_cast<std::uint32_t>(rank),
                          static_cast<RangeKind>(kind), offset, rangeBytes[kind],
                          exports_[kind]};
  }

  // A rank-zero Unix-socket broker transfers real file descriptors. MPI only
  // carries the unique socket path; copying raw handle bytes via MPI is invalid
  // when the handle contains a file descriptor from another process.
  SocketPath path;
  char socketName[108]{};
  Fd listener;
  if (rank == 0) {
    std::strcpy(path.directory, "/tmp/bulk-fifo-ring-XXXXXX");
    socketCheck(::mkdtemp(path.directory) != nullptr, "mkdtemp");
    path.socket = std::string(path.directory) + "/ipc";
    std::strcpy(socketName, path.socket.c_str());
    listener = newSocket();
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strcpy(address.sun_path, socketName);
    socketCheck(::bind(listener.get(), reinterpret_cast<sockaddr*>(&address),
                       sizeof(address)) == 0, "bind IPC broker");
    socketCheck(::listen(listener.get(), world) == 0, "listen IPC broker");
  }
  MPI_Bcast(socketName, sizeof(socketName), MPI_CHAR, 0, communicator);

  const int previous = (rank + world - 1) % world;
  const int next = (rank + 1) % world;
  auto importRange = [&](const Metadata& metadata, unsigned index) {
    const auto expectedOwner = index == 0 ? previous : next;
    const auto expectedKind = static_cast<RangeKind>(index);
    if (metadata.owner != static_cast<unsigned>(expectedOwner) ||
        metadata.kind != expectedKind || metadata.bytes != rangeBytes[index])
      throw std::runtime_error("unexpected owner, kind, or size in IPC range");
    int receivedFd;
    std::memcpy(&receivedFd, &metadata.handle, sizeof(receivedFd));
    Fd importFd(::fcntl(receivedFd, F_DUPFD_CLOEXEC, 0));
    socketCheck(importFd.get() >= 0, "duplicate IPC import descriptor");
    auto importHandle = metadata.handle;
    const int descriptor = importFd.get();
    std::memcpy(&importHandle, &descriptor, sizeof(descriptor));
    zeCheck(zeMemOpenIpcHandle(context_, device, importHandle,
                              ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED,
                              &imports_[index]), "zeMemOpenIpcHandle");
    // Linux NEO retains this descriptor in the imported allocation and closes
    // it with zeMemCloseIpcHandle. Broker/client Fd objects own separate copies.
    importFd.release();
    ranges_[index] = static_cast<unsigned char*>(imports_[index]) +
                     metadata.offset;
    const auto alignment = index == 2 ? DataAlignment : ControlAlignment;
    if (reinterpret_cast<std::uintptr_t>(ranges_[index]) % alignment)
      throw std::runtime_error("imported logical range is misaligned");
  };

  if (rank != 0) {
    Fd client = newSocket();
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strcpy(address.sun_path, socketName);
    socketCheck(::connect(client.get(), reinterpret_cast<sockaddr*>(&address),
                          sizeof(address)) == 0, "connect IPC broker");
    for (const auto& metadata : local) sendRange(client.get(), metadata);
    for (unsigned kind = 0; kind < 3; ++kind) {
      auto received = receiveRange(client.get());
      importRange(received.metadata, kind);
    }
  } else {
    std::vector<std::array<Metadata, 3>> all(world);
    std::vector<std::array<Fd, 3>> receivedFds(world);
    std::vector<Fd> clients(world);
    all[0] = local;
    for (int count = 1; count < world; ++count) {
      Fd client(::accept4(listener.get(), nullptr, nullptr, SOCK_CLOEXEC));
      socketCheck(client.get() >= 0, "accept IPC client");
      unsigned owner = 0;
      for (unsigned kind = 0; kind < 3; ++kind) {
        auto received = receiveRange(client.get());
        const auto& metadata = received.metadata;
        if (kind == 0) owner = metadata.owner;
        if (!owner || owner >= static_cast<unsigned>(world) ||
            metadata.owner != owner || metadata.kind != static_cast<RangeKind>(kind) ||
            metadata.bytes != rangeBytes[kind] || clients[owner].get() >= 0)
          throw std::runtime_error("invalid client IPC metadata");
        all[owner][kind] = metadata;
        receivedFds[owner][kind] = std::move(received.fd);
      }
      clients[owner] = std::move(client);
    }
    for (int destination = 0; destination < world; ++destination) {
      for (unsigned kind = 0; kind < 3; ++kind) {
        const int owner = kind == 0 ? (destination + world - 1) % world
                                   : (destination + 1) % world;
        if (destination == 0) importRange(all[owner][kind], kind);
        else sendRange(clients[destination].get(), all[owner][kind]);
      }
    }
  }
  MPI_Barrier(communicator);
}

RingIpc::~RingIpc() {
  for (auto* imported : imports_)
    if (imported) zeMemCloseIpcHandle(context_, imported);
  for (unsigned i = 0; i < 3; ++i)
    if (exported_[i]) zeMemPutIpcHandle(context_, exports_[i]);
}

} // namespace bulk_fifo
