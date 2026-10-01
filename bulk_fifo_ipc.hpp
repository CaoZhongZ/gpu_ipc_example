#pragma once

#include <array>
#include <mpi.h>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "bulk_fifo_layout.hpp"

namespace bulk_fifo {

void zeCheck(ze_result_t result, const char* operation);

// Only the three peer ranges used by this Ring rank are opened. Each range
// retains its own exported allocation base/offset, including pooled USM.
class RingIpc {
public:
  RingIpc(sycl::queue& queue, MPI_Comm communicator, int rank, int world,
          const std::array<void*, 3>& localRanges,
          const std::array<std::size_t, 3>& rangeBytes);
  ~RingIpc();
  RingIpc(const RingIpc&) = delete;
  RingIpc& operator=(const RingIpc&) = delete;

  unsigned char* previousSendControl() const { return ranges_[0]; }
  unsigned char* nextReceiveControl() const { return ranges_[1]; }
  unsigned char* nextData() const { return ranges_[2]; }

private:
  ze_context_handle_t context_;
  std::array<ze_ipc_mem_handle_t, 3> exports_{};
  std::array<bool, 3> exported_{};
  std::array<void*, 3> imports_{};
  std::array<unsigned char*, 3> ranges_{};
};

} // namespace bulk_fifo
