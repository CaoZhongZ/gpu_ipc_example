#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace bulk_fifo {

constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = 1024 * KiB;
constexpr unsigned MaxRanks = 16;
constexpr unsigned Slots = 8;
constexpr unsigned SubgroupSize = 16;
constexpr unsigned PackBytes = 16;
constexpr unsigned MinTransferBytes = SubgroupSize * PackBytes;
constexpr std::size_t ControlAlignment = 4 * KiB;
constexpr std::size_t DataAlignment = 2 * MiB;
constexpr std::size_t ControlTableBytes = 4 * KiB;
constexpr std::size_t ControlEntryBytes = 256;

enum class RangeKind : std::uint32_t { SendControl, ReceiveControl, Data };

enum class StoreCache : unsigned {
  UcUcUc, WbWbUc, WtWbUc, StWbUc, UcWbUc, StUcUc, WbUcUc
};
inline constexpr std::array<const char*, 7> StoreCacheNames{
    "uc.uc.uc", "wb.wb.uc", "wt.wb.uc", "st.wb.uc",
    "uc.wb.uc", "st.uc.uc", "wb.uc.uc"};
constexpr const char* storeCacheName(StoreCache cache) {
  return StoreCacheNames[static_cast<unsigned>(cache)];
}

struct alignas(ControlEntryBytes) CounterEntry {
  std::uint64_t step;
  unsigned char reserved[ControlEntryBytes - sizeof(step)];
};
static_assert(sizeof(CounterEntry) == ControlEntryBytes);
static_assert(MaxRanks * sizeof(CounterEntry) == ControlTableBytes);

constexpr bool isPowerOfTwo(std::size_t n) { return n && !(n & (n - 1)); }

// Also callable in a kernel for a validBytes value bounded by StepBytes.
constexpr std::size_t transferBytes(std::size_t validBytes) {
  if (!validBytes) return 0;
  std::size_t result = MinTransferBytes;
  while (result < validBytes) result *= 2;
  return result;
}

inline std::size_t nextPowerOfTwo(std::size_t n) {
  if (!n) return 0;
  std::size_t result = 1;
  while (result < n) {
    if (result > std::numeric_limits<std::size_t>::max() / 2)
      throw std::overflow_error("bulk FIFO allocation size overflow");
    result *= 2;
  }
  return result;
}

constexpr std::size_t controlOffset(unsigned channel, unsigned peer) {
  return channel * ControlTableBytes + peer * ControlEntryBytes;
}
constexpr std::size_t ringFifoOffset(unsigned channel, std::size_t fifoBytes) {
  return channel * fifoBytes;  // One incoming connection per Ring channel.
}
constexpr std::size_t stepsFor(std::size_t bytes, std::size_t stepBytes) {
  return bytes / stepBytes + (bytes % stepBytes != 0);
}

struct RingGeometry {
  std::size_t fifoBytes;
  std::size_t stepBytes;
  unsigned channels;
  std::size_t controlAllocationBytes;
  std::size_t dataAllocationBytes;

  explicit RingGeometry(std::size_t fifo = 4 * MiB, unsigned count = 1)
      : fifoBytes(fifo), stepBytes(fifo / Slots), channels(count),
        controlAllocationBytes(0), dataAllocationBytes(0) {
    if (fifo != 2 * MiB && fifo != 4 * MiB)
      throw std::invalid_argument("FIFO must be 2 MiB or 4 MiB");
    if (!count || count > std::numeric_limits<std::size_t>::max() / fifo)
      throw std::invalid_argument("invalid channel count");
    controlAllocationBytes = nextPowerOfTwo(count * ControlTableBytes);
    dataAllocationBytes = nextPowerOfTwo(count * fifo);
  }
};

// Pattern changes with byte position, source rank, and iteration.
constexpr unsigned char pattern(unsigned rank, unsigned iteration,
                                std::size_t offset) {
  std::uint64_t x = offset ^ ((std::uint64_t(rank) + 1) << 48) ^
                    ((std::uint64_t(iteration) + 1) * 0x9e3779b97f4a7c15ULL);
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return static_cast<unsigned char>(x ^ (x >> 31));
}

} // namespace bulk_fifo
