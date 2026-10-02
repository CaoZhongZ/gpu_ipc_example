#include "../bulk_fifo_layout.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

namespace bf = bulk_fifo;

int main() {
  assert(bf::nextPowerOfTwo(0) == 0);
  assert(bf::nextPowerOfTwo(12 * bf::MiB) == 16 * bf::MiB);
  bool overflow = false;
  try { bf::nextPowerOfTwo(std::numeric_limits<std::size_t>::max()); }
  catch (const std::overflow_error&) { overflow = true; }
  assert(overflow);
  for (unsigned channels : {1, 3, 8}) {
    for (std::size_t bytes : {std::size_t(0), std::size_t(16), std::size_t(32),
                              std::size_t(272), std::size_t(1008),
                              5 * bf::MiB + 128, 32 * bf::MiB, 128 * bf::MiB}) {
      std::size_t end = 0;
      std::size_t minimum = std::numeric_limits<std::size_t>::max(), maximum = 0;
      for (unsigned channel = 0; channel < channels; ++channel) {
        const auto part = bf::channelPartition(bytes, channel, channels);
        assert(part.offset == end && part.offset % bf::PackBytes == 0);
        assert(part.bytes % bf::PackBytes == 0 && part.bytes <= bytes - end);
        end += part.bytes;
        minimum = std::min(minimum, part.bytes);
        maximum = std::max(maximum, part.bytes);
      }
      assert(end == bytes && maximum - minimum <= bf::PackBytes);
    }
  }
  for (const auto fifo : {2 * bf::MiB, 4 * bf::MiB}) {
    for (unsigned channels : {1, 3, 8}) {
      const bf::RingGeometry geometry(fifo, channels);
      assert(bf::isPowerOfTwo(geometry.controlAllocationBytes));
      assert(bf::isPowerOfTwo(geometry.dataAllocationBytes));
      assert(geometry.controlAllocationBytes >= channels * bf::ControlTableBytes);
      assert(geometry.dataAllocationBytes >= channels * fifo);
      for (unsigned channel = 0; channel < channels; ++channel) {
        assert(bf::ringFifoOffset(channel, fifo) % bf::DataAlignment == 0);
        for (unsigned peer = 0; peer < bf::MaxRanks; ++peer)
          assert(bf::controlOffset(channel, peer) + sizeof(bf::CounterEntry) <=
                 geometry.controlAllocationBytes);
      }
    }
    const auto stepBytes = fifo / bf::Slots;
    for (std::size_t bytes : {std::size_t(0), std::size_t(16), std::size_t(8192),
                              stepBytes - bf::PackBytes, stepBytes,
                              stepBytes + bf::PackBytes, 5 * bf::MiB + 128}) {
      for (unsigned workers : {16, 64, 192, 1024}) {
        std::size_t copied = 0;
        for (std::size_t step = 0; step < bf::stepsFor(bytes, stepBytes); ++step) {
          const auto valid = std::min(bytes - copied, stepBytes);
          assert(valid > 0 && valid <= stepBytes && valid % bf::PackBytes == 0);
          std::vector<unsigned> visits(valid / bf::PackBytes, 0);
          for (unsigned worker = 0; worker < workers; ++worker)
            for (std::size_t offset = worker * bf::PackBytes; offset < valid;
                 offset += workers * bf::PackBytes)
              ++visits[offset / bf::PackBytes];
          for (unsigned visitsPerPack : visits) assert(visitsPerPack == 1);
          copied += valid;
        }
        assert(copied == bytes);
      }
    }
  }
  assert(bf::pattern(0, 0, 0) != bf::pattern(0, 1, 0));
  std::cout << "PASS: layout, rounding overflow, transfer bounds, worker coverage\n";
}
