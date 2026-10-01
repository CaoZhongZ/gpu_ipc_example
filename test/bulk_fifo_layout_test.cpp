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
    for (std::size_t bytes : {std::size_t(0), std::size_t(1), std::size_t(8192),
                              stepBytes - 1, stepBytes, stepBytes + 1,
                              5 * bf::MiB + 123}) {
      for (unsigned workers : {16, 64, 192, 1024}) {
        std::size_t copied = 0;
        for (std::size_t step = 0; step < bf::stepsFor(bytes, stepBytes); ++step) {
          const auto valid = std::min(bytes - copied, stepBytes);
          const auto span = bf::transferBytes(valid);
          assert(bf::isPowerOfTwo(span));
          assert(span >= valid && span <= stepBytes && span >= 256);
          std::vector<unsigned> visits(span / bf::PackBytes, 0);
          for (unsigned worker = 0; worker < workers; ++worker)
            for (std::size_t offset = worker * bf::PackBytes; offset < span;
                 offset += workers * bf::PackBytes)
              ++visits[offset / bf::PackBytes];
          for (unsigned visitsPerPack : visits) assert(visitsPerPack == 1);
          copied += valid;
        }
        assert(copied == bytes);
      }
    }
  }
  assert(bf::transferBytes(0) == 0);
  assert(bf::transferBytes(1000) == 1024);
  assert(bf::pattern(0, 0, 0) != bf::pattern(0, 1, 0));
  std::cout << "PASS: layout, rounding overflow, transfer bounds, worker coverage\n";
}
