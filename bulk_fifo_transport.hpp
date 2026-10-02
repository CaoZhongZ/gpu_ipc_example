#pragma once

#include <type_traits>

#include "bulk_fifo_memory.hpp"

namespace bulk_fifo {

struct RingConnection {
  std::uint64_t* localHead;
  std::uint64_t* localTail;
  std::uint64_t* remoteHead; // Previous rank's send-control entry for us.
  std::uint64_t* remoteTail; // Next rank's receive-control entry for us.
  unsigned char* localFifo;
  unsigned char* remoteFifo;

  RingConnection forChannel(unsigned channel, std::size_t fifoBytes) const {
    const auto counterOffset =
        channel * (ControlTableBytes / sizeof(std::uint64_t));
    const auto fifoOffset = ringFifoOffset(channel, fifoBytes);
    return {localHead + counterOffset, localTail + counterOffset,
            remoteHead + counterOffset, remoteTail + counterOffset,
            localFifo + fifoOffset, remoteFifo + fifoOffset};
  }
};

enum class WaitPhase : unsigned { None, Credit, Ready, FinalCredit };

struct alignas(64) KernelStatus {
  unsigned phase = 0;
  std::uint64_t expected = 0;
  std::uint64_t observed = 0;
  std::uint64_t polls = 0;
};

inline bool waitFor(sycl::nd_item<1> item, const std::uint64_t* counter,
                    std::uint64_t target, std::uint64_t spinLimit,
                    WaitPhase phase, KernelStatus* status) {
  unsigned success = 1;
  if (item.get_local_linear_id() == 0) {
    std::uint64_t observed = 0;
    std::uint64_t polls = 0;
    do {
      observed = loadCounter(counter);
      ++polls;
    } while (observed < target && polls < spinLimit);
    status->polls += polls;
    success = observed >= target;
    if (!success) {
      status->phase = static_cast<unsigned>(phase);
      status->expected = target;
      status->observed = observed;
    }
  }
  // Every work-item takes the same return path, including those without data.
  return sycl::group_broadcast(item.get_group(), success, 0) != 0;
}

template <StoreCache Cache>
inline void sendSlice(sycl::nd_item<1> item, RingConnection connection,
                      const Pack* input, std::size_t valid,
                      std::size_t stepBytes, std::uint64_t step) {
  auto* destination = connection.remoteFifo + (step % Slots) * stepBytes;
  for (std::size_t offset = item.get_local_linear_id() * PackBytes;
       offset < valid; offset += item.get_local_range(0) * PackBytes)
    storePack<Cache>(destination + offset, input[offset / PackBytes]);
  // A leader fence alone does not cover other workers' stores.
  systemRelease();
  sycl::group_barrier(item.get_group());
  if (item.get_local_linear_id() == 0) {
    storeCounter(connection.remoteTail, step + 1);
    systemRelease();
  }
  sycl::group_barrier(item.get_group());
}

inline void receiveSlice(sycl::nd_item<1> item, RingConnection connection,
                         Pack* output, std::size_t valid,
                         std::size_t stepBytes, std::uint64_t step) {
  // Each worker establishes freshness after the leader's successful poll.
  systemAcquire();
  sycl::group_barrier(item.get_group());
  const auto* source = connection.localFifo + (step % Slots) * stepBytes;
  for (std::size_t offset = item.get_local_linear_id() * PackBytes;
       offset < valid; offset += item.get_local_range(0) * PackBytes)
    output[offset / PackBytes] = loadPack(source + offset);
  // Finish every worker's reads before granting permission to reuse the slot.
  systemRelease();
  sycl::group_barrier(item.get_group());
  if (item.get_local_linear_id() == 0) {
    storeCounter(connection.remoteHead, step + 1);
    systemRelease();
  }
  sycl::group_barrier(item.get_group());
}

template <StoreCache Cache>
struct RingCopyKernel {
  RingConnection connection;
  // User buffers are 16-byte-aligned uint4 arrays; bytes is a multiple of 16.
  const Pack* input;
  Pack* output;
  std::size_t bytes;
  std::size_t stepBytes;
  unsigned channels;
  const std::uint64_t* firstSteps;
  std::uint64_t spinLimit;
  KernelStatus* status;

  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<1> item) const {
    const auto channel = static_cast<unsigned>(item.get_group_linear_id());
    const auto partition = channelPartition(bytes, channel, channels);
    const auto channelConnection =
        connection.forChannel(channel, stepBytes * Slots);
    const auto* channelInput = input + partition.offset / PackBytes;
    auto* channelOutput = output + partition.offset / PackBytes;
    auto* channelStatus = status + channel;
    const auto firstStep = firstSteps[channel];
    const std::size_t steps = stepsFor(partition.bytes, stepBytes);
    // Send a window, then receive it. All ranks start with send credits;
    // no rank waits for its predecessor before posting its first window.
    // At a later window, credit polling can overlap a slower consumer.
    for (std::size_t base = 0; base < steps; base += Slots) {
      const std::size_t window = (steps - base < Slots) ? steps - base : Slots;
      for (std::size_t j = 0; j < window; ++j) {
        const std::uint64_t step = firstStep + base + j;
        const std::uint64_t minimumHead =
            step + 1 > Slots ? step + 1 - Slots : 0;
        if (!waitFor(item, channelConnection.localHead, minimumHead, spinLimit,
                     WaitPhase::Credit, channelStatus)) return;
        systemAcquire(); // The previous receiver has released this slot.
        const std::size_t offset = (base + j) * stepBytes;
        const std::size_t valid = partition.bytes - offset < stepBytes
                                     ? partition.bytes - offset : stepBytes;
        sendSlice<Cache>(item, channelConnection, channelInput + offset / PackBytes,
                         valid, stepBytes, step);
      }
      for (std::size_t j = 0; j < window; ++j) {
        const std::uint64_t step = firstStep + base + j;
        if (!waitFor(item, channelConnection.localTail, step + 1, spinLimit,
                     WaitPhase::Ready, channelStatus)) return;
        const std::size_t offset = (base + j) * stepBytes;
        const std::size_t valid = partition.bytes - offset < stepBytes
                                     ? partition.bytes - offset : stepBytes;
        receiveSlice(item, channelConnection, channelOutput + offset / PackBytes,
                     valid, stepBytes, step);
      }
    }
    // Make both counters' final values observable before host validation.
    if (steps)
      waitFor(item, channelConnection.localHead, firstStep + steps, spinLimit,
              WaitPhase::FinalCredit, channelStatus);
  }
};

// Dispatch on the host to distinct compiled kernels, keeping policy branches
// out of the copy loop. The same dispatch selects the kernel for limit checks.
template <typename Function>
auto withStoreCache(StoreCache cache, Function function) {
  switch (cache) {
  case StoreCache::UcUcUc:
    return function(std::integral_constant<StoreCache, StoreCache::UcUcUc>{});
  case StoreCache::WbWbUc:
    return function(std::integral_constant<StoreCache, StoreCache::WbWbUc>{});
  case StoreCache::WtWbUc:
    return function(std::integral_constant<StoreCache, StoreCache::WtWbUc>{});
  case StoreCache::StWbUc:
    return function(std::integral_constant<StoreCache, StoreCache::StWbUc>{});
  case StoreCache::UcWbUc:
    return function(std::integral_constant<StoreCache, StoreCache::UcWbUc>{});
  case StoreCache::StUcUc:
    return function(std::integral_constant<StoreCache, StoreCache::StUcUc>{});
  case StoreCache::WbUcUc:
    return function(std::integral_constant<StoreCache, StoreCache::WbUcUc>{});
  }
  throw std::invalid_argument("invalid payload-store cache policy");
}

template <StoreCache Cache>
inline sycl::event launchRingCopyWithCache(
    sycl::queue& queue, RingConnection connection, const Pack* input,
    Pack* output, std::size_t bytes, std::size_t stepBytes,
    unsigned workItems, unsigned channels, const std::uint64_t* firstSteps,
    std::uint64_t spinLimit, KernelStatus* status) {
  const RingCopyKernel<Cache> kernel{
      connection, input, output, bytes, stepBytes, channels, firstSteps,
      spinLimit, status};
  return queue.parallel_for<RingCopyKernel<Cache>>(
      sycl::nd_range<1>(std::size_t(workItems) * channels, workItems), kernel);
}

inline sycl::event launchRingCopy(sycl::queue& queue, RingConnection connection,
                                 const Pack* input, Pack* output,
                                 std::size_t bytes, std::size_t stepBytes,
                                 unsigned workItems, unsigned channels,
                                 const std::uint64_t* firstSteps,
                                 std::uint64_t spinLimit, KernelStatus* status,
                                 StoreCache cache) {
  return withStoreCache(cache, [&](auto selection) {
    return launchRingCopyWithCache<decltype(selection)::value>(
        queue, connection, input, output, bytes, stepBytes, workItems, channels,
        firstSteps, spinLimit, status);
  });
}

} // namespace bulk_fifo
