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
};

enum class WaitPhase : unsigned { None, Credit, Ready, FinalCredit };

struct KernelStatus {
  unsigned phase = 0;
  unsigned paddingError = 0;
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
                      const unsigned char* input, std::size_t valid,
                      std::size_t stepBytes, std::uint64_t step) {
  const std::size_t span = transferBytes(valid);
  auto* destination = connection.remoteFifo + (step % Slots) * stepBytes;
  for (std::size_t offset = item.get_local_linear_id() * PackBytes;
       offset < span; offset += item.get_local_range(0) * PackBytes) {
    Pack pack{0, 0, 0, 0};
    for (unsigned word = 0; word < 4; ++word) {
      std::uint32_t value = 0;
      for (unsigned byte = 0; byte < 4; ++byte) {
        const std::size_t i = offset + word * 4 + byte;
        if (i < valid) value |= std::uint32_t(input[i]) << (byte * 8);
      }
      pack[word] = value;
    }
    storePack<Cache>(destination + offset, pack);
  }
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
                         unsigned char* output, std::size_t valid,
                         std::size_t stepBytes, std::uint64_t step,
                         KernelStatus* status) {
  // Each worker establishes freshness after the leader's successful poll.
  systemAcquire();
  sycl::group_barrier(item.get_group());
  const std::size_t span = transferBytes(valid);
  const auto* source = connection.localFifo + (step % Slots) * stepBytes;
  unsigned badPadding = 0;
  for (std::size_t offset = item.get_local_linear_id() * PackBytes;
       offset < span; offset += item.get_local_range(0) * PackBytes) {
    const Pack pack = loadPack(source + offset);
    for (unsigned word = 0; word < 4; ++word) {
      const std::uint32_t value = pack[word];
      for (unsigned byte = 0; byte < 4; ++byte) {
        const std::size_t i = offset + word * 4 + byte;
        const auto data = static_cast<unsigned char>(value >> (byte * 8));
        if (i < valid) output[i] = data;
        else badPadding |= data != 0;
      }
    }
  }
  const unsigned anyPadding = sycl::any_of_group(item.get_group(),
                                               badPadding != 0);
  // Finish every worker's reads before granting permission to reuse the slot.
  systemRelease();
  sycl::group_barrier(item.get_group());
  if (item.get_local_linear_id() == 0) {
    status->paddingError |= anyPadding;
    storeCounter(connection.remoteHead, step + 1);
    systemRelease();
  }
  sycl::group_barrier(item.get_group());
}

template <StoreCache Cache> class RingCopyKernel;

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
inline sycl::event launchRingCopyWithCache(sycl::queue& queue, RingConnection connection,
                                 const unsigned char* input,
                                 unsigned char* output, std::size_t bytes,
                                 std::size_t stepBytes, unsigned workItems,
                                 std::uint64_t firstStep,
                                 std::uint64_t spinLimit,
                                 KernelStatus* status) {
  return queue.submit([=](sycl::handler& handler) {
    handler.parallel_for<RingCopyKernel<Cache>>(
        sycl::nd_range<1>(workItems, workItems),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const std::size_t steps = stepsFor(bytes, stepBytes);
          // Send a window, then receive it. All ranks start with send credits;
          // no rank waits for its predecessor before posting its first window.
          // At a later window, credit polling can overlap a slower consumer.
          for (std::size_t base = 0; base < steps; base += Slots) {
            const std::size_t window =
                (steps - base < Slots) ? steps - base : Slots;
            for (std::size_t j = 0; j < window; ++j) {
              const std::uint64_t step = firstStep + base + j;
              const std::uint64_t minimumHead =
                  step + 1 > Slots ? step + 1 - Slots : 0;
              if (!waitFor(item, connection.localHead, minimumHead, spinLimit,
                           WaitPhase::Credit, status)) return;
              systemAcquire(); // The previous receiver has released this slot.
              const std::size_t offset = (base + j) * stepBytes;
              const std::size_t valid =
                  bytes - offset < stepBytes ? bytes - offset : stepBytes;
              sendSlice<Cache>(item, connection, input + offset, valid,
                               stepBytes, step);
            }
            for (std::size_t j = 0; j < window; ++j) {
              const std::uint64_t step = firstStep + base + j;
              if (!waitFor(item, connection.localTail, step + 1, spinLimit,
                           WaitPhase::Ready, status)) return;
              const std::size_t offset = (base + j) * stepBytes;
              const std::size_t valid =
                  bytes - offset < stepBytes ? bytes - offset : stepBytes;
              receiveSlice(item, connection, output + offset, valid,
                           stepBytes, step, status);
            }
          }
          // Make both counters' final values observable before host validation.
          if (steps)
            waitFor(item, connection.localHead, firstStep + steps, spinLimit,
                    WaitPhase::FinalCredit, status);
        });
  });
}

inline sycl::event launchRingCopy(sycl::queue& queue, RingConnection connection,
                                 const unsigned char* input,
                                 unsigned char* output, std::size_t bytes,
                                 std::size_t stepBytes, unsigned workItems,
                                 std::uint64_t firstStep,
                                 std::uint64_t spinLimit, KernelStatus* status,
                                 StoreCache cache) {
  return withStoreCache(cache, [&](auto selection) {
    return launchRingCopyWithCache<decltype(selection)::value>(
        queue, connection, input, output, bytes, stepBytes, workItems,
        firstStep, spinLimit, status);
  });
}

} // namespace bulk_fifo
