#include "bulk_fifo_ipc.hpp"
#include "bulk_fifo_transport.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if !defined(CRI)
#error "Build this standalone test with make ARCH=cri bulk_fifo_ring_test"
#endif

namespace bf = bulk_fifo;

namespace {

struct Options {
  std::size_t bytes = 5 * bf::MiB + 128;
  std::size_t fifoBytes = 4 * bf::MiB;
  unsigned workItems = 1024;
  unsigned channels = 1;
  unsigned iterations = 3;
  std::uint64_t spinLimit = 100000000;
  bf::StoreCache storeCache = bf::StoreCache::UcUcUc;
  std::vector<int> devices;
  bool help = false;
  bool list = false;
};

std::size_t parseSize(std::string text) {
  std::size_t multiplier = 1;
  if (text.empty() || text[0] == '-')
    throw std::invalid_argument("expected a nonnegative size");
  if (text.back() == 'K' || text.back() == 'k') {
    multiplier = bf::KiB;
    text.pop_back();
  } else if (text.back() == 'M' || text.back() == 'm') {
    multiplier = bf::MiB;
    text.pop_back();
  } else if (text.back() == 'G' || text.back() == 'g') {
    multiplier = 1024 * bf::MiB;
    text.pop_back();
  }
  std::size_t end = 0;
  const auto n = std::stoull(text, &end);
  if (end != text.size() ||
      n > std::numeric_limits<std::size_t>::max() / multiplier)
    throw std::invalid_argument("invalid or overflowing size");
  return n * multiplier;
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string name = argv[i];
    if (name == "--help") { options.help = true; continue; }
    if (name == "--list-devices") { options.list = true; continue; }
    const auto equal = name.find('=');
    std::string value;
    if (equal != std::string::npos) {
      value = name.substr(equal + 1);
      name.resize(equal);
    } else {
      if (++i == argc) throw std::invalid_argument("missing option value");
      value = argv[i];
    }
    if (name == "--store-cache") {
      const auto match = std::find_if(
          bf::StoreCacheNames.begin(), bf::StoreCacheNames.end(),
          [&](const char* policy) { return value == policy; });
      if (match == bf::StoreCacheNames.end())
        throw std::invalid_argument("unsupported --store-cache policy: " + value);
      options.storeCache = static_cast<bf::StoreCache>(
          match - bf::StoreCacheNames.begin());
    } else if (name == "--devices") {
      std::istringstream stream(value);
      std::string index;
      while (std::getline(stream, index, ',')) {
        const auto n = parseSize(index);
        if (n > std::numeric_limits<int>::max())
          throw std::invalid_argument("device index exceeds int range");
        options.devices.push_back(static_cast<int>(n));
      }
      if (options.devices.empty()) throw std::invalid_argument("empty device list");
    } else {
      const auto n = parseSize(value);
      if (name == "--bytes") options.bytes = n;
      else if (name == "--fifo-bytes") options.fifoBytes = n;
      else if (name == "--spin-limit") options.spinLimit = n;
      else if (name == "--work-items" || name == "--channels" || name == "--iterations") {
        if (n > std::numeric_limits<unsigned>::max())
          throw std::invalid_argument("option exceeds unsigned range");
        if (name == "--work-items") options.workItems = static_cast<unsigned>(n);
        else if (name == "--channels") options.channels = static_cast<unsigned>(n);
        else options.iterations = static_cast<unsigned>(n);
      } else throw std::invalid_argument("unknown option: " + name);
    }
  }
  if (!options.help && !options.list && options.bytes % bf::PackBytes)
    throw std::invalid_argument("--bytes must be a multiple of 16 (uint4)");
  return options;
}

std::vector<sycl::device> enumerateDevices() {
  std::vector<sycl::device> devices;
  for (const auto& platform : sycl::platform::get_platforms())
    if (platform.get_backend() == sycl::backend::ext_oneapi_level_zero)
      for (const auto& device : platform.get_devices(sycl::info::device_type::gpu))
        devices.push_back(device);
  return devices;
}

class Allocations {
public:
  explicit Allocations(sycl::queue& queue) : queue_(queue) {}
  ~Allocations() {
    for (auto* pointer : allocations_) sycl::free(pointer, queue_);
  }
  unsigned char* allocate(std::size_t alignment, std::size_t bytes) {
    auto* pointer = static_cast<unsigned char*>(
        sycl::aligned_alloc_device(alignment, bytes, queue_));
    if (!pointer) throw std::runtime_error("device USM allocation failed");
    allocations_.push_back(pointer);
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignment)
      throw std::runtime_error("device USM allocation is misaligned");
    return pointer;
  }
private:
  sycl::queue& queue_;
  std::vector<void*> allocations_;
};

void checkSeparateRanges(const std::array<void*, 3>& ranges,
                         const std::array<std::size_t, 3>& sizes) {
  for (unsigned i = 0; i < 3; ++i)
    for (unsigned j = i + 1; j < 3; ++j) {
      const auto a = reinterpret_cast<std::uintptr_t>(ranges[i]);
      const auto b = reinterpret_cast<std::uintptr_t>(ranges[j]);
      if (a < b + sizes[j] && b < a + sizes[i])
        throw std::runtime_error("control/data logical allocations overlap");
    }
}

void compareBytes(const std::vector<unsigned char>& actual,
                  const std::vector<unsigned char>& expected,
                  const char* range) {
  const auto mismatch = std::mismatch(actual.begin(), actual.end(),
                                      expected.begin());
  if (mismatch.first != actual.end()) {
    const auto offset = mismatch.first - actual.begin();
    std::ostringstream message;
    message << range << " mismatch at byte " << offset << ": expected "
            << unsigned(*mismatch.second) << ", received "
            << unsigned(*mismatch.first);
    throw std::runtime_error(message.str());
  }
}

// Useful outgoing bytes only. The kernel also consumes incoming bytes.
// Decimal GB/s = bytes / (milliseconds * 10^6).
double sendGoodput(std::size_t bytes, double milliseconds) {
  return milliseconds > 0 ? double(bytes) / (milliseconds * 1e6) : 0;
}

void run(const Options& options, int rank, int world) {
  const auto devices = enumerateDevices();
  if (options.list) {
    if (rank == 0)
      for (std::size_t i = 0; i < devices.size(); ++i)
        std::cout << i << ": " << devices[i].get_info<sycl::info::device::name>()
                  << '\n';
    return;
  }
  if (world < 2 || world > static_cast<int>(bf::MaxRanks))
    throw std::invalid_argument("use 2-16 MPI ranks, one rank per GPU");
  if (!options.iterations || !options.spinLimit)
    throw std::invalid_argument("iterations and spin-limit must be positive");
  if (!options.workItems || options.workItems % bf::SubgroupSize)
    throw std::invalid_argument("work-items must be a positive multiple of 16");
  bf::RingGeometry geometry(options.fifoBytes, options.channels);
  std::vector<std::size_t> channelSteps(options.channels);
  for (unsigned channel = 0; channel < options.channels; ++channel) {
    const auto partition = bf::channelPartition(options.bytes, channel, options.channels);
    channelSteps[channel] = bf::stepsFor(partition.bytes, geometry.stepBytes);
    if (channelSteps[channel] >
        (std::numeric_limits<std::uint64_t>::max() - bf::Slots) / options.iterations)
      throw std::invalid_argument("step counters would overflow");
  }
  if (options.channels > std::numeric_limits<std::size_t>::max() / options.workItems)
    throw std::invalid_argument("global work-item count would overflow");
  if (options.bytes > std::numeric_limits<std::size_t>::max() - 512)
    throw std::invalid_argument("input allocation size would overflow");

  MPI_Comm shared;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &shared);
  int localWorld;
  MPI_Comm_size(shared, &localWorld);
  MPI_Comm_free(&shared);
  if (localWorld != world) throw std::invalid_argument("all ranks must be on one host");
  if (!options.devices.empty() && options.devices.size() != unsigned(world))
    throw std::invalid_argument("--devices must contain one index per rank");
  const int index = options.devices.empty() ? rank : options.devices[rank];
  if (index < 0 || std::size_t(index) >= devices.size())
    throw std::invalid_argument("selected Level Zero GPU index does not exist");
  std::vector<int> mapping(world);
  MPI_Allgather(&index, 1, MPI_INT, mapping.data(), 1, MPI_INT, MPI_COMM_WORLD);
  auto sorted = mapping;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    throw std::invalid_argument("each rank must select a distinct GPU");
  const unsigned cacheCode = static_cast<unsigned>(options.storeCache);
  std::vector<unsigned> cacheCodes(world);
  MPI_Allgather(&cacheCode, 1, MPI_UNSIGNED, cacheCodes.data(), 1, MPI_UNSIGNED,
                 MPI_COMM_WORLD);
  // Worker counts may differ; geometry and the transfer schedule must agree.
  const std::array<std::uint64_t, 4> schedule{
      options.bytes, options.fifoBytes, options.iterations, options.channels};
  std::vector<std::uint64_t> schedules(world * schedule.size());
  MPI_Allgather(schedule.data(), schedule.size(), MPI_UINT64_T, schedules.data(),
                schedule.size(), MPI_UINT64_T, MPI_COMM_WORLD);
  for (int r = 0; r < world; ++r)
    if (!std::equal(schedule.begin(), schedule.end(),
                    schedules.begin() + r * schedule.size()))
      throw std::invalid_argument("ranks disagree on bytes, FIFO, iterations, or channels");

  sycl::queue queue(devices[index],
                    sycl::property_list{sycl::property::queue::in_order{},
                                        sycl::property::queue::enable_profiling{}});
  if (!devices[index].has(sycl::aspect::usm_device_allocations))
    throw std::runtime_error("selected GPU lacks device USM support");
  const auto maximum =
      devices[index].get_info<sycl::info::device::max_work_group_size>();
  const auto oneDim =
      devices[index].get_info<sycl::info::device::max_work_item_sizes<1>>()[0];
  if (options.workItems > maximum || options.workItems > oneDim)
    throw std::invalid_argument("work-items exceed the selected GPU's limit");
  const auto subgroupSizes =
      devices[index].get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(subgroupSizes.begin(), subgroupSizes.end(), bf::SubgroupSize) ==
      subgroupSizes.end())
    throw std::runtime_error("selected GPU does not support SIMD16 subgroups");
  const auto kernelId = bf::withStoreCache(options.storeCache, [](auto selection) {
    return sycl::get_kernel_id<bf::RingCopyKernel<decltype(selection)::value>>();
  });
  const auto kernel = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      queue.get_context(), {devices[index]}, {kernelId}).get_kernel(kernelId);
  const auto compiledMaximum =
      kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(
          devices[index]);
  if (options.workItems > compiledMaximum)
    throw std::invalid_argument("work-items exceed the compiled kernel's limit");

  const int next = (rank + 1) % world;
  const int previous = (rank + world - 1) % world;
  const auto native =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[index]);
  for (const int peer : {next, previous}) {
    const auto remote =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[mapping[peer]]);
    ze_bool_t accessible = false;
    bf::zeCheck(zeDeviceCanAccessPeer(native, remote, &accessible),
                "zeDeviceCanAccessPeer");
    if (!accessible) throw std::runtime_error("required Ring P2P link unavailable");
  }

  Allocations memory(queue);
  auto* sendControl = memory.allocate(bf::ControlAlignment,
                                      geometry.controlAllocationBytes);
  auto* receiveControl = memory.allocate(bf::ControlAlignment,
                                         geometry.controlAllocationBytes);
  auto* data = memory.allocate(bf::DataAlignment, geometry.dataAllocationBytes);
  const std::array<void*, 3> ranges{sendControl, receiveControl, data};
  const std::array<std::size_t, 3> sizes{geometry.controlAllocationBytes,
                                        geometry.controlAllocationBytes,
                                        geometry.dataAllocationBytes};
  checkSeparateRanges(ranges, sizes);
  queue.memset(sendControl, 0, sizes[0]);
  queue.memset(receiveControl, 0, sizes[1]);
  queue.memset(data, 0x5a, sizes[2]);
  queue.wait_and_throw();
  bf::RingIpc ipc(queue, MPI_COMM_WORLD, rank, world, ranges, sizes);

  auto counter = [](unsigned char* base, int peer) {
    return reinterpret_cast<std::uint64_t*>(base + bf::controlOffset(0, peer));
  };
  const bf::RingConnection connection{
      counter(sendControl, next), counter(receiveControl, previous),
      counter(ipc.previousSendControl(), rank),
      counter(ipc.nextReceiveControl(), rank), data, ipc.nextData()};

  constexpr std::size_t guard = 256;
  const auto userAllocation = bf::nextPowerOfTwo(options.bytes + 2 * guard);
  auto* input = memory.allocate(256, userAllocation);
  auto* output = memory.allocate(256, userAllocation);
  auto* status = reinterpret_cast<bf::KernelStatus*>(
      memory.allocate(64, bf::nextPowerOfTwo(options.channels * sizeof(bf::KernelStatus))));
  auto* deviceFirstSteps = reinterpret_cast<std::uint64_t*>(
      memory.allocate(64, bf::nextPowerOfTwo(options.channels * sizeof(std::uint64_t))));
  std::vector<unsigned char> expectedInput(userAllocation, 0xa5);
  std::vector<unsigned char> expectedOutput(userAllocation, 0xcc);
  std::vector<unsigned char> actual(userAllocation);
  std::vector<unsigned char> expectedFifo(geometry.dataAllocationBytes, 0x5a);
  std::vector<std::uint64_t> firstSteps(options.channels, 0);
  double totalMilliseconds = 0;
  double minimumMilliseconds = std::numeric_limits<double>::infinity();
  double maximumMilliseconds = 0;
  double totalMaxRankMilliseconds = 0;
  double minimumMaxRankMilliseconds = std::numeric_limits<double>::infinity();
  double maximumMaxRankMilliseconds = 0;

  if (rank == 0) {
    std::cout << "CRI Ring " << world << " GPUs, " << options.channels
              << " channels/groups, " << options.workItems
              << " work-items/group, " << options.bytes
              << " user bytes/rank, " << options.iterations << " iterations\n"
              << "send-control=" << sizes[0] << " receive-control=" << sizes[1]
              << " data=" << sizes[2] << " step=" << geometry.stepBytes
              << " slots=" << bf::Slots
              << "; control/FIFO-load cache=uc.uc.uc\n";
    for (unsigned channel = 0; channel < options.channels; ++channel) {
      const auto partition = bf::channelPartition(options.bytes, channel, options.channels);
      std::cout << "channel " << channel << " input_offset=" << partition.offset
                << " user_bytes=" << partition.bytes
                << " steps/iteration=" << channelSteps[channel] << '\n';
    }
    for (int r = 0; r < world; ++r)
      std::cout << "rank " << r << " GPU " << mapping[r] << " -> rank "
                << (r + 1) % world << " store_cache="
                << bf::storeCacheName(static_cast<bf::StoreCache>(cacheCodes[r]))
                << '\n';
    std::cout.flush();
  }
  for (unsigned iteration = 0; iteration < options.iterations; ++iteration) {
    for (std::size_t i = 0; i < options.bytes; ++i) {
      expectedInput[guard + i] = bf::pattern(rank, iteration, i);
      expectedOutput[guard + i] = bf::pattern(previous, iteration, i);
    }
    queue.memcpy(input, expectedInput.data(), userAllocation);
    queue.memset(output, 0xcc, userAllocation);
    queue.memset(status, 0, options.channels * sizeof(bf::KernelStatus));
    queue.memcpy(deviceFirstSteps, firstSteps.data(),
                  options.channels * sizeof(std::uint64_t));
    queue.wait_and_throw();
    MPI_Barrier(MPI_COMM_WORLD);
    auto event = bf::launchRingCopy(
        queue, connection, reinterpret_cast<const bf::Pack*>(input + guard),
        reinterpret_cast<bf::Pack*>(output + guard), options.bytes,
        geometry.stepBytes, options.workItems, options.channels, deviceFirstSteps,
        options.spinLimit, status,
        options.storeCache);
    event.wait_and_throw();
    const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    if (end < start) throw std::runtime_error("invalid event profiling timestamps");
    const double milliseconds = double(end - start) / 1e6;
    totalMilliseconds += milliseconds;
    minimumMilliseconds = std::min(minimumMilliseconds, milliseconds);
    maximumMilliseconds = std::max(maximumMilliseconds, milliseconds);
    std::vector<bf::KernelStatus> results(options.channels);
    queue.memcpy(results.data(), status, results.size() * sizeof(bf::KernelStatus))
        .wait_and_throw();
    for (unsigned channel = 0; channel < options.channels; ++channel) {
      const auto& result = results[channel];
      if (result.phase) {
        std::ostringstream message;
        message << "GPU wait timed out, channel=" << channel
                << " phase=" << result.phase
                << " expected=" << result.expected << " observed=" << result.observed;
        throw std::runtime_error(message.str());
      }
    }
    queue.memcpy(actual.data(), output, userAllocation).wait_and_throw();
    compareBytes(actual, expectedOutput, "output/guards");
    queue.memcpy(actual.data(), input, userAllocation).wait_and_throw();
    compareBytes(actual, expectedInput, "input/guards");
    for (unsigned channel = 0; channel < options.channels; ++channel)
      firstSteps[channel] += channelSteps[channel];

    // Check both complete control ranges, including inactive entries/padding.
    for (unsigned kind = 0; kind < 2; ++kind) {
      std::vector<unsigned char> expected(sizes[kind], 0), control(sizes[kind]);
      for (unsigned channel = 0; channel < options.channels; ++channel) {
        const auto offset = bf::controlOffset(channel, kind == 0 ? next : previous);
        std::memcpy(expected.data() + offset, &firstSteps[channel],
                     sizeof(std::uint64_t));
      }
      queue.memcpy(control.data(), ranges[kind], sizes[kind]).wait_and_throw();
      compareBytes(control, expected, kind == 0 ? "send-control" : "receive-control");
    }
    // Track the exact final FIFO image, including untouched bytes in each slot.
    for (unsigned channel = 0; channel < options.channels; ++channel) {
      const auto partition = bf::channelPartition(options.bytes, channel, options.channels);
      for (std::size_t step = 0; step < channelSteps[channel]; ++step) {
        const auto channelOffset = step * geometry.stepBytes;
        const auto userOffset = partition.offset + channelOffset;
        const auto valid = std::min(partition.bytes - channelOffset, geometry.stepBytes);
        const auto fifoOffset = bf::ringFifoOffset(channel, geometry.fifoBytes) +
            ((firstSteps[channel] - channelSteps[channel] + step) % bf::Slots) *
                geometry.stepBytes;
        for (std::size_t i = 0; i < valid; ++i)
          expectedFifo[fifoOffset + i] =
              bf::pattern(previous, iteration, userOffset + i);
      }
    }
    // Reduce durations, not timestamps: device clocks need not be synchronized.
    // All MPI, host staging, and validation occur outside the profiled event.
    double maxRankMilliseconds = 0;
    MPI_Reduce(&milliseconds, &maxRankMilliseconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                MPI_COMM_WORLD);
    if (rank == 0) {
      totalMaxRankMilliseconds += maxRankMilliseconds;
      minimumMaxRankMilliseconds =
          std::min(minimumMaxRankMilliseconds, maxRankMilliseconds);
      maximumMaxRankMilliseconds =
          std::max(maximumMaxRankMilliseconds, maxRankMilliseconds);
      const auto goodput = sendGoodput(options.bytes, maxRankMilliseconds);
      std::cout << std::fixed << std::setprecision(6)
                << "Event iteration " << iteration + 1
                << ": max_rank_kernel_ms=" << maxRankMilliseconds
                << " send_goodput_GBps=" << goodput
                << " ring_send_goodput_GBps=" << world * goodput << '\n';
      std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  std::vector<unsigned char> actualFifo(sizes[2]);
  queue.memcpy(actualFifo.data(), data, sizes[2]).wait_and_throw();
  compareBytes(actualFifo, expectedFifo, "FIFO/untouched storage");
  const std::array<double, 4> localTiming{
      minimumMilliseconds, totalMilliseconds / options.iterations,
      maximumMilliseconds, totalMilliseconds};
  std::vector<double> rankTimings(world * localTiming.size());
  MPI_Gather(localTiming.data(), localTiming.size(), MPI_DOUBLE, rankTimings.data(),
              localTiming.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<unsigned> workers(world);
  MPI_Gather(&options.workItems, 1, MPI_UNSIGNED, workers.data(), 1, MPI_UNSIGNED,
              0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "PASS: predecessor patterns, input/output guards, "
                 "untouched storage, and per-channel head/tail:";
    for (auto step : firstSteps) std::cout << ' ' << step;
    std::cout << '\n' << "work-items/group per rank:";
    for (auto count : workers) std::cout << ' ' << count;
    std::cout << '\n';
    for (int r = 0; r < world; ++r) {
      const auto offset = r * localTiming.size();
      std::cout << "Rank " << r << " events: min_ms=" << rankTimings[offset]
                << " avg_ms=" << rankTimings[offset + 1]
                << " max_ms=" << rankTimings[offset + 2]
                << " total_ms=" << rankTimings[offset + 3] << '\n';
    }
    const auto averageMaxRankMilliseconds =
        totalMaxRankMilliseconds / options.iterations;
    const auto goodput = sendGoodput(options.bytes, averageMaxRankMilliseconds);
    std::cout << "Event summary: max_rank_min_ms=" << minimumMaxRankMilliseconds
              << " max_rank_avg_ms=" << averageMaxRankMilliseconds
              << " max_rank_max_ms=" << maximumMaxRankMilliseconds
              << " sum_max_rank_ms=" << totalMaxRankMilliseconds
              << " send_goodput_GBps=" << goodput
              << " ring_send_goodput_GBps=" << world * goodput << '\n';
  }
  // Stop all kernels and finish validation before closing imported allocations.
  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace

int main(int argc, char** argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank, world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  try {
    const auto options = parseOptions(argc, argv);
    if (options.help) {
      if (rank == 0)
        std::cout << "Standalone CRI bulk FIFO Ring test (2-16 GPUs)\n"
                     "  --bytes N[K|M|G]       multiple of 16 (default 5243008)\n"
                     "  --fifo-bytes 2M|4M     eight-slot FIFO (default 4M)\n"
                     "  --work-items W        multiple of 16 (default 1024)\n"
                     "  --channels C          workgroups/FIFOs per rank (default 1)\n"
                     "  --iterations I        preserve counters (default 3)\n"
                     "  --devices 0,1,...      numeric Level Zero GPU indices\n"
                     "  --spin-limit N        maximum polls per wait (default 100000000)\n"
                     "  --store-cache POLICY  payload stores (default uc.uc.uc)\n"
                     "    wb.wb.uc | wt.wb.uc | st.wb.uc | uc.wb.uc | st.uc.uc | wb.uc.uc\n"
                     "  --list-devices        print the device indices\n";
    } else run(options, rank, world);
    MPI_Finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
