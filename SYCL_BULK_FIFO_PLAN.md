# Separate SYCL bulk FIFO prototype

## Scope

Build a new bulk FIFO transport for **2–16 GPUs on one host**, with separate readiness and credit counters, using the buffer organization studied in `NCCL_SIMPLE_PROTOCOL_STUDY.md`. Use one MPI rank per selected GPU and a runtime rank count. Two GPUs are the first validation case, not an architectural limit.

The prototype will have its own send-control, receive-control, and data allocations, kernels, benchmark, and protocol state. These occupy separate allocation ranges with separate bases and IPC handle metadata. It will not place counters or payload inside the existing `ipcbuf0`/`ipcbuf1` scatter/gather buffers, change their offsets, or reuse their embedded flags. Existing `simple`, `simple_pcie`, and other transport paths remain available for comparison.

The standalone CRI Ring copy test is implemented in `bulk_fifo_ring_test.cpp`, with its own layout, memory-ordering, transport, and IPC files. It supports multiple channels; collective operations remain later work. Hardware validation results are recorded below.

### First CRI Ring test

Build and run independently of the original benchmark:

```bash
make ARCH=cri bulk_fifo_ring_test
mpirun -n 2 ./bulk_fifo_ring_test
# Explicit numeric device mapping, uint4 byte count, and smaller FIFO:
mpirun -n 2 ./bulk_fifo_ring_test --devices 0,1 --bytes 3145856 --fifo-bytes 2M
# Boundary-size and wrap checks:
bash test/test_bulk_fifo.sh
```

Use 2–16 ranks on one host, with a distinct Level Zero root GPU per rank. The default mapping is GPU index equal to MPI rank; `--devices` accepts comma-separated numeric indices, including 10–15. `--list-devices` prints the available indices. Increase `mpirun -n` and supply the corresponding device list when additional GPUs are available.

The executable runs **one workgroup per channel per rank**, with **one channel, 1024 work-items per group, and SIMD16** by default. `--channels C` launches `C` groups in a single event: global range `C * W`, local range `W`. `--work-items W` accepts other supported positive multiples of 16, including 192; it checks both device and compiled-kernel limits. Geometry, channel count, and the byte-count/iteration schedule must agree across ranks, while workgroup sizes may differ.

Split the total per-rank user input into contiguous channel ranges in units of complete uint4 packs. Ranges differ by at most one pack; channels with zero packs schedule no transfers. Each channel has its own incoming FIFO, send/receive control tables, progress, and 64-byte-aligned status record. The kernel derives the channel from its workgroup ID and uses that channel's input/output range and connection offsets.

For **eight groups of 1024 work-items**, each GPU launches 8192 work-items. With the default FIFO size, each GPU allocates **32 MiB data + 32 KiB send control + 32 KiB receive control**, keeping all three ranges separate. A **32 MiB** user input gives each channel **4 MiB / eight steps per iteration**; **128 MiB** gives each channel **16 MiB / 32 steps per iteration**. Ten iterations end at head/tail 80 or 320 per channel, respectively:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu I_MPI_FABRICS=shm \
  mpirun -n 2 ./bulk_fifo_ring_test --channels 8 --work-items 1024 \
  --bytes 32M --iterations 10
# Repeat with --bytes 128M.
```

User input and output are **16-byte-aligned `uint4` arrays**, with byte counts divisible by 16. The transport takes `Pack*` pointers (`Pack = sycl::vec<std::uint32_t, 4>`) and copies complete vectors. The test rejects a `--bytes` value that is not a multiple of 16. Each slice copies exactly its valid bytes and leaves the remaining FIFO slot bytes untouched.

`RingCopyKernel<StoreCache>` in `bulk_fifo_transport.hpp` is the kernel functor. It holds the connection bases, input/output pointers, transfer geometry, channel count, per-channel progress pointer, poll limit, and status pointer; its `operator()` contains the GPU implementation and the SIMD16 attribute. The launcher constructs this functor and passes it directly to `queue.parallel_for`, which returns the profiled event for all channels.

Each rank sends its own input to `(rank + 1) % world` and verifies input from `(rank + world - 1) % world`. A window sends up to eight one-step transfers, then receives the same number. Each send checks credit, copies valid bytes with 16-byte stores, joins all workers after system-release ordering, and publishes the receiver's tail. Each receive checks readiness, establishes freshness for every worker, copies valid bytes to user output, joins after completing consumption, and returns head credit. Bounded GPU polls and the test script's process timeout catch failed progress.

The backend uses CRI **uncached counter accesses and FIFO loads at L1, L2, and L3**, plus per-worker `lsc_fence.ugm.evict.sysrel` release and `lsc_fence.ugm.evict.sysacq` acquire ordering. The vISA scope spelling is `sysrel` (the compiler rejects `system`).

Select FIFO payload stores with `--store-cache wb.wb.uc|wt.wb.uc|st.wb.uc|uc.wb.uc|st.uc.uc|wb.uc.uc`. The original fully uncached `uc.uc.uc` option remains the default. The three qualifiers name L1/L2/L3 policies, respectively. Selection dispatches on the host to separate AOT kernel specializations, so the payload loop contains the selected store instruction with no runtime policy branch. Control stores and all FIFO loads remain `uc.uc.uc`; fences and counter semantics are the same for every payload-store setting. Different ranks may select different policies, and the startup report prints each rank's setting. Use `STORE_CACHE=policy bash test/test_bulk_fifo.sh` for boundary/wrap validation of a specific policy.

The default input is **5 MiB + 128 bytes**, repeated three times with changing rank/iteration/offset patterns. With one channel, a 4 MiB FIFO has eight 512 KiB slots, so each iteration consumes 11 steps; counters finish at 33 rather than resetting. With multiple channels, each advances by its own partition's step count. Validation checks all user input/output and surrounding guard storage, every channel's complete final FIFO image including untouched bytes, the complete control ranges including inactive peer entries and allocation padding, and per-channel head/tail values. Set `CHANNELS=8 bash test/test_bulk_fifo.sh` to run the boundary cases across eight channels.

### Event timing

The queue enables profiling. For each Ring kernel event covering all channels, use `command_end - command_start` on that GPU to obtain execution time; compare durations across ranks rather than absolute timestamps from different GPU clocks.

Each iteration prints its maximum-rank kernel duration. At the end, each rank reports minimum, average, maximum, and total duration, followed by minimum/average/maximum of the iteration maxima and their sum. `sum_max_rank_ms` is `sum(iterations: max(ranks: kernel_ms))`, which can differ from taking the maximum of per-rank totals.

`send_goodput_GBps = userBytes / (max_rank_kernel_ms * 10^6)` counts useful outgoing bytes once per rank, in decimal GB/s. `ring_send_goodput_GBps = rankCount * send_goodput_GBps` counts useful sends across all ranks. The summary divides total useful bytes by the sum of iteration maxima, equivalently using their average for a fixed input size. Zero bytes or a zero-duration event report zero goodput.

The profiled event includes GPU input-to-peer-FIFO copies, local FIFO-to-output copies, counter polling, fences, and barriers. Host input staging, initialization, verification, and MPI operations are outside the event. Control traffic is excluded from the goodput numerator; this is useful-data throughput, not PCIe wire bandwidth. All requested iterations are included, with no separate warmup.

Measured on **2026-10-01**, before the uint4 copy change, on the two-CRI-GPU host below: **8 MiB per rank**, one channel, **1024 work-items**, **4 MiB FIFO**, ten iterations per policy including the first. The time columns summarize each iteration's maximum-rank duration; goodput uses their average.

| FIFO store policy | Min ms | Average ms | Max ms | Send GB/s per rank |
|---|---:|---:|---:|---:|
| `wb.wb.uc` | 5.103 | 5.292 | 5.430 | 1.585 |
| `wt.wb.uc` | 5.349 | 5.458 | 5.810 | 1.537 |
| `st.wb.uc` | 5.236 | 5.364 | 5.516 | 1.564 |
| `uc.wb.uc` | 5.402 | 5.536 | 5.986 | 1.515 |
| `st.uc.uc` | 5.277 | 5.380 | 5.648 | 1.559 |
| `wb.uc.uc` | 5.203 | 5.348 | 5.498 | 1.568 |
| `uc.uc.uc` (default) | 5.368 | 5.494 | 5.671 | 1.527 |

All seven runs passed correctness checks and ended at head/tail **160**. With `uc.wb.uc` and a **2 MiB FIFO**, the same 8 MiB input takes 32 steps per iteration instead of 16: average time was **8.389 ms**, send goodput **1.000 GB/s per rank**, and counters ended at **320**. Aggregate Ring send goodput is twice the per-rank figure in these two-GPU runs.

Run a measurement of the current implementation with:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:gpu I_MPI_FABRICS=shm \
  mpirun -n 2 ./bulk_fifo_ring_test --bytes 8M --iterations 10 \
  --fifo-bytes 4M --store-cache wb.wb.uc
```

### Validation

Historical validation before the uint4 contract, on **2026-10-01**, using two CRI root GPUs on `10.99.62.220`, oneAPI 2026.1, Intel MPI, and the `spir64_gen -device cri-a0` AOT target:

| Check | Result |
|---|---|
| Host layout, rounding overflow, transfer bounds, and pack coverage for 16/64/192/1024 workers | Pass |
| CRI AOT compilation of counter/pack accesses and system fences | Pass |
| All 17 script cases at 1024 work-items, including zero/partial bytes, slot boundaries, and wraps with both FIFO capacities | Pass |
| 5 MiB + 123 bytes, three iterations, at 16 and at 192 work-items | Pass; head/tail end at 33 |
| 3 MiB + 123 bytes, 2 MiB FIFO, 192 work-items on rank 0 and 1024 on rank 1 | Pass; head/tail end at 39 |
| 5 MiB + 123 bytes, 20 iterations, 1024 work-items | Pass; head/tail end at 220 |
| All seven store-policy AOT kernels; 8 MiB, ten iterations per policy | Pass; head/tail end at 160 |
| All six requested store policies with a partial final transfer: 5 MiB + 123 bytes, three iterations | Pass; head/tail end at 33 |
| Mixed `wb.wb.uc`/`uc.wb.uc` policies with 192/1024 workers, 2 MiB FIFO, 3 MiB + 123 bytes | Pass; head/tail end at 39 |
| Event statistics, throughput formulas, counter totals, and zero-byte rates across all 17 timing-check runs | Pass |
| Functor launch: 8 KiB with all seven store policies; 5 MiB + 123 bytes, two iterations, at 1024 and 192 work-items | Pass; wrap runs end at head/tail 22 |

Initial **uint4 validation**, before removing the padding loops, passed the host layout checks and CRI AOT compilation of all seven store-policy kernels; the CLI also rejected `--bytes 17` before GPU setup. Two-GPU execution was blocked at kernel submission by `UR_RESULT_ERROR_OUT_OF_RESOURCES` at 16, 192, and 1024 work-items, including a zero-byte run. A runtime trace reported `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` from `zeCommandListAppendLaunchKernelWithArguments`, followed by device loss during cleanup. A freshly rebuilt baseline from commit `21758d0` also failed with the resource error at 8 KiB.

After removing padding, the updated host layout/coverage checks pass, and CRI AOT compilation succeeds for all seven cache policies. The two-GPU smoke attempt at 5 MiB + 128 bytes, three iterations, and 1024 work-items still fails with `UR_RESULT_ERROR_OUT_OF_RESOURCES` before producing an event result. The updated GPU boundary/wrap cases remain unvalidated.

The **eight-channel benchmark attempts** at **32 MiB and 128 MiB per rank**, 1024 work-items per group, and ten requested iterations also fail at the first peer Ring kernel submission with `UR_RESULT_ERROR_OUT_OF_RESOURCES`; neither produces an event measurement. Host checks cover balanced channel partitioning, including empty channels and both requested sizes, and all seven CRI kernels compile. Local diagnostic kernels, including the Ring functor with local buffers, complete at eight groups of 1024 items. Changing the Level Zero launch entry point, disabling the USM pool, and selecting the v1 adapter with `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` do not resolve peer submission.

IPC imports now receive independent duplicated descriptors: Linux NEO retains an import descriptor with its allocation, while broker/client descriptor objects own separate copies. This lifetime correction does not resolve the submission error. The current multi-channel peer data path, wraps, and counters remain unvalidated. See the [benchmark report and raw logs](benchmark_results/bulk_fifo_8groups_32M_128M_20261001/README.md).

Historical GPU validation covers **two GPUs**. Execution on 3–16 GPUs and deliberate consumer-delay/failure-injection tests remain future checks.

## 1. Define an independent memory layout

Use **three independent device USM allocations per rank**:

- A **send-control allocation**, with **4 KiB per channel**, containing outgoing credit counters.
- A **receive-control allocation**, with **4 KiB per channel**, containing incoming readiness counters.
- A **data allocation**, aligned to **2 MiB**, containing only the incoming payload FIFOs for those channels.

These are distinct ranges, not offsets within one combined allocation. Keep separate `sendCtrlBase`, `recvCtrlBase`, and `dataBase` pointers, sizes, IPC handle metadata, imported bases, and cleanup state. Do not derive one base from another or assume that the ranges are adjacent. Use 4 KiB alignment for control allocations; only data allocation/FIFO bases require 2 MiB alignment.

A directed connection is identified by `(channel, source rank, destination rank)` and has its own counters, FIFO, and progress state. For each channel, reserve **4 KiB in each control allocation** and an eight-step FIFO per active incoming connection in the data allocation. The FIFO capacity is configurable: **4 MiB by default**, with **2 MiB** as the first smaller configuration. Both sizes preserve 2 MiB-aligned FIFO bases.

Intermediate allocation sizes, FIFO capacities, and slot capacities are powers of two. The actual copied payload length may be any multiple of 16, and workgroup sizes need not be powers of two. If active channels/connections require a non-power-of-two aggregate allocation size, round the allocation up to the next power of two and leave the additional space unused; do not add protocol slots or credits because allocation padding is available.

Each 4 KiB control table has **16 rank-indexed entries with a 256-byte stride**. Each entry contains an aligned 64-bit counter followed by reserved bytes/padding. Thus `16 * 256 = 4096` bytes supports up to 15 peers plus an unused self entry. Send controls hold `head` counters; receive controls hold `tail` counters. This is 4 KiB per table per channel, not 4 KiB per peer.

Multiple senders must not write into one shared incoming FIFO or advance one shared `tail`. Each directed connection has one producer and one consumer. If concurrently operating phases require separate connections, include a phase identifier in the connection key.

For a Ring with `N` ranks, rank `r` sends to `next = (r + 1) % N` and receives from `prev = (r + N - 1) % N`. One Ring channel requests **4 MiB + 8 KiB total per rank across three separate ranges**:

```text
Rank r — one Ring channel; 2 <= N <= 16
Control bases aligned to 0x1000 (4 KiB)
Data base aligned to 0x200000 (2 MiB)

SEND CONTROL RANGE: sendCtrlBase, length 4 KiB
Offsets below are relative to sendCtrlBase

0x000000  ┌──────────────────────────────────────────┐
          │ Send control table — 4 KiB              │
          │ 16 entries, 256 bytes per entry         │
          │ Active: head[r → next]                  │
0x001000  └──────────────────────────────────────────┘

RECEIVE CONTROL RANGE: recvCtrlBase, length 4 KiB
Offsets below are relative to recvCtrlBase

0x000000  ┌──────────────────────────────────────────┐
          │ Receive control table — 4 KiB           │
          │ 16 entries, 256 bytes per entry         │
          │ Active: tail[prev → r]                  │
0x001000  └──────────────────────────────────────────┘

DATA RANGE: dataBase, length 4 MiB
Offsets below are relative to dataBase
Separate allocation; no assumed address relationship to either control range

0x000000  ┌──────────────────────────────────────────┐
          │ Incoming FIFO[prev → r] — 4 MiB         │
          │ slot 0: 512 KiB                         │
          │ slot 1: 512 KiB                         │
          │ slot 2: 512 KiB                         │
          │ slot 3: 512 KiB                         │
          │ slot 4: 512 KiB                         │
          │ slot 5: 512 KiB                         │
          │ slot 6: 512 KiB                         │
          │ slot 7: 512 KiB                         │
0x400000  └──────────────────────────────────────────┘

Total: 4 KiB send control + 4 KiB receive control + 4 MiB data
Separate input/output allocations belong to the new benchmark.
```

For multiple active peers and channels, expand the three ranges independently:

```text
Rank r's SEND CONTROL allocation
  Channel 0 SendControl[peer rank]         16 × 256 bytes = 4 KiB
  Channel 1 SendControl[peer rank]         16 × 256 bytes = 4 KiB
  ...

Rank r's RECEIVE CONTROL allocation
  Channel 0 RecvControl[peer rank]         16 × 256 bytes = 4 KiB
  Channel 1 RecvControl[peer rank]         16 × 256 bytes = 4 KiB
  ...

Rank r's DATA allocation
  Channel 0
    FIFO[incoming peer 0 → r]              4 MiB
    FIFO[incoming peer 1 → r]              4 MiB
    ...
  Channel 1
    Its own incoming FIFOs                4 MiB each
    ...
```

Outgoing and incoming peer lists may differ. Allocate payload FIFOs only for active incoming connections; the fixed control tables support all 16 ranks. Each channel owns independent regions within each allocation. A control-channel stride is 4 KiB and a FIFO stride is 4 MiB, preserving the respective control and data alignment when concatenating regions.

For a one-way test rank with no incoming connections, omit its empty data allocation and data handle. Its send/receive control allocations still exist. Connection descriptors and IPC setup must support this case.

For the default FIFO size, the requested per-rank allocation sizes, including counter padding but excluding input/output, are:

```text
sendControlUsedBytes       = nChannels * 4 KiB
recvControlUsedBytes       = nChannels * 4 KiB
dataUsedBytes              = sum over channels: incomingConnections * FifoBytes

sendControlAllocationBytes = nextPowerOfTwo(sendControlUsedBytes)
recvControlAllocationBytes = nextPowerOfTwo(recvControlUsedBytes)
dataAllocationBytes        = dataUsedBytes == 0 ? 0 : nextPowerOfTwo(dataUsedBytes)
totalBytes                 = sendControlAllocationBytes
                           + recvControlAllocationBytes
                           + dataAllocationBytes
```

For a single channel with the default 4 MiB FIFO:

| Topology at 16 GPUs | Outgoing / incoming connections per GPU | Control ranges per GPU | Data allocation per GPU | Total |
|---|---:|---:|---:|---:|
| Unidirectional Ring | 1 / 1 | 4 KiB send + 4 KiB receive | 4 MiB | 4 MiB + 8 KiB |
| Two neighbors with bidirectional connections | 2 / 2 | 4 KiB send + 4 KiB receive | 8 MiB | 8 MiB + 8 KiB |
| Full mesh | 15 / 15 | 4 KiB send + 4 KiB receive | 64 MiB: 60 MiB FIFOs + 4 MiB unused | 64 MiB + 8 KiB |

For example, three Ring channels use 12 KiB of each control range and 12 MiB of data. Allocate 16 KiB for each control range and 16 MiB for data, retaining the original 4 KiB control-table and 4 MiB FIFO offsets.

Only the bulk protocol is allocated here; no LL or LL128 buffers are needed. The 256-byte counter-entry stride and 4 KiB control tables are layout choices, not atomic transfer units. For a given directed connection, its `head` lives in the sender's send-control allocation and its `tail` lives in the receiver's receive-control allocation.

The 2 MiB alignment applies to data addresses. Requesting it through SYCL does not itself establish the driver's physical page size or huge-page backing. Likewise, requesting 4 KiB of control storage does not determine the driver's backing allocation size. Validate the logical ranges and their non-overlap, record each Level Zero allocation base/offset for IPC, check control bases against 4 KiB alignment, and check data/FIFO bases against 2 MiB alignment. The payload loop starts each SIMD16 group of FIFO packs on a 256-byte boundary. User buffers and individual packs require 16-byte alignment.

Use separate control/data offsets and pointer construction:

```text
ControlAlignment  = 4 KiB = 0x1000
DataAlignment     = 2 MiB = 0x200000
ControlTableBytes = 4 KiB
MaxRanks          = 16
ControlEntryBytes = 256
FifoBytes         = configurable: 4 MiB default, 2 MiB smaller configuration
StepBytes         = FifoBytes / 8; 512 KiB or 256 KiB respectively

controlChannelOffset(c) = c * ControlTableBytes
dataChannelOffset(c)    = sum over channels i < c: incomingConnections[i] * FifoBytes

headOffset(peer)         = peer * ControlEntryBytes
tailOffset(peer)         = peer * ControlEntryBytes
fifoOffset(incomingSlot) = incomingSlot * FifoBytes

headPtr(c, peer) = sendCtrlBase + controlChannelOffset(c) + headOffset(peer)
tailPtr(c, peer) = recvCtrlBase + controlChannelOffset(c) + tailOffset(peer)
fifoPtr(c, slot) = dataBase    + dataChannelOffset(c)    + fifoOffset(slot)
```

The pointer expressions above use byte-pointer arithmetic. `incomingSlot` is the compact index of an active incoming peer, not its rank number; share that mapping with peers. Round future configurable FIFO sizes to retain 2 MiB-aligned FIFO bases; keep control tables at their separate 4 KiB alignment.

Define layout constants, padded control types, connection keys, and byte-offset helpers in a new `bulk_fifo_layout.hpp`. Include a checked `nextPowerOfTwo()` helper compatible with the repository's C++17 build. Use compile-time checks for control sizes and runtime checks for computed offsets, allocation sizes, rounding overflow, and local/imported pointer alignment. Avoid applying the existing payload-element pointer arithmetic to control regions.

Build host-side offset tables keyed by channel and peer rank, with an explicit send-control/receive-control/data range identifier. Share the offset metadata so each rank can find the appropriate regions within the correct imported allocation. Use a checked maximum of 16 ranks for descriptor capacity, with runtime active counts; do not add a dispatch restricted to two ranks or only powers of two.

For rank `r` and peer `p`, the connection pointers are:

| Operation on rank `r` | Pointer used | Physical owner |
|---|---|---|
| Wait for outgoing credit to `p` | Local `sendControl[channel][p].head` | Rank `r` |
| Write outgoing payload to `p` | Peer `recvFIFO[channel][r]` | Rank `p` |
| Publish outgoing readiness to `p` | Peer `recvControl[channel][r].tail` | Rank `p` |
| Wait for incoming readiness from `p` | Local `recvControl[channel][p].tail` | Rank `r` |
| Read incoming payload from `p` | Local `recvFIFO[channel][p]` | Rank `r` |
| Return incoming credit to `p` | Peer `sendControl[channel][r].head` | Rank `p` |

These are logical table lookups: `head` pointers use the owning rank's send-control base, `tail` pointers use its receive-control base, and FIFO pointers use its data base. Physical offsets come from the respective allocation metadata. In a Ring with more than two ranks, the outgoing peer and incoming peer are different.

### 1.1 Channel ownership and memory budget

A channel is the persistent communication state: its peer connections, FIFOs, and counters. A workgroup is the team of work-items executing that state. In this design, `item.get_group(0)` selects channel `c`, with one workgroup per active channel on each GPU. Workgroup `c` on the sender uses the corresponding channel `c` connection on the receiver.

Two workgroups therefore use two independent sets of connection counters and FIFO regions. Both sets reside in their GPU's allocations, indexed by channel and peer. A barrier joins only the work-items in its workgroup; one group's posting work-item cannot use that barrier to publish another group's payload.

For a Ring, there is one incoming FIFO per GPU per channel:

| FIFO capacity per channel | FIFO slots | Bytes per step | Data storage for 8 channels per GPU | Send + receive control storage |
|---|---:|---:|---:|---:|
| 4 MiB | 8 | 512 KiB | 32 MiB | 32 KiB + 32 KiB |
| 2 MiB | 8 | 256 KiB | 16 MiB | 32 KiB + 32 KiB |

This is device global-memory storage, reused across operations. With one step reserved per transfer, a 2 MiB channel payload needs four maximum-sized transfers with the 4 MiB FIFO, or eight with the 2 MiB FIFO. Each transfer adds another payload completion barrier, publication ordering, and head/tail update. With two steps reserved per slice, the same comparison is two versus four slices. A small payload that already fits one transfer need not gain additional synchronization when the FIFO shrinks.

## 2. Allocate and exchange the separate control/data ranges

Start with the standalone `bulk_fifo_ring_test.cpp`, with runtime support for 2–16 ranks and its own device selection and IPC setup. A later benchmark can extend this entry point.

Expose `--work-items W` with default `1024`. Validate the requested size against the selected device and compiled kernel limits, including the required SIMD16 subgroup specialization. Workgroup size is a launch parameter, independent of the control/data allocation layout.

`--channels C` selects the active workgroup/channel count; `--fifo-bytes B` defaults to 4 MiB per channel. Validate the 2 MiB and 4 MiB FIFO configurations. FIFO size, channel count, and workgroup size are independent settings; both peers must agree on the connection's FIFO geometry and transfer schedule.

1. Validate the runtime rank count and an explicit numeric device list, including device indices 10–15. Enumerate the intended root GPUs or tiles consistently and validate distinct rank/device assignments; avoid the original benchmark's single-character device parsing.
2. Construct the active topology and check P2P access for the links it requires.
3. Compute the three power-of-two allocation sizes, keeping used sizes separate from allocated sizes. Allocate send and receive control tables separately with `sycl::aligned_alloc_device(4096, respectiveControlAllocationBytes, queue)`. Allocate nonempty data storage with `sycl::aligned_alloc_device(2 * 1024 * 1024, dataAllocationBytes, queue)`. Check allocation success, the respective alignment, and range separation.
4. Initialize active `head` counters in send-control storage, `tail` counters in receive-control storage, and debug payload patterns in data storage; wait for initialization to finish.
5. Export send-control, receive-control, and data IPC handle metadata. Include a range-kind tag, the underlying allocation offset, and the requested size for each range; apply the correct returned peer allocation offset before adding connection offsets. Omit the data handle when that rank has no data allocation.
6. For an outgoing connection, import the peer's receive-control and data ranges; for an incoming connection, import its send-control range. Maintain distinct `peerSendCtrlBases[]`, `peerRecvCtrlBases[]`, and `peerDataBases[]` tables and construct explicit send/receive connection descriptors. Check 4 KiB control alignment and 2 MiB data/FIFO alignment after applying the respective exchanged allocation offsets.
7. Synchronize participating MPI ranks before launching communicating kernels.
8. Wait for all users of all three ranges to finish before closing their imported handles and freeing the send-control, receive-control, and data allocations independently.

Use the existing Level Zero IPC and Unix-domain file-descriptor exchange approach as a reference. The current `open_all_ipc_mems()` imports every rank's allocation; a separate `bulk_fifo_ipc.hpp`/`bulk_fifo_ipc.cpp` helper should import only required peer ranges so a Ring does not require full-mesh mapping/access. Give the new helper its own socket namespace and unambiguous rank/instance/range-kind identifiers.

Require device P2P access for active connections in the first version and report the selected devices, topology, and allocation type. Sixteen local GPUs do not imply every pair supports direct access. Choose a viable Ring where possible and report unsupported required links explicitly. Test any host-memory variant separately; the current main benchmark can fall back to host allocations, which would change the meaning of the experiment.

Add a separate `bulk_fifo_bench` Makefile target using the existing compiler and architecture flags. The original `main` target and its buffer setup stay intact.

## 3. Establish counter and payload visibility

This is the first correctness milestone. The memory layout itself is straightforward; publication across separate devices is the main technical dependency.

Payload publication orders writes in the data allocation before a `tail` update in the peer's receive-control allocation. Credit return orders completed reads from the data allocation before a `head` update in the peer's send-control allocation. The backend must establish these contracts across the separate ranges.

Create `bulk_fifo_memory.hpp` with a small interface for:

- Loading progress counters with the required visibility.
- Publishing progress counters with the required ordering.
- Publishing payload writes from all workers.
- Making newly published payload visible to all consuming workers.
- Completing consumption before returning credit.

Check device capabilities for 64-bit counter accesses and supported atomic scopes/orders. A possible SYCL implementation uses system-scope `atomic_ref<uint64_t>` with acquire/release operations, but the scope spelling alone does not establish support for Level Zero imported device allocations across two devices.

The existing `tvisa/include/gateway.hpp` exposes system release/acquire LSC fence instructions that can be evaluated for the CRI backend. Verify the generated instructions and their documented guarantees before selecting a backend. Keep any new backend wrappers outside the existing protocols.

Two contracts need explicit validation:

```text
Every worker's payload writes become visible
    BEFORE the receiver can observe the new tail.

Every consumer worker finishes reading the payload
    BEFORE the sender can observe the returned head.
```

Do not assume a local group barrier or a fence executed only by the posting thread covers all workers' remote accesses. A conservative backend may require each worker to fence its writes before the group joins and the leader publishes progress. Likewise, acquiring readiness in the leader must be paired with a visibility scheme that works for every worker's payload reads.

The test defaults to fully uncached CRI accesses through separate control and payload wrappers and offers the six payload-store policy variants listed above. The original repository's `uc` policy uses write-back L2 stores and cached L2 loads; selecting `uc.wb.uc` here changes payload stores while keeping FIFO loads uncached. Disabling L1 caching alone does not prove the contracts above. Address compiler ordering as well as hardware ordering when using inline assembly.

Start with a two-GPU publication test: every worker writes a distinct sequence-dependent region, the sender publishes a counter, and the receiver checks every region. Repeat with slot reuse and delayed credit return, then run the same test for the active connection types in larger topologies. Passing these tests supports a backend choice; it does not replace establishing the platform's ordering guarantees.

## 4. Implement one-way bulk copy

Create `bulk_fifo_transport.hpp` around explicit connection descriptors, not a hardcoded `1 - rank` peer. Begin validation with a two-GPU Ring, then increase the runtime rank count when hardware is available. Use a **configurable workgroup size `W`, defaulting to 1024**, with a required subgroup size `S = 16` in the initial backend. Use one workgroup per participating GPU per channel.

The per-GPU launch has local range `W` and global range `W * activeChannels`, with `W / S` subgroups per workgroup. At the default, this is 1024 work-items and 64 SIMD16 subgroups; one channel on 16 GPUs launches 16,384 work-items. Accept positive multiples of `S` within the device and compiled kernel limits, including supported sizes that are not powers of two. Report an unsupported requested configuration instead of silently changing it.

All `W` work-items participate in payload work. Work-item 0 handles polling and posting at defined points; all work-items execute the same group barriers, including work-items with no payload elements in a short transfer. Derive worker counts and copy strides from `nd_item.get_local_range(0)` instead of hardcoding 1024 or 64 subgroups. The initial transport must not require a fixed workgroup-size kernel attribute.

For example, treating each logical element as one uint4, the distribution is:

```text
workerCount = item.get_local_range(0)
workerId    = item.get_local_id(0)

for element = workerId; element < validElements; element += workerCount:
    load the user element
    write that element into the peer FIFO

all work-items join the required barrier
```

The receiver reads valid uint4 packs from the FIFO and writes/reduces them into user output. User buffers and slice offsets are 16-byte aligned; every valid span is a whole number of packs. FIFO capacity, slot offsets, slice step reservation, counter increments, and connection identity do not depend on `W`. Producer and consumer workgroups may use different supported sizes if they agree on the valid byte count and protocol schedule.

Use this explicit distribution for complete 16-byte vector copies:

```text
W         = item.get_local_range(0)
localId   = item.get_local_id(0)
PackBytes = 16

for offset = localId * PackBytes;
    offset < validBytes;
    offset += W * PackBytes:
  load input[offset / PackBytes] as one uint4
  store the 16-byte pack at fifoSliceBase + offset

all work-items complete the required publication ordering and group joins
the posting work-item publishes readiness for the completed slice
```

At `W=1024`, work-item 0 visits byte offsets `0, 16384, 32768, ...`; work-item 1 visits `16, 16400, 32784, ...`. Work-item 1023 starts at byte offset 16368. Adjacent work-items handle adjacent packs, so each SIMD16 subgroup covers 256 contiguous bytes and a full workgroup round covers 16 KiB.

| Transfer span | Work-items with payload at W=1024 | Copy rounds | Bytes per active work-item |
|---|---:|---:|---:|
| 8 KiB | 512: 32 SIMD16 subgroups | One partial workgroup round | 16 |
| 16 KiB | 1024 | 1 | 16 |
| 512 KiB | 1024 | 32 | 512 |
| 1 MiB | 1024 | 64 | 1024 |

The 1 MiB row describes a single slice only when two steps are reserved with the default FIFO. The initial one-step configuration sends a 1 MiB user span as two 512 KiB transfers.

There is no group barrier or progress publication after each 16-byte pack or copy-loop round. All work-items join at slice completion, including those with no payload. The backend establishes visibility for every worker's stores before the posting work-item publishes `tail`. Receive workers copy complete valid packs to output and complete consumption before returning credit.

A dedicated synchronization subgroup and split barriers can be added after the basic protocol works. If that variant reserves one subgroup, compute its payload worker count from `W - S` and validate that enough workers remain; derive role indices and barrier participation from the selected configuration.

Keep eight FIFO steps. Start the copy protocol with one step per slice, as in Simple point-to-point operations; later support two steps per slice for the nominal Ring organization.

### 4.1 Uint4 transfers and FIFO step reservations

Track two distinct quantities:

- `validBytes`: bytes belonging to user input/output.
- `reservedSteps`: FIFO credit consumed by that transfer, independent of valid byte count or workgroup size.

For the initial SIMD16 backend, use 16-byte vector packs per work-item, giving a 256-byte access span for a fully active subgroup. User sizes may be non-power-of-two multiples of 16. Both peers know the valid byte count, so each copies exactly that count:

```text
Q                = 8
StepBytes        = FifoBytes / Q; 512 KiB with the default FIFO
reservedSteps    = 1 initially; 2 in the later Ring configuration
TransferCapacity = reservedSteps * StepBytes

For remainingBytes > 0:
  validBytes    = min(remainingBytes, TransferCapacity)

  assert validBytes <= TransferCapacity
  assert validBytes % 16 == 0
  copy validBytes from the user input as complete uint4 vectors
  publish readiness after all valid packs are stored
  read validBytes from the FIFO and write/reduce them into user output

  userOffset    += validBytes
  remainingBytes -= validBytes
  connectionStep += reservedSteps
```

The remaining bytes in each reserved slot retain their previous contents. Future collective partitions must preserve complete 16-byte packs and operate only on valid elements.

Each transfer occupies a fresh reserved FIFO position even when `validBytes` is smaller than its reserved capacity. For example, a 1008-byte transfer still consumes one 512 KiB FIFO step in the initial configuration. Its untransferred remainder is not part of the payload. This keeps slot addressing and credit independent of the user size while copying only the requested data.

Both peers derive valid sizes from the agreed operation schedule initially, so a new payload header is unnecessary. If a later mode allows the producer to choose sizes independently, put the required length metadata in the receive-control entry and order it before readiness publication.

For a zero-byte standalone copy, both peers agree to schedule no transfers and leave counters unchanged. An explicitly scheduled empty collective slice still executes its agreed barriers and step publication/credit return, even though it transfers no data. Do not skip a scheduled slice on only one peer.

For a slice reserving `k` steps, starting at step `s`:

```text
Q = 8
slot = s % Q
payload address = fifoBase + slot * stepBytes

Sender:
  leader waits until head + Q >= s + k
  group joins
  workers collectively copy validBytes into the peer FIFO
  all workers complete the required publication ordering
  group joins
  leader publishes peer tail = s + k
  advance sender step to s + k

Receiver:
  leader waits until tail >= s + k
  establish payload visibility for the consuming workers
  group joins
  workers copy/reduce validBytes from the local FIFO into user output
  all workers complete consumption and required credit ordering
  group joins
  leader publishes peer head = s + k
  advance receiver step to s + k
```

This is logical pseudocode; the backend from step 3 determines the actual fence, counter, and barrier sequence.

Maintain `s` independently for each directed connection and channel. One peer's progress must not grant credit or publish readiness for another peer.

The counter publication granularity is one completed group-wide slice, regardless of the number of worker stores. In the initial `k=1` configuration, a completed 8 KiB transfer advances the sender's published `tail` by one step; after consuming it, the receiver advances the published `head` by one step. Neither counter counts bytes, packs, work-items, or loop rounds. `head` is the credit-return counter; a third credit counter is unnecessary.

For simultaneous transfers in both directions, maintain two directed connections. GPU 0 publishes GPU 1's receive `tail` for `0→1`, and GPU 1 returns credit through GPU 0's send `head`. The `1→0` direction uses its own tail/head pair and progress. Each channel has another independent pair of directed connections.

Use strided vector copy across workers for exactly `validBytes`. Payload contains no embedded flags. FIFO allocations and slot capacities remain powers of two; actual copied lengths may be any multiple of 16.

Ensure a reserved slice does not straddle the FIFO end. The initial `k=1` schedule and aligned `k=2` schedule divide the eight-step ring naturally. Keep producer/consumer steps consistent for partial and scheduled empty slices.

### 4.2 Scenarios to review

The following standalone-copy examples use one connection, `reservedSteps=1`, a 512 KiB maximum transfer, and 16-byte packs:

| User input | Intermediate transfers | FIFO steps consumed |
|---|---|---:|
| 0 bytes | None | 0 |
| 16 bytes | One 16-byte transfer | 1 |
| 272 bytes (17 uint4s) | One 272-byte transfer | 1 |
| 1008 bytes (63 uint4s) | One 1008-byte transfer | 1 |
| 512 KiB | One 512 KiB transfer | 1 |
| 700 KiB | One 512 KiB transfer + one 188 KiB transfer | 2 |
| 5 MiB + 128 bytes | Ten 512 KiB transfers + one 128-byte transfer | 11 |

For the last example, transfers 1–8 use slots 0–7. Transfer 9 reuses slot 0 only after `head >= 1`; transfer 10 reuses slot 1 after `head >= 2`; transfer 11 reuses slot 2 after `head >= 3`. Readiness ends at step 11. Once all data has been consumed, credit also reaches step 11. The next operation starts at step 11, slot 3, without resetting progress.

With 1024 work-items, a full 512 KiB transfer distributes an average of 512 bytes per work-item. With 256 work-items, it distributes an average of 2 KiB. A short 1024-byte transfer has 64 vector packs of 16 bytes, so only a subset of the default workgroup carries payload, while every work-item still joins the barriers.

For a 16-GPU Ring, each rank still has one incoming 4 MiB FIFO and two independent 4 KiB control ranges per channel. For a 16-GPU full mesh with one channel, each rank has 15 incoming FIFOs using 60 MiB inside a 64 MiB data allocation. The final 4 MiB is unused allocation padding; each connection retains exactly eight slots.

For a collective input of 1000 uint4 packs per rank on a 16-GPU, single-channel Ring, split by pack count: eight chunks have 63 packs and eight have 62. Their transfers copy exactly 1008 or 992 bytes, respectively. Reduce-scatter and all-gather each take 15 hops. Maintain per-connection progress across both phases.

## 5. Validate reuse and expand to 16 GPUs

Add `test/test_bulk_fifo.sh` for the new executable, with bounded process timeouts.

Cover:

- Zero bytes, one vector, slot boundaries, and short final slices containing complete vectors.
- Power-of-two allocations/slot capacities with valid byte counts divisible by 16; check output bounds, untouched slot bytes, and unused aggregate allocation space.
- More transfers than FIFO capacity, including many repeated wraps.
- Sequence-dependent payloads that expose stale reads and overwritten slots.
- Delayed consumers, a full FIFO, and confirmation that the producer waits for returned credit.
- Consecutive operations with progress preserved instead of resetting counters.
- Multiworker publication and simultaneous traffic in both directions.
- Independent peer/channel counters and offsets, including checks for writes into another connection's storage.
- Separate send-control/receive-control/data range metadata, correct IPC handle selection, and checks that each counter/payload access stays within its designated range.
- Rank counts 2, 3, 4, 8, and 16 when the corresponding hardware is available.
- Device selection with multi-digit indices and asymmetric incoming/outgoing peer lists.
- Ring traffic at increasing rank counts, followed by full-mesh tests only where the required P2P links exist.
- Supported workgroup sizes such as 16, 64, 128, 256, 512, and 1024, plus a non-power-of-two size such as 192; repeat short-slice and wrap tests at each tested size.
- Different supported producer/consumer workgroup sizes, confirming that progress and buffer ownership are independent of worker count.
- The 16-byte pack distribution at short, full, and multi-round transfer sizes; ensure idle work-items still join and counter publication happens once per slice.
- Independent progress with two and eight channels, and 2 MiB versus 4 MiB FIFO capacities.

For bidirectional exchange, use each connection's outgoing `head` and incoming `tail`/FIFO independently. Begin with an agreed send/receive schedule that lets both peers make progress. Extend to a runtime Ring, initially with one workgroup per rank managing that channel's send and receive operations in an agreed order. Full-mesh validation can use pairwise rounds to keep the number of simultaneously active connections bounded.

Do not launch one indefinitely polling workgroup per peer/channel without establishing that its producers and consumers can be scheduled. No workgroup should depend on an unscheduled workgroup on the same GPU to release a polled condition. Larger rank counts and more channels must preserve a progress-safe schedule.

Once correct, enable two steps per slice. With the default 4 MiB FIFO, this gives a nominal 1 MiB slice and two such slices per nominal 2 MiB chunk. With a 2 MiB FIFO, the corresponding amounts are 512 KiB per slice and 1 MiB per chunk. These amounts are shared across the workgroup.

Report validation results by tested GPU count; a passing two-GPU test is not evidence that the 16-GPU schedule has been validated.

## 6. Add collective operations and tune

Implement a separate runtime-`N` Ring reduce-scatter/all-gather path for 2–16 ranks using the new FIFO. Each phase has `N - 1` hops. Handle uneven input partitions and arbitrary supported rank counts. Preserve per-connection progress across phases and ensure storage is released before reuse. If phases need separate FIFOs, allocate them in the new transport's data allocation and put their counters in the respective send/receive control allocations.

The existing `AllReduce` template expects message and embedded-flag operations, so the first bulk implementation should have its own kernel entry point. Add a distinct selector only after the standalone transport is validated and the caller supports its separate control/data allocations.

Then measure and tune:

1. Configurable workgroup size, register usage, occupancy, and copy unrolling, with 1024 work-items as the default.
2. Slice sizes and one versus two reserved steps.
3. Multiple channels, each with independent counters, FIFO storage, and progress.
4. A dedicated posting subgroup and overlap, where device forward progress supports it.
5. Store/cache policy variants and direct user-buffer paths.
6. Scaling from 2 to 16 GPUs and connection memory requirements for each topology.

Compare against the existing protocols using the same useful bytes per rank and maximum-rank kernel time. Report device P2P and host-memory results separately.

## Proposed file boundaries

| File | Responsibility |
|---|---|
| `bulk_fifo_layout.hpp` | Separate ranges, power-of-two allocation/FIFO sizing, per-connection offsets for up to 16 ranks |
| `bulk_fifo_ipc.hpp`, `bulk_fifo_ipc.cpp` | Three range kinds for handle exchange, tagged metadata, required peer-range imports, independent cleanup |
| `bulk_fifo_memory.hpp` | Counter access and CRI publication/consumption ordering |
| `bulk_fifo_transport.hpp` | Bulk slice copy, waits, publication, and credit return |
| `bulk_fifo_ring_test.cpp` | First CRI Ring test: numeric device selection, separate allocations, runtime topology, launches, verification, timing |
| `test/test_bulk_fifo.sh` | Correctness and timeout checks from 2 to 16 GPUs, subject to available hardware |
| `test/bulk_fifo_layout_test.cpp` | Host checks for layout, rounding overflow, and worker coverage |
| `Makefile` | Add only a separate benchmark target initially |

Use `ipc_exchange.cpp` and `sycl_misc.*` as references for IPC and device setup. Reuse utility functions only where their behavior fits the selected active-peer topology and numeric device mapping. The existing `rt_bulk.hpp` is an empty scaffold; the proposed new files keep this experiment separate from the original transport interfaces.

The Ring test combines separate send-control, receive-control, and data allocations and IPC setup with repeated FIFO wraps across configurable channels. Larger topology validation, deliberate consumer-delay tests, and runtime-`N` collective integration follow.

For review, begin with `bulk_fifo_layout.hpp`, then `bulk_fifo_memory.hpp`, `bulk_fifo_transport.hpp`, the IPC files, and `bulk_fifo_ring_test.cpp`. Check `channelPartition()` for input ranges, `RingConnection::forChannel()` for independent FIFO/control offsets, and the per-channel verification in the executable.
