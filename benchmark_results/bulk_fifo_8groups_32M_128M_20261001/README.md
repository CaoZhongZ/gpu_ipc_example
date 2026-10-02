# CRI bulk FIFO: eight groups of 1024 work-items

Attempted on 2026-10-01 (America/Los_Angeles; 2026-10-02 UTC) on
`10.99.62.220`, using two CRI root GPUs, oneAPI 2026.1, Intel MPI, and the
`spir64_gen -device cri-a0` target.

Both requested sizes failed at the first Ring kernel submission with
`UR_RESULT_ERROR_OUT_OF_RESOURCES` (40), before any event timing or payload
validation result. No latency or bandwidth measurement is available.

| User input per GPU | Input per group | Steps per group per iteration | Result |
|---|---|---|---|
| 32 MiB | 4 MiB | 8 | Submission failed; exit 1 |
| 128 MiB | 16 MiB | 32 | Submission failed; exit 1 |

Each GPU launches eight workgroups of 1024 work-items (global range 8192,
SIMD16). Total input per GPU is split into eight contiguous uint4 ranges.
Each group has independent head/tail counters and a 4 MiB FIFO with eight
512 KiB slots. Per GPU, the three separate protocol ranges are 32 KiB send
control, 32 KiB receive control, and 32 MiB data aligned to 2 MiB.

Both attempts requested ten measured iterations, with no separate warmup.
FIFO payload stores, FIFO loads, and counters used `uc.uc.uc`.
The kernel sends input to the next GPU's FIFO and copies the predecessor's
data from its local FIFO into output. Profiling covers the complete kernel,
including both copies, polling, fences, and barriers.

```bash
make ARCH=cri bulk_fifo_ring_test
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu I_MPI_FABRICS=shm
timeout --kill-after=5s 60s mpirun -n 2 ./bulk_fifo_ring_test \
  --channels 8 --work-items 1024 --bytes 32M --iterations 10
timeout --kill-after=5s 60s mpirun -n 2 ./bulk_fifo_ring_test \
  --channels 8 --work-items 1024 --bytes 128M --iterations 10
```

The final build includes independent descriptors for IPC imports. Each
import receives a duplicated FD whose lifetime belongs to the imported
allocation, separate from the broker/client descriptor. The available Linux
NEO source retains and closes that FD with the allocation. This change did
not resolve the submission failure.

Final attempt logs: [32 MiB](benchmark_32M_fd_fix.log) and
[128 MiB](benchmark_128M_fd_fix.log).
Earlier attempts are in [32 MiB](benchmark_32M.log) and
[128 MiB](benchmark_128M.log).

Diagnostics:

- Host channel partition/layout/worker coverage checks and compilation of
  all seven CRI store-policy kernels pass.
- Local device copies and group synchronization pass at eight groups of
  1024 work-items: [copy](resource_probe.log) and
  [synchronization](resource_probe_sync.log).
- The same Ring kernel functor completes using local buffers on each GPU:
  [log](resource_ring_local.log). This small diagnostic checks only its first
  output pack and does not validate the peer transport.
- A 272-byte peer smoke run fails after the descriptor change:
  [log](fd_fix_smoke.log).
- The older Level Zero launch entry point also fails:
  [runtime trace](fallback_trace.log). The underlying call reports
  `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`; this alone does not establish
  physical memory exhaustion.
- Disabling the USM pool does not resolve the failure:
  [log](no_pool.log).
- Selecting the v1 Level Zero adapter with
  `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` also fails:
  [log](adapter_v1_aligned_smoke.log). This scratch diagnostic used an
  oversized allocation and an aligned subrange because that adapter rejects
  allocation alignment above 64 KiB. Its logical data range remains
  2 MiB-aligned, and its three protocol ranges remain separate.

The peer submission failure's cause remains unresolved. Multi-channel
payload correctness, FIFO wraps, final counters, and performance remain
unvalidated on the two-GPU path.
