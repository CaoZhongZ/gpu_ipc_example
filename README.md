# Intel GPU IPC communication collectives

This repository contains single-kernel SYCL all-reduce and all-gather
prototypes for Intel GPUs. It supports PVC, Xe2/BMG, DG2, and CRI builds.

## Requirements

1. An Intel SYCL compiler that supports the selected AOT target
2. A compatible Level Zero driver
3. MPI

## Build the Benchmark

```sh
source /opt/intel/oneapi/setvars.sh
git submodule update --init
make main
```

When Intel MPI is available, the Makefile selects `mpiicpx` so compilation and
launch use the same MPI ABI. An explicit `CXX=...` still overrides this choice.

Build for CRI:

```sh
make ARCH=cri CRI_STORE_L1_CACHE=uc main
```

The default `CRI_TARGET=legacy` uses `spir64_gen -device cri-a0`, which is
supported by the oneAPI 2026.1 compiler. Newer compilers can use
`CRI_TARGET=dedicated` for `-fsycl-targets=intel_gpu_cri`.

CRI builds enable `XE_PLUS` and `ATOB_SUPPORT`, including native split barriers.
Automatic all-reduce and all-gather launch sizing uses `maxSS=32`, and parallel,
sequential, and ring transmitters use `maxLaunch=64*32` (2048 subgroups).
The benchmark's explicit `-g` and `-w` settings still determine its launch size.

Communication loads always use an uncached L1 policy so polling observes peer
writes. `CRI_STORE_L1_CACHE` controls only the communication-store L1 policy:

- `uc` (default): uncached L1 store
- `wt`: write-through L1 store
- `st`: streaming L1 store

All three retain the same write-back L2 and uncached L3 store policy. L1
write-back is intentionally unsupported. Use `make -B` or clean between policy
changes.

## Run

```sh
mpirun -np 8 ./main -n \<number of elements in half\> [-w sub-group] [-g group] [-a small | simple | bisect]
```

CRI cards expose root devices without PVC-style subdevices. CRI builds compile
only the two-card, SIMD16 topology and route the normal `small` and `simple`
names to `Rt64_PCIE` and `Rt64_128_PCIE`, respectively. Thus they use the
Xe2/BMG protocols rather than the PVC protocols. Each protocol stores 16 bytes
per lane, so SIMD16 produces one 256-byte message. Compile-time assertions and
runtime checks enforce that transaction size, and all local and imported IPC
buffer bases are allocated and checked at 256-byte alignment.

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  I_MPI_FABRICS=shm mpirun -np 2 \
  ./main -l 01 -a small -s 16 -n 8M -v
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  I_MPI_FABRICS=shm mpirun -np 2 \
  ./main -l 01 -a simple -s 16 -n 8M -v
```

The explicit `small_pcie_p` and `simple_pcie_p` names remain aliases for these
same parallel Xe2/BMG protocol paths.

### Separate CRI bulk FIFO Ring test

`bulk_fifo_ring_test` uses its own FIFO and separate send/receive control
allocations. It runs one channel with 1024 work-items by default, sends each
rank's input to the next rank (last to first), and verifies its predecessor's
input across repeated FIFO wraps. It accepts 2–16 ranks on one host, with one
distinct Level Zero root GPU per rank.

```sh
make ARCH=cri bulk_fifo_ring_test
ONEAPI_DEVICE_SELECTOR=level_zero:gpu I_MPI_FABRICS=shm \
  mpirun -n 2 ./bulk_fifo_ring_test --devices 0,1
bash test/test_bulk_fifo.sh
```

Use `--bytes`, `--iterations`, `--work-items`, and `--fifo-bytes 2M|4M` to
adjust the test; `--list-devices` prints numeric GPU indices.
`--store-cache` selects the FIFO payload-store policy:
`wb.wb.uc`, `wt.wb.uc`, `st.wb.uc`, `uc.wb.uc`, `st.uc.uc`, or `wb.uc.uc`.
The default remains `uc.uc.uc`. Controls and FIFO loads use `uc.uc.uc`;
each store policy has its own AOT kernel. For example:

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu I_MPI_FABRICS=shm \
  mpirun -n 2 ./bulk_fifo_ring_test --bytes 8M --iterations 10 \
  --store-cache uc.wb.uc
# Run boundary/wrap checks for a selected policy:
STORE_CACHE=uc.wb.uc bash test/test_bulk_fifo.sh
```

SYCL event start/end timestamps report each iteration's maximum-rank kernel
time and each rank's minimum, average, maximum, and total kernel time. Send
goodput is useful outgoing bytes per rank divided by the maximum-rank duration,
in decimal GB/s; Ring send goodput multiplies this by the rank count. The
summary uses the sum of each iteration's maximum duration. These times include
GPU send/receive copies, polling, and synchronization; host staging, validation,
and MPI are excluded. All iterations are measured, including the first.

See [SYCL_BULK_FIFO_PLAN.md](SYCL_BULK_FIFO_PLAN.md) for the layout, ordering,
verification, and validation status.

### Performance report

Without `-v`, each rank reports its kernel execution time and goodput in
decimal GB/s (10^9 bytes/s). For both all-reduce and all-gather, goodput is the
per-rank input payload, `nelems * sizeof(element)`, divided by that rank's
kernel time. It excludes protocol flags and padding, is not summed across
ranks, and is not PCIe wire bandwidth. A zero-duration event reports zero
goodput.

For example, `-n 16M` uses 33,554,432 bytes of BF16 input per rank. A 5 ms kernel
reports `Goodput: 6.71089 GB/s`.

Run the two-GPU goodput report checks with `bash test/test_2tile_goodput.sh`.

### CRI cache-policy results

The following was measured on 2026-09-30 on the two-card CRI host
`10.99.62.220` with oneAPI 2026.1 and Intel MPI. Loads were fixed at
`uc.ca.uc`; only the L1 store policy changed. Functionality used
`-n 1M -g 1 -w 4 -v`, and performance is the median of three maximum-rank
kernel times using `-n 64M -g 64 -w 4`.

| L1 store | Store control | `small` | `simple` | Functionality |
|---|---|---:|---:|---|
| UC | `uc.wb.uc` | 13.468 ms | 9.999 ms | Pass |
| WT | `wt.wb.uc` | 13.579 ms (+0.82%) | 10.033 ms (+0.35%) | Pass |
| ST | `st.wb.uc` | N/A | N/A | Hangs; 60-second timeout |

UC remains the default. WT provided no performance gain in this test. ST hung
for both protocols even though each store was 256 bytes and 256-byte aligned.
