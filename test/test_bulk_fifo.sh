#!/usr/bin/env bash
set -euo pipefail

# Build first: make ARCH=cri bulk_fifo_ring_test
# WORLD=16 DEVICES=0,1,...,15 ./test/test_bulk_fifo.sh
cd "$(dirname "$0")/.."
world=${WORLD:-2}
workers=${WORK_ITEMS:-1024}
store_cache=${STORE_CACHE:-uc.uc.uc}
deadline=${TIMEOUT_SECONDS:-90}
launcher=${MPIEXEC:-mpirun}
binary=${BULK_FIFO_BINARY:-./bulk_fifo_ring_test}
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export I_MPI_FABRICS=${I_MPI_FABRICS:-shm}
device_args=()
if [[ -n ${DEVICES:-} ]]; then device_args=(--devices "$DEVICES"); fi

run_case() {
  echo "Ring case: ranks=$world work-items=$workers store-cache=$store_cache fifo=$1 bytes=$2"
  timeout --kill-after=10s "${deadline}s" "$launcher" -n "$world" "$binary" \
    --work-items "$workers" --store-cache "$store_cache" \
    --fifo-bytes "$1" --bytes "$2" --iterations 3 \
    "${device_args[@]}"
}

# Arbitrary lengths, partial packs, minimum transfer, a full slot, and wraps.
for bytes in 0 1 15 16 17 255 256 257 8192 524287 524288 524289 5243003; do
  run_case 4M "$bytes"
done
for bytes in 262143 262144 262145 3145851; do
  run_case 2M "$bytes"
done
