#!/bin/bash
set -euo pipefail

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export I_MPI_FABRICS=shm

logfile=$(mktemp)
trap 'rm -f "$logfile"' EXIT

for algorithm in small simple allgather_small allgather_simple; do
  for melements in 1 16; do
    echo "Check goodput: $algorithm ${melements}M elements"
    mpirun -np 2 ./main -l 01 -a "$algorithm" -s 16 \
      -n "${melements}M" -g 32 -w 4 2>&1 | tee "$logfile"

    awk -v payload_bytes="$((melements * 1024 * 1024 * 2))" '
      /Running time:/ {
        elapsed_ns = ""
        goodput = ""
        unit = ""
        for (field = 1; field <= NF; ++field) {
          if ($field == "time:") {
            elapsed_ns = $(field + 1)
            sub(/ns,$/, "", elapsed_ns)
          }
          if ($field == "Goodput:") {
            goodput = $(field + 1)
            unit = $(field + 2)
          }
        }
        if (elapsed_ns !~ /^[0-9]+$/ ||
            goodput !~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ ||
            unit != "GB/s") {
          print "Invalid performance report: " $0 > "/dev/stderr"
          exit 1
        }
        expected = elapsed_ns == 0 ? 0 : payload_bytes / elapsed_ns
        difference = goodput - expected
        if (difference < 0)
          difference = -difference
        if (difference > expected * 0.00001 + 0.000001) {
          print "Incorrect goodput: " $0 > "/dev/stderr"
          exit 1
        }
        ++reports
      }
      END {
        if (reports != 2) {
          print "Expected two rank performance reports, got " reports > "/dev/stderr"
          exit 1
        }
      }
    ' "$logfile"
  done
done

echo "Goodput checks passed"
