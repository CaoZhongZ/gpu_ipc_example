#!/bin/bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export PATH=/home/caozhong/Workspace/ccl/release/_install/bin:$PATH
export LD_LIBRARY_PATH=/home/caozhong/Workspace/ccl/release/_install/lib:$LD_LIBRARY_PATH

# debug_s0="gdbserver :44222"
# debug_s1="gdbserver :44444"
debug_s2="gdbserver :44555"

mpirun -disable-auto-cleanup \
  -np 1 $debug_s1 ./fill_remote -c 16 -t fp16 : \
  -np 1 $debug_s2 ./fill_remote -c 16 -t fp16
