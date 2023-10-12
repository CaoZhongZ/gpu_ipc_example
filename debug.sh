#!/bin/bash
export FI_PROVIDER=tcp
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# export PATH=/home/caozhong/Workspace/ccl/release/_install/bin:$PATH
# export LD_LIBRARY_PATH=/home/caozhong/Workspace/ccl/release/_install/lib:$LD_LIBRARY_PATH

mpirun -disable-auto-cleanup \
  -np 2 ./copy_atomicctl $@ : \
  -np 1 ./copy_atomicctl $@ : \
  -np 1 ./copy_atomicctl $@ : \
  -np 1 gdbserver :44555 ./copy_atomicctl $@ : \
  -np 3 ./copy_atomicctl $@
