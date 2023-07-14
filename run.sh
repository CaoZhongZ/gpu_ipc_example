#!/bin/bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export PATH=/home/caozhong/Workspace/ccl/release/_install/bin:$PATH
export LD_LIBRARY_PATH=/home/caozhong/Workspace/ccl/release/_install/lib:$LD_LIBRARY_PATH

mpirun -np 8 ./fill_remote -g 16384 -l 32 -i $@
