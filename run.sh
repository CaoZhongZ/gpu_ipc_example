#!/bin/bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export PATH=../ccl/release/_install/bin:$PATH
export LD_LIBRARY_PATH=../ccl/release/_install/lib:$LD_LIBRARY_PATH
export NEOReadDebugKeys=1
export EnableImplicitScaling=0

mpirun -np 1 ./fill_remote -c 16 -t fp16 : -np 1 ./fill_remote -c 16 -t fp16
