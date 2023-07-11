#!/bin/bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export PATH=../ccl/release/_install/bin:$PATH
export LD_LIBRARY_PATH=../ccl/release/_install/lib:$LD_LIBRARY_PATH

export NEOReadDebugKeys=1
export EnableImplicitScaling=0

export NEO_BIN=/home/caozhong/Workspace/neo/bin
export NEO_LIB=/home/caozhong/Workspace/neo/lib
export PATH=$NEO_BIN:$PATH
export LD_LIBRARY_PATH=$(pwd)/_install/lib:$NEO_LIB:$LD_LIBRARY_PATH
export OCL_ICD_VENDORS=/home/caozhong/Workspace/neo/etc/OpenCL/vendors/intel.icd

mpirun -disable-auto-cleanup -np 1 gdbserver :44444 ./fill_remote -c 16 -t fp16 : -np 1 gdbserver :44555 ./fill_remote -c 16 -t fp16
