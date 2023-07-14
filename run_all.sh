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
export SYCL_ENABLE_DEFAULT_CONTEXTS=0

# mpirun -np 1 ./fill_remote -c 16 -t fp16 : -np 1 ./fill_remote -c 16 -t fp16
# trace_cmd="unitrace --demangle --chrome-device-timeline --chrome-call-logging --chrome-no-thread-on-device --chrome-no-engine-on-device"
trace_cmd=""
# $trace_cmd mpirun -np 2 ./fill_remote -c 16 -t fp16
mpirun -np 8 $trace_cmd ./fill_remote -c 16 -t fp16
