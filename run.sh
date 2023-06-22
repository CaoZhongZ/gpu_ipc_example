#!/bin/bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
mpirun -np 1 ./fill_remote -c 16 -t fp16 : -np 1 ./fill_remote -c 16 -t fp16
