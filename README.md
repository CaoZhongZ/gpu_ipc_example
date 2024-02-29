## Intel MLPerf Prototypes on Peak Performance Communication Collectives

This SYCL implementation of All-Reduce shows how to achieve peak performance on Intel PVC system with XeLinks. The concurrent full utilization of all link bandwidth is the key to achieve designed peak performance in single launch of kernel. Implementation demonstrated in half precision at the moment.

## Requirements
1. Intel SYCL Compiler
2. Most up-to-date drivers for PVC
3. MPI

## Build the Benchmark
git submodule update --init
make main

## Run
mpirun -np 8 ./main -n \<number of elements in half\> [-w sub-group] [-g group]
