## Intel MLPerf Prototypes on Peak Performance Communication Collectives

This SYCL implementation of All-Reduce shows how to achieve peak performance on Intel PVC system with XeLinks. A launch of single kernel that concurrently utilizes of all available internel (Cross Tile and XeLinks) bandwidth of links is the key to achieve designed peak performance. Implementation demonstrated in half precision at the moment.

## Requirements
1. Intel SYCL Compiler
2. Most up-to-date drivers for PVC
3. MPI

## Build the Benchmark
```
git submodule update --init
make main
```

## Run
```
mpirun -np 8 ./main -n \<number of elements in half\> [-w sub-group] [-g group]
```
