Dependencies:
  1. MPI
  2. Level-Zero
  3. SYCL enabled compiler

Build:
  make

Run:
  ```mpirun -np <N> fill_remote -g 4096 -l 32 -i 256```
