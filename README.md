Dependencies:
  1. MPI
  2. Level-Zero
  3. SYCL enabled compiler

Build:
  make

Run:
  bash run.sh -n 32M -s 1 -e 16

Options:
-n test size
-s sync mode (0, no sync, 1, atomic)
-e group number (only support 16)
