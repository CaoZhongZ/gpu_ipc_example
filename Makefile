CC=clang
CXX=clang++

OPT=-O3 -fno-strict-aliasing
# OPT=-g -fno-strict-aliasing

SYCLFLAGS=-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc"

# CCL_ROOT=../ccl/release/_install
# INCLUDES=-I$(CCL_ROOT)/include
# LIBRARIES=-L$(CCL_ROOT)/lib -lmpi -lze_loader

INCLUDES=
LIBRARIES=-lmpi -lze_loader

CXXFLAGS=-std=c++17 $(SYCLFLAGS) $(OPT) -Wall $(INCLUDES) $(LIBRARIES)

all : copy_atomicctl copy_baseline

copy_atomicctl : copy_atomicctl.cpp ipc_exchange.cpp sycl_misc.cpp

copy_baseline : copy_baseline.cpp sycl_misc.cpp

clean:
	rm -f fill_remote atomic_2020 linearize allreduce list_device copy_atomicctl
