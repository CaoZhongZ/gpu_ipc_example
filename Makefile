CC=clang
CXX=clang++

OPT=-O3 -fno-strict-aliasing -g
# OPT=-g -fno-strict-aliasing

SYCLFLAGS=-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc"

# CCL_ROOT=../ccl/release/_install
# INCLUDES=-I$(CCL_ROOT)/include
# LIBRARIES=-L$(CCL_ROOT)/lib -lmpi -lze_loader

INCLUDES=-Itvisa/include
LIBRARIES=-lmpi -lze_loader

CXXFLAGS=-std=c++17 $(SYCLFLAGS) $(OPT) -Wall -Wno-vla-cxx-extension $(INCLUDES) $(LIBRARIES)

main : main.cpp ipc_exchange.cpp sycl_misc.cpp transmit.cpp

all : main

clean:
	rm -f main
