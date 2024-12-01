CXX ?= icpx
ARCH ?= pvc

ifeq ($(ARCH), bmg)
arch_string=bmg-g21-a0
arch_support=-DXE_PLUS -DATOB_SUPPORT
endif

ifeq ($(ARCH), pvc)
arch_string=pvc
arch_support=-DXE_PLUS
endif

ifeq ($(ARCH), arc770)
arch_string=ats-m150
arch_support=-DDG2
endif

OPT=-O3 -fno-strict-aliasing
# OPT=-g -fno-strict-aliasing
# VERBOSE=-D__enable_device_verbose__
#

SYCLFLAGS=-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device $(arch_string)"

.PRECIOUS: %.o

# CCL_ROOT=../ccl/release/_install
# INCLUDES=-I$(CCL_ROOT)/include
# LIBRARIES=-L$(CCL_ROOT)/lib -lmpi -lze_loader

INCLUDES=-Itvisa/include
LIBRARIES=-lmpi -lze_loader

CXXFLAGS=-std=c++17 -fopenmp $(SYCLFLAGS) $(OPT) $(VERBOSE) -Wall -Wno-vla-cxx-extension -Wno-deprecated-declarations $(INCLUDES) $(LIBRARIES) $(arch_support)

main : ipc_exchange.cpp sycl_misc.cpp allreduce.cpp main.cpp

all : main

clean:
	rm -f main
