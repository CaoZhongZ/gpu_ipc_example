ifeq ($(origin CXX), default)
ifneq ($(shell command -v mpiicpx 2>/dev/null),)
CXX := mpiicpx
else
CXX := icpx
endif
endif

ARCH ?= pvc
CRI_STORE_L1_CACHE ?= uc
CRI_TARGET ?= legacy

ifeq ($(ARCH), bmg)
arch_string=bmg-g21-a0
arch_support=-DXE_PLUS -DATOB_SUPPORT -DBMG
endif

ifeq ($(ARCH), pvc)
arch_string=pvc
arch_support=-DXE_PLUS -DPVC
endif

ifeq ($(ARCH), arc770)
arch_string=ats-m150
arch_support=-DDG2
endif

ifeq ($(ARCH), cri)
arch_support=-DXE_PLUS -DCRI -D__SYCL_TARGET_INTEL_GPU_CRI__ \
	-D__SYCL_USE_LIBSYCL8_VEC_IMPL=1
ifeq ($(CRI_STORE_L1_CACHE), uc)
arch_support += -DCRI_STORE_L1_CACHE_POLICY=0
else ifeq ($(CRI_STORE_L1_CACHE), wt)
arch_support += -DCRI_STORE_L1_CACHE_POLICY=1
else ifeq ($(CRI_STORE_L1_CACHE), st)
arch_support += -DCRI_STORE_L1_CACHE_POLICY=2
else
$(error CRI_STORE_L1_CACHE must be one of: uc, wt, st)
endif
endif

OPT=-O3 -fno-strict-aliasing -DLOCAL_TEST
# OPT=-g -fno-strict-aliasing
# VERBOSE=-D__enable_device_verbose__
#

ifeq ($(ARCH), cri)
ifeq ($(CRI_TARGET), legacy)
SYCLFLAGS=-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device cri-a0"
else ifeq ($(CRI_TARGET), dedicated)
SYCLFLAGS=-fsycl -fsycl-targets=intel_gpu_cri
else
$(error CRI_TARGET must be one of: legacy, dedicated)
endif
else
SYCLFLAGS=-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device $(arch_string)"
endif

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
