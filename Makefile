CC=icx
CXX=icpx

OPT=-O3

SYCLFLAGS=-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc -internal_options -ze-intel-has-buffer-offset-arg -internal_options -cl-intel-greater-than-4GB-buffer-required"
CCL_ROOT=../ccl/release/_install

INCLUDES=-I$(CCL_ROOT)/include
LIBRARIES=-L$(CCL_ROOT)/lib -lmpi -lze_loader

CXXFLAGS=-std=c++17 $(SYCLFLAGS) $(OPT) -Wall $(INCLUDES) $(LIBRARIES)

all : fill_remote

clean:
	rm -f fill_remote atomic_2020
