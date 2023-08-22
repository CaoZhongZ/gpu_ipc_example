#pragma once

template <typename T, int Size, int subGroupWidth = 16, int nSubGroup = 2, int nAtomics = 16>
struct Cell {
public:
  union {
    char bytes[ Size * subGroupWidth * nSubGroup ];

    uint16_t words[ sizeof(bytes)/sizeof(uint16_t) ];
    sycl::half halfs[ sizeof(bytes)/sizeof(sycl::half) ];

    uint32_t dwords[ sizeof(bytes]/sizeof(uint32_t) ];
    float floats[ sizeof(bytes)/sizeof(float) ];

    T data[ sizeof(bytes) / sizeof(T) ];
  };
  uint32_t atomics[nAtomics];
};
