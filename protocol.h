#pragma once

template <int subGroupWidth = 16, int nSubGroup = 2, int nAtomics = 16>
struct Cell {
public:
  union {
    char bytes[ subGroupWidth * nSubGroup ];

    uint16_t words[ sizeof(bytes)/(sizeof(uint16_t)) ];
    sycl::half halfs[ sizeof(bytes)/sizeof(sycl::half) ];

    uint32_t dwords[ sizeof(bytes]/(sizeof(uint32_t) ];
    float floats[ sizeof(bytes)/sizeof(float) ];
  };
  uint32_t atomics[nAtomics];
};

template <int hwGroups, int subGroupWidth = 16, int nSubGroup = 2, int nAtomics = 16>
struct Scratch {
  static constexpr int nBuffer = 4;
public:
  Cell<subGroupWidth, nSubGroup, nAtomics> cells[nBuffer][hwGroups];
};
