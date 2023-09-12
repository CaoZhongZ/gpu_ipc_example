#pragma once

#include <sycl/sycl.hpp>

struct launchConfig1 {
  // 4x32 group, process 1024 sycl::half elements
  static constexpr size_t laneWidth = 16;
  static constexpr size_t groupY = 4;
  static constexpr size_t groupX = 32;
  static constexpr size_t subGroupSize = 16;
  static constexpr size_t maxRoleGroups = 128;
};

template <typename T, typename groupTrait>
struct Cell {
public:
  union {
    char bytes[ groupTrait::laneWidth * groupTrait::groupY * groupTrait::groupX ];
    uint16_t words[ sizeof(bytes)/sizeof(uint16_t) ];
    sycl::half halfs[ sizeof(bytes)/sizeof(sycl::half) ];
    uint32_t dwords[ sizeof(bytes)/sizeof(uint32_t) ];
    float floats[ sizeof(bytes)/sizeof(float) ];

    sycl::vec<T, groupTrait::laneWidth/sizeof(T)>
      data[groupTrait::groupY][groupTrait::groupX];
  };

  uint32_t atomics[ 128 /* cache line size */ / sizeof(uint32_t) ];
};

template <typename T, typename groupTrait>
struct stepCell {
public:
  union {
    char bytes[
      groupTrait::maxRoleGroups * groupTrait::laneWidth
        * groupTrait::groupY * groupTrait::groupX
    ];
    uint16_t words[ sizeof(bytes)/sizeof(uint16_t) ];
    sycl::half halfs[ sizeof(bytes)/sizeof(sycl::half) ];
    uint32_t dwords[ sizeof(bytes)/sizeof(uint32_t) ];
    float floats[ sizeof(bytes)/sizeof(float) ];

    sycl::vec<T, groupTrait::laneWidth/sizeof(T)>
      data[groupTrait::maxRoleGroups][groupTrait::groupY][groupTrait::groupX];
  };

  uint32_t atomics[ 128 /* cache line size */ / sizeof(uint32_t) ];
};
