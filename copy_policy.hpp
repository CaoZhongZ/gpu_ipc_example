#pragma once

#include <sycl/sycl.hpp>

//
// Next: Copy in chunks of certain size for maximized group occupation
//
// building block for 4 , 8, 16, 32, 64 elems copy
//
template <typename T, size_t NElems>
struct chunk_copy {
  static constexpr size_t v_lane = NElems * sizeof(T) < 16 ? 8 : 16;
  using v_T = sycl::vec<T, v_lane/sizeof(T)>;
  static_assert(NElems % v_T::size() == 0);
  static constexpr size_t n_loop = NElems / v_T::size();

  static inline size_t chunk_size(sycl::nd_item<1> pos) {
    return n_loop * pos.get_global_range(0);
  }

  static inline size_t chunk_group(sycl::nd_item<1> pos) {
    return n_loop * pos.get_local_range(0);
  }

  static inline size_t item_size() {
    return NElems;
  }

  static inline void run(
      sycl::nd_item<1> pos,
      T* dst, const T* src, size_t nelems
  ) {
    auto* v_dst = reinterpret_cast<v_T *>(dst);
    auto* v_src = reinterpret_cast<const v_T *>(src);
    auto bound = nelems / v_T::size();

    size_t off = pos.get_global_id(0);

    while (off < bound) {
#     pragma unroll
      for (int n = 0; n < n_loop; ++ n) {
        if (off < bound) {
          v_dst[off] = v_src[off];
          off += pos.get_global_range(0);
        }
      }
    }
  }

  static inline void run(
      T* dst, const T* src, size_t off, size_t step, size_t nelems
  ) {
    auto* v_dst = reinterpret_cast<v_T *>(dst);
    auto* v_src = reinterpret_cast<const v_T *>(src);
    auto bound = nelems / v_T::size();
#   pragma unroll
    for (int n = 0; n < n_loop; ++ n) {
      if (off < bound) {
        v_dst[off] = v_src[off];
        off += step;
      }
    }
  }

  static inline void run(
      sycl::nd_item<1> pos,
      T* dst, const T* src,
      size_t start, size_t nelems
  ) {
    auto* v_dst = reinterpret_cast<v_T *>(dst);
    auto* v_src = reinterpret_cast<const v_T *>(src);
    auto bound = nelems / v_T::size();
    auto off = pos.get_global_id(0) + start;
    auto step = pos.get_global_range(0);

#   pragma unroll
    for (int n = 0; n < n_loop; ++ n) {
      if (off < bound)
        v_dst[off] = v_src[off];
      off += step;
    }
  }

  static inline void reduce(
      T* dst, const T* src0, const T* src1,
      size_t dst_off, size_t src_off,
      size_t stride, size_t nelems
  ) {
    auto* v_dst = reinterpret_cast<v_T *>(dst);
    auto* v_src0 = reinterpret_cast<const v_T *>(src0);
    auto* v_src1 = reinterpret_cast<const v_T *>(src1);
    auto bound = nelems / v_T::size();
#   pragma unroll
    for (int n = 0; n < n_loop; ++ n) {
      v_T intermediate {};
      if (src_off < bound) {
        intermediate = v_src0[src_off] + v_src1[src_off];
        src_off += stride;
      }

      if (dst_off < bound) {
        v_dst[dst_off] = intermediate;
        dst_off += stride;
      }
    }
  }

  // Make sure use 1/4 group size for this case
  template <int Series, typename std::enable_if<Series != n_loop>>
  static inline void reduce_gather(
      T* dsts[], const T* src,
      size_t dst_off, size_t src_off, size_t stride, size_t nelems
  ) {
    auto bound = nelems / v_T::size();
    auto* v_src = reinterpret_cast<const v_T *>(src);

#   pragma unroll
    for (int n = 0; n < n_loop; ++ n) {
      v_T intermediate {};
#     pragma unroll
      for (int s = 0; s < Series; ++ s) {
        if (src_off + stride * s < bound) {
          intermediate += v_src[src_off + stride * s];
          src_off += stride * s;
        }
      }

      if (dst_off < bound) {
#       pragma unroll
        for (int s = 0; s < Series; ++ s) {
          auto* v_dst = reinterpret_cast<v_T *>(dsts[s]);
          v_dst[dst_off] = intermediate;
        }
      }
    }
  }

  // Use same group size
  template <int Series, typename std::enable_if<Series == n_loop>>
  static inline void reduce_gather(
      T* dsts[], const T* src,
      size_t dst_off, size_t src_off, size_t stride, size_t nelems
  ) {
    auto bound = nelems / v_T::size();
    auto* v_src = reinterpret_cast<const v_T *>(src);

    v_T intermediate {};
#   pragma unroll
    for (int s = 0; s < Series; ++ s) {
      if (src_off + stride * s < bound) {
        intermediate += v_src[src_off + stride * s];
        src_off += stride * s;
      }
    }

    if (dst_off < bound) {
#     pragma unroll
      for (int s = 0; s < Series; ++ s) {
        auto* v_dst = reinterpret_cast<v_T *>(dsts[s]);
        v_dst[dst_off] = intermediate;
      }
    }
  }
};
