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

  static inline size_t cover(size_t range) {
    return n_loop * range;
  }

  static inline size_t item_size() {
    return NElems;
  }

  static inline void run(
      T* dst, const T* src, size_t off, size_t global_range, size_t nelems
  ) {
    auto* v_dst = reinterpret_cast<v_T *>(dst);
    auto* v_src = reinterpret_cast<const v_T *>(src);
    auto bound = nelems / v_T::size();
#   pragma unroll
    for (int n = 0; n < n_loop; ++ n) {
      if (off < bound) {
        v_dst[off] = v_src[off];
        off += global_range;
      }
    }
  }

  static inline void run(
      T* dst, const T* src0, const T* src1,
      size_t off0, size_t off1, size_t step, size_t nelems
  ) {
    auto* v_dst = reinterpret_cast<v_T *>(dst);
    auto* v_src0 = reinterpret_cast<const v_T *>(src0);
    auto* v_src1 = reinterpret_cast<const v_T *>(src1);
    auto bound = nelems / v_T::size();
#   pragma unroll
    for (int n = 0; n < n_loop; ++ n) {
      if (off0 < bound) {
        v_dst[off0] = v_src0[off0];
        off0 += step;
      }
      if (off1 < bound) {
        v_dst[off1] = v_src1[off1];
        off1 += step;
      }
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

      auto s_off = src_off + n * stride;
      if (s_off < bound)
        intermediate = v_src0[s_off] + v_src1[s_off];

      auto d_off = dst_off + n * stride;
      if (d_off < bound)
        v_dst[d_off] = intermediate;
    }
  }

  template <int Series>
  static inline void reduce_gather(
      T* const dsts[], const T* src,
      size_t dst_off, size_t src_off,
      size_t stride, size_t nelems,
      int scramble
  ) {
    auto bound = nelems / v_T::size();
    auto* v_src = reinterpret_cast<const v_T *>(src);

    v_T intermediate {};
#   pragma unroll
    for (int i = 0; i < Series; ++ i) {
      if (src_off + stride * i < bound)
        intermediate += v_src[src_off + stride * i];
    }

    if (dst_off < bound) {
#     pragma unroll
      for (int i = 0; i < Series; ++ i) {
        auto* v_dst = reinterpret_cast<v_T *>(
            dsts[(i + scramble) % Series]);
        v_dst[dst_off] = intermediate;
      }
    }
  }
};
