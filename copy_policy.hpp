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
      auto v_off = off + n * global_range;
      if (v_off < bound)
        v_dst[v_off] = v_src[v_off];
    }
  }

  static inline void merge(
      T* dst, const T* src0, const T* src1,
      size_t off0, size_t off1, size_t step, size_t nelems
  ) {
    auto* v_dst = reinterpret_cast<v_T *>(dst);
    auto* v_src0 = reinterpret_cast<const v_T *>(src0);
    auto* v_src1 = reinterpret_cast<const v_T *>(src1);
    auto bound = nelems / v_T::size();
#   pragma unroll
    for (int n = 0; n < n_loop; ++ n) {
      auto v_off0 = off0 + n * step;
      auto v_off1 = off1 + n * step;

      if (v_off0 < bound) {
        v_dst[v_off0] = v_src0[v_off1];
      }

      if (v_off1 < bound) {
        v_dst[v_off1] = v_src1[v_off1];
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
      size_t stride, size_t nelems
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
        auto* v_dst = reinterpret_cast<v_T *>(dsts[i]);
        v_dst[dst_off] = intermediate;
      }
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
