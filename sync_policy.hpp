#pragma once

#include <sycl/sycl.hpp>

template <typename T>
class atomic_ref: public sycl::atomic_ref<T,
                      sycl::memory_order::relaxed,
                      sycl::memory_scope::device,
                      sycl::access::address_space::global_space> {
public:
  atomic_ref(T& r) : sycl::atomic_ref<T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space> (r)
  {}
};

template <typename T>
class slm_atomic_ref : public sycl::atomic_ref<T,
                      sycl::memory_order::relaxed,
                      sycl::memory_scope::device,
                      sycl::access::address_space::local_space> {
public:
  slm_atomic_ref(T& r) : sycl::atomic_ref<T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::local_space> (r)
  {}
};

class dummy_sync {
public:
  static constexpr int local_count = 0;
  // do nothing
  static inline void init_slm_flags(
      sycl::nd_item<1>,
      sycl::local_ptr<uint32_t>,
      sycl::local_ptr<uint32_t>,
      size_t
  ) {
    // sycl::group_barrier(pos.get_group());
  }

  static inline void init_slm_flags(
      size_t,
      sycl::local_ptr<uint32_t>,
      sycl::local_ptr<uint32_t>,
      size_t
  ) {}

  static inline void block_if_eq(
      sycl::nd_item<1>, uint32_t*, uint32_t,
      sycl::local_ptr<uint32_t>,
      sycl::local_ptr<uint32_t>
  ) {}

  static inline void block_if_not(
      sycl::nd_item<1> pos,
      uint32_t* flag, uint32_t target,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {}

  static inline void signal(
      sycl::nd_item<1> pos,
      uint32_t* flag, uint32_t value,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {}

  static inline void finish(
      sycl::nd_item<1>, uint32_t, uint32_t*, uint32_t*,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {};

  static inline void reset(
      sycl::nd_item<1> pos,
      uint32_t* flag,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {}

  static inline size_t get_target(
      sycl::nd_item<1>
  ) {
    return 0;
  }
};

//
// Global sync protocol, up half of 32-bit is for remote
//
class flat_sync {
public:
  static constexpr int local_count = 0;
  // do nothing
  static inline void init_slm_flags(
      sycl::nd_item<1>,
      sycl::local_ptr<uint32_t>,
      sycl::local_ptr<uint32_t>,
      size_t
  ) {}

  static inline void init_slm_flags(
      size_t,
      sycl::local_ptr<uint32_t>,
      sycl::local_ptr<uint32_t>,
      size_t
  ) {}

  static inline void block_if_eq(
      sycl::nd_item<1> pos,
      uint32_t* flag, uint32_t target,
      sycl::local_ptr<uint32_t>,
      sycl::local_ptr<uint32_t>
  ) {
    atomic_ref g_flag(*flag);

    if (pos.get_sub_group().leader()) {
      while((g_flag.load() & 0xffff) == target);
    }

    sycl::group_barrier(pos.get_sub_group());
  }

  static inline void signal(
      sycl::nd_item<1> pos, uint32_t* flag, uint32_t value,
      sycl::local_ptr<uint32_t>, sycl::local_ptr<uint32_t>) {
    atomic_ref g_flag(*flag);
    if (pos.get_sub_group().leader())
      g_flag ++;

    sycl::group_barrier(pos.get_sub_group());
  }

  static inline void finish(
      sycl::nd_item<1> pos, uint32_t target,
      uint32_t* flag, uint32_t *remote,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {
    atomic_ref g_flag(*flag);
    atomic_ref r_flag(*remote);

    if (pos.get_sub_group().leader()) {
      uint32_t count = g_flag++;

      if ((count & 0xffff) == target - 1)
        r_flag |= (target << 16);
    }

    sycl::group_barrier(pos.get_sub_group());
  };

  static inline void reset(
      sycl::nd_item<1> pos,
      uint32_t* flag,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {
  }

  static inline size_t get_target(
      sycl::nd_item<1> pos
  ) {
    return pos.get_sub_group().get_group_range()[0]
      * pos.get_group_range()[0];
  }
};

class hierarchy_sync {
  static inline bool group_leader(
      sycl::local_ptr<uint32_t> local_counter
  ) {
    slm_atomic_ref l_c(*local_counter);
    return (l_c ++ == 0);
  }

  static inline bool group_tail(
      sycl::nd_item<1> pos,
      sycl::local_ptr<uint32_t> local_counter
  ) {
    slm_atomic_ref l_c(*local_counter);
    return (l_c ++ == 2*pos.get_sub_group().get_group_range()[0] -1);
  }

  static inline bool group_tail(
      size_t local_size,
      sycl::local_ptr<uint32_t> local_counter
  ) {
    slm_atomic_ref l_c(*local_counter);
    return (l_c ++ == 2 * local_size -1);
  }

public:
  // safe if larger than sub-group number in this group
  static constexpr int local_count = 128;
  // slow, group barrier inside
  static inline void init_slm_flags(
      sycl::nd_item<1> pos,
      sycl::local_ptr<uint32_t> counter_array,
      sycl::local_ptr<uint32_t> wait_array,
      size_t n_slot
  ) {
    auto off = pos.get_local_range(0);
    if (off < n_slot) {
      counter_array[off] = 0;
      wait_array[off] = true;
    }

    sycl::group_barrier(pos.get_group());
  }

  // put group barrier outside!
  static inline void init_slm_flags(
      size_t off,
      sycl::local_ptr<uint32_t> counter_array,
      sycl::local_ptr<uint32_t> wait_array,
      size_t n_slot
  ) {
    if (off < n_slot) {
      counter_array[off] = 0;
      wait_array[off] = true;
    }
  }

  static inline void block_if_eq(
      sycl::nd_item<1> pos,
      uint32_t* flag, uint32_t target,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {
    slm_atomic_ref l_wait(*local_wait);
    slm_atomic_ref l_c(*local_counter);

    atomic_ref g_flag(*flag);

    if (pos.get_sub_group().leader()) {
      if (group_leader(local_counter)) {
        while(g_flag.load() == target);
        l_wait.store(false);
      } else {
        while(l_wait.load());
      }
    }

    sycl::group_barrier(pos.get_sub_group());
  }

  static inline void block_if_not(
      sycl::nd_item<1> pos,
      uint32_t* flag, uint32_t target,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {
    slm_atomic_ref l_wait(*local_wait);
    slm_atomic_ref l_c(*local_counter);

    atomic_ref g_flag(*flag);

    if (pos.get_sub_group().leader()) {
      if (group_leader(local_counter)) {
        while(g_flag.load() != target);
        l_wait.store(false);
      } else {
        while(l_wait.load());
      }
    }

    sycl::group_barrier(pos.get_sub_group());
  }

  static inline void finish(
      sycl::nd_item<1> pos,
      uint32_t target, uint32_t* flag, uint32_t *remote,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {
    atomic_ref g_flag(*flag);
    atomic_ref r_flag(*remote);
    slm_atomic_ref l_c(*local_counter);
    slm_atomic_ref l_w(*local_wait);

    if (pos.get_sub_group().leader()) {
      if (group_tail(pos, local_counter)) {
        uint32_t count = g_flag++;

        if ((count & 0xffff) == target - 1)
          r_flag |= target << 16;

        l_c.store(0);
        l_w.store(true);
      }
    }

    sycl::group_barrier(pos.get_sub_group());
  };

  static inline void signal(
      sycl::nd_item<1> pos,
      uint32_t* flag, uint32_t value,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {
    atomic_ref g_flag(*flag);
    slm_atomic_ref l_c(*local_counter);
    slm_atomic_ref l_w(*local_wait);
    if (pos.get_sub_group().leader()) {
      if (group_tail(pos, local_counter)) {
        g_flag += value;

        l_c.store(0);
        l_w.store(true);
      }
    }
    sycl::group_barrier(pos.get_sub_group());
  }

  static inline void reset(
      sycl::nd_item<1> pos,
      uint32_t* flag,
      sycl::local_ptr<uint32_t> local_counter,
      sycl::local_ptr<uint32_t> local_wait
  ) {
  }

  static inline size_t get_target(
      sycl::nd_item<1> pos
  ) {
    return pos.get_group_range(0);
  }
};
