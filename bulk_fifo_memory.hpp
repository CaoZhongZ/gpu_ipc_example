#pragma once

#include <atomic>
#include <cstring>
#include <sycl/sycl.hpp>

#include "bulk_fifo_layout.hpp"

namespace bulk_fifo {

// Controls and FIFO loads use UC at all three CRI cache levels.
// Payload-store policies are independent of the original protocol's settings.
// Keep compiler ordering as well as the GPU's system ordering explicit.
inline std::uint64_t loadCounter(const std::uint64_t* address) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  std::uint64_t value;
  asm volatile("lsc_load.ugm.uc.uc.uc (M1, 1) %0:d64 flat[%1]:a64\n"
               : "=rw"(value) : "rw"(address) : "memory");
  return value;
#else
  return *address;
#endif
}

inline void storeCounter(std::uint64_t* address, std::uint64_t value) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  asm volatile("lsc_store.ugm.uc.uc.uc (M1, 1) flat[%0]:a64 %1:d64\n"
               : : "rw"(address), "rw"(value) : "memory");
#else
  *address = value;
#endif
}

using Pack = sycl::vec<std::uint32_t, 4>;

inline Pack loadPack(const unsigned char* address) {
  Pack value;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  auto& native = reinterpret_cast<Pack::vector_t&>(value);
  asm volatile("lsc_load.ugm.uc.uc.uc (M1, 16) %0:d32x4 flat[%1]:a64\n"
               : "=rw"(native) : "rw"(address) : "memory");
#else
  std::memcpy(&value, address, PackBytes);
#endif
  return value;
}

template <StoreCache Cache>
inline void storePack(unsigned char* address, const Pack& value) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  const auto& native = reinterpret_cast<const Pack::vector_t&>(value);
  if constexpr (Cache == StoreCache::UcUcUc)
    asm volatile("lsc_store.ugm.uc.uc.uc (M1, 16) flat[%0]:a64 %1:d32x4\n"
                 : : "rw"(address), "rw"(native) : "memory");
  else if constexpr (Cache == StoreCache::WbWbUc)
    asm volatile("lsc_store.ugm.wb.wb.uc (M1, 16) flat[%0]:a64 %1:d32x4\n"
                 : : "rw"(address), "rw"(native) : "memory");
  else if constexpr (Cache == StoreCache::WtWbUc)
    asm volatile("lsc_store.ugm.wt.wb.uc (M1, 16) flat[%0]:a64 %1:d32x4\n"
                 : : "rw"(address), "rw"(native) : "memory");
  else if constexpr (Cache == StoreCache::StWbUc)
    asm volatile("lsc_store.ugm.st.wb.uc (M1, 16) flat[%0]:a64 %1:d32x4\n"
                 : : "rw"(address), "rw"(native) : "memory");
  else if constexpr (Cache == StoreCache::UcWbUc)
    asm volatile("lsc_store.ugm.uc.wb.uc (M1, 16) flat[%0]:a64 %1:d32x4\n"
                 : : "rw"(address), "rw"(native) : "memory");
  else if constexpr (Cache == StoreCache::StUcUc)
    asm volatile("lsc_store.ugm.st.uc.uc (M1, 16) flat[%0]:a64 %1:d32x4\n"
                 : : "rw"(address), "rw"(native) : "memory");
  else if constexpr (Cache == StoreCache::WbUcUc)
    asm volatile("lsc_store.ugm.wb.uc.uc (M1, 16) flat[%0]:a64 %1:d32x4\n"
                 : : "rw"(address), "rw"(native) : "memory");
#else
  std::memcpy(address, &value, PackBytes);
#endif
}

inline void systemRelease() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  asm volatile("lsc_fence.ugm.evict.sysrel\n" : : : "memory");
#else
  std::atomic_thread_fence(std::memory_order_release);
#endif
}

inline void systemAcquire() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  asm volatile("lsc_fence.ugm.evict.sysacq\n" : : : "memory");
#else
  std::atomic_thread_fence(std::memory_order_acquire);
#endif
}

} // namespace bulk_fifo
