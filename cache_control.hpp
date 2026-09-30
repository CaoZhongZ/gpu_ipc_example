#pragma once

#include <gen_visa_templates.hpp>

#define CRI_STORE_L1_CACHE_UC 0
#define CRI_STORE_L1_CACHE_WT 1
#define CRI_STORE_L1_CACHE_ST 2

#ifndef CRI_STORE_L1_CACHE_POLICY
#define CRI_STORE_L1_CACHE_POLICY CRI_STORE_L1_CACHE_UC
#endif

namespace ipc_cache {

#if defined(__SYCL_TARGET_INTEL_GPU_CRI__)

constexpr auto CommReadCacheCtrl = CacheCtrl::L1UC_L2C_L3UC;

#if CRI_STORE_L1_CACHE_POLICY == CRI_STORE_L1_CACHE_UC
constexpr auto CommWriteCacheCtrl = CacheCtrl::L1UC_L2WB_L3UC;
constexpr const char *CommCachePolicyName = "uc";
#elif CRI_STORE_L1_CACHE_POLICY == CRI_STORE_L1_CACHE_WT
constexpr auto CommWriteCacheCtrl = CacheCtrl::L1WT_L2WB_L3UC;
constexpr const char *CommCachePolicyName = "wt";
#elif CRI_STORE_L1_CACHE_POLICY == CRI_STORE_L1_CACHE_ST
constexpr auto CommWriteCacheCtrl = CacheCtrl::L1S_L2WB_L3UC;
constexpr const char *CommCachePolicyName = "st";
#else
#error "Unsupported CRI_STORE_L1_CACHE_POLICY"
#endif

constexpr auto PcieCommReadCacheCtrl = CommReadCacheCtrl;
constexpr auto PcieCommWriteCacheCtrl = CommWriteCacheCtrl;

#else

constexpr auto CommReadCacheCtrl = CacheCtrl::L1UC_L3C;
constexpr auto CommWriteCacheCtrl = CacheCtrl::L1UC_L3WB;
constexpr const char *CommCachePolicyName = "uc";

#if defined(XE_PLUS)
constexpr auto PcieCommReadCacheCtrl = CacheCtrl::L1UC_L3C;
constexpr auto PcieCommWriteCacheCtrl = CacheCtrl::L1UC_L3WB;
#else
constexpr auto PcieCommReadCacheCtrl = CacheCtrl::L1UC_L3UC;
constexpr auto PcieCommWriteCacheCtrl = CacheCtrl::L1UC_L3UC;
#endif

#endif

} // namespace ipc_cache

#if defined(__SYCL_TARGET_INTEL_GPU_CRI__)
#define IPC_LSC_DEFAULT_CACHE_CTRL "df.df.df"
#else
#define IPC_LSC_DEFAULT_CACHE_CTRL "df.df"
#endif
