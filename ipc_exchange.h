#pragma once

#include <level_zero/ze_api.h>

void *mmap_host(size_t map_size, ze_ipc_mem_handle_t ipc_handle);

ze_ipc_mem_handle_t open_peer_ipc_mems(
    void *ptr, int rank, int world, void *peer_bases[], size_t offsets[]
);

