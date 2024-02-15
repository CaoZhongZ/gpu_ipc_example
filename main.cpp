#include <mpi.h>
#include <sys/mman.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include "ipc_exchange.h"

#include "cxxopts.hpp"
#include "utils.hpp"
#include "ze_exception.hpp"
#include "sycl_misc.hpp"
#include "transmit.hpp"

size_t parse_nelems(const std::string& nelems_string) {
  size_t base = 1;
  size_t pos = nelems_string.rfind("K");
  if (pos != std::string::npos) {
    base = 1024ull;
  } else {
    pos = nelems_string.rfind("M");
    if (pos != std::string::npos)
      base = 1024 * 1024ull;
    else {
      pos = nelems_string.rfind("G");
      if (pos != std::string::npos)
        base = 1024 * 1024 * 1024ull;
    }
  }

  return stoull(nelems_string) * base;
}

template <typename T>
void extract_profiling(sycl::event e) {
  e.wait();
};

void fill_pattern(uint32_t *input, int rank, size_t n) {
  for (int i = 0; i < n; ++ i)
    input[i] = i % 32 | rank << 16;
}

int main(int argc, char* argv[]) {
  cxxopts::Options opts(
      "GPU IPC access",
      "Extremely Optimized GPU IPC Examples"
  );

  opts.allow_unrecognised_options();
  opts.add_options()
    ("n,nelems", "Number of elements, in half",
     cxxopts::value<std::string>()->default_value("8MB"))
    ("g,groups", "Number of groups",
     cxxopts::value<size_t>()->default_value("1"))
    ("w,subgroups", "Number of sub-groups",
     cxxopts::value<size_t>()->default_value("1"))
    ("f,flag", "Transmit flag identify steps",
     cxxopts::value<uint32_t>()->default_value("0xe00f100f"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems = parse_nelems(parsed_opts["nelems"].as<std::string>());
  auto groups = parsed_opts["groups"].as<size_t>();
  auto subgroups = parsed_opts["subgroups"].as<size_t>();
  auto flag = parsed_opts["flag"].as<size_t>();

  auto ret = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    std::cout<<"MPI init error"<<std::endl;
    return -1;
  }
  __scope_guard MPI_Exit([] {MPI_Finalize();});
  zeCheck(zeInit(0));

  int rank, world;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  using test_type = sycl::half;
  size_t alloc_size = nelems * sizeof(test_type);
  size_t interm_size = 32 * 1024 * 1024; // fix at 32M for now.

  auto queue = currentQueue(rank / 2, rank & 1);

  auto* input = (test_type *)sycl::malloc_device(alloc_size, queue);
  //
  // We need double buffer for both scatter and gather
  // We only need single IPC exchange
  //
  auto* host_init = (test_type *) sycl::malloc_host(alloc_size, queue);
  auto* host_verify = (test_type *)sycl::malloc_host(interm_size, queue);

  auto* ipcbuf0 = (test_type *)sycl::malloc_device(interm_size * 2, queue);
  auto* ipcbuf1 = (test_type *)((uintptr_t)ipcbuf0 + interm_size);

  __scope_guard free_pointers([&]{
      free(host_init, queue);
      free(host_verify, queue);
      free(ipcbuf0, queue);
  });

  fill_pattern((uint32_t *)host_init, rank, alloc_size/sizeof(uint32_t));
  queue.memcpy(input, host_init, alloc_size);

  void *peer_bases[world];
  size_t offsets[world];
  auto ipc_handle = open_peer_ipc_mems(ipcbuf0, rank, world, peer_bases, offsets);

  test_type *peerbuf0[world];
  test_type *peerbuf1[world];
  std::transform(peer_bases, peer_bases+world, offsets, peerbuf0,
  [](void* p, size_t off) {
      return (test_type *)((uintptr_t)p + off);
  });
  std::transform(peerbuf0, peerbuf0 + world, peerbuf1,
  [&](void *p){
      return (test_type *)((uintptr_t)p + interm_size);
  });

  auto l0_ctx = sycl::get_native<
    sycl::backend::ext_oneapi_level_zero>(queue.get_context());

  auto host_view = mmap_host(interm_size * 2, ipc_handle);

  __scope_guard release_handles([&] {
      munmap(host_view, interm_size * 2);

      for (int i = 0;i < world; ++ i) {
        if (i != rank) zeCheck(zeMemCloseIpcHandle(l0_ctx, peer_bases[i]));
      }
      (void)ipc_handle; // Put IPC handle in the future
  });


  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int SG_SZ = 16;

  auto local_size = subgroups * SG_SZ;
  auto global_size = groups * local_size;

  auto e = testSimpleTransmit<test_type, SG_SZ>(
      {sycl::range<1>(global_size), sycl::range<1>(local_size)},
      input, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
      nelems, rank, world, flag, queue
  );

  MPI_Barrier(MPI_COMM_WORLD);
  extract_profiling<test_type>(e);

  // XXX: over simplification
  auto workNelems = nelems / world;

  queue.memcpy(host_verify, ipcbuf0, interm_size).wait();
  return verifyTransmit<test_type, SG_SZ>(
      (uint32_t *)host_verify, flag, rank, world, workNelems
  );
}
