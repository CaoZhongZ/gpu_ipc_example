#include <mpi.h>
#include <sycl/sycl.hpp>

#include "cxxopts.hpp"
#include "utils.hpp"

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
void extract_profiling(sycl::event e) {};

int main(int argc, char* argv[]) {
  cxxopts::Options opts(
      "GPU IPC access",
      "Extremely Optimized GPU IPC Examples"
  );

  opts.allow_unrecognised_options();
  opts.add_options()
    ("n,nelems", "Number of elements, in half",
     cxxopts::value<std::string>()->default_value("16MB"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems = parse_nelems(parsed_opts["nelems"].as<std::string>());
  auto ret = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    std::cout<<"MPI init error"<<std::endl;
    return -1;
  }
  __scope_guard MPI_Finalize([] {MPI_Finalize();});
  zeCheck(zeInit(0));

  int rank, world;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  using test_type = sycl::half;
  size_t alloc_size = nelems * sizeof(test_type);
  size_t interm_size = 32 * 1024 * 1024; // fix at 32M for now.

  auto queue = currentQueue(rank / 2, rank & 1);

  void* input = sycl::malloc_device(alloc_size, queue);
  //
  // We need double buffer for both scatter and gather
  // We only need single IPC exchange
  //
  test_type* host_init = sycl::malloc_host(alloc_size);

  void* ipcbuf0 = sycl::malloc_device(interm_size * 2, queue);
  void* ipcbuf1 = (void *)((uintptr_t)ipc_buf0 + interm_size);

  __scope_guard free_pointers([&]{
      free(host_init, queue);
      free(ipcbuf0, queue);
      free(ipcbuf1, queue);
  });

  void *peer_bases[world];
  size_t offsets[world];
  auto ipc_handle = open_peer_ipc_mems(ipcbuf0, rank, world, peer_bases, offsets);

  void *peerbuf0[world];
  void *peerbuf1[world];
  std::transform(peer_bases, peer_bases+world, offsets, peerbuf0,
  [](void* p, size_t off) {
      return (void *)((uintptr_t)p + off);
  });
  std::transform(peerbuf0, peerbuf0 + world, peerbuf1,
  [&](void *p){
      return (void *)((uintptr_t)p + interm_size);
  });

  auto l0_ctx = sycl::get_native<
    sycl::backend::ext_oneapi_level_zero>(queue.get_context());

  __scope_guard release_handles([&] {
      for (int i = 0;i < world; ++ i) {
        if (i != rank) zeCheck(zeMemCloseIpcHandle(l0_ctx, peer_bases[i]));
      }
  });

  auto e = testSimpleTransmit<test_type>(
      {sycl::range<1>(1), sycl::range<1>(16)},
      input, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
      alloc_size, rank, world, queue
  );
  extract_profiling<test_type>(e);
}
