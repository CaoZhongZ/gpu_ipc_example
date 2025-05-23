#include <mpi.h>
#include <sys/mman.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include "ipc_exchange.h"

#include "cxxopts.hpp"
#include "utils.hpp"
#include "ze_exception.hpp"
#include "sycl_misc.hpp"
#include "utils.hpp"

#include "allreduce.hpp"
#include "allgather.hpp"

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
void extract_profiling(sycl::event e, int rank) {
  e.wait();
  auto start = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  std::cout<<"["<<rank<<"] Running time: "<<(end - start)<<"ns"<<std::endl;
};

using test_type = sycl::ext::oneapi::bfloat16;

template <typename T>
static void benchAllreduce(
    std::string transmitType,
    sycl::nd_range<1> launchParam,
    T* input, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t simd, uint32_t flag, sycl::queue queue,
    bool verify = false
) {
  size_t alloc_size = nelems * sizeof(T);
  size_t interm_size = world * 8 * 1024 * 1024; // rough estimation

  auto* host_init = (test_type *) sycl::malloc_host(alloc_size, queue);
  auto* host_verify = (test_type *)sycl::malloc_host(interm_size * 2, queue);

  __scope_guard free_pointers([&]{
      free(host_init, queue);
      free(host_verify, queue);
  });

  auto e = testTransmit<test_type>(
      transmitType, launchParam,
      input, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
      nelems, rank, world, flag, simd, queue
  );

  e.wait();
  // extract_profiling<test_type>(e);
  if (verify) {
    queue.memcpy(host_verify, ipcbuf0, interm_size * 2);
    queue.memcpy(host_init, input, alloc_size).wait();

    verifyTransmit<test_type>(
        host_verify, host_init, flag, rank, world, simd, nelems
    );
    std::cout<<std::dec;
    return;
  }

  testTransmit<test_type>(
      transmitType, launchParam,
      input, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
      nelems, rank, world, flag + 100, simd, queue
  );

  testTransmit<test_type>(
      transmitType, launchParam,
      input, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
      nelems, rank, world, flag + 150, simd, queue
  );

  if (rank == 0)
    std::cout<<"---------last run------------------"<<std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  e = testTransmit<test_type>(
      transmitType, launchParam,
      input, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
      nelems, rank, world, flag + 200, simd, queue
  );
  extract_profiling<test_type>(e, rank);
}

template <typename T>
static void benchAllgather(
    std::string transmitType,
    sycl::nd_range<1> launchParam,
    T* input, T* output, T* ipcbuf0, T* ipcbuf1,
    T* const peerbuf0[], T* const peerbuf1[], size_t nelems,
    int rank, int world, uint32_t simd, uint32_t flag, sycl::queue queue,
    bool verify = false
) {
  size_t alloc_size = nelems * sizeof(T);
  auto* host_init = (test_type *) sycl::malloc_host(alloc_size * world, queue);

  __scope_guard free_pointers([&]{
      free(host_init, queue);
  });

  auto e = testAllgather<test_type>(
      transmitType, launchParam,
      input, output, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
      nelems, rank, world, flag, simd, queue
  );

  e.wait();
  // extract_profiling<test_type>(e);
  if (verify) {
    queue.memcpy(host_init, output, alloc_size * world).wait();

    verifyAllgather<test_type>(
        host_init, rank, world, nelems
    );
    std::cout<<std::dec;
    return;
  }

  testAllgather<test_type>(
      transmitType, launchParam,
      input, output, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
      nelems, rank, world, flag + 100, simd, queue
  );

  testAllgather<test_type>(
      transmitType, launchParam,
      input, output, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
      nelems, rank, world, flag + 150, simd, queue
  );

  if (rank == 0)
    std::cout<<"---------last run------------------"<<std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  e = testAllgather<test_type>(
      transmitType, launchParam,
      input, output, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
      nelems, rank, world, flag + 200, simd, queue
  );
  extract_profiling<test_type>(e, rank);
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
     cxxopts::value<size_t>()->default_value("4"))
    ("f,flag", "Transmit flag identify steps",
     cxxopts::value<size_t>()->default_value("0xe00f100f"))
    ("s,simd", "Transmit flag identify steps",
     cxxopts::value<uint32_t>()->default_value("16"))
    ("v,verify", "Do verification or performance",
     cxxopts::value<bool>()->default_value("false"))
    ("a,algo", "Which algorithm is tested",
     cxxopts::value<std::string>()->default_value("small"))
    ("i,instance", "Instance offset, used for multiple instances",
     cxxopts::value<uint32_t>()->default_value("0"))
    ("l,devices", "Device lists for communication",
     cxxopts::value<std::string>()->default_value("012345678"))
    ("p,pcie", "Use PCIE to do the communication",
     cxxopts::value<bool>()->default_value("false"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto nelems = parse_nelems(parsed_opts["nelems"].as<std::string>());
  auto groups = parsed_opts["groups"].as<size_t>();
  auto subgroups = parsed_opts["subgroups"].as<size_t>();
  auto flag = parsed_opts["flag"].as<size_t>();
  auto simd = parsed_opts["simd"].as<uint32_t>();
  auto verify = parsed_opts["verify"].as<bool>();
  auto algo = parsed_opts["algo"].as<std::string>();
  auto instance = parsed_opts["instance"].as<uint32_t>();
  auto devices = parsed_opts["devices"].as<std::string>();
  auto pcie = parsed_opts["pcie"].as<bool>();

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

  size_t alloc_size = nelems * sizeof(test_type);
  size_t interm_size = world * 8 * 1024 * 1024; // rough estimation
  int device = devices[rank] - '0';
  auto queue = currentQueue(device / 2, device & 1);

  auto* input = (test_type *)sycl::malloc_device(alloc_size, queue);
  //
  // We need double buffer for both scatter and gather
  // We only need single IPC exchange
  //
  auto* host_init = (test_type *) sycl::malloc_host(alloc_size, queue);

  auto p2p = pcie ? false : canAccessPeer(rank, world);

  if (!pcie && !p2p)
    std::cout<<"Request p2p but not support, revert back to host"<<std::endl;

  auto* ipcbuf0 = !p2p ? (test_type *)sycl::malloc_host(interm_size * 2, queue)
	  : (test_type *)sycl::malloc_device(interm_size * 2, queue);
  auto* ipcbuf1 = (test_type *)((uintptr_t)ipcbuf0 + interm_size);

  __scope_guard free_pointers([&]{
      free(host_init, queue);
      free(ipcbuf0, queue);
  });

  fill_pattern(host_init, rank, nelems);
  queue.memset(ipcbuf0, 0, interm_size * 2);
  queue.memcpy(input, host_init, alloc_size);

  void *peer_bases[world];
  size_t offsets[world];
  auto ipc_handle = open_all_ipc_mems(
      queue, ipcbuf0, rank, world, peer_bases, offsets, instance
  );

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

  // barrier
  queue.wait();
  MPI_Barrier(MPI_COMM_WORLD);

  if (subgroups % 4 != 0) {
    throw std::logic_error("Subgroup numbers must be multiple of 4");
  }

  auto local_size = subgroups * simd;
  auto global_size = groups * local_size;

  if (algo.find("allgather") != std::string::npos) {
    auto* output = (test_type *)sycl::malloc_device(alloc_size * world, queue);
    queue.memset(output, 1, alloc_size * world);
    __scope_guard free_pointers([&]{
        free(output, queue);
    });

    benchAllgather(
        algo,
        {sycl::range<1>(global_size), sycl::range<1>(local_size)},
        input, output, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
        nelems, rank, world, simd, flag, queue, verify);
  } else {
    benchAllreduce(
        algo,
        {sycl::range<1>(global_size), sycl::range<1>(local_size)},
        input, ipcbuf0, ipcbuf1, peerbuf0, peerbuf1,
        nelems, rank, world, simd, flag, queue, verify);
  }

  return 0;
}
