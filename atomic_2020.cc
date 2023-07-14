#include <cassert>
#include <CL/sycl.hpp>

#include "cxxopts.hpp"
#include "sycl_misc.hpp"

template <typename T>
using __slm__ = sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>;

template <typename T>
class atomic_ref : public sycl::atomic_ref<T,
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

class global_sync {
public:

  void mark_group_finished(sycl::nd_item<1> pos, sycl::local_ptr<bool> finished) const {
    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);

    if (pos.get_local_linear_id() == 0) {
      auto flag = atomic_ref(flags_[0]);
      int num_group_finished = flag++;
      finished[0] = (num_group_finished == pos.get_group_range(0) - 1);
      // cout<<num_group_finished<<'\n';
    }

    sycl::group_barrier(pos.get_group(), sycl::memory_scope::work_group);
  }

  void operator() (sycl::nd_item<1> pos) const {
    mark_group_finished(pos, finished_);

    if (pos.get_local_linear_id() == 0 && finished_[0])
      cout<<"Show on ["<<pos.get_group()[0]<<"]\n";
  }

  global_sync(int *flags, __slm__<bool> finished, sycl::stream s)
    : flags_(flags), finished_(finished), cout(s)
  {}

  static void run(sycl::queue q, size_t group_num) {
    // TODO: RAII
    auto* flags = sycl::aligned_alloc_device<int>(4096, group_num, q);
    q.submit([&](sycl::handler &h) {
        __slm__<bool> finished(sycl::range<1> {1}, h);
        sycl::stream s(4096, 32, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1> {group_num * 32},sycl::range<1> {32}),
            global_sync(flags, finished, s));
    });
    free(flags, q);
  }
private:
  int* flags_;
  __slm__<bool> finished_;
  sycl::stream cout;
};

int main(int argc, char **argv) {
  cxxopts::Options opts("atomic_2020", "Atomic global sync test");
  opts.add_options()
    ("d,dev", "Device", cxxopts::value<uint32_t>()->default_value("1"))
    ("g,group_number", "Group numbers", cxxopts::value<uint32_t>()->default_value("1"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto dev_num = parsed_opts["dev"].as<uint32_t>();
  auto grp_num = parsed_opts["group_number"].as<uint32_t>();

  auto q = currentQueue(dev_num/ 2, dev_num & 1);
  global_sync::run(q, grp_num);
}
