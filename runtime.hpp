#include <CL/sycl.hpp>
#include <iostream>

template <int ndev>
sycl::device getDevice() {
  static auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
  static auto dev = devs[ndev];
  return dev;
}

template <int ndev, int nsub>
sycl::device getSubDevice() {
  auto dev = getDevice<ndev>();
  try {
    static auto subs = dev.template create_sub_devices<
      sycl::info::partition_property::partition_by_affinity_domain>(
          sycl::info::partition_affinity_domain::numa);

    return subs[nsub];
  } catch (sycl::exception &e) {
    std::cout<<e.what()<<std::endl;
    return dev;
  };
}

template <int ndev, int nsub>
sycl::queue getQueue() {
  static sycl::queue q(
      getSubDevice<ndev, nsub>(),
      sycl::property_list {
        sycl::property::queue::enable_profiling(),
        sycl::property::queue::in_order()
      });
  return q;
}

sycl::queue currentQueue(int ndev, int nsub) {
  switch(ndev) {
  case 0:
    if (nsub == 0) return getQueue<0,0>();
    else return getQueue<0,1>();
  case 1:
    if (nsub == 0) return getQueue<1,0>();
    else return getQueue<1,1>();
  case 2:
    if (nsub == 0) return getQueue<2,0>();
    else return getQueue<2,1>();
  case 3:
    if (nsub == 0) return getQueue<3,0>();
    else return getQueue<3,1>();
  }
  throw std::exception();
}

sycl::device currentDevice(int ndev) {
  switch(ndev) {
  case 0:
    return getDevice<0>();
  case 1:
    return getDevice<1>();
  case 2:
    return getDevice<2>();
  case 3:
    return getDevice<3>();
  }
  throw std::exception();
}

sycl::device currentSubDevice(int ndev, int nsub) {
  switch(ndev) {
  case 0:
    if (nsub == 0) return getSubDevice<0,0>();
    else return getSubDevice<0,1>();
  case 1:
    if (nsub == 0) return getSubDevice<1,0>();
    else return getSubDevice<1,1>();
  case 2:
    if (nsub == 0) return getSubDevice<2,0>();
    else return getSubDevice<2,1>();
  case 3:
    if (nsub == 0) return getSubDevice<3,0>();
    else return getSubDevice<3,1>();
  }
  throw std::exception();
}

static uint32_t g_dev_num = 1;
static uint32_t g_part_num = 0;

sycl::device currentSubDevice() {
  return currentSubDevice(g_dev_num, g_part_num);
}

sycl::queue currentQueue() {
  return currentQueue(g_dev_num, g_part_num);
}
