#pragma once

#include <sycl/sycl.hpp>
#include <gen_visa_templates.hpp>

#define divUp(x, m)  \
  ((x + m -1) / m)

#define alignUp(x, c) \
  (divUp(x, c) * c)

#include "small_transmit.hpp"
#include "simple_transmit.hpp"
#include "throughput.hpp"
