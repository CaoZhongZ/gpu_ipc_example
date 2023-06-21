#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include "ze_exception.hpp"

int main(int argc, char* arg[]) {
  cxxopts::Options opts("IPC exchange example", "Exchange IPC handle to cross rank");
  opts.allow_unrecognised_options();
  opts.add_options()
    ("i,size", "GPU allocation size", cxxopts::value<size_t>()->default_value("8192"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto allocSize = parsed_opts["size"].as<size_t>();

  zeCheck(zeInit(0));
}
