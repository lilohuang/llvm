// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx90a %s -o %t-gfx90a
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1102 %s -o %t-gfx1102

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
  // Compile-time query to validate the matrix parameters
  using myparams = matrix_params<architecture::amd_gpu_gfx90a, int8_t, int8_t,
                                 int32_t, int32_t, 32, 32, 8>;

  static_assert(myparams::M == 32);
  static_assert(myparams::N == 32);
  static_assert(myparams::K == 8);

  // Sizes-only compile-time query: types are given, generate default sizes
  using myparams2 = matrix_params<architecture::amd_gpu_gfx90a, int8_t, int8_t,
                                  int32_t, int32_t>;
  static_assert(myparams2::M == 16);
  static_assert(myparams2::N == 16);
  static_assert(myparams2::K == 4);

  using gfx11_params = matrix_params<architecture::amd_gpu_gfx1102, int8_t,
                                     int8_t, int32_t, int32_t, 16, 16, 16>;
  static_assert(gfx11_params::M == 16);
  static_assert(gfx11_params::N == 16);
  static_assert(gfx11_params::K == 16);

  using gfx11_defaults =
      matrix_params<architecture::amd_gpu_gfx1102, half, half, float, float>;
  static_assert(gfx11_defaults::M == 16);
  static_assert(gfx11_defaults::N == 16);
  static_assert(gfx11_defaults::K == 16);

  return 0;
};
