#ifndef GGML_SYCL_ARGSORT_HPP
#define GGML_SYCL_ARGSORT_HPP

#include "common.hpp"

void ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_ARGSORT_HPP
