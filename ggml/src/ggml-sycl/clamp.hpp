#ifndef GGML_SYCL_CLAMP_HPP
#define GGML_SYCL_CLAMP_HPP

#include "common.hpp"

void ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_CLAMP_HPP
