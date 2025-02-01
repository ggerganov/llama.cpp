#ifndef GGML_SYCL_ARGMAX_HPP
#define GGML_SYCL_ARGMAX_HPP

#include "common.hpp"

void ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_ARGMAX_HPP