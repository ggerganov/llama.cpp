#ifndef GGML_SYCL_SUM_HPP
#define GGML_SYCL_SUM_HPP

#include "common.hpp"

void ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_SUM_HPP
