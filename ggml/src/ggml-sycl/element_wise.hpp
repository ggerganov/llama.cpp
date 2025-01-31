#ifndef GGML_SYCL_ELEMENTWISE_HPP
#define GGML_SYCL_ELEMENTWISE_HPP

#include "common.hpp"

void ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_ELEMENTWISE_HPP
