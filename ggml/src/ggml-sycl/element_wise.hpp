#ifndef GGML_SYCL_ELEMENTWISE_HPP
#define GGML_SYCL_ELEMENTWISE_HPP

#include "common.hpp"

static __dpct_inline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __dpct_inline__ float op_add(const float a, const float b) {
    return a + b;
}

static __dpct_inline__ float op_sub(const float a, const float b) {
    return a - b;
}

static __dpct_inline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __dpct_inline__ float op_div(const float a, const float b) {
    return a / b;
}

inline void ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream);

inline void ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream);

inline void ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                    const ggml_tensor *src1, ggml_tensor *dst,
                                    const float *src0_dd, const float *src1_dd,
                                    float *dst_dd,
                                    const queue_ptr &main_stream);

inline void ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream);

inline void ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const queue_ptr &main_stream);

inline void ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                     const ggml_tensor *src1, ggml_tensor *dst,
                                     const float *src0_dd, const float *src1_dd,
                                     float *dst_dd,
                                     const queue_ptr &main_stream);

inline void ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream);

inline void ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream);

inline void ggml_sycl_op_log(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream);

inline void ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream);

inline void ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream);

inline void ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream);

inline void ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream);

inline void ggml_sycl_op_step(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream);

inline void ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd, const queue_ptr &main_stream);

inline void ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                    const ggml_tensor *src1, ggml_tensor *dst,
                                    const float *src0_dd, const float *src1_dd,
                                    float *dst_dd,
                                    const queue_ptr &main_stream);

inline void ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream);

inline void ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 const float *src0_dd, const float *src1_dd,
                                 float *dst_dd,
                                 const queue_ptr &main_stream);

inline void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream);

inline void ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const queue_ptr &main_stream);


void ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_sin(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_cos(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_acc(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_gelu(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);     

void ggml_sycl_silu(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_tanh(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_relu(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);


void ggml_sycl_exp(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_log(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_neg(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_step(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_sqr(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_pad(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_add(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_sycl_div(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

#endif // GGML_SYCL_ELEMENTWISE_HPP