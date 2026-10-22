#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "unary.cuh"

#if defined(GGML_USE_HIP)

template <bool write_dots = false>
static __global__ void hc_down_inject_bf16_f32(
        const nv_bfloat16 * down, const nv_bfloat16 * inject, const float * xn,
        float * lo, float * coeff, float * dots = nullptr) {
    __shared__ float partial[32];
    const int tid = threadIdx.x;
    const bool is_inject = blockIdx.x >= 320;
    const int row = is_inject ? blockIdx.x - 320 : blockIdx.x;
    const int * w = reinterpret_cast<const int *>((is_inject ? inject : down) + int64_t(row) * 10240);
    const float2 * x = reinterpret_cast<const float2 *>(xn);
    if (tid < 32) {
        partial[tid] = 0.0f;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int col2 = tid; col2 < 5120; col2 += 256) {
        const int pair = w[col2];
        const float2 xv = x[col2];
        const float w0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&pair)[0]);
        const float w1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&pair)[1]);
        ggml_cuda_mad(sum, w0, xv.x);
        ggml_cuda_mad(sum, w1, xv.y);
    }
    sum = warp_reduce_sum<32>(sum);
    if (tid % 32 == 0) {
        partial[tid / 32] = sum;
    }
    __syncthreads();
    if (tid < 32) {
        sum = warp_reduce_sum<32>(partial[tid]);
    }
    if (tid == 0) {
        if constexpr (write_dots) {
            dots[blockIdx.x] = sum;
        }
        sum *= 0.25f;
        sum = is_inject ? 1.0f / (1.0f + expf(-sum)) : ggml_cuda_op_silu_single(sum);
        sum *= is_inject ? 2.0f : 1.0f;
        (is_inject ? coeff : lo)[row] = sum;
    }
}

#endif
