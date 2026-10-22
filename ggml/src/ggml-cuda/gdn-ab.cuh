#pragma once

#include "common.cuh"
#include "convert.cuh"

#if defined(GGML_USE_HIP)

template <bool write_dots = false>
static __global__ void gdn_ab_bf16_f32(
        const nv_bfloat16 * alpha, const nv_bfloat16 * beta, const float * x,
        const float * dt, const float * a, float * gate, float * sigmoid, float * dots = nullptr) {
    __shared__ float partial[32];
    const int tid = threadIdx.x;
    const bool is_beta = blockIdx.x >= 48;
    const int row = is_beta ? blockIdx.x - 48 : blockIdx.x;
    const int * w = reinterpret_cast<const int *>((is_beta ? beta : alpha) + int64_t(row) * 2560);
    const float2 * input = reinterpret_cast<const float2 *>(x);
    if (tid < 32) {
        partial[tid] = 0.0f;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int col2 = tid; col2 < 1280; col2 += 256) {
        const int pair = w[col2];
        const float2 value = input[col2];
        const float w0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&pair)[0]);
        const float w1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&pair)[1]);
        ggml_cuda_mad(sum, w0, value.x);
        ggml_cuda_mad(sum, w1, value.y);
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
        if (is_beta) {
            sigmoid[row] = 1.0f / (1.0f + expf(-sum));
        } else {
            sum += dt[row];
            sum = sum > 20.0f ? sum : logf(1.0f + expf(sum));
            gate[row] = sum * a[row];
        }
    }
}

#endif
