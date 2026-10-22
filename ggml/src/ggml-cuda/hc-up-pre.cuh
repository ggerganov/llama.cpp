#pragma once

#include "common.cuh"
#include "convert.cuh"

#if defined(GGML_USE_HIP)

template <bool write_gates = false>
static __global__ void hc_up_pre_bf16_f32(
        const nv_bfloat16 * weights, const float * lo, const float * xn, float * dst, float * gates = nullptr) {
    __shared__ float2 lo_pair[160];
    __shared__ float gate[8];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int col = 2 * blockIdx.x + warp % 2;
    const int row = (warp / 2) * 2560 + col;

    if (tid < 160) {
        lo_pair[tid] = reinterpret_cast<const float2 *>(lo)[tid];
    }
    __syncthreads();

    const int * w_pair = reinterpret_cast<const int *>(weights + int64_t(row) * 320);
    float partial[5];
#pragma unroll
    for (int v = 0; v < 5; ++v) {
        const int pair = lane + 32 * v;
        const int w = w_pair[pair];
        const float2 x = lo_pair[pair];
        const float w0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&w)[0]);
        const float w1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&w)[1]);
        partial[v] = 0.0f;
        ggml_cuda_mad(partial[v], w0, x.x);
        ggml_cuda_mad(partial[v], w1, x.y);
        partial[v] = warp_reduce_sum<32>(partial[v]);
    }

    float sum = 0.0f;
#pragma unroll
    for (int v = 0; v < 5; ++v) {
        if (lane == v) {
            sum = partial[v];
        }
    }
    sum = warp_reduce_sum<32>(sum);
    if (lane == 0) {
        gate[warp] = sum;
        if constexpr (write_gates) {
            gates[row] = sum;
        }
    }
    __syncthreads();

    if (tid < 2) {
        const int d = 2 * blockIdx.x + tid;
        float mixed = 0.0f;
        for (int c = 0; c < 4; ++c) {
            const float xv = xn[c * 2560 + d];
            const float wv = 1.0f / (1.0f + expf(-gate[2 * c + tid]));
            mixed += xv * wv;
        }
        dst[d] = 0.25f * mixed;
    }
}

#endif
