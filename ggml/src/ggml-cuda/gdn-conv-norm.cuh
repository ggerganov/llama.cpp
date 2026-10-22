#pragma once

#include "common.cuh"
#include "unary.cuh"

#if defined(GGML_USE_HIP)

template <bool write_intermediates = false, bool direct_history = false>
static __global__ void gdn_conv_norm_f32(
        const float * x, const float * w, float * silu, float * q, float * k,
        float eps_q, float eps_k, float scale_out, float * raw = nullptr, float * norms = nullptr,
        float * history = nullptr, const int32_t * state_id = nullptr) {
    __shared__ float values_norm[128];
    __shared__ float scale_norm;
    const int tid = threadIdx.x;
    const int group = blockIdx.x;
    const int channel = group*128 + tid;
    if constexpr (direct_history) {
        if (*state_id != 0) {
            if (tid < 128) {
                silu[channel] = NAN;
                if (group < 32) (group < 16 ? q : k)[(group % 16)*128 + tid] = NAN;
            }
            return;
        }
    }
    float y = 0.0f;
    if (tid < 128) {
        float values[4];
        if constexpr (direct_history) {
            values[0] = history[channel*3 + 0];
            values[1] = history[channel*3 + 1];
            values[2] = history[channel*3 + 2];
            values[3] = x[channel];
        }
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumf += (direct_history ? values[j] : x[channel*4 + j]) * w[channel*4 + j];
        }
        sumf += 0.0f;
        if constexpr (direct_history) {
            history[channel*3 + 0] = values[1];
            history[channel*3 + 1] = values[2];
            history[channel*3 + 2] = values[3];
        }
        if constexpr (write_intermediates) raw[channel] = sumf;
        y = ggml_cuda_op_silu_single(sumf);
        silu[channel] = y;
    }
    if (group < 32) {
        if (tid < 128) values_norm[tid] = y;
        __syncthreads();
        if (tid < 32) {
            const float4 v = make_float4(values_norm[4*tid], values_norm[4*tid + 1], values_norm[4*tid + 2], values_norm[4*tid + 3]);
            float tmp = 0.0f;
            tmp += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
            tmp = block_reduce<block_reduce_method::SUM, 32>(tmp, static_cast<float *>(nullptr));
            if (tid == 0) scale_norm = rsqrtf(tmp / 128 + (group < 16 ? eps_q : eps_k));
        }
        __syncthreads();
        const float scale = scale_norm;
        if (tid < 128) {
            if constexpr (write_intermediates) norms[channel] = scale*y;
            float * dst = group < 16 ? q : k;
            dst[(group % 16)*128 + tid] = scale_out * (scale*y);
        }
    }
}

#endif
