#pragma once

#include "common.cuh"
#include "convert.cuh"

#if defined(GGML_USE_HIP)

static __global__ void qsa_mask_f16(
        const int32_t * selected, const float * scores, const int32_t * top, const int32_t * tail,
        const half * causal, half * dst, int n_kv, int n_pool, int n_top) {
#pragma clang fp contract(off)
    __shared__ unsigned int bitmap[256];
    __shared__ unsigned int invalid;
    const int tid = threadIdx.x;
    const int n_sel = 4*n_top + 3;
    bitmap[tid] = 0;
    if (tid == 0) invalid = 0;
    __syncthreads();

    for (int slot = tid; slot < n_sel; slot += 256) {
        const int32_t cell = selected[slot];
        float live;
        if (slot < 4*n_top) {
            const int32_t pool = top[slot/4];
            if (pool < 0 || pool >= n_pool) {
                atomicExch(&invalid, 1u);
                continue;
            }
            live = fminf(fmaxf(scores[pool] + 1.0f, 0.0f), 1.0f);
        } else {
            const int32_t cell_tail = tail[slot - 4*n_top];
            if (cell_tail < 0 || cell_tail > n_kv) {
                atomicExch(&invalid, 1u);
                continue;
            }
            live = fminf(fmaxf(-float(cell_tail) + float(n_kv), 0.0f), 1.0f);
        }
        if (cell < 0 || cell > n_kv) {
            atomicExch(&invalid, 1u);
            continue;
        }
        const float dump = float(slot + 1) + float(n_kv - 1);
        const float delta = float(cell) - dump;
        const float product = delta * live;
        const int32_t index = int32_t(product + dump);
        if (index >= 0 && index < n_kv) {
            atomicOr(&bitmap[index/32], 1u << (index%32));
        }
    }
    __syncthreads();

    for (int cell = tid; cell < n_kv; cell += 256) {
        const float value = (bitmap[cell/32] & (1u << (cell%32))) ? 0.0f : -INFINITY;
        dst[cell] = ggml_cuda_cast<half>(invalid ? NAN : value + ggml_cuda_cast<float>(causal[cell]));
    }
}

#endif
