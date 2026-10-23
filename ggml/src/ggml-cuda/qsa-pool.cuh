#pragma once

#include "common.cuh"
#include "convert.cuh"

#if defined(GGML_USE_HIP)

static __device__ __forceinline__ float qsa_pool_sum4(float a, float b, float c, float d) {
#pragma clang fp reassociate(off)
    const float ab = a + b;
    const float abc = ab + c;
    return abc + d;
}

static __global__ void qsa_pool_f16(const half * cache, const int32_t * ids, float * dst,
                                   int n_rows, float scale, float bias) {
    const int d = threadIdx.x;
    const int32_t r0 = ids[0], r1 = ids[1], r2 = ids[2], r3 = ids[3];
    if (r0 < 0 || r0 >= n_rows || r1 < 0 || r1 >= n_rows || r2 < 0 || r2 >= n_rows || r3 < 0 || r3 >= n_rows) {
        dst[d] = NAN;
        return;
    }
    const float sum = qsa_pool_sum4(
            ggml_cuda_cast<float>(cache[size_t(r0)*256 + d]), ggml_cuda_cast<float>(cache[size_t(r1)*256 + d]),
            ggml_cuda_cast<float>(cache[size_t(r2)*256 + d]), ggml_cuda_cast<float>(cache[size_t(r3)*256 + d]));
    dst[d] = scale * ggml_cuda_cast<float>(sum) + bias;
}

#endif
