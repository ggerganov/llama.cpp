#pragma once

#include "common.cuh"

#if defined(GGML_USE_HIP)

static __device__ __forceinline__ float moe_shared_tail_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float moe_shared_tail_add(float routed, float shared, float gate, float & product) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    product = shared * gate;
    return routed + product;
}

template <bool diagnostics>
static __global__ void moe_shared_tail_f32(const float * gate_input, const float * shared, const float * routed,
                                          float * dst, float * gate_output, float * product_output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2560) return;
    const float gate = moe_shared_tail_sigmoid(gate_input[0]);
    float product;
    dst[i] = moe_shared_tail_add(routed[i], shared[i], gate, product);
    if constexpr (diagnostics) {
        if (i == 0) gate_output[0] = gate;
        product_output[i] = product;
    }
}

#endif
