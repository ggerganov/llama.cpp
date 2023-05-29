#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

constant int k_digits [[function_constant(0)]];

kernel void kernel_add(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    dst[gid] = src0[gid] + src1[gid];
}

kernel void kernel_relu(
        device const float * src,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    dst[gid] = max(0.0f, src[gid]);
}

kernel void kernel_soft_max(
        device const float * src,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    float max = 0.0f;
    for (int i = 0; i < k_digits; i++) {
        max = MAX(max, src[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < k_digits; i++) {
        dst[i] = exp(src[i] - max);
        sum += dst[i];
    }
    for (int i = 0; i < k_digits; i++) {
        dst[i] /= sum;
    }
}
