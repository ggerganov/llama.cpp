#include <stdint.h>
#include <cuda_fp16.h>
#include "ggml-cuda.h"

typedef uint16_t ggml_fp16_t;
static_assert(sizeof(__half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define QK4_0 32
typedef struct {
    float   d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(float) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    float   d;              // delta
    float   m;              // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(float) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK4_2 16
typedef struct {
    __half d;               // delta
    uint8_t qs[QK4_2 / 2];  // nibbles / quants
} block_q4_2;
static_assert(sizeof(block_q4_2) == sizeof(ggml_fp16_t) + QK4_2 / 2, "wrong q4_2 block size/padding");


static __global__ void dequantize_block_q4_0(const void * vx, float * y) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int i = blockIdx.x;

    const float d = x[i].d;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_0; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = (vi0 - 8)*d;
        const float v1 = (vi1 - 8)*d;

        y[i*QK4_0 + l + 0] = v0;
        y[i*QK4_0 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q4_1(const void * vx, float * y) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int i = blockIdx.x;

    const float d = x[i].d;
    const float m = x[i].m;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_1; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = vi0*d + m;
        const float v1 = vi1*d + m;

        y[i*QK4_1 + l + 0] = v0;
        y[i*QK4_1 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q4_2(const void * vx, float * y) {
    const block_q4_2 * x = (const block_q4_2 *) vx;

    const int i = blockIdx.x;

    const float d = x[i].d;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_2; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = (vi0 - 8)*d;
        const float v1 = (vi1 - 8)*d;

        y[i*QK4_2 + l + 0] = v0;
        y[i*QK4_2 + l + 1] = v1;
    }
}

extern "C" {
    __host__ void dequantize_row_q4_0_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
        const int nb = k / QK4_0;
        dequantize_block_q4_0<<<nb, 1, 0, stream>>>(vx, y);
    }

    __host__ void dequantize_row_q4_1_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
        const int nb = k / QK4_1;
        dequantize_block_q4_1<<<nb, 1, 0, stream>>>(vx, y);
    }

    __host__ void dequantize_row_q4_2_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
        const int nb = k / QK4_2;
        dequantize_block_q4_2<<<nb, 1, 0, stream>>>(vx, y);
    }
}
