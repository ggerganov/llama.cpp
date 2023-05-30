#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define QK4_0 32
#define QR4_0 2
typedef struct {
    half    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

static void dequantize_row_q4_0(device const block_q4_0 * x, device float * y, int k) {
    const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const half d = x[i].d;

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

// TODO: not needed
constant int nsoftmax [[function_constant(0)]];

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

// TODO: broken
kernel void kernel_soft_max(
        device const float * src,
        device       float * dst) {
    float max = 0.0f;
    for (int i = 0; i < nsoftmax; i++) {
        max = MAX(max, src[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < nsoftmax; i++) {
        dst[i] = exp(src[i] - max);
        sum += dst[i];
    }
    for (int i = 0; i < nsoftmax; i++) {
        dst[i] /= sum;
    }
}

kernel void kernel_get_rows_q4_0(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint gid[[thread_position_in_grid]]) {
    device const block_q4_0 * src = (device const block_q4_0 *)src0;

    const int i = gid;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q4_0(
            (device const block_q4_0 *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}
