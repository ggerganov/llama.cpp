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
        device const float * src0,
        device       float * dst,
        uint gid[[thread_position_in_grid]]) {
    dst[gid] = max(0.0f, src0[gid]);
}

// TODO: broken
kernel void kernel_soft_max(
        device const float * src0,
        device       float * dst) {
    float max = 0.0f;
    for (int i = 0; i < nsoftmax; i++) {
        max = MAX(max, src0[i]);
    }
    float sum = 0.0f;
    for (int i = 0; i < nsoftmax; i++) {
        dst[i] = exp(src0[i] - max);
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
    const int i = gid;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q4_0(
            (device const block_q4_0 *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_rms_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        uint gid[[thread_position_in_grid]]) {
    device const float * x = (device const float *) ((device const char *) src0 + gid*nb01);

    float sum = 0.0f;
    for (int i00 = 0; i00 < ne00; i00++) {
        sum += x[i00] * x[i00];
    }

    const float mean  = sum/ne00;
    const float scale = 1.0f/sqrt(mean + eps);

    device float * y = dst + gid*ne00;
    for (int i00 = 0; i00 < ne00; i00++) {
        y[i00] = x[i00] * scale;
    }
}
