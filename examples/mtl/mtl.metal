#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define QK4_0 32
#define QR4_0 2
typedef struct {
    half    d;             // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
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
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig];
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_mul(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig % ne00];
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
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
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
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
        uint tpig[[thread_position_in_grid]]) {
    device const float * x = (device const float *) ((device const char *) src0 + tpig*nb01);

    float sum = 0.0f;
    for (int i00 = 0; i00 < ne00; i00++) {
        sum += x[i00] * x[i00];
    }

    const float mean  = sum/ne00;
    const float scale = 1.0f/sqrt(mean + eps);

    device float * y = dst + tpig*ne00;
    for (int i00 = 0; i00 < ne00; i00++) {
        y[i00] = x[i00] * scale;
    }
}

kernel void kernel_mul_mat_q4_0(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2  tpig[[thread_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2  tptg[[threads_per_threadgroup]]) {
    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    const int qk = QK4_0;
    const int nb = ne00/qk;

    device const block_q4_0 * x = (device const block_q4_0 *) (src0) + r0*nb;
    device const float      * y = (device const float      *) (src1) + r1*ne10;

    threadgroup float sum[32]; // TODO: should be equal to threadgroup size
    sum[tpitg.x] = 0.0f;

    for (int i = 0; i < nb; i += tptg.x) {
        device const uint4  * x0p = (device const  uint4 *) (x + i)->qs;
        device const float4 * y0p = (device const float4 *) (y + i*qk);

        const uint4 x0 = *x0p;

        const uint4 x0l = (x0 & uint4(0x0F0F0F0F));
        const uint4 x0h = (x0 & uint4(0xF0F0F0F0)) >> 4;

        thread const char * x0lsb = (thread const char *) &x0l;
        thread const char * x0hsb = (thread const char *) &x0h;

        const float4 y00 = *(y0p + 0);
        const float4 y01 = *(y0p + 1);
        const float4 y02 = *(y0p + 2);
        const float4 y03 = *(y0p + 3);
        const float4 y04 = *(y0p + 4);
        const float4 y05 = *(y0p + 5);
        const float4 y06 = *(y0p + 6);
        const float4 y07 = *(y0p + 7);

        const half d = (x + i)->d;

        sum[tpitg.x] += (
                (x0lsb[ 0] - 8)*y00[0] + (x0lsb[ 1] - 8)*y00[1] + (x0lsb[ 2] - 8)*y00[2] + (x0lsb[ 3] - 8)*y00[3] +
                (x0lsb[ 4] - 8)*y01[0] + (x0lsb[ 5] - 8)*y01[1] + (x0lsb[ 6] - 8)*y01[2] + (x0lsb[ 7] - 8)*y01[3] +
                (x0lsb[ 8] - 8)*y02[0] + (x0lsb[ 9] - 8)*y02[1] + (x0lsb[10] - 8)*y02[2] + (x0lsb[11] - 8)*y02[3] +
                (x0lsb[12] - 8)*y03[0] + (x0lsb[13] - 8)*y03[1] + (x0lsb[14] - 8)*y03[2] + (x0lsb[15] - 8)*y03[3] +
                (x0hsb[ 0] - 8)*y04[0] + (x0hsb[ 1] - 8)*y04[1] + (x0hsb[ 2] - 8)*y04[2] + (x0hsb[ 3] - 8)*y04[3] +
                (x0hsb[ 4] - 8)*y05[0] + (x0hsb[ 5] - 8)*y05[1] + (x0hsb[ 6] - 8)*y05[2] + (x0hsb[ 7] - 8)*y05[3] +
                (x0hsb[ 8] - 8)*y06[0] + (x0hsb[ 9] - 8)*y06[1] + (x0hsb[10] - 8)*y06[2] + (x0hsb[11] - 8)*y06[3] +
                (x0hsb[12] - 8)*y07[0] + (x0hsb[13] - 8)*y07[1] + (x0hsb[14] - 8)*y07[2] + (x0hsb[15] - 8)*y07[3]
                ) * d;
    }

    // accumulate the sum from all threads in the threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tptg.x/2; i > 0; i /= 2) {
        if (tpitg.x < i) {
            sum[tpitg.x] += sum[tpitg.x + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dst[r1*ne0 + r0] = sum[0];
}
