#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

kernel void kernel_add(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig];
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig % ne00];
}

kernel void kernel_mul(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig];
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_mul_row(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig % ne00];
}

kernel void kernel_scale(
        device const float * src0,
        device       float * dst,
        constant     float & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_silu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    float x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

constant float GELU_COEF_A    = 0.044715f;
constant float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

kernel void kernel_gelu(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    float x = src0[tpig];

    // BEWARE !!!
    // Simply using "tanh" instead of "precise::tanh" will sometimes results in NaNs!
    // This was observed with Falcon 7B and 40B models
    //
    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_soft_max(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        threadgroup float  * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    device const float * psrc0 = src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    device       float * pdst  = dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    // parallel max
    buf[tpitg[0]] = -INFINITY;
    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        buf[tpitg[0]] = MAX(buf[tpitg[0]], psrc0[i00]);
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg[0]/2; i > 0; i /= 2) {
        if (tpitg[0] < i) {
            buf[tpitg[0]] = MAX(buf[tpitg[0]], buf[tpitg[0] + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // broadcast
    if (tpitg[0] == 0) {
        buf[0] = buf[0];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float max = buf[0];

    // parallel sum
    buf[tpitg[0]] = 0.0f;
    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        buf[tpitg[0]] += exp(psrc0[i00] - max);
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg[0]/2; i > 0; i /= 2) {
        if (tpitg[0] < i) {
            buf[tpitg[0]] += buf[tpitg[0] + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // broadcast
    if (tpitg[0] == 0) {
        buf[0] = buf[0];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sum = buf[0];

    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        pdst[i00] = exp(psrc0[i00] - max) / sum;
    }
}

kernel void kernel_diag_mask_inf(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant       int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i02 = tpig[2];
    const int64_t i01 = tpig[1];
    const int64_t i00 = tpig[0];

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

kernel void kernel_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * sum [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * x = (device const float *) ((device const char *) src0 + tgpig*nb01);
    // MEAN
    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += x[i00];
    }
    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // broadcast
    if (tpitg == 0) {
        sum[0] /= ne00;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean  = sum[0];

    // recenter
    device float * y = dst + tgpig*ne00;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = x[i00] - mean;
    }

    // VARIANCE
    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += y[i00] * y[i00];
    }
    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // broadcast
    if (tpitg == 0) {
        sum[0] /= ne00;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float variance = sum[0];

    const float scale = 1.0f/sqrt(variance + eps);
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = y[i00] * scale;
    }
}


kernel void kernel_rms_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * sum [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float4 * x = (device const float4 *) ((device const char *) src0 + tgpig*nb01);
    device const float * x_scalar = (device const float *) x;
    float4 sumf=0;
    float all_sum=0;

    // parallel sum
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    if (tiisg == 0) {
        sum[sgitg] = all_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // broadcast, simd group number is ntg / 32
    for (uint i = ntg / 32 / 2; i > 0; i /= 2) {
       if (tpitg < i) {
           sum[tpitg] += sum[tpitg + i];
       }
    }
    if (tpitg == 0) {
        for (int i = 4 * (ne00 / 4); i < ne00; i++) {sum[0] += x_scalar[i];}
        sum[0] /= ne00;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean  = sum[0];
    const float scale = 1.0f/sqrt(mean + eps);

    device float4 * y = (device float4 *) (dst + tgpig*ne00);
    device float * y_scalar = (device float *) y;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
    if (tpitg == 0) {
        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {y_scalar[i00] = x_scalar[i00] * scale;}
    }
}

kernel void kernel_mul_mat_f16_f32(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        threadgroup float  * sum [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tpig[[thread_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    device const half  * x = (device const half  *) (src0 + r0*nb01 + im/(ne12/ne02)*nb02);
    device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);

    sum[tpitg.x] = 0.0f;

    for (int i = tpitg.x; i < ne00; i += tptg.x) {
        sum[tpitg.x] += (float) x[i] * (float) y[i];
    }

    // accumulate the sum from all threads in the threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tptg.x/2; i > 0; i /= 2) {
        if (tpitg.x < i) {
            sum[tpitg.x] += sum[tpitg.x + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tpitg.x == 0) {
        dst[im*ne1*ne0 + r1*ne0 + r0] = sum[0];
    }
}

kernel void kernel_alibi_f32(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        constant      float & m0,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
    float m_k = pow(m0, i2 + 1);
    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0] + m_k * (i00 - ne00 + 1);
    }
}

kernel void kernel_rope(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        constant       int & n_past,
        constant       int & n_dims,
        constant       int & mode,
        constant     float & freq_base,
        constant     float & freq_scale,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i3 = tpig[2];
    const int64_t i2 = tpig[1];
    const int64_t i1 = tpig[0];

    const bool is_neox = mode & 2;
    const float theta_scale = pow(freq_base, -2.0f/n_dims);

    const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);

    float theta = freq_scale * (float)p;

    if (!is_neox) {
        for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cos(theta);
            const float sin_theta = sin(theta);

            theta *= theta_scale;

            device const float * const src = (device float *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       float * dst_data  = (device float *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        }
    } else {
        for (int64_t ib = 0; ib < ne0/n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
                const float cos_theta = cos(theta);
                const float sin_theta = sin(theta);

                theta *= theta_scale;

                const int64_t i0 = ib*n_dims + ic/2;

                device const float * const src = (device float *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                device       float * dst_data  = (device float *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                const float x0 = src[0];
                const float x1 = src[n_dims/2];

                dst_data[0]        = x0*cos_theta - x1*sin_theta;
                dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
            }
        }
    }
}

kernel void kernel_cpy_f16_f16(
        device const half * src0,
        device       half * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const half * src = (device half *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f16(
        device const float * src0,
        device        half * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f32(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

//============================================ quant blocks ======================================================

#define QK4_0 32
#define QR4_0 2
typedef struct {
    half    d;             // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK4_1 32
typedef struct {
    half d;          // delta
    half m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;

#define QK8_0 32
typedef struct {
    half    d;         // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

#ifndef QK_K
#define QK_K 256
#else
static_assert(QK_K == 256 || QK_K == 64, "QK_K must be 256 or 64");
#endif

#if QK_K == 256
#define K_SCALE_SIZE 12
#else
#define K_SCALE_SIZE 4
#endif

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half d;           // super-block scale for quantized scales
    half dmin;        // super-block scale for quantized mins
} block_q2_K;
// 84 bytes / block

typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
#if QK_K == 64
    uint8_t scales[2];
#else
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
#endif
    half d;             // super-block scale
} block_q3_K;

#if QK_K == 64
typedef struct {
    half    d[2];          // super-block scales/mins
    uint8_t scales[2];
    uint8_t qs[QK_K/2];    // 4-bit quants
} block_q4_K;
#else
typedef struct {
    half d;             // super-block scale for quantized scales
    half dmin;          // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
#endif

#if QK_K == 64
typedef struct {
    half  d;                     // super-block scales/mins
    int8_t  scales[QK_K/16];     // 8-bit block scales
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
#else
typedef struct {
    half d;                      // super-block scale for quantized scales
    half dmin;                   // super-block scale for quantized mins
    uint8_t scales[3*QK_K/64];   // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
// 176 bytes / block
#endif

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half d;                  // super-block scale
} block_q6_K;
// 210 bytes / block

//============================================ quant drivers ======================================================
// load quantized blocks from device/threadgroup memory, dequantize 16 weights to a half4x4 or float4x4 type
// init(il) : prepare some values that can be reused as long as il doesn't change.
// dequantize(...) : dequantize 16 continuous weights.
// inner_product_pre(il, yl) : multiply yl elements by a factor to speed up inner product calculations.
// inner_product(...) : do inner product, may not use continuous weights.

static inline void fix_y_v1(thread float & sumy, thread float4x4 & yl) {
    sumy = 0.f;
    for (int i = 0; i < 8; i += 2) {
        sumy += yl[  i/4][i%4]; sumy += yl[  i/4][i%4+1];
        sumy += yl[2+i/4][i%4]; sumy += yl[2+i/4][i%4+1];
        yl[i/4  ][i%4  ] =            yl[  i/4][i%4];
        yl[i/4  ][i%4+1] = 1/256.f  * yl[  i/4][i%4+1];
        yl[i/4+2][i%4  ] = 1/16.f   * yl[2+i/4][i%4];
        yl[i/4+2][i%4+1] = 1/4096.f * yl[2+i/4][i%4+1];
    }
}

static inline void fix_y_v2(thread float & coef1, thread float & coef2, thread float & sumy, thread float4x4 & yl) {
    sumy = 0.f;
    for (int i = 0; i < 16; i += 2) {
        sumy += yl[i/4][i%4];
        sumy += yl[i/4][i%4+1];
        yl[i/4][i%4]   = coef1 * yl[i/4][i%4];
        yl[i/4][i%4+1] = coef2 * yl[i/4][i%4+1];
    }
}

template <typename addr_uint16_p,typename addr_block_q_p, typename type4x4>
class q4_0_driver {
    public:
        uint16_t mask1, mask2, q_offset;
        float coef1, coef2, sumy;

        void init(int il) {
            mask1 = il ? 0x00F0 : 0x000F; mask2 = mask1 << 8;
            coef1 = il ? 1/16.f : 1.f;    coef2 = coef1 / 256.f;
            q_offset = il ? 4 : 0;
        }

        void inner_product_pre(int il, thread float4x4 & yl){
            fix_y_v1(sumy, yl);
        }

        void inner_product(addr_block_q_p xb, int il, thread float4x4 & yl, thread float & sum){
            const half d = xb->d;
            addr_uint16_p q = (addr_uint16_p)xb->qs + q_offset;
            for (int i = 0; i < 8; i += 2) {
                sum += yl[i/4  ][i%4]   * (q[i/2] & 0x000F);
                sum += yl[i/4  ][i%4+1] * (q[i/2] & 0x0F00);
                sum += yl[i/4+2][i%4]   * (q[i/2] & 0x00F0);
                sum += yl[i/4+2][i%4+1] * (q[i/2] & 0xF000);
            }
            sum = d * (sum - 8.f * sumy);
        }

        void dequantize(addr_block_q_p xb, int il, thread type4x4 & reg) {
            const half d = xb->d;
            addr_uint16_p q = (addr_uint16_p)xb->qs;
            for (int i = 0; i < 16; i += 2) {
                reg[i/4][i%4]   = d * (coef1 * (q[i/2] & mask1) - 8.f);
                reg[i/4][i%4+1] = d * (coef2 * (q[i/2] & mask2) - 8.f);
            }
        }
};

template <typename addr_uint16_p,typename addr_block_q_p, typename type4x4>
class q4_1_driver {
    public:
        uint16_t mask1, mask2, q_offset;
        float coef1, coef2, sumy;

        void init(int il) {
            mask1 = il ? 0x00F0 : 0x000F; mask2 = mask1 << 8;
            coef1 = il ? 1/16.f : 1.f;    coef2 = coef1 / 256.f;
            q_offset = il ? 4 : 0;
        }

        void inner_product_pre(int il, thread float4x4 & yl){
            fix_y_v1(sumy, yl);
        }

        void inner_product(addr_block_q_p xb, int il, thread float4x4 & yl, thread float & sum){
            const half d = xb->d;
            const half m = xb->m;
            addr_uint16_p q = (addr_uint16_p)xb->qs + q_offset;
            for (int i = 0; i < 8; i += 2) {
                sum += yl[i/4  ][i%4]   * (q[i/2] & 0x000F);
                sum += yl[i/4  ][i%4+1] * (q[i/2] & 0x0F00);
                sum += yl[i/4+2][i%4]   * (q[i/2] & 0x00F0);
                sum += yl[i/4+2][i%4+1] * (q[i/2] & 0xF000);
            }
            sum = d * sum + m * sumy;
        }

        void dequantize(addr_block_q_p xb, int il, thread type4x4 & reg) {
            const half d = xb->d;
            const half m = xb->m;
            addr_uint16_p q = (addr_uint16_p)xb->qs;
            for (int i = 0; i < 16; i += 2) {
                reg[i/4][i%4]   = d * coef1 * (q[i/2] & mask1) + m;
                reg[i/4][i%4+1] = d * coef2 * (q[i/2] & mask2) + m;
            }
        }
};

template <typename addr_uint16_p,typename addr_block_q_p, typename type4x4>
class q8_0_driver {
    public:
        uint16_t mask1, mask2, q_offset;
        float coef1, coef2, sumy;

        void init(int il) {
            q_offset = il * 16;
        }

        void inner_product_pre(int il, thread float4x4 & yl){
        }

        void inner_product(addr_block_q_p xb, int il, thread float4x4 & yl, thread float & sum){
            const half d = xb->d;
            for (int i = 0; i < 16; i++) {
                sum += yl[i/4][i%4] * xb->qs[i + q_offset];
            }
            sum = d * sum;
        }

        void dequantize(addr_block_q_p xb, int il, thread type4x4 & reg) {
            const half d = xb->d;
            for (int i = 0; i < 16; i++) {
                reg[i/4][i%4] = (xb->qs[i + q_offset] * d);
            }
        }
};

template <typename addr_uint16_p,typename addr_block_q_p, typename type4x4>
class f16_driver {
    public:
        void init(int il) {}

        void inner_product_pre(int il, thread float4x4 & yl) {}

        void inner_product(addr_block_q_p xb, int il, thread float4x4 & yl, thread float & sum){
            half4x4 temp = *xb;
            for (int i = 0; i < 16; i++){
                sum += yl[i/4][i%4] * temp[i/4][i%4];
            }
        }

        void dequantize(addr_block_q_p xb, int il, thread type4x4 & reg) {
            half4x4 temp = *xb;
            for (int i = 0; i < 16; i++){
                reg[i/4][i%4] = temp[i/4][i%4];
            }
        }
};

template <typename addr_uint16_p,typename addr_block_q_p, typename type4x4>
class q2_K_driver {
    public:
        uint16_t mask1, mask2, q_offset;
        float coef1, coef2, sumy;

        void init(int il) {
            #if QK_K == 256
                q_offset = 16*(il/8) + 8*(il&1);
                il = (il/2)%4;
            #else
                q_offset = 0;
            #endif
            coef1 = il>1 ? (il>2 ? 1/64.f : 1/16.f) : (il>0 ? 1/4.f : 1.f); coef2 = coef1/256.f;
            mask1 = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);   mask2 = mask1 << 8;
        }

        void get_scales(addr_block_q_p xb, int il, thread float & dl, thread float & ml) {
            const float d = (float)(xb->d);
            const float min = (float)(xb->dmin);
            dl = d * (xb->scales[il] & 0xF), ml = min * (xb->scales[il] >> 4);
        }

        void inner_product_pre(int il, thread float4x4 & yl){
            fix_y_v2(coef1, coef2, sumy, yl);
        }

        void inner_product(addr_block_q_p xb, int il, thread float4x4 & yl, thread float & sum){
            float dl, ml;
            get_scales(xb, il, dl, ml);
            addr_uint16_p q = (addr_uint16_p)xb->qs + q_offset;
            for (int i = 0; i < 16; i += 2) {
                sum += yl[i/4][i%4  ] * (q[i/2] & mask1);
                sum += yl[i/4][i%4+1] * (q[i/2] & mask2);
            }
            sum = dl * sum - ml * sumy;
        }

        void dequantize(addr_block_q_p xb, int il, thread type4x4 & reg) {
            float dl, ml;
            get_scales(xb, il, dl, ml);
            addr_uint16_p q = (addr_uint16_p)xb->qs + q_offset;
            for (int i = 0; i < 16; i += 2) {
                reg[i/4][i%4  ] = coef1 * dl * (q[i/2] & mask1) - ml;
                reg[i/4][i%4+1] = coef2 * dl * (q[i/2] & mask2) - ml;
            }
        }
};

template <typename addr_uint16_p,typename addr_block_q_p, typename type4x4>
class q3_K_driver {
    public:
        uint16_t m1, m2, kmask1, kmask2 ,mask1, mask2;
        float coef1, coef2, sumy;
        float4x4 yl_str;
        uint16_t q_offset, h_offset, d_loc1, d_loc2;

        void init(int il) {
#if QK_K == 256
            d_loc1 = 8 + il%4; d_loc2 = il%8;
            q_offset = 16 * (il/8) + 8 * (il&1); h_offset = 8 * (il&1);
            kmask1 = (il/4)>1   ? ((il/4)>2   ? 192    : 48)     : ((il/4)>0 ? 12  : 3);       kmask2 = il/8 ? 0xF0 : 0x0F;
            m1 = 1 << (il/2); m2 = m1 << 8;
            il = (il/2)%4;
#else
            m1 = 1 << (il*2); m2 = m1 << 8;
            q_offset = 0;     h_offset = 0;
            kmask1 = il&1 ? 0xF0 : 0x0F;
            d_loc1 = il/2;;
#endif
            coef1  = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h); coef2  = coef1 / 256.h;
            mask1  = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);   mask2  = mask1 << 8;
        }

        void get_scales(addr_block_q_p xb, int il, thread float & dl) {
            const half d_all = xb->d;
#if QK_K == 256
            uint16_t scale_1 = xb->scales[d_loc1], scale_2 = xb->scales[d_loc2];
            int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2) : \
                                         (scale_2&kmask2) | ((scale_1&kmask1) << 4);
            dl = il < 8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
#else
            float kcoef = il&1 ? 1.f/16.f : 1.f;
            dl = d_all * ((xb->scales[d_loc1] & kmask1) * kcoef - 8);
#endif
        }

        void inner_product_pre(int il, thread float4x4 & yl){
            fix_y_v2(coef1, coef2, sumy, yl);
        }

        void inner_product(addr_block_q_p xb, int il, thread float4x4 & yl, thread float & sum){
            float dl;
            get_scales(xb, il, dl);
            addr_uint16_p q = (addr_uint16_p)xb->qs + q_offset;
            addr_uint16_p h = (addr_uint16_p)xb->hmask + h_offset;
#if QK_K == 256
            for (int i = 0; i < 16; i += 2) {
                sum += yl[i/4][i%4  ] * ((q[i/2] & mask1) - ((h[i/2] & m1) ? 0 : 4.f/coef1));
                sum += yl[i/4][i%4+1] * ((q[i/2] & mask2) - ((h[i/2] & m2) ? 0 : 4.f/coef2));
            }
#else
            for (int i = 0; i < 8; i += 2) {
                sum +=yl[i/4  ][i%4  ] * ((q[i/2  ] & mask1) - (h[i/2] & m1     ? 0 : 4.f/coef1));
                sum +=yl[i/4  ][i%4+1] * ((q[i/2  ] & mask2) - (h[i/2] & m2     ? 0 : 4.f/coef2));
                sum +=yl[i/4+2][i%4  ] * ((q[i/2+4] & mask1) - (h[i/2] & (2*m1) ? 0 : 4.f/coef1));
                sum +=yl[i/4+2][i%4+1] * ((q[i/2+4] & mask2) - (h[i/2] & (2*m2) ? 0 : 4.f/coef2));
            }
#endif
            sum = dl * sum;
        }

        void dequantize(addr_block_q_p xb, int il, thread type4x4 & reg) {
            float dl;
            get_scales(xb, il, dl);
            addr_uint16_p q = (addr_uint16_p)xb->qs + q_offset;
            addr_uint16_p h = (addr_uint16_p)xb->hmask + h_offset;
#if QK_K == 256
            for (int i = 0; i < 16; i += 2) {
                reg[i/4][i%4]   = coef1 * dl * ((q[i/2] & mask1) - ((h[i/2] & m1) ? 0 : 4.f/coef1));
                reg[i/4][i%4+1] = coef2 * dl * ((q[i/2] & mask2) - ((h[i/2] & m2) ? 0 : 4.f/coef2));
            }
#else
            for (int i = 0; i < 8; i += 2) {
                reg[i/4  ][i%4  ] = coef1 * dl * ((q[i/2  ] & mask1) - (h[i/2] & m1     ? 0 : 4.f/coef1));
                reg[i/4  ][i%4+1] = coef2 * dl * ((q[i/2  ] & mask2) - (h[i/2] & m2     ? 0 : 4.f/coef2));
                reg[i/4+2][i%4  ] = coef1 * dl * ((q[i/2+4] & mask1) - (h[i/2] & (2*m1) ? 0 : 4.f/coef1));
                reg[i/4+2][i%4+1] = coef2 * dl * ((q[i/2+4] & mask2) - (h[i/2] & (2*m2) ? 0 : 4.f/coef2));
            }
#endif
        }
};

template <typename addr_uint16_p,typename addr_block_q_p, typename type4x4>
class q4_K_driver {
    public:
        uint16_t d_mask1, d_mask2, m_mask1, mask1, mask2;
        float coef1, coef2, sumy1, sumy2;
        uint16_t d_loc1, d_loc2, m_loc1, m_loc2, q_offset;

        void init(int il) {
            q_offset = (il/4) * 16 + 4 * (il%4);
            d_mask1 = il < 8 ? 0x3F3F : 0x0F0F;      d_mask2 = il < 8 ? 0x0000 : 0xC0C0;
            d_loc1  = il < 8 ? il/4   : il/4 + 2;    d_loc2  = il < 8 ? il/4   : il/4 - 2;
            m_mask1 = il < 8 ? 0x3F3F : 0xF0F0;
            m_loc1  = il/4 + 2;                      m_loc2  = il/4;
        }

        void get_scales(addr_block_q_p xb, int il, thread float & dl1, thread float & ml1, thread float & dl2, thread float & ml2) {
 #if QK_K == 256
            const float    d = (float)(xb->d);
            const float  min = (float)(xb->dmin);
            addr_uint16_p sc = (addr_uint16_p)xb->scales;
            uint16_t d_int = (sc[d_loc1] & d_mask1) | ((sc[d_loc2] & d_mask2) >> 2);
            uint16_t m_int = il < 8 ? (sc[m_loc1] & m_mask1) : ((sc[m_loc1] & m_mask1) >> 4);
            m_int = m_int | ((sc[m_loc2] & d_mask2) >> 2);
            dl1 = as_type<uchar2>(d_int)[0] * d, ml1 = as_type<uchar2>(m_int)[0] * min;
            dl2 = as_type<uchar2>(d_int)[1] * d, ml2 = as_type<uchar2>(m_int)[1] * min;
#else
            dl1 = (float)(xb->d[0]) * (xb->scales[0]&0xF); dl2 = (float)(xb->d[0]) * (xb->scales[1]&0xF);
            ml1 = (float)(xb->d[1]) * (xb->scales[0]>>4);  ml2 = (float)(xb->d[1]) * (xb->scales[1]>>4);
#endif
        }

        void get_scales2(addr_block_q_p xb, int il, thread float & dl, thread float & ml) {
            q_offset = (il/4) * 16 + 8 * (il&1);
            mask1 = (il%4) < 2 ? 0x000F : 0x00F0; mask2 = mask1 << 8;
            coef1 = (il%4) < 2 ? 1.f    : 1/16.f; coef2 = coef1 / 256.f;
#if QK_K == 256
            d_mask1 = il < 8 ? 63   : 0x0F;       d_mask2 = il < 8 ? 0    : 192;
            d_loc1  = il < 8 ? il/2 : 4 + il/2;   d_loc2  = il < 8 ? il/2 : il/2 - 4;
            m_mask1 = il < 8 ? 63 : 0xF0;
            m_loc1  = il/2 + 4;                   m_loc2  = il/2;
            const float d = (float)(xb->d);
            const float min = (float)(xb->dmin);
            uint16_t d_int = (xb->scales[d_loc1] & d_mask1) | ((xb->scales[d_loc2] & d_mask2) >> 2);
            uint16_t m_int = il < 8 ? (xb->scales[m_loc1] & m_mask1) : ((xb->scales[m_loc1] & m_mask1) >> 4);
            m_int = m_int | ((xb->scales[m_loc2] & d_mask2) >> 2);
            dl = d_int * d, ml = m_int * min;
#else
            dl = il<2 ? (float)(xb->d[0]) * (xb->scales[0]&0xF) : (float)(xb->d[0]) * (xb->scales[1]&0xF);
            ml = il<2 ? (float)(xb->d[1]) * (xb->scales[0]>>4)  : (float)(xb->d[1]) * (xb->scales[1]>>4);
#endif
        }

        void inner_product_pre(int il, thread float4x4 & yl){
            sumy1 = 0.f; sumy2 = 0.f;
            for (int i = 0; i < 8; i += 2) {
                sumy1 += yl[i/4  ][i%4]; sumy1 += yl[i/4  ][i%4+1];
                sumy2 += yl[2+i/4][i%4]; sumy2 += yl[2+i/4][i%4+1];
                yl[i/4  ][i%4  ] = yl[i/4][i%4];
                yl[i/4  ][i%4+1] = 1/256.f  * yl[i/4][i%4+1];
                yl[i/4+2][i%4  ] = 1/16.f   * yl[2+i/4][i%4];
                yl[i/4+2][i%4+1] = 1/4096.f * yl[2+i/4][i%4+1];
            }
        }

        void inner_product(addr_block_q_p xb, int il, thread float4x4 & yl, thread float & sum){
            float dl1, ml1, dl2, ml2;
            float sum2 = 0.f;
            get_scales(xb, il, dl1, ml1, dl2, ml2);
            addr_uint16_p q = (addr_uint16_p)xb->qs + q_offset;
            for (int i = 0; i < 8; i += 2) {
                sum  += yl[i/4  ][i%4  ] * ((q[i/2]&0x000F));
                sum  += yl[i/4  ][i%4+1] * ((q[i/2]&0x0F00));
                sum2 += yl[i/4+2][i%4  ] * ((q[i/2]&0x00F0));
                sum2 += yl[i/4+2][i%4+1] * ((q[i/2]&0xF000));
            }
            sum = dl1 * sum - ml1 * sumy1 + dl2 * sum2 - ml2 * sumy2;
        }

        void dequantize(addr_block_q_p xb, int il, thread type4x4 & reg) {
            float dl, ml;
            get_scales2(xb, il, dl, ml);
            addr_uint16_p q = (addr_uint16_p)xb->qs + q_offset;
            for (int i = 0; i < 16; i += 2) {
                reg[i/4][i%4]   = coef1 * dl * (q[i/2] & mask1) - ml;
                reg[i/4][i%4+1] = coef2 * dl * (q[i/2] & mask2) - ml;
            }
        }
};

template <typename addr_uint16_p,typename addr_block_q_p, typename type4x4>
class q5_K_driver {
    public:
        uint16_t m1, m2, d_mask1, d_mask2, m_mask1, mask1, mask2;
        float coef1, coef2, sumy;
        uint16_t d_loc1, d_loc2, m_loc1, m_loc2, q_offset, h_offset;

        void init(int il) {
            d_mask1 = il < 8 ? 63 : 0x0F;           d_mask2 = il < 8 ? 0 : 192;
            d_loc1  = il < 8 ? il/2 : 4 + il/2;     d_loc2  = il < 8 ? il/2 : il/2 - 4;
            m_mask1 = il < 8 ? 63 : 0xF0;
            m_loc1  = il/2 + 4;                     m_loc2  = il/2;
            mask1   = (il%4) < 2 ? 0x000F : 0x00F0; mask2 = mask1 << 8;
            coef1   = (il%4) < 2 ? 1.f : 1/16.f;    coef2 = coef1 / 256.f;
#if QK_K == 256
            q_offset = (il/4) * 16 + 8 * (il&1);  h_offset = 8 * (il&1);
            m1 = 1 << (il/2);                     m2 = m1 << 8;
#else
            q_offset = 8 * (il&1);                h_offset = 0;
            m1 = 1 << (il*2);                     m2 = m1 << 8;
#endif
        }

        void get_scales(addr_block_q_p xb, int il, thread float & dl, thread float & ml) {
#if QK_K == 256
            uint16_t d_int = (xb->scales[d_loc1] & d_mask1) | ((xb->scales[d_loc2] & d_mask2) >> 2);
            uint16_t m_int = il < 8 ? (xb->scales[m_loc1] & m_mask1) : ((xb->scales[m_loc1] & m_mask1) >> 4);
            m_int = m_int | ((xb->scales[m_loc2] & d_mask2) >> 2);
            dl = d_int * xb->d, ml = m_int * xb->dmin;
#else
            dl = (float)(xb->d) * xb->scales[il]; ml = 0.f;
#endif
        }

        void inner_product_pre(int il, thread float4x4 & yl){
            fix_y_v2(coef1, coef2, sumy, yl);
        }

        void inner_product(addr_block_q_p xb, int il, thread float4x4 & yl, thread float & sum){
            float dl, ml;
            get_scales(xb, il, dl, ml);
            addr_uint16_p q = (addr_uint16_p)xb->qs + q_offset;
            addr_uint16_p h = (addr_uint16_p)xb->qh + h_offset;
#if QK_K == 256
            for (int i = 0; i < 16; i += 2) {
                sum += yl[i/4][i%4  ] * ((q[i/2] & mask1) + ((h[i/2] & m1) ? 16.f/coef1 : 0));
                sum += yl[i/4][i%4+1] * ((q[i/2] & mask2) + ((h[i/2] & m2) ? 16.f/coef2 : 0));
            }
#else
            for (int i = 0; i < 8; i += 2) {
                sum += yl[i/4  ][i%4  ] * ((q[i/2  ] & mask1) - (h[i/2] & m1     ? 0 : 16.f/coef1));
                sum += yl[i/4  ][i%4+1] * ((q[i/2  ] & mask2) - (h[i/2] & m2     ? 0 : 16.f/coef2));
                sum += yl[i/4+2][i%4  ] * ((q[i/2+4] & mask1) - (h[i/2] & (2*m1) ? 0 : 16.f/coef1));
                sum += yl[i/4+2][i%4+1] * ((q[i/2+4] & mask2) - (h[i/2] & (2*m2) ? 0 : 16.f/coef2));
            }
#endif
            sum = dl * sum - ml * sumy;
        }

        void dequantize(addr_block_q_p xb, int il, thread type4x4 & reg) {
            float dl, ml;
            get_scales(xb, il, dl, ml);
            addr_uint16_p q = (addr_uint16_p)xb->qs + q_offset;
            addr_uint16_p h = (addr_uint16_p)xb->qh + h_offset;
#if QK_K == 256
            for (int i = 0; i < 16; i += 2) {
                reg[i/4][i%4  ] = coef1 * dl * ((q[i/2] & mask1) + (h[i/2] & m1 ? 16.f/coef1 : 0)) - ml;
                reg[i/4][i%4+1] = coef2 * dl * ((q[i/2] & mask2) + (h[i/2] & m2 ? 16.f/coef2 : 0)) - ml;
            }
#else
            for (int i = 0; i < 8; i += 2) {
                reg[i/4  ][i%4  ] = coef1 * dl * ((q[i/2  ] & mask1) - (h[i/2] & m1     ? 0 : 16.f/coef1));
                reg[i/4  ][i%4+1] = coef2 * dl * ((q[i/2  ] & mask2) - (h[i/2] & m2     ? 0 : 16.f/coef2));
                reg[i/4+2][i%4  ] = coef1 * dl * ((q[i/2+4] & mask1) - (h[i/2] & (2*m1) ? 0 : 16.f/coef1));
                reg[i/4+2][i%4+1] = coef2 * dl * ((q[i/2+4] & mask2) - (h[i/2] & (2*m2) ? 0 : 16.f/coef2));
            }
#endif
        }
};

template <typename addr_uint16_p,typename addr_block_q_p, typename type4x4>
class q6_K_driver {
    public:
        uint16_t hmask1, hmask2, lmask1, lmask2;
        float coef1, coef2, h_coef, sumy1, sumy2;
        uint16_t d_loc, q_offset, h_offset;

        void init(int il) {
            d_loc    = il;
#if QK_K == 256
            q_offset = 32*(il/8) + 8*(il%4); h_offset = 16*(il/8) + 8*(il&1);
            il = (il/2)%4;
#else
            q_offset = 8 * (il&1);           h_offset = 0;
#endif
            hmask1 = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3), hmask2 = hmask1 << 8;
            lmask1 = il>1 ? 0xF0              : 0x0F,            lmask2 = lmask1 << 8;
            h_coef = il&1 ? 4.h : 16.h;
            coef1  = il>1 ? 1.f/16.f          : 1.f,             coef2  = coef1 / 256.f;
        }

        void get_scales(addr_block_q_p xb, int il, thread float & dl) {
            dl = (float)(xb->d) * xb->scales[d_loc];
        }

        void inner_product_pre(int il, thread float4x4 & yl){
            sumy1 = 0.f; sumy2 = 0.f;
            for (int i = 0; i < 8; i += 2) {
                sumy1 += yl[i/4  ][i%4]; sumy1 += yl[i/4  ][i%4+1];
                sumy2 += yl[2+i/4][i%4]; sumy2 += yl[2+i/4][i%4+1];
                yl[i/4  ][i%4  ] = yl[i/4][i%4];
                yl[i/4  ][i%4+1] = 1/256.f  * yl[i/4][i%4+1];
                yl[i/4+2][i%4  ] = 1/16.f   * yl[2+i/4][i%4];
                yl[i/4+2][i%4+1] = 1/4096.f * yl[2+i/4][i%4+1];
            }
#if QK_K == 256
            q_offset = 32*(il/8)  + 4*(il%8); h_offset = 16*(il/8)  + 4*(il%4);
            hmask1 = (il%8)<4 ? 3 : 12;       hmask2 = hmask1 << 8;
            h_coef = (il%8)<4 ? 16.f : 4.f;   d_loc = 8*(il/8) + (il%8)/2;
#else
            q_offset = 4*il;                  h_offset = 4*(il&1);
            hmask1 = il<2 ? 3 : 12;           hmask2 = hmask1 << 8;
            h_coef = il<2 ? 16.f : 4.f;       d_loc = il/2;
#endif
        }

        void inner_product(addr_block_q_p xb, int il, thread float4x4 & yl, thread float & sum){
            float dl1, dl2;
            float sum2 = 0.f;
#if QK_K == 256
            dl1 = (float)(xb->d) * xb->scales[d_loc]; dl2 = (float)(xb->d) * xb->scales[d_loc+4];
#else
            dl1 = (float)(xb->d) * xb->scales[d_loc]; dl2 = (float)(xb->d) * xb->scales[d_loc+2];
#endif
            addr_uint16_p ql = (addr_uint16_p)xb->ql + q_offset;
            addr_uint16_p qh = (addr_uint16_p)xb->qh + h_offset;
            for (int i = 0; i < 8; i+=2) {
                sum  += yl[i/4  ][i%4  ] * ((ql[i/2]&0x000F) + (qh[i/2]&hmask1)      * h_coef);
                sum  += yl[i/4  ][i%4+1] * ((ql[i/2]&0x0F00) + (qh[i/2]&hmask2)      * h_coef);
                sum2 += yl[i/4+2][i%4  ] * ((ql[i/2]&0x00F0) + (qh[i/2]&(hmask1<<4)) * h_coef);
                sum2 += yl[i/4+2][i%4+1] * ((ql[i/2]&0xF000) + (qh[i/2]&(hmask2<<4)) * h_coef);
            }
            sum = dl1 * (sum - 32.h * sumy1) + dl2 * (sum2 - 32.h * sumy2);
        }

        void dequantize(addr_block_q_p xb, int il, thread type4x4 & reg) {
            float dl;
            get_scales(xb, il, dl);
            addr_uint16_p ql = (addr_uint16_p)xb->ql + q_offset;
            addr_uint16_p qh = (addr_uint16_p)xb->qh + h_offset;
            for (int i = 0; i < 16; i+=2) {
                reg[i/4][i%4  ] = dl * (((ql[i/2]&lmask1) + (qh[i/2]&hmask1) * h_coef) * coef1 - 32.f);
                reg[i/4][i%4+1] = dl * (((ql[i/2]&lmask2) + (qh[i/2]&hmask2) * h_coef) * coef2 - 32.f);
            }
        }
};

//============================= templates and their specializations =============================
#define N_SIMDWIDTH 32

template<typename block_q_type, short nl, template<typename, typename, typename> class quant_dri>
kernel void kernel_get_rows(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint                 tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint                 tptg[[threads_per_threadgroup]]) {
    const int i = tgpig;
    const int r = ((device int32_t *) src1)[i];

    for (int ind = tiitg; ind < ne00/16; ind += tptg) {
        float4x4 temp;
        quant_dri<device const uint16_t *, device const block_q_type *, float4x4> dequan_worker;
        dequan_worker.init(ind%nl);
        dequan_worker.dequantize(
            ((device const block_q_type *) ((device char *) src0 + r*nb01)) + ind/nl, ind%nl, temp);
        *(((device float4x4 *) ((device char *) dst + i*nb1)) + ind) = temp;
    }
}

// nl: Each block has 16*nl weights
// n_shift: Each thread deals with 16 dequantized weights. However, the 16 weights may not be continuous.
//          n_shift is the difference between the address of the first 8 weights and the last 8 weights.
//          (i.e. n_shift=8 means 16 continuous weights)
template<typename block_q_type, int nr, int nsg, int nl, int n_shift, template<typename, typename, typename> class quant_dri>
kernel void kernel_mat_mv(device const   void * src0,
                          device const  float * src1,
                          device        float * dst,
                          constant    int64_t & ne00,
                          constant    int64_t & ne01,
                          constant    int64_t & ne02,
                          constant    int64_t & ne10,
                          constant    int64_t & ne12,
                          constant    int64_t & ne0,
                          constant    int64_t & ne1,
                          constant    uint    & gqa,
                          threadgroup uint   * shared_memory[[threadgroup(0)]],
                          uint3  tgpig[[threadgroup_position_in_grid]],
                          uint   tiisg[[thread_index_in_simdgroup]],
                          uint   sgitg[[simdgroup_index_in_threadgroup]]) {
    const int nb = ne00/(nl * 16);
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int ix = tiisg / nl;
    const int il = tiisg % nl;
    const short blocks_size_aligned = ((N_SIMDWIDTH / nl) * sizeof(block_q_type) + 4 * (N_SIMDWIDTH / nr) - 1) \
                                      / (4 * (N_SIMDWIDTH / nr)) * (N_SIMDWIDTH / nr);
    const short need_align_fix = ((sizeof(block_q_type) % 4) / 2) * (nb % 2) * sgitg;
    const int   first_row           = (r0 * nsg) * nr + sgitg + nsg * (tiisg / (N_SIMDWIDTH/nr));
    const uint  offset0             = first_row * nb + im/gqa*(nb*ne0);
    const uint  offset1             = r1*ne10 + im*ne00*ne1 + ix * (nl * 16) + (il/(n_shift/8))*16*(n_shift/8) + (il%(n_shift/8)) * 8;

    device const block_q_type * x    = (device const block_q_type *) src0 + offset0;
    device const float        * y    = (device const float        *) src1 + offset1;
    threadgroup  uint         * x_st = shared_memory + blocks_size_aligned * nr * sgitg \
                                       + blocks_size_aligned * (tiisg / (N_SIMDWIDTH / nr));
    threadgroup  uint16_t     * x_ld = ((threadgroup uint16_t *)(shared_memory + blocks_size_aligned * nr * sgitg)) \
                                       + need_align_fix + ix * sizeof(block_q_type) / 2;

    float4x4 yl;       // src1 vector cache
    float sumf[nr] = {0.f};

    quant_dri<threadgroup const uint16_t *, threadgroup const block_q_type *, half4x4> dequan_worker;
    dequan_worker.init(il);

    // each thread in a SIMD group deals with 16 dequantized weights.
    for (int ib = ix; ib < (nb + (N_SIMDWIDTH / nl) - 1)/(N_SIMDWIDTH / nl)*(N_SIMDWIDTH / nl) ; ib += N_SIMDWIDTH / nl) {
        #pragma unroll(MIN(blocks_size_aligned / (N_SIMDWIDTH / nr), 16))
        for (int i = tiisg % (N_SIMDWIDTH / nr); i < blocks_size_aligned; i += N_SIMDWIDTH / nr) {
            *(x_st + i) = *((device const uint *)x + i);
        }
        yl[0] = *((device const float4 *)y);
        yl[1] = *((device const float4 *)y + 1);
        yl[2] = *((device const float4 *)y + n_shift/4);
        yl[3] = *((device const float4 *)y + n_shift/4 + 1);

        dequan_worker.inner_product_pre(il, yl);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll(nr)
        for (int row = 0; row < nr; row++) {
            float sum_temp = 0.f;
            simdgroup_barrier(mem_flags::mem_none);
            dequan_worker.inner_product((threadgroup block_q_type *) \
                                (x_ld + 2 * blocks_size_aligned * row), il, yl, sum_temp);
            sumf[row] += ib<nb ? sum_temp : 0;
        }
        x  += N_SIMDWIDTH / nl;
        y  += N_SIMDWIDTH * 16;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + nsg * row < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + nsg * row] = tot;
        }
    }
}

#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix A
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// each block_q contains 16*nl weights
template<typename block_q_type, short nl, template<typename, typename, typename> class quant_dri>
kernel void kernel_mul_mm(device const  uchar * src0,
                           device const  float * src1,
                           device        float * dst,
                           constant    int64_t & ne00,
                           constant    int64_t & ne02,
                           constant    int64_t & nb01,
                           constant    int64_t & nb02,
                           constant    int64_t & ne12,
                           constant    int64_t & ne0,
                           constant    int64_t & ne1,
                           constant    uint & gqa,
                           threadgroup   uchar * shared_memory [[threadgroup(0)]],
                           uint3                 tgpig[[threadgroup_position_in_grid]],
                           uint                  tiitg[[thread_index_in_threadgroup]],
                           uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half * sa = ((threadgroup half *)shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;
    const uint im = tgpig.z;
    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;
    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_half8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);
    uint offset0 = im/gqa*nb02; ushort offset1 = il/nl;
    device const block_q_type  * x = (device const block_q_type  *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0) + offset1;
    device const float * y = src1 + (r1 * BLOCK_SIZE_N + thread_col) * ne00 \
                             + BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL) + im * ne00 * ne1;



    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        //load data and store to threadgroup memory
        half4x4 temp_a;
        quant_dri<device const uint16_t *, device const block_q_type *, half4x4> dequan_worker;
        dequan_worker.init(il);
        dequan_worker.dequantize(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll(16)
        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            + 16 * (tiitg % THREAD_PER_ROW) + 8 * (i / 8)) \
            + (tiitg / THREAD_PER_ROW) % 8 + (i & 7) * 8) = temp_a[i/4][i%4];
        }
        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) \
                = *((device float2x4 *)y);
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        //load matrices from threadgroup memory and conduct outer products
        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));
        #pragma unroll(4)
        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i],lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i],lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;
            #pragma unroll(8)
            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= ne0 && (r1 + 1) * BLOCK_SIZE_N <= ne1) {
        device float *C = dst + BLOCK_SIZE_M * r0 + 32 * (sgitg&1) \
                          + (BLOCK_SIZE_N * r1 + 16 * (sgitg>>1)) * ne0 + im*ne1*ne0;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float *temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        device float *C = dst + BLOCK_SIZE_M * r0 + (BLOCK_SIZE_N * r1) * ne0 + im*ne1*ne0;
        if (sgitg==0) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = tiitg; j< n_cols; j += BLOCK_SIZE_N) {
                    *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}

#if QK_K == 256
#define QK_NL 16
#else
#define QK_NL 4
#endif

typedef void (get_rows_t)(device const void *, device const int *, device float *, constant int64_t &, \
                          constant uint64_t &, constant uint64_t &, uint, uint, uint);

template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_t kernel_get_rows<half4x4,    1, f16_driver>;
template [[host_name("kernel_get_rows_q4_0")]] kernel get_rows_t kernel_get_rows<block_q4_0, 2, q4_0_driver>;
template [[host_name("kernel_get_rows_q4_1")]] kernel get_rows_t kernel_get_rows<block_q4_1, 2, q4_1_driver>;
template [[host_name("kernel_get_rows_q8_0")]] kernel get_rows_t kernel_get_rows<block_q8_0, 2, q8_0_driver>;
template [[host_name("kernel_get_rows_q2_K")]] kernel get_rows_t kernel_get_rows<block_q2_K, QK_NL, q2_K_driver>;
template [[host_name("kernel_get_rows_q3_K")]] kernel get_rows_t kernel_get_rows<block_q3_K, QK_NL, q3_K_driver>;
template [[host_name("kernel_get_rows_q4_K")]] kernel get_rows_t kernel_get_rows<block_q4_K, QK_NL, q4_K_driver>;
template [[host_name("kernel_get_rows_q5_K")]] kernel get_rows_t kernel_get_rows<block_q5_K, QK_NL, q5_K_driver>;
template [[host_name("kernel_get_rows_q6_K")]] kernel get_rows_t kernel_get_rows<block_q6_K, QK_NL, q6_K_driver>;

typedef void (mat_mm_t)(device const uchar *, device const float *, device float *, constant int64_t &,\
                             constant int64_t &, constant int64_t &, constant int64_t &, constant int64_t &, \
                             constant int64_t &, constant int64_t &, constant uint &, threadgroup uchar *, uint3, uint, uint);

template [[host_name("kernel_mul_mm_f16_f32")]]  kernel mat_mm_t kernel_mul_mm<half4x4,    1, f16_driver>;
template [[host_name("kernel_mul_mm_q4_0_f32")]] kernel mat_mm_t kernel_mul_mm<block_q4_0, 2, q4_0_driver>;
template [[host_name("kernel_mul_mm_q4_1_f32")]] kernel mat_mm_t kernel_mul_mm<block_q4_1, 2, q4_1_driver>;
template [[host_name("kernel_mul_mm_q8_0_f32")]] kernel mat_mm_t kernel_mul_mm<block_q8_0, 2, q8_0_driver>;
template [[host_name("kernel_mul_mm_q2_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q2_K, QK_NL, q2_K_driver>;
template [[host_name("kernel_mul_mm_q3_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q3_K, QK_NL, q3_K_driver>;
template [[host_name("kernel_mul_mm_q4_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q4_K, QK_NL, q4_K_driver>;
template [[host_name("kernel_mul_mm_q5_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q5_K, QK_NL, q5_K_driver>;
template [[host_name("kernel_mul_mm_q6_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q6_K, QK_NL, q6_K_driver>;

typedef void (mat_mv_t)(device const void *, device const float *, device float *, constant int64_t &,\
                             constant int64_t &, constant int64_t &, constant int64_t &, constant int64_t &, \
                             constant int64_t &, constant int64_t &, constant uint &, threadgroup uint *, uint3, uint, uint);

#define N_DST 4
#define N_SIMDGROUP 2
template [[host_name("kernel_mul_mv_f16_f32" )]] kernel mat_mv_t kernel_mat_mv<half4x4,    N_DST, N_SIMDGROUP, 1,     8,  f16_driver>;
template [[host_name("kernel_mul_mv_q4_0_f32")]] kernel mat_mv_t kernel_mat_mv<block_q4_0, N_DST, N_SIMDGROUP, 2,     16, q4_0_driver>;
template [[host_name("kernel_mul_mv_q4_1_f32")]] kernel mat_mv_t kernel_mat_mv<block_q4_1, N_DST, N_SIMDGROUP, 2,     16, q4_1_driver>;
template [[host_name("kernel_mul_mv_q8_0_f32")]] kernel mat_mv_t kernel_mat_mv<block_q8_0, N_DST, N_SIMDGROUP, 2,     8,  q8_0_driver>;
template [[host_name("kernel_mul_mv_q2_K_f32")]] kernel mat_mv_t kernel_mat_mv<block_q2_K, N_DST, N_SIMDGROUP, QK_NL, 8,  q2_K_driver>;
template [[host_name("kernel_mul_mv_q3_K_f32")]] kernel mat_mv_t kernel_mat_mv<block_q3_K, N_DST, N_SIMDGROUP, QK_NL, 8,  q3_K_driver>;
template [[host_name("kernel_mul_mv_q4_K_f32")]] kernel mat_mv_t kernel_mat_mv<block_q4_K, N_DST, N_SIMDGROUP, QK_NL, 32,  q4_K_driver>;
template [[host_name("kernel_mul_mv_q5_K_f32")]] kernel mat_mv_t kernel_mat_mv<block_q5_K, N_DST, N_SIMDGROUP, QK_NL, 8,  q5_K_driver>;
#if QK_K == 256
template [[host_name("kernel_mul_mv_q6_K_f32")]] kernel mat_mv_t kernel_mat_mv<block_q6_K, N_DST, N_SIMDGROUP, QK_NL, 64, q6_K_driver>;
#else
template [[host_name("kernel_mul_mv_q6_K_f32")]] kernel mat_mv_t kernel_mat_mv<block_q6_K, N_DST, N_SIMDGROUP, QK_NL, 32, q6_K_driver>;
#endif
