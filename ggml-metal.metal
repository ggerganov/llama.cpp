#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define QK4_0 32
#define QR4_0 2
typedef struct {
    half    d;             // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK4_1 32
typedef struct {
    half d;                 // delta
    half m;                 // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;

#define QK5_0 32
typedef struct {
    half d;                // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;

#define QK5_1 32
typedef struct {
    half d;                 // delta
    half m;                 // min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;

#define QK8_0 32
typedef struct {
    half    d;         // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

enum ggml_sort_order {
    GGML_SORT_ASC,
    GGML_SORT_DESC,
};

// general-purpose kernel for addition, multiplication and division of two tensors
// pros: works for non-contiguous tensors, supports broadcast across all dims
// cons: not very efficient
kernel void kernel_add(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        constant  int64_t & offs,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01 + offs;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1  + offs;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) + *((device float *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_mul(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) * *((device float *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_div(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) / *((device float *)(src1_ptr + i10*nb10));
    }
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig % nb];
}

kernel void kernel_mul_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb  [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig % nb];
}

kernel void kernel_div_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb  [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] / src1[tpig % nb];
}

kernel void kernel_scale(
        device const float * src0,
        device       float * dst,
        constant     float & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_scale_4(
        device const float4 * src0,
        device       float4 * dst,
        constant     float  & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

kernel void kernel_tanh(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = precise::tanh(x);
}

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

kernel void kernel_gelu(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    // BEWARE !!!
    // Simply using "tanh" instead of "precise::tanh" will sometimes results in NaNs!
    // This was observed with Falcon 7B and 40B models
    //
    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_quick(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_silu(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_sqr(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src0[tpig];
}

kernel void kernel_sum_rows(
        device const float * src0,
        device       float * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tpig[[thread_position_in_grid]]) {
    int64_t i3 = tpig.z;
    int64_t i2 = tpig.y;
    int64_t i1 = tpig.x;

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    device const float * src_row = (device const float *) ((device const char *) src0 + i1*nb01 + i2*nb02 + i3*nb03);
    device       float * dst_row = (device       float *) ((device       char *) dst  + i1*nb1  + i2*nb2  + i3*nb3);

    float row_sum = 0;

    for (int64_t i0 = 0; i0 < ne00; i0++) {
        row_sum += src_row[i0];
    }

    dst_row[0] = row_sum;
}

kernel void kernel_soft_max(
        device const float * src0,
        device const float * src1,
        device const float * src2,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float * psrc0 =         src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    device const float * pmask = src1 != src0 ? src1                               + i01*ne00 : nullptr;
    device const float * ppos  = src2 != src0 ? src2                                          : nullptr;
    device       float * pdst  =         dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    float slope = 0.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = -INFINITY;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        lmax = MAX(lmax, psrc0[i00]*scale + (pmask ? pmask[i00] : 0.0f) + (ppos ? slope*ppos[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        const float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? pmask[i00] : 0.0f) + (ppos ? slope*ppos[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        pdst[i00] *= inv_sum;
    }
}

kernel void kernel_soft_max_4(
        device const float * src0,
        device const float * src1,
        device const float * src2,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float4 * psrc4 =                (device const float4 *)(src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    device const float4 * pmask = src1 != src0 ? (device const float4 *)(src1 +                                      i01*ne00) : nullptr;
    device const float4 * ppos  = src2 != src0 ? (device const float4 *)(src2)                                                 : nullptr;
    device       float4 * pdst4 =                (device       float4 *)(dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float slope = 0.0f;

    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = -INFINITY;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + (pmask ? pmask[i00] : 0.0f) + (ppos ? slope*ppos[i00] : 0.0f));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + (pmask ? pmask[i00] : 0.0f) + (ppos ? slope*ppos[i00] : 0.0f)) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        pdst4[i00] *= inv_sum;
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

kernel void kernel_diag_mask_inf_8(
        device const float4 * src0,
        device       float4 * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne01,
        constant        int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {

    const int64_t i = 2*tpig[0];

    dst[i+0] = src0[i+0];
    dst[i+1] = src0[i+1];
    int64_t i4 = 4*i;
    const int64_t i02 = i4/(ne00*ne01); i4 -= i02*ne00*ne01;
    const int64_t i01 = i4/(ne00);      i4 -= i01*ne00;
    const int64_t i00 = i4;
    for (int k = 3; k >= 0; --k) {
        if (i00 + 4 + k <= n_past + i01) {
            break;
        }
        dst[i+1][k] = -INFINITY;
        if (i00 + k > n_past + i01) {
            dst[i][k] = -INFINITY;
        }
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
    const float mean  = sum[0] / ne00;

    // recenter and VARIANCE
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * y = dst + tgpig*ne00;
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = x[i00] - mean;
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
    const float variance = sum[0] / ne00;

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
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float4 * x = (device const float4 *) ((device const char *) src0 + tgpig*nb01);

    float4 sumf = 0;
    float all_sum = 0;

    // parallel sum
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = all_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        all_sum = buf[tiisg];
        all_sum = simd_sum(all_sum);
    }

    const float mean  = all_sum/ne00;
    const float scale = 1.0f/sqrt(mean + eps);

    device float4 * y = (device float4 *) (dst + tgpig*ne00);
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
}

kernel void kernel_group_norm(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int32_t & n_groups,
        constant     float & eps,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    const int64_t ne = ne00*ne01*ne02;
    const int64_t gs = ne00*ne01*((ne02 + n_groups - 1) / n_groups);

    int start = tgpig * gs;
    int end   = start + gs;

    start += tpitg;

    if (end >= ne) {
        end = ne;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += ntg) {
        tmp += src0[j];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float mean = tmp / gs;
    tmp = 0.0f;

    for (int j = start; j < end; j += ntg) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float variance = tmp / gs;
    const float scale = 1.0f/sqrt(variance + eps);
    for (int j = start; j < end; j += ntg) {
        dst[j] *= scale;
    }
}

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float2 acc = 0.f;

    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float2 acc = 0.f;

    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 2 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (acc[0] + acc[1]) + sumy * m;
}

// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float2 acc = 0.f;

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010))
                + yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[1] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100))
                + yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }
    return d * (sumy * -16.f + acc[0] + acc[1]);
}

// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float2 acc = 0.f;

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010))
                + yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[1] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100))
                + yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }
    return d * (acc[0] + acc[1]) + sumy * m;
}

// putting them in the kernel cause a significant performance penalty
#define N_DST 4        // each SIMD group works on 4 rows
#define N_SIMDGROUP 2  // number of SIMD groups in a thread group
//Note: This is a template, but strictly speaking it only applies to
//      quantizations where the block size is 32. It also does not
//      guard against the number of rows not being divisible by
//      N_DST, so this is another explicit assumption of the implementation.
template<typename block_q_type, int nr, int nsg, int nw>
void mul_vec_q_n_f32_impl(
        device const void  * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3 tgpig, uint tiisg, uint sgitg) {
    const int nb = ne00/QK4_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q_type * x = (device const block_q_type *) src0 + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16]; // src1 vector cache
    float sumf[nr] = {0.f};

    const int ix = (tiisg/2);
    const int il = (tiisg%2)*8;

    device const float * yb = y + ix * QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1]/256.f;

            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16]/16.f;
            yl[i+9] = yb[i+17]/4096.f;
        }

        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q_n_dot_y(x+ib+row*nb, sumy, yl, il);
        }

        yb += QK4_0 * 16;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[im*ne0*ne1 + r1*ne0 + first_row + row] = tot;
        }
    }
}

kernel void kernel_mul_mv_q4_0_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q4_1_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32_impl<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q5_0_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q5_1_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,tgpig,tiisg,sgitg);
}


#define NB_Q8_0 8

void kernel_mul_mv_q8_0_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = ne00/QK8_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q8_0 * x = (device const block_q8_0 *) src0 + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[NB_Q8_0];
    float sumf[nr]={0.f};

    const int ix = tiisg/4;
    const int il = tiisg%4;

    device const float * yb = y + ix * QK8_0 + NB_Q8_0*il;

    // each thread in a SIMD group deals with NB_Q8_0 quants at a time
    for (int ib = ix; ib < nb; ib += nw/4) {
        for (int i = 0; i < NB_Q8_0; ++i) {
            yl[i] = yb[i];
        }

        for (int row = 0; row < nr; row++) {
            device const int8_t * qs = x[ib+row*nb].qs + NB_Q8_0*il;
            float sumq = 0.f;
            for (int iq = 0; iq < NB_Q8_0; ++iq) {
                sumq += qs[iq] * yl[iq];
            }
            sumf[row] += sumq*x[ib+row*nb].d;
        }

        yb += NB_Q8_0 * nw;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,tgpig,tiisg,sgitg);
}

#define N_F32_F32 4

void kernel_mul_mv_f32_f32_impl(
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t rb = tgpig.y*N_F32_F32;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const float * x = (device const float *) (src0 + offset0);

    if (ne00 < 128) {
        for (int row = 0; row < N_F32_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);

            float sumf = 0;
            for (int i = tiisg; i < ne00; i += 32) {
                sumf += (float) x[i] * (float) y[i];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        device const float4 * x4 = (device const float4 *)x;
        for (int row = 0; row < N_F32_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            device const float  * y  = (device const float  *) (src1 + r1*nb11 + im*nb12);
            device const float4 * y4 = (device const float4 *) y;

            float sumf = 0;
            for (int i = tiisg; i < ne00/4; i += 32) {
                for (int k = 0; k < 4; ++k) sumf += (float) x4[i][k] * y4[i][k];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) x[i] * y[i];
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

[[host_name("kernel_mul_mv_f32_f32")]]
kernel void kernel_mul_mv_f32_f32(
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_f32_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb00, nb01, nb02, ne10, ne11, ne12, nb10, nb11, nb12, ne0, ne1, r2, r3, tgpig, tiisg);
}

#define N_F16_F16 4

kernel void kernel_mul_mv_f16_f16(
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t rb = tgpig.y*N_F16_F16;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const half * x = (device const half *) (src0 + offset0);

    if (ne00 < 128) {
        for (int row = 0; row < N_F16_F16; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            device const half * y = (device const half *) (src1 + r1*nb11 + im*nb12);

            float sumf = 0;
            for (int i = tiisg; i < ne00; i += 32) {
                sumf += (half) x[i] * (half) y[i];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        device const half4 * x4 = (device const half4 *)x;
        for (int row = 0; row < N_F16_F16; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            device const half  * y  = (device const half  *) (src1 + r1*nb11 + im*nb12);
            device const half4 * y4 = (device const half4 *) y;

            float sumf = 0;
            for (int i = tiisg; i < ne00/4; i += 32) {
                for (int k = 0; k < 4; ++k) sumf += (half) x4[i][k] * y4[i][k];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (half) x[i] * y[i];
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

void kernel_mul_mv_f16_f32_1row_impl(
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const half  * x = (device const half  *) (src0 + offset0);
    device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);

    float sumf = 0;
    if (ne00 < 128) {
        for (int i = tiisg; i < ne00; i += 32) {
            sumf += (float) x[i] * (float) y[i];
        }
        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    } else {
        device const half4  * x4 = (device const half4  *) x;
        device const float4 * y4 = (device const float4 *) y;
        for (int i = tiisg; i < ne00/4; i += 32) {
            for (int k = 0; k < 4; ++k) sumf += (float)x4[i][k] * y4[i][k];
        }
        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) x[i] * y[i];
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_f16_f32_1row")]]
kernel void kernel_mul_mv_f16_f32_1row(
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_f16_f32_1row_impl(src0, src1, dst, ne00, ne01, ne02, nb00, nb01, nb02, ne10, ne11, ne12, nb10, nb11, nb12, ne0, ne1, r2, r3, tgpig, tiisg);
}

#define N_F16_F32 4

void kernel_mul_mv_f16_f32_impl(
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t rb = tgpig.y*N_F16_F32;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const half * x = (device const half *) (src0 + offset0);

    if (ne00 < 128) {
        for (int row = 0; row < N_F16_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);

            float sumf = 0;
            for (int i = tiisg; i < ne00; i += 32) {
                sumf += (float) x[i] * (float) y[i];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        device const half4 * x4 = (device const half4 *)x;
        for (int row = 0; row < N_F16_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            device const float  * y  = (device const float  *) (src1 + r1*nb11 + im*nb12);
            device const float4 * y4 = (device const float4 *) y;

            float sumf = 0;
            for (int i = tiisg; i < ne00/4; i += 32) {
                for (int k = 0; k < 4; ++k) sumf += (float) x4[i][k] * y4[i][k];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) x[i] * y[i];
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

[[host_name("kernel_mul_mv_f16_f32")]]
kernel void kernel_mul_mv_f16_f32(
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_f16_f32_impl(src0, src1, dst, ne00, ne01, ne02, nb00, nb01, nb02, ne10, ne11, ne12, nb10, nb11, nb12, ne0, ne1, r2, r3, tgpig, tiisg);
}

// Assumes row size (ne00) is a multiple of 4
kernel void kernel_mul_mv_f16_f32_l4(
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]]) {

    const int nrows = ne11;
    const int64_t r0 = tgpig.x;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const half4 * x4 = (device const half4 *) (src0 + offset0);

    for (int r1 = 0; r1 < nrows; ++r1) {
        device const float4 * y4 = (device const float4 *) (src1 + r1*nb11 + im*nb12);

        float sumf = 0;
        for (int i = tiisg; i < ne00/4; i += 32) {
            for (int k = 0; k < 4; ++k) sumf += (float) x4[i][k] * y4[i][k];
        }

        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
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
        constant     float & m0,
        constant     float & m1,
        constant       int & n_heads_log2_floor,
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
  //const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    const int64_t k = i3*ne3 + i2;

    float m_k;
    if (k < n_heads_log2_floor) {
        m_k = pow(m0, k + 1);
    } else {
        m_k = pow(m1, 2 * (k - n_heads_log2_floor) + 1);
    }

    device       char * dst_row = (device char *) dst + i3*nb3 + i2*nb2 + i1*nb1;
    device const char * src_row = (device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01;
    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        const  float   src_v = *(device float *)(src_row + i00*nb00);
        device float * dst_v =  (device float *)(dst_row + i00*nb0);
        *dst_v = i00 * m_k + src_v;
    }
}

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta
) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_orig_ctx, float n_rot, float base) {
    return n_dims * log(n_orig_ctx / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_orig_ctx, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_orig_ctx, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_orig_ctx, beta_slow, freq_base)));
}

typedef void (rope_t)(
        device const    void * src0,
        device const int32_t * src1,
        device         float * dst,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant     int64_t & ne03,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant    uint64_t & nb03,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant     int64_t & ne2,
        constant     int64_t & ne3,
        constant    uint64_t & nb0,
        constant    uint64_t & nb1,
        constant    uint64_t & nb2,
        constant    uint64_t & nb3,
        constant         int & n_past,
        constant         int & n_dims,
        constant         int & mode,
        constant         int & n_orig_ctx,
        constant       float & freq_base,
        constant       float & freq_scale,
        constant       float & ext_factor,
        constant       float & attn_factor,
        constant       float & beta_fast,
        constant       float & beta_slow,
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        uint3 tgpig[[threadgroup_position_in_grid]]);

template<typename T>
kernel void kernel_rope(
        device const    void * src0,
        device const int32_t * src1,
        device         float * dst,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant     int64_t & ne03,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant    uint64_t & nb03,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant     int64_t & ne2,
        constant     int64_t & ne3,
        constant    uint64_t & nb0,
        constant    uint64_t & nb1,
        constant    uint64_t & nb2,
        constant    uint64_t & nb3,
        constant         int & n_past,
        constant         int & n_dims,
        constant         int & mode,
        constant         int & n_orig_ctx,
        constant       float & freq_base,
        constant       float & freq_scale,
        constant       float & ext_factor,
        constant       float & attn_factor,
        constant       float & beta_fast,
        constant       float & beta_slow,
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        uint3 tgpig[[threadgroup_position_in_grid]]) {
    const int64_t i3 = tgpig[2];
    const int64_t i2 = tgpig[1];
    const int64_t i1 = tgpig[0];

    const bool is_neox = mode & 2;

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, corr_dims);

    device const int32_t * pos = src1;

    const int64_t p = pos[i2];

    const float theta_0 = (float)p;
    const float inv_ndims = -1.f/n_dims;

    if (!is_neox) {
        for (int64_t i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x) {

            const float theta = theta_0 * pow(freq_base, inv_ndims*i0);
            float cos_theta, sin_theta;
            rope_yarn(theta, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            const T x0 = src[0];
            const T x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        }
    } else {
        for (int64_t ic = 2*tiitg; ic < ne0; ic += 2*tptg.x) {
            if (ic < n_dims) {
                const int64_t ib = 0;

                // simplified from `(ib * n_dims + ic) * inv_ndims`
                const float cur_rot = inv_ndims*ic - ib;

                const float theta = theta_0 * pow(freq_base, cur_rot);
                float cos_theta, sin_theta;
                rope_yarn(theta, freq_scale, corr_dims, cur_rot, ext_factor, attn_factor, &cos_theta, &sin_theta);

                const int64_t i0 = ib*n_dims + ic/2;

                device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                const float x0 = src[0];
                const float x1 = src[n_dims/2];

                dst_data[0]        = x0*cos_theta - x1*sin_theta;
                dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
            } else {
                const int64_t i0 = ic;

                device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                dst_data[0] = src[0];
                dst_data[1] = src[1];
            }
        }
    }
}

template [[host_name("kernel_rope_f32")]] kernel rope_t kernel_rope<float>;
template [[host_name("kernel_rope_f16")]] kernel rope_t kernel_rope<half>;

typedef void (im2col_t)(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t iiw = tgpig[2] * s0 + tpitg[2] * d0 - p0;
    const int32_t iih = tgpig[1] * s1 + tpitg[1] * d1 - p1;

    const int32_t offset_dst =
        (tpitg[0] * tgpg[1] * tgpg[2] + tgpig[1] * tgpg[2] + tgpig[2]) * CHW +
        (tgpig[0] * (ntg[1] * ntg[2]) + tpitg[1] * ntg[2] + tpitg[2]);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int32_t offset_src = tpitg[0] * ofs0 + tgpig[0] * ofs1;
        pdst[offset_dst] = x[offset_src + iih * IW + iiw];
    }
}

template [[host_name("kernel_im2col_f32")]] kernel im2col_t kernel_im2col<float>;
template [[host_name("kernel_im2col_f16")]] kernel im2col_t kernel_im2col<half>;

kernel void kernel_upscale_f32(
    device  const char * src0,
    device        char * dst,
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
    constant   int32_t & sf,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1/sf;

    device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1);

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dst_ptr[i0] = src0_ptr[i0/sf];
    }
}

kernel void kernel_pad_f32(
    device  const char * src0,
    device        char * dst,
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

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1);

    if (i1 < ne01 && i2 < ne02 && i3 < ne03) {
        for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
            if (i0 < ne00) {
                dst_ptr[i0] = src0_ptr[i0];
            } else {
                dst_ptr[i0] = 0.0f;
            }
        }

        return;
    }

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dst_ptr[i0] = 0.0f;
    }
}

// bitonic sort implementation following the CUDA kernels as reference
typedef void (argsort_t)(
        device const float * x,
        device     int32_t * dst,
        constant   int64_t & ncols,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
        device const float   * x,
        device       int32_t * dst,
        constant     int64_t & ncols,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    // bitonic sort
    int col = tpitg[0];
    int row = tgpig[1];

    if (col >= ncols) return;

    device const float   * x_row   = x   + row * ncols;
    device       int32_t * dst_row = dst + row * ncols;

    // initialize indices
    if (col < ncols) {
        dst_row[col] = col;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ncols; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (order == GGML_SORT_ASC ? x_row[dst_row[col]] > x_row[dst_row[ixj]] : x_row[dst_row[col]] < x_row[dst_row[ixj]]) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (order == GGML_SORT_ASC ? x_row[dst_row[col]] < x_row[dst_row[ixj]] : x_row[dst_row[col]] > x_row[dst_row[ixj]]) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template [[host_name("kernel_argsort_f32_i32_asc")]]  kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ASC>;
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_DESC>;

kernel void kernel_leaky_relu_f32(
        device const float * src0,
        device       float * dst,
        constant     float & slope,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] > 0.0f ? src0[tpig] : src0[tpig] * slope;
}

kernel void kernel_cpy_f16_f16(
        device  const half * src0,
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
        device const half * src = (device half *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f16_f32(
        device  const half * src0,
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

kernel void kernel_cpy_f32_q8_0(
        device const float * src0,
        device        void * dst,
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
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK8_0;

    device block_q8_0 * dst_data = (device block_q8_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK8_0; i00 < ne00; i00 += ntg.x*QK8_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = src[j];
            amax = MAX(amax, fabs(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK8_0].d = d;

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = src[j]*id;

            dst_data[i00/QK8_0].qs[j] = round(x0);
        }
    }
}

kernel void kernel_cpy_f32_q4_0(
        device const float * src0,
        device        void * dst,
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
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_0;

    device block_q4_0 * dst_data = (device block_q4_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_0; i00 < ne00; i00 += ntg.x*QK4_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK4_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK4_0].d = d;

        for (int j = 0; j < QK4_0/2; ++j) {
            const float x0 = src[0       + j]*id;
            const float x1 = src[QK4_0/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            dst_data[i00/QK4_0].qs[j]  = xi0;
            dst_data[i00/QK4_0].qs[j] |= xi1 << 4;
        }
    }
}

kernel void kernel_cpy_f32_q4_1(
        device const float * src0,
        device        void * dst,
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
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_1;

    device block_q4_1 * dst_data = (device block_q4_1 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_1; i00 < ne00; i00 += ntg.x*QK4_1) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < QK4_1; j++) {
            const float v = src[j];
            if (min > v) min = v;
            if (max < v) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK4_1].d = d;
        dst_data[i00/QK4_1].m = min;

        for (int j = 0; j < QK4_1/2; ++j) {
            const float x0 = (src[0       + j] - min)*id;
            const float x1 = (src[QK4_1/2 + j] - min)*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            dst_data[i00/QK4_1].qs[j]  = xi0;
            dst_data[i00/QK4_1].qs[j] |= xi1 << 4;
        }
    }
}

kernel void kernel_concat(
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne10,
    constant   int64_t & ne11,
    constant   int64_t & ne12,
    constant   int64_t & ne13,
    constant  uint64_t & nb10,
    constant  uint64_t & nb11,
    constant  uint64_t & nb12,
    constant  uint64_t & nb13,
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

    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01 + tpitg.x*nb00;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11 + tpitg.x*nb10;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1  + tpitg.x*nb0;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        if (i02 < ne02) {
            ((device float *)dst_ptr)[0] = ((device float *)src0_ptr)[0];
            src0_ptr += ntg.x*nb00;
        } else {
            ((device float *)dst_ptr)[0] = ((device float *)src1_ptr)[0];
            src1_ptr += ntg.x*nb10;
        }
        dst_ptr += ntg.x*nb0;
    }
}

//============================================ k-quants ======================================================

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

typedef struct {
    half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;
// 66 bytes / block for QK_K = 256, so 2.0625 bpw

typedef struct {
    half d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;
// 74 bytes / block for QK_K = 256, so 2.3125 bpw

typedef struct {
    half d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;
// 98 bytes / block for QK_K = 256, so 3.0625 bpw

typedef struct {
    half d;
    uint8_t qs[QK_K/8];
    uint8_t scales[QK_K/16];
} block_iq1_s;

// Non-linear quants
#define QK4_NL 32
typedef struct {
    half    d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;

//====================================== dot products =========================

void kernel_mul_mv_q2_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q2_K * x = (device const block_q2_K *) src0 + ib_row + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q2_K) * nb;

#if QK_K == 256
    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3
    const int is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += step/2;
            sc += step;
            dh += step/2;
        }

        y4 += 4 * QK_K;
    }
#else
    const int ix = tiisg/2;  // 0...15
    const int it = tiisg%2;  // 0...1

    device const float * y4 = y + ix * QK_K + 8 * it;

    for (int ib = ix; ib < nb; ib += 16) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+16]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+32]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+48]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 4 * it;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[1] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[2] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[3] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] >> 4) + sumy[1] * (sc[1] >> 4) + sumy[2] * (sc[2] >> 4) + sumy[3] * (sc[3] >> 4));

            qs += step/2;
            sc += step;
            dh += step/2;
        }

        y4 += 16 * QK_K;
    }
#endif

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, tgpig, tiisg, sgitg);
}

#if QK_K == 256
void kernel_mul_mv_q3_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q3_K * x = (device const block_q3_K *) src0 + first_row*nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];

    //const uint16_t kmask1 = 0x3030;
    //const uint16_t kmask2 = 0x0f0f;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int ip  = tid/4;          // 0 or 1
    const int il  = 2*((tid%4)/2);  // 0 or 2
    const int ir  = tid%2;
    const int n   = 8;
    const int l0  = n*ir;

    // One would think that the Metal compiler would figure out that ip and il can only have
    // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it
    // with these two tales.
    //
    // Possible masks for the high bit
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

    // Possible masks for the low 2 bits
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const int shift = 2*il;
    const float    v1 = il == 0 ? 4.f : 64.f;
    const float    v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const int q_offset = 32*ip + l0;
    const int y_offset = 128*ip + 32*il + l0;

    const int step = sizeof(block_q3_K) * nb / 2;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[2] = {0.f};
    float sumf2[2] = {0.f};
    for (int i = ix; i < nb; i += 4) {

        for (int l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (int row = 0; row < 2; ++row) {

            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (int l = 0; l < n; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (int l = 0; l < n; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += step;
            h  += step;
            a  += step;
            dh += step;

        }

        y1 += 4 * QK_K;

    }

    for (int row = 0; row < 2; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }
    if (tiisg == 0) {
        for (int row = 0; row < 2; ++row) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = sumf1[row];
        }
    }
}
#else
void kernel_mul_mv_q3_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    const int row = 2 * r0 + sgitg;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q3_K * x = (device const block_q3_K *) src0 + row*nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/4;
    const int il = 4 * (tiisg%4);// 0, 4, 8, 12
    const int iq = il/8;         // 0, 0, 1, 1
    const int in = il%8;         // 0, 4, 0, 4

    float2 sum = {0.f, 0.f};

    for (int i = ix; i < nb; i += 8) {

        const float d_all = (float)(x[i].d);

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + il);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + in);
        device const uint16_t * s = (device const uint16_t *)(x[i].scales);
        device const float    * y = yy + i * QK_K + il;

        const float d1 = d_all * ((int32_t)(s[0] & 0x000F) - 8);
        const float d2 = d_all * ((int32_t)(s[0] & 0x00F0) - 128) * 1.f/64.f;
        const float d3 = d_all * ((int32_t)(s[0] & 0x0F00) - 2048) * 1.f/4096.f;
        const float d4 = d_all * ((int32_t)(s[0] & 0xF000) - 32768) * 1.f/262144.f;

        for (int l = 0; l < 4; l += 2) {
            const uint16_t hm = h[l/2] >> iq;
            sum[0] += y[l+ 0] * d1 * ((int32_t)(q[l/2] & 0x0003) - ((hm & 0x0001) ? 0 :  4))
                    + y[l+16] * d2 * ((int32_t)(q[l/2] & 0x000c) - ((hm & 0x0004) ? 0 : 16))
                    + y[l+32] * d3 * ((int32_t)(q[l/2] & 0x0030) - ((hm & 0x0010) ? 0 : 64))
                    + y[l+48] * d4 * ((int32_t)(q[l/2] & 0x00c0) - ((hm & 0x0040) ? 0 : 256));
            sum[1] += y[l+ 1] * d1 * ((int32_t)(q[l/2] & 0x0300) - ((hm & 0x0100) ? 0 : 1024))
                    + y[l+17] * d2 * ((int32_t)(q[l/2] & 0x0c00) - ((hm & 0x0400) ? 0 : 4096))
                    + y[l+33] * d3 * ((int32_t)(q[l/2] & 0x3000) - ((hm & 0x1000) ? 0 : 16384))
                    + y[l+49] * d4 * ((int32_t)(q[l/2] & 0xc000) - ((hm & 0x4000) ? 0 : 65536));
        }

    }
    const float sumf = sum[0] + sum[1] * 1.f/256.f;

    const float tot = simd_sum(sumf);
    if (tiisg == 0) {
        dst[r1*ne0 + im*ne0*ne1 + row] = tot;
    }

}
#endif

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, tgpig, tiisg, sgitg);
}

#if QK_K == 256
void kernel_mul_mv_q4_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    //const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int first_row = r0 * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];
    float yh[16];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q4_K) * nb / 2;

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);
                acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);
                acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);
                acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += step;
            sc += step;
            dh += step;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}
#else
void kernel_mul_mv_q4_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const int ix = tiisg/4;  // 0...7
    const int it = tiisg%4;  // 0...3

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = r0 * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[8];
    float yh[8];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q4_K) * nb / 2;

    device const float * y4 = y + ix * QK_K + 8 * it;

    uint16_t sc16[4];

    for (int ib = ix; ib < nb; ib += 8) {

        float2 sumy = {0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i] = y4[i+ 0]; sumy[0] += yl[i];
            yh[i] = y4[i+32]; sumy[1] += yh[i];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 4 * it;
        device const half     * dh = x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            sc16[0] = sc[0] & 0x000f;
            sc16[1] = sc[0] & 0x0f00;
            sc16[2] = sc[0] & 0x00f0;
            sc16[3] = sc[0] & 0xf000;

            float2 acc1 = {0.f, 0.f};
            float2 acc2 = {0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+0] * (qs[i/2] & 0x000F);
                acc1[1] += yl[i+1] * (qs[i/2] & 0x0F00);
                acc2[0] += yh[i+0] * (qs[i/2] & 0x00F0);
                acc2[1] += yh[i+1] * (qs[i/2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc16[0] +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc16[1] * 1.f/4096.f) -
                         dmin * 1.f/16.f * (sumy[0] * sc16[2] + sumy[1] * sc16[3] * 1.f/256.f);

            qs += step;
            sc += step;
            dh += step;
        }

        y4 += 8 * QK_K;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}
#endif

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q5_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q5_K * x = (device const block_q5_K *) src0 + first_row*nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf[2]={0.f};

    const int step = sizeof(block_q5_K) * nb;

#if QK_K == 256
#
    float yl[16], yh[16];

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int iq  = tid/4;
    const int ir  = tid%4;
    const int n   = 8;

    const int l0 = n*ir;
    const int q_offset = 32*iq + l0;
    const int y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {

        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (int row = 0; row < 2; ++row) {

            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            for (int l = 0; l < n; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }
            const float dall = dh[0];
            const float dmin = dh[1];
            sumf[row] += dall * (sc8[0] * (acc1[0] +  16.f*acc2[0]) +
                                 sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                 sc8[4] * (acc1[2] +  16.f*acc2[2]) +
                                 sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += step;
            qh += step;
            dh += step/2;
            a  += step/2;

        }

        y1 += 4 * QK_K;

    }
#else
    float yl[8], yh[8];

    const int il = 4 * (tiisg/8);  // 0, 4, 8, 12
    const int ix = tiisg%8;
    const int iq = il/8;         // 0, 0, 1, 1
    const int in = il%8;         // 0, 4, 0, 4

    device const float * y = yy + ix*QK_K + il;

    for (int i = ix; i < nb; i += 8) {

        for (int l = 0; l < 4; ++l) {
            yl[l+0] = y[l+ 0];
            yl[l+4] = y[l+16];
            yh[l+0] = y[l+32];
            yh[l+4] = y[l+48];
        }

        device const half * dh = &x[i].d;
        device const uint8_t * q = x[i].qs + il;
        device const uint8_t * h = x[i].qh + in;
        device const int8_t  * s = x[i].scales;

        for (int row = 0; row < 2; ++row) {

            const float d = dh[0];

            float2 acc = {0.f, 0.f};
            for (int l = 0; l < 4; ++l) {
                const uint8_t hl = h[l] >> iq;
                acc[0] += yl[l+0] * s[0] * ((int16_t)(q[l+ 0] & 0x0F) - (hl & 0x01 ? 0 : 16))
                        + yl[l+4] * s[1] * ((int16_t)(q[l+16] & 0x0F) - (hl & 0x04 ? 0 : 16));
                acc[1] += yh[l+0] * s[2] * ((int16_t)(q[l+ 0] & 0xF0) - (hl & 0x10 ? 0 : 256))
                        + yh[l+4] * s[3] * ((int16_t)(q[l+16] & 0xF0) - (hl & 0x40 ? 0 : 256));
            }
            sumf[row] += d * (acc[0] + 1.f/16.f * acc[1]);

            q += step;
            h += step;
            s += step;
            dh += step/2;

        }

        y += 8 * QK_K;
    }
#endif

    for (int row = 0; row < 2; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q6_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int     im = tgpig.z;

    const int row = 2 * r0 + sgitg;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q6_K * x = (device const block_q6_K *) src0 + row * nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf = 0;

#if QK_K == 256
    const int tid  = tiisg/2;
    const int ix   = tiisg%2;
    const int ip   = tid/8;         // 0 or 1
    const int il   = tid%8;
    const int n    = 4;
    const int l0   = n*il;
    const int is   = 8*ip + l0/16;

    const int y_offset = 128*ip + l0;
    const int q_offset_l = 64*ip + l0;
    const int q_offset_h = 32*ip + l0;

    for (int i = ix; i < nb; i += 2) {

        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;

        device const float * y = yy + i * QK_K + y_offset;

        const float dall = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            sums[0] += y[l+ 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += y[l+32] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += y[l+64] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            sums[3] += y[l+96] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }

        sumf += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

    }

#else
    const int ix  = tiisg/4;
    const int il  = 4*(tiisg%4);

    for (int i = ix; i < nb; i += 8) {
        device const float * y = yy + i * QK_K + il;
        device const uint8_t * ql = x[i].ql + il;
        device const uint8_t * qh = x[i].qh + il;
        device const int8_t  * s  = x[i].scales;

        const float d = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 4; ++l) {
            sums[0] += y[l+ 0] * ((int8_t)((ql[l+ 0] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += y[l+16] * ((int8_t)((ql[l+16] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += y[l+32] * ((int8_t)((ql[l+ 0] >>  4) | ((qh[l] & kmask3) >> 0)) - 32);
            sums[3] += y[l+48] * ((int8_t)((ql[l+16] >>  4) | ((qh[l] & kmask4) >> 2)) - 32);
        }
        sumf += d * (sums[0] * s[0] + sums[1] * s[1] + sums[2] * s[2] + sums[3] * s[3]);
    }

#endif

    const float tot = simd_sum(sumf);
    if (tiisg == 0) {
        dst[r1*ne0 + im*ne0*ne1 + row] = tot;
    }
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, tgpig, tiisg, sgitg);
}

// ======================= "True" 2-bit

constexpr constant static uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

constexpr constant static uint64_t iq2xs_grid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

constexpr constant static uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

#define NGRID_IQ1S 512
constexpr constant static uint64_t iq1s_grid[NGRID_IQ1S] = {
    0xffffffffffff0101, 0xffffffffff01ff00, 0xffffffffff010100, 0xffffffff00000000,
    0xffffffff01ff00ff, 0xffffffff01ff0001, 0xffffffff0101ffff, 0xffffffff0101ff01,
    0xffffff00ff000000, 0xffffff000000ff00, 0xffffff00000000ff, 0xffffff0000000100,
    0xffffff0000010000, 0xffffff0001000000, 0xffffff01ffff00ff, 0xffffff01ff01ff00,
    0xffffff01ff010100, 0xffffff0100000001, 0xffffff0101ffff00, 0xffffff0101ff0101,
    0xffffff0101010100, 0xffff00ffff00ff01, 0xffff00ffff0000ff, 0xffff00ff00ff0100,
    0xffff00ff0100ff00, 0xffff00ff010001ff, 0xffff0000ff0101ff, 0xffff000000ffff00,
    0xffff000000000000, 0xffff00000001ff01, 0xffff000001000101, 0xffff0000010100ff,
    0xffff0001ffff0100, 0xffff00010000ff00, 0xffff000100010101, 0xffff000101000000,
    0xffff01ffffff0000, 0xffff01ffff01ffff, 0xffff01ffff010100, 0xffff01ff00000000,
    0xffff01ff01ffffff, 0xffff01ff01ff0001, 0xffff01ff0101ffff, 0xffff01ff01010001,
    0xffff0100ffffff01, 0xffff01000000ffff, 0xffff010000000100, 0xffff010001ff01ff,
    0xffff010001000000, 0xffff0101ff000000, 0xffff0101000101ff, 0xffff010101ffff01,
    0xffff01010101ff00, 0xff00ffffff000000, 0xff00ffff00ffff00, 0xff00ffff00000001,
    0xff00ffff000001ff, 0xff00ffff01010000, 0xff00ff00ffff0000, 0xff00ff00ff00ff00,
    0xff00ff00ff0000ff, 0xff00ff00ff000100, 0xff00ff00ff010001, 0xff00ff0000ff0001,
    0xff00ff000000ffff, 0xff00ff0000000000, 0xff00ff000001ff00, 0xff00ff0000010100,
    0xff00ff0001ff0000, 0xff00ff000100ff00, 0xff00ff0001000100, 0xff00ff01ff000000,
    0xff00ff0100ff0000, 0xff00ff01000001ff, 0xff00ff0101010001, 0xff0000ff00000000,
    0xff0000ff0001ff00, 0xff0000ff00010100, 0xff000000ffff0101, 0xff000000ff000000,
    0xff000000ff01ff00, 0xff00000000ff0000, 0xff0000000000ff00, 0xff000000000000ff,
    0xff00000000000000, 0xff00000000000001, 0xff00000000000100, 0xff0000000001ffff,
    0xff00000000010000, 0xff00000001000000, 0xff00000001010100, 0xff000001ff00ff01,
    0xff000001ff0100ff, 0xff00000100000000, 0xff0000010001ff00, 0xff00000101ff0100,
    0xff0000010100ff00, 0xff0001ff00ff00ff, 0xff0001ff00000101, 0xff0001ff000100ff,
    0xff0001ff01000000, 0xff000100ff0001ff, 0xff0001000000ff01, 0xff00010000000000,
    0xff00010000010001, 0xff00010000010100, 0xff00010001ffff00, 0xff00010001ff0101,
    0xff00010001010000, 0xff000101ffffffff, 0xff000101ff000101, 0xff00010101ff00ff,
    0xff00010101000001, 0xff000101010100ff, 0xff01ffffff000101, 0xff01ffffff01ffff,
    0xff01ffffff01ff01, 0xff01ffffff0101ff, 0xff01ffff00000000, 0xff01ffff01ff0001,
    0xff01ffff0101ff01, 0xff01ff00ff000000, 0xff01ff0000ff0100, 0xff01ff000000ff01,
    0xff01ff0000010000, 0xff01ff00010000ff, 0xff01ff01ff01ff00, 0xff01ff0100000101,
    0xff0100ffffff0000, 0xff0100ffff010000, 0xff0100ff01ff00ff, 0xff0100ff01000100,
    0xff0100ff010100ff, 0xff010000ffffff01, 0xff01000000000000, 0xff0100000101ff00,
    0xff010001ffff00ff, 0xff010001ff000100, 0xff01000100ffff00, 0xff01000100010001,
    0xff01000101ff0001, 0xff010001010001ff, 0xff0101ffffffffff, 0xff0101ffff01ffff,
    0xff0101ffff010101, 0xff0101ff0000ff00, 0xff0101ff01010001, 0xff010100ff000000,
    0xff010100ff01ff01, 0xff01010000ff0001, 0xff01010000000100, 0xff01010001000000,
    0xff0101010100ffff, 0x00ffffff0000ff01, 0x00ffffff000000ff, 0x00ffffff00000100,
    0x00ffffff00010000, 0x00ffff00ffff0001, 0x00ffff00ff0000ff, 0x00ffff00ff000100,
    0x00ffff0000000000, 0x00ffff0001000100, 0x00ffff0001010001, 0x00ffff01ff00ff01,
    0x00ffff0100ff0100, 0x00ffff010000ff00, 0x00ffff01000100ff, 0x00ffff0101ff00ff,
    0x00ffff010101ff00, 0x00ff00ffffffffff, 0x00ff00ffffff01ff, 0x00ff00ffff000101,
    0x00ff00ff00000000, 0x00ff00ff000101ff, 0x00ff00ff01010101, 0x00ff0000ff000000,
    0x00ff0000ff01ffff, 0x00ff000000ff0000, 0x00ff00000000ff00, 0x00ff0000000000ff,
    0x00ff000000000000, 0x00ff000000000001, 0x00ff000000000100, 0x00ff000000010000,
    0x00ff000001ffff01, 0x00ff000001000000, 0x00ff0001ff000101, 0x00ff000100ffffff,
    0x00ff000100000000, 0x00ff0001010001ff, 0x00ff01ffff000000, 0x00ff01ff0001ff00,
    0x00ff01ff01ff0100, 0x00ff0100ff01ff01, 0x00ff010000ff00ff, 0x00ff010000ff0101,
    0x00ff010000000000, 0x00ff010000010101, 0x00ff01000100ff00, 0x00ff010001010000,
    0x00ff0101ffffff00, 0x00ff01010000ff01, 0x00ff010100000100, 0x00ff010101ff0000,
    0x0000ffffffff0100, 0x0000ffffff00ff00, 0x0000ffffff0000ff, 0x0000ffffff010000,
    0x0000ffff00000000, 0x0000ffff00010101, 0x0000ffff01ffff01, 0x0000ffff01000100,
    0x0000ff00ff000000, 0x0000ff00ff01ff00, 0x0000ff00ff0101ff, 0x0000ff0000ff0000,
    0x0000ff000000ff00, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
    0x0000ff0000000100, 0x0000ff0000010000, 0x0000ff0001ffffff, 0x0000ff0001ff01ff,
    0x0000ff0001000000, 0x0000ff000101ffff, 0x0000ff01ffff0101, 0x0000ff01ff010000,
    0x0000ff0100000000, 0x0000ff0101000101, 0x000000ffffff0001, 0x000000ffff000000,
    0x000000ff00ff0000, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
    0x000000ff00000001, 0x000000ff00000100, 0x000000ff00010000, 0x000000ff01000000,
    0x000000ff0101ff00, 0x00000000ffff0000, 0x00000000ff00ff00, 0x00000000ff0000ff,
    0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff010000,
    0x0000000000ffff00, 0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001,
    0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01,
    0x00000000000000ff, 0x0000000000000001, 0x00000000000001ff, 0x0000000000000100,
    0x0000000000000101, 0x000000000001ff00, 0x00000000000100ff, 0x0000000000010000,
    0x0000000000010001, 0x0000000000010100, 0x0000000001ff0000, 0x000000000100ff00,
    0x00000000010000ff, 0x0000000001000000, 0x0000000001000001, 0x0000000001000100,
    0x0000000001010000, 0x00000001ffff01ff, 0x00000001ff000000, 0x0000000100ff0000,
    0x000000010000ff00, 0x00000001000000ff, 0x0000000100000000, 0x0000000100000001,
    0x0000000100000100, 0x0000000100010000, 0x0000000101000000, 0x000001ffff00ff00,
    0x000001ffff010001, 0x000001ffff0101ff, 0x000001ff00ffff01, 0x000001ff0000ffff,
    0x000001ff00000000, 0x000001ff010000ff, 0x000001ff01010100, 0x00000100ffff0100,
    0x00000100ff000000, 0x0000010000ff0000, 0x000001000000ff00, 0x00000100000000ff,
    0x0000010000000000, 0x0000010000000001, 0x0000010000000100, 0x0000010000010000,
    0x0000010001000000, 0x000001000101ff01, 0x00000101ffff0001, 0x00000101ff01ffff,
    0x0000010100000000, 0x0000010101010100, 0x0001ffffff000000, 0x0001ffff00ffffff,
    0x0001ffff00000100, 0x0001ffff0001ff00, 0x0001ffff01000000, 0x0001ff00ffffff00,
    0x0001ff00ffff01ff, 0x0001ff00ff010000, 0x0001ff0000000000, 0x0001ff0000010001,
    0x0001ff0001ff0000, 0x0001ff0001010100, 0x0001ff01ff0000ff, 0x0001ff01ff000001,
    0x0001ff0100ffffff, 0x0001ff010001ffff, 0x0001ff01000101ff, 0x0001ff010100ff01,
    0x000100ffff00ffff, 0x000100ffff00ff01, 0x000100ffff000100, 0x000100ff00000000,
    0x000100ff000101ff, 0x000100ff01ff0101, 0x000100ff0100ffff, 0x000100ff01010101,
    0x00010000ff000000, 0x00010000ff010100, 0x0001000000ff0000, 0x000100000000ff00,
    0x00010000000000ff, 0x0001000000000000, 0x0001000000000001, 0x0001000000000100,
    0x0001000000010000, 0x0001000001ffff01, 0x0001000001000000, 0x0001000100ff0101,
    0x0001000100000000, 0x00010001010100ff, 0x000101ffffff01ff, 0x000101ffffff0101,
    0x000101ff00010000, 0x000101ff01ff0000, 0x000101ff0100ff01, 0x00010100ffff0000,
    0x0001010000000000, 0x000101000001ffff, 0x0001010000010101, 0x00010100010001ff,
    0x00010101ff00ff00, 0x00010101ff010001, 0x0001010100ffffff, 0x0001010100ff01ff,
    0x00010101000101ff, 0x0001010101ff0000, 0x000101010100ff01, 0x0001010101000101,
    0x01ffffffffff0101, 0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff,
    0x01ffffffff010101, 0x01ffffff00000000, 0x01ffffff01ff01ff, 0x01ffffff01000101,
    0x01ffffff0101ff01, 0x01ffffff010100ff, 0x01ffff000000ff00, 0x01ffff0000000001,
    0x01ffff00000001ff, 0x01ffff0000010000, 0x01ffff0001ff0000, 0x01ffff01ffffffff,
    0x01ffff01ffff01ff, 0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff0101ff,
    0x01ffff010100ffff, 0x01ff00ffffff0000, 0x01ff00ffff010000, 0x01ff00ff00ffff01,
    0x01ff0000ff0000ff, 0x01ff000000000000, 0x01ff00000001ff01, 0x01ff000001ffffff,
    0x01ff000001010100, 0x01ff0001ffffff01, 0x01ff0001ff010001, 0x01ff000101ff0100,
    0x01ff000101000001, 0x01ff0001010100ff, 0x01ff01ffff00ffff, 0x01ff01ff00010001,
    0x01ff01ff01000000, 0x01ff01ff010101ff, 0x01ff0100ff000001, 0x01ff010000ffff00,
    0x01ff010000000100, 0x01ff010001ff01ff, 0x01ff01000101ffff, 0x01ff0101ffff00ff,
    0x01ff0101ffff0101, 0x01ff0101ff0101ff, 0x01ff010100010000, 0x0100ffff00ff00ff,
    0x0100ffff00ff0001, 0x0100ffff00000100, 0x0100ffff0100ff00, 0x0100ff00ffff0000,
    0x0100ff00ff00ffff, 0x0100ff00ff00ff01, 0x0100ff00ff000100, 0x0100ff00ff010000,
    0x0100ff0000000000, 0x0100ff00000100ff, 0x0100ff0001ff0101, 0x0100ff0001010101,
    0x0100ff0100ff00ff, 0x0100ff0100ff0001, 0x0100ff0100000100, 0x0100ff0100010001,
    0x0100ff0101000000, 0x010000ffff00ff00, 0x010000ff0000ffff, 0x010000ff00000000,
    0x010000ff010001ff, 0x010000ff01010001, 0x01000000ffffff00, 0x01000000ffff0101,
    0x01000000ff000000, 0x01000000ff0100ff, 0x01000000ff010101, 0x0100000000ff0000,
    0x010000000000ff00, 0x01000000000000ff, 0x0100000000000000, 0x0100000000000001,
    0x0100000000000100, 0x0100000000010000, 0x0100000001000000, 0x0100000100000000,
    0x01000001000101ff, 0x0100000101ffff01, 0x010001ffff000101, 0x010001ff00ff0100,
    0x010001ff0000ff00, 0x010001ff000100ff, 0x010001ff01ffffff, 0x01000100ffff0000,
    0x01000100ff0001ff, 0x0100010000000000, 0x010001000001ff00, 0x0100010001ff0000,
    0x01000100010000ff, 0x0100010001000101, 0x01000101ff00ff01, 0x0100010100ff0100,
    0x010001010000ffff, 0x0100010101010001, 0x0101ffffffff0101, 0x0101ffffff0001ff,
    0x0101ffffff01ffff, 0x0101ffffff010101, 0x0101ffff00000000, 0x0101ffff0101ffff,
    0x0101ffff010101ff, 0x0101ff00ff000000, 0x0101ff0000ff0100, 0x0101ff000000ff00,
    0x0101ff0000010000, 0x0101ff00010000ff, 0x0101ff0001000001, 0x0101ff01ff010101,
    0x0101ff0100000000, 0x0101ff010101ff00, 0x010100ffffff0000, 0x010100ffff010000,
    0x010100ff00ff01ff, 0x010100ff000000ff, 0x010100ff00000101, 0x010100ff01ffff00,
    0x01010000ffffff01, 0x01010000ff000100, 0x01010000ff01ff01, 0x0101000000000000,
    0x01010000000100ff, 0x010100000101ff01, 0x01010001ffff0000, 0x01010001ff00ffff,
    0x01010001ff010000, 0x0101000101ffffff, 0x0101000101ff01ff, 0x0101000101010101,
    0x010101ffff01ffff, 0x010101ff00000000, 0x010101ff0001ff01, 0x010101ff0101ffff,
    0x010101ff010101ff, 0x01010100ffffffff, 0x01010100ff000001, 0x010101000000ff00,
    0x0101010001010000, 0x0101010100ff0001, 0x010101010001ff01, 0x010101010101ffff,
};

constexpr constant static uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

constexpr constant static uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

void kernel_mul_mv_iq2_xxs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) src0 + ib_row + offset0;
    device const float         * y = (device const float         *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#if QK_K == 256
    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + aux8[l]);
                const uint8_t signs = shared_signs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += nb*sizeof(block_iq2_xxs)/2;
            q2 += nb*sizeof(block_iq2_xxs)/2;
        }

        y4 += 32 * 32;
    }
#else
    (void) x;
    (void) y;
    (void) yl;
    (void) nb32;
#endif

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xxs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq2_xs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq2_xs * x = (device const block_iq2_xs *) src0 + ib_row + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 512);
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq2xs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#if QK_K == 256
    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (int l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + (q2[l] & 511));
                const uint8_t signs = shared_signs[(q2[l] >> 9)];
                for (int j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (int l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + (q2[l] & 511));
                const uint8_t signs = shared_signs[(q2[l] >> 9)];
                for (int j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += nb*sizeof(block_iq2_xs)/2;
            q2 += nb*sizeof(block_iq2_xs)/2;
            sc += nb*sizeof(block_iq2_xs);
        }

        y4 += 32 * 32;
    }
#else
    (void) x;
    (void) y;
    (void) yl;
    (void) nb32;
#endif

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq3_xxs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) src0 + ib_row + offset0;
    device const float         * y = (device const float         *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * values = (threadgroup uint32_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#if QK_K == 256
    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(values + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(values + q3[2*l+1]);
                const uint8_t signs = shared_signs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += nb*sizeof(block_iq3_xxs)/2;
            q3  += nb*sizeof(block_iq3_xxs);
            gas += nb*sizeof(block_iq3_xxs)/2;
        }

        y4 += 32 * 32;
    }
#else
    (void) x;
    (void) y;
    (void) yl;
    (void) nb32;
#endif

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.5f;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq1_s_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq1_s * x = (device const block_iq1_s *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

#if QK_K == 256
    const int ix = tiisg/2;
    const int il = tiisg%2;

    device const float * y4 = y + 32 * ix + 16 * il;

    for (int ib32 = ix; ib32 < nb32; ib32 += 16) {

        for (int i = 0; i < 16; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib + 2 * il;
        device const uint8_t * sc = xr->scales + 2 * ib + il;
        device const half    * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            constant int8_t * grid1 = (constant int8_t *)(iq1s_grid + (qs[0] | ((sc[0] & 0x08) << 5)));
            constant int8_t * grid2 = (constant int8_t *)(iq1s_grid + (qs[1] | ((sc[0] & 0x80) << 1)));

            float2 sum = {0};
            for (int j = 0; j < 8; ++j) {
                sum[0] += yl[j+ 0] * grid1[j];
                sum[1] += yl[j+ 8] * grid2[j];
            }
            sumf[row] += (float)dh[0] * (sum[0] * (2*(sc[0] & 7) + 1) + sum[1] * (2*((sc[0] >> 4) & 7) + 1));

            dh += nb*sizeof(block_iq1_s)/2;
            qs += nb*sizeof(block_iq1_s);
            sc += nb*sizeof(block_iq1_s);
        }

        y4 += 16 * 32;
    }
#else
    (void) x;
    (void) y;
    (void) yl;
    (void) nb32;
#endif

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

void kernel_mul_mv_iq4_nl_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne10,
        constant   int64_t & ne12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup float  * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK4_NL;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq4_nl * x = (device const block_iq4_nl *) src0 + ib_row + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/2;  // 0...15
    const int it = tiisg%2;  // 0 or 1

    shared_values[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK4_NL + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < nb; ib += 16) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        for (int row = 0; row < 2; ++row) {

            device const block_iq4_nl & xb = x[row*nb + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

        }

        yb += 16 * QK4_NL;
    }

    for (int row = 0; row < 2; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        device const  void * src0,
        device const float * src1,
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
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup float * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

//============================= templates and their specializations =============================

// NOTE: this is not dequantizing - we are simply fitting the template
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    float4x4 temp = *(((device float4x4 *)src));
    for (int i = 0; i < 16; i++){
        reg[i/4][i%4] = temp[i/4][i%4];
    }
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    half4x4 temp = *(((device half4x4 *)src));
    for (int i = 0; i < 16; i++){
        reg[i/4][i%4] = temp[i/4][i%4];
    }
}

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i=0;i<8;i++) {
        reg[i/2][2*(i%2)+0] = d1 * (qs[i] & mask0) + md;
        reg[i/2][2*(i%2)+1] = d2 * (qs[i] & mask1) + md;
    }
}

template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i=0;i<8;i++) {
        reg[i/2][2*(i%2)+0] = ((qs[i] & mask0) * d1) + m;
        reg[i/2][2*(i%2)+1] = ((qs[i] & mask1) * d2) + m;
    }
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[i/2][2*(i%2)+0] = d * x0 + md;
        reg[i/2][2*(i%2)+1] = d * x1 + md;
    }
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[i/2][2*(i%2)+0] = d * x0 + m;
        reg[i/2][2*(i%2)+1] = d * x1 + m;
    }
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const half d = xb->d;

    for (int i = 0; i < 16; i++) {
        reg[i/4][i%4] = (qs[i + 16*il] * d);
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

#if QK_K == 256
    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;
#endif
    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

#if QK_K == 256
    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
#else
    float    kcoef = il&1 ? 1.f/16.f : 1.f;
    uint16_t kmask = il&1 ? 0xF0     : 0x0F;
    float    dl = d_all * ((scales[il/2] & kmask) * kcoef - 8);
    float    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uint8_t  mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    uint8_t  m = 1<<(il*2);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = coef * dl * ((q[i] & mask) - ((h[i%8] & (m * (1 + i/8))) ? 0 : 4.f/coef));
    }
#endif
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

#if QK_K == 256
    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
#else
    (void) get_scale_min_k4_just2;

    q = q + 16 * (il&1);
    device const uint8_t * s = xb->scales;
    device const half2 * dh = (device const half2 *)xb->d;
    const float2 d = (float2)dh[0];
    const float dl = il<2 ? d[0] * (s[0]&0xF) : d[0] * (s[1]&0xF)/16.h;
    const float ml = il<2 ? d[1] * (s[0]>>4)  : d[1] * (s[1]>>4);
#endif
    const ushort mask = il<2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

#if QK_K == 256
    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
#else
    q = q + 16 * (il&1);
    device const int8_t * s = xb->scales;
    const float dl = xb->d * s[il];
    uint8_t m = 1<<(il*2);
    const float  coef = il<2 ? 1.f  : 1.f/16.f;
    const ushort mask = il<2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = coef * dl * ((q[i] & mask) - (qh[i%8] & (m*(1+i/8)) ? 0.f : 16.f/coef));
    }
#endif
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * ql = (device const uint8_t *)xb->ql;
    device const uint8_t * qh = (device const uint8_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

#if QK_K == 256
    ql = ql + 64*(il/8) + 32*((il/2)&1) + 16*(il&1);
    qh = qh + 32*(il/8) + 16*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;
#else
    ql = ql + 16 * (il&1);
    float sc = scales[il];
#endif
    const uint16_t  kmask1 = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);
    const uint16_t  kmask2 = il>1 ? 0xF0              : 0x0F;
    const float       coef = il>1 ? 1.f/16.f          : 1.f;
    const float ml = d_all * sc * 32.f;
    const float dl = d_all * sc * coef;
    for (int i = 0; i < 16; ++i) {
        const half q = il&1 ? ((ql[i] & kmask2) | ((qh[i] & kmask1) << 2))
                            : ((ql[i] & kmask2) | ((qh[i] & kmask1) << 4));
        reg[i/4][i%4] = dl * q - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    device const uint8_t * qs = xb->qs + 2*il;
    device const uint8_t * sc = xb->scales + il;
    const float dl1 = d * (2*(sc[0] & 7) + 1);
    const float dl2 = d * (2*((sc[0] >> 4) & 7) + 1);
    constant int8_t * grid1 = (constant int8_t *)(iq1s_grid + (qs[0] | ((sc[0] & 0x08) << 5)));
    constant int8_t * grid2 = (constant int8_t *)(iq1s_grid + (qs[1] | ((sc[0] & 0x80) << 1)));
    for (int i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = dl1 * grid1[i];
        reg[i/4+2][i%4] = dl2 * grid2[i];
    }
}

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_get_rows(
        device const  void * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    //const int64_t i = tgpig;
    //const int64_t r = ((device int32_t *) src1)[i];

    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((device int32_t *) ((device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int64_t ind = tiitg; ind < ne00/16; ind += tptg.x) {
        float4x4 temp;
        dequantize_func(
            ((device const block_q *) ((device char *) src0 + r*nb01 + i02*nb02)) + ind/nl, ind%nl, temp);
        *(((device float4x4 *) ((device char *) dst + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

kernel void kernel_get_rows_f32(
        device const  void * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((device int32_t *) ((device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((device float *) ((device char *) dst + i11*nb2 + i10*nb1))[ind] =
            ((device float *) ((device char *) src0 + r*nb01 + i02*nb02))[ind];
    }
}

kernel void kernel_get_rows_f16(
        device const  void * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((device int32_t *) ((device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((device float *) ((device char *) dst + i11*nb2 + i10*nb1))[ind] =
            ((device half *) ((device char *) src0 + r*nb01 + i02*nb02))[ind];
    }
}

kernel void kernel_get_rows_i32(
        device const  void * src0,
        device const  char * src1,
        device     int32_t * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((device int32_t *) ((device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((device int32_t *) ((device char *) dst + i11*nb2 + i10*nb1))[ind] =
            ((device int32_t *) ((device char *) src0 + r*nb01 + i02*nb02))[ind];
    }
}


#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// each block_q contains 16*nl weights
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
void kernel_mul_mm_impl(device const  uchar * src0,
                        device const  uchar * src1,
                        device        float * dst,
                        constant    int64_t & ne00,
                        constant    int64_t & ne02,
                        constant   uint64_t & nb01,
                        constant   uint64_t & nb02,
                        constant    int64_t & ne12,
                        constant   uint64_t & nb10,
                        constant   uint64_t & nb11,
                        constant   uint64_t & nb12,
                        constant    int64_t & ne0,
                        constant    int64_t & ne1,
                        constant       uint & r2,
                        constant       uint & r3,
                        threadgroup   uchar * shared_memory [[threadgroup(0)]],
                        uint3                 tgpig[[threadgroup_position_in_grid]],
                        uint                  tiitg[[thread_index_in_threadgroup]],
                        uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shared_memory);
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

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    uint   offset0 = (i12/r2)*nb02 + (i13/r3)*(nb02*ne02);
    ushort offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb12 * im
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        half4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
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
        device float * C = dst + (BLOCK_SIZE_M * r0 + 32 * (sgitg &  1)) \
                               + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * ne0 + im*ne1*ne0;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = dst + (BLOCK_SIZE_M * r0) + (BLOCK_SIZE_N * r1) * ne0 + im*ne1*ne0;
        if (sgitg == 0) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                    *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}

// same as kernel_mul_mm_impl, but src1 and dst are accessed via indices stored in src1ids
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
void kernel_mul_mm_id_impl(
        device const  uchar * src0,
        device const  uchar * src1,
        thread        short * src1ids,
        device        float * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne02,
        constant   uint64_t & nb01,
        constant   uint64_t & nb02,
        constant    int64_t & ne12,
        constant   uint64_t & nb10,
        constant   uint64_t & nb11,
        constant   uint64_t & nb12,
        constant    int64_t & ne0,
                    int64_t   ne1,
        constant       uint & r2,
        constant       uint & r3,
        threadgroup   uchar * shared_memory,
        uint3                 tgpig[[threadgroup_position_in_grid]],
        uint                  tiitg[[thread_index_in_threadgroup]],
        uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;
    const uint im = tgpig.z;

    if (r1 * BLOCK_SIZE_N >= ne1) return;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    uint   offset0 = (i12/r2)*nb02 + (i13/r3)*(nb02*ne02);
    ushort offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb12 * im
        + nb11 * src1ids[r1 * BLOCK_SIZE_N + thread_col]
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        half4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i],lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i],lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = dst + (BLOCK_SIZE_M * r0) + im*ne1*ne0;
        if (sgitg == 0) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                    *(C + i + src1ids[j + r1*BLOCK_SIZE_N] * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm(device const  uchar * src0,
                          device const  uchar * src1,
                          device        float * dst,
                          constant    int64_t & ne00,
                          constant    int64_t & ne02,
                          constant   uint64_t & nb01,
                          constant   uint64_t & nb02,
                          constant    int64_t & ne12,
                          constant   uint64_t & nb10,
                          constant   uint64_t & nb11,
                          constant   uint64_t & nb12,
                          constant    int64_t & ne0,
                          constant    int64_t & ne1,
                          constant       uint & r2,
                          constant       uint & r3,
                          threadgroup   uchar * shared_memory [[threadgroup(0)]],
                          uint3                 tgpig[[threadgroup_position_in_grid]],
                          uint                  tiitg[[thread_index_in_threadgroup]],
                          uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_impl<block_q, nl, dequantize_func>(
        src0,
        src1,
        dst,
        ne00,
        ne02,
        nb01,
        nb02,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        ne1,
        r2,
        r3,
        shared_memory,
        tgpig,
        tiitg,
        sgitg);
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm_id(
        device const   uchar * ids,
        device const   uchar * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne02,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const   uchar * src00,
        device const   uchar * src01,
        device const   uchar * src02,
        device const   uchar * src03,
        device const   uchar * src04,
        device const   uchar * src05,
        device const   uchar * src06,
        device const   uchar * src07,
        threadgroup    uchar * shared_memory [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const uchar * src0s[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    // expert id
    const int32_t id = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    // row indices of src1 for expert id
    int64_t _ne1 = 0;
    short src1ids[512];

    for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (((device int32_t *) (ids + i1*nbi1))[idx] == id) {
            src1ids[_ne1++] = i1;
        }
    }

    kernel_mul_mm_id_impl<block_q, nl, dequantize_func>(
        src0s[id],
        src1,
        src1ids,
        dst,
        ne00,
        ne02,
        nb01,
        nb02,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        _ne1,
        r2,
        r3,
        shared_memory,
        tgpig,
        tiitg,
        sgitg);
}

#if QK_K == 256
#define QK_NL 16
#else
#define QK_NL 4
#endif

//
// get rows
//

typedef void (get_rows_t)(
        device const void * src0,
        device const char * src1,
        device      float * dst,
        constant  int64_t & ne00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant  int64_t & ne10,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        uint3, uint, uint3);

//template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_t kernel_get_rows<float4x4,   1, dequantize_f32>;
//template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_t kernel_get_rows<half4x4,    1, dequantize_f16>;
template [[host_name("kernel_get_rows_q4_0")]] kernel get_rows_t kernel_get_rows<block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_get_rows_q4_1")]] kernel get_rows_t kernel_get_rows<block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_get_rows_q5_0")]] kernel get_rows_t kernel_get_rows<block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_get_rows_q5_1")]] kernel get_rows_t kernel_get_rows<block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_get_rows_q8_0")]] kernel get_rows_t kernel_get_rows<block_q8_0, 2, dequantize_q8_0>;
template [[host_name("kernel_get_rows_q2_K")]] kernel get_rows_t kernel_get_rows<block_q2_K, QK_NL, dequantize_q2_K>;
template [[host_name("kernel_get_rows_q3_K")]] kernel get_rows_t kernel_get_rows<block_q3_K, QK_NL, dequantize_q3_K>;
template [[host_name("kernel_get_rows_q4_K")]] kernel get_rows_t kernel_get_rows<block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("kernel_get_rows_q5_K")]] kernel get_rows_t kernel_get_rows<block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("kernel_get_rows_q6_K")]] kernel get_rows_t kernel_get_rows<block_q6_K, QK_NL, dequantize_q6_K>;
template [[host_name("kernel_get_rows_iq2_xxs")]] kernel get_rows_t kernel_get_rows<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_get_rows_iq2_xs")]]  kernel get_rows_t kernel_get_rows<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_get_rows_iq3_xxs")]] kernel get_rows_t kernel_get_rows<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_get_rows_iq1_s")]]   kernel get_rows_t kernel_get_rows<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_get_rows_iq4_nl")]]  kernel get_rows_t kernel_get_rows<block_iq4_nl,  2, dequantize_iq4_nl>;

//
// matrix-matrix multiplication
//

typedef void (mat_mm_t)(
        device const  uchar * src0,
        device const  uchar * src1,
        device        float * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne02,
        constant   uint64_t & nb01,
        constant   uint64_t & nb02,
        constant    int64_t & ne12,
        constant   uint64_t & nb10,
        constant   uint64_t & nb11,
        constant   uint64_t & nb12,
        constant    int64_t & ne0,
        constant    int64_t & ne1,
        constant       uint & r2,
        constant       uint & r3,
        threadgroup   uchar *,
        uint3, uint, uint);

template [[host_name("kernel_mul_mm_f32_f32")]]  kernel mat_mm_t kernel_mul_mm<float4x4,   1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_f16_f32")]]  kernel mat_mm_t kernel_mul_mm<half4x4,    1,     dequantize_f16>;
template [[host_name("kernel_mul_mm_q4_0_f32")]] kernel mat_mm_t kernel_mul_mm<block_q4_0, 2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_q4_1_f32")]] kernel mat_mm_t kernel_mul_mm<block_q4_1, 2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_q5_0_f32")]] kernel mat_mm_t kernel_mul_mm<block_q5_0, 2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mm_q5_1_f32")]] kernel mat_mm_t kernel_mul_mm<block_q5_1, 2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mm_q8_0_f32")]] kernel mat_mm_t kernel_mul_mm<block_q8_0, 2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_q2_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q2_K, QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_q3_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q3_K, QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_q4_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_q5_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_q6_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q6_K, QK_NL, dequantize_q6_K>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mat_mm_t kernel_mul_mm<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mat_mm_t kernel_mul_mm<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mat_mm_t kernel_mul_mm<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mat_mm_t kernel_mul_mm<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mat_mm_t kernel_mul_mm<block_iq4_nl,  2, dequantize_iq4_nl>;

//
// indirect matrix-matrix multiplication
//

typedef void (mat_mm_id_t)(
        device const   uchar * ids,
        device const   uchar * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne02,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const   uchar * src00,
        device const   uchar * src01,
        device const   uchar * src02,
        device const   uchar * src03,
        device const   uchar * src04,
        device const   uchar * src05,
        device const   uchar * src06,
        device const   uchar * src07,
        threadgroup    uchar *,
        uint3, uint, uint);

template [[host_name("kernel_mul_mm_id_f32_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<float4x4,   1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<half4x4,    1,     dequantize_f16>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_q4_0, 2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_q4_1, 2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_q5_0, 2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_q5_1, 2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_q8_0, 2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_q2_K, QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_q3_K, QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_q6_K, QK_NL, dequantize_q6_K>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq4_nl,  2, dequantize_iq4_nl>;

//
// matrix-vector multiplication
//

[[host_name("kernel_mul_mv_id_f32_f32")]]
kernel void kernel_mul_mv_id_f32_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_f32_f32_impl(
        src0[id],
        src1 + bid*nb11,
        dst  + bid*ne0,
        ne00,
        ne01,
        ne02,
        nb00,
        nb01,
        nb02,
        ne10,
        ne11,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg);
}

[[host_name("kernel_mul_mv_id_f16_f32")]]
kernel void kernel_mul_mv_id_f16_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_f16_f32_impl(
        src0[id],
        src1 + bid*nb11,
        dst  + bid*ne0,
        ne00,
        ne01,
        ne02,
        nb00,
        nb01,
        nb02,
        ne10,
        ne11,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg);
}

[[host_name("kernel_mul_mv_id_q8_0_f32")]]
kernel void kernel_mul_mv_id_q8_0_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_q8_0_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_q4_0_f32")]]
kernel void kernel_mul_mv_id_q4_0_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_q4_1_f32")]]
kernel void kernel_mul_mv_id_q4_1_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    mul_vec_q_n_f32_impl<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_q5_0_f32")]]
kernel void kernel_mul_mv_id_q5_0_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    mul_vec_q_n_f32_impl<block_q5_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_q5_1_f32")]]
kernel void kernel_mul_mv_id_q5_1_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    mul_vec_q_n_f32_impl<block_q5_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_q2_K_f32")]]
kernel void kernel_mul_mv_id_q2_K_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_q2_K_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_q3_K_f32")]]
kernel void kernel_mul_mv_id_q3_K_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_q3_K_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_q4_K_f32")]]
kernel void kernel_mul_mv_id_q4_K_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_q4_K_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_q5_K_f32")]]
kernel void kernel_mul_mv_id_q5_K_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_q5_K_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_q6_K_f32")]]
kernel void kernel_mul_mv_id_q6_K_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_q6_K_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_iq2_xxs_f32")]]
kernel void kernel_mul_mv_id_iq2_xxs_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        threadgroup int8_t   * shared_values [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_iq2_xxs_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        shared_values,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_iq2_xs_f32")]]
kernel void kernel_mul_mv_id_iq2_xs_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        threadgroup int8_t   * shared_values [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_iq2_xs_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        shared_values,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_iq3_xxs_f32")]]
kernel void kernel_mul_mv_id_iq3_xxs_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        threadgroup int8_t   * shared_values [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_iq3_xxs_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        shared_values,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_iq1_s_f32")]]
kernel void kernel_mul_mv_id_iq1_s_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_iq1_s_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg,
        sgitg);
}

[[host_name("kernel_mul_mv_id_iq4_nl_f32")]]
kernel void kernel_mul_mv_id_iq4_nl_f32(
        device const    char * ids,
        device const    char * src1,
        device         float * dst,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        constant        uint & r2,
        constant        uint & r3,
        constant         int & idx,
        device const    char * src00,
        device const    char * src01,
        device const    char * src02,
        device const    char * src03,
        device const    char * src04,
        device const    char * src05,
        device const    char * src06,
        device const    char * src07,
        threadgroup float    * shared_values [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    device const char * src0[8] = {src00, src01, src02, src03, src04, src05, src06, src07};

    const int64_t bid = tgpig.z/(ne12*ne13);

    tgpig.z = tgpig.z%(ne12*ne13);

    const int32_t id = ((device int32_t *) (ids + bid*nbi1))[idx];

    kernel_mul_mv_iq4_nl_f32_impl(
        src0[id],
        (device const float *) (src1 + bid*nb11),
        dst + bid*ne0,
        ne00,
        ne01,
        ne02,
        ne10,
        ne12,
        ne0,
        ne1,
        r2,
        r3,
        shared_values,
        tgpig,
        tiisg,
        sgitg);
}
