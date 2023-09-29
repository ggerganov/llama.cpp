#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

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

// general-purpose kernel for addition of two tensors
// pros: works for non-contiguous tensors, supports broadcast across dims 1, 2 and 3
// cons: not very efficient
kernel void kernel_add(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant  int64_t & nb00,
        constant  int64_t & nb01,
        constant  int64_t & nb02,
        constant  int64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant  int64_t & nb10,
        constant  int64_t & nb11,
        constant  int64_t & nb12,
        constant  int64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant  int64_t & nb0,
        constant  int64_t & nb1,
        constant  int64_t & nb2,
        constant  int64_t & nb3,
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
        ((device float *)dst_ptr)[0] = ((device float *)src0_ptr)[0] + ((device float *)src1_ptr)[0];

        src0_ptr += ntg.x*nb00;
        src1_ptr += ntg.x*nb10;
        dst_ptr  += ntg.x*nb0;
    }
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant    int64_t & nb [[buffer(27)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig % nb];
}

kernel void kernel_mul(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig];
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_mul_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant    int64_t & nb,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig % nb];
}

kernel void kernel_scale(
        device const float4 * src0,
        device       float4 * dst,
        constant     float & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_silu(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];
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

kernel void kernel_soft_max(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    device const float * psrc0 = src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    device       float * pdst  = dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    // parallel max
    float lmax = tpitg[0] < ne00 ? psrc0[tpitg[0]] : -INFINITY;
    for (int i00 = tpitg[0] + ntg[0]; i00 < ne00; i00 += ntg[0]) {
        lmax = MAX(lmax, psrc0[i00]);
    }
    const float max = simd_max(lmax);

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        const float exp_psrc0 = exp(psrc0[i00] - max);
        lsum += exp_psrc0;
        // Remember the result of exp here. exp is expensive, so we really do not
        // whish to compute it twice.
        pdst[i00] = exp_psrc0;
    }

    const float sum = simd_sum(lsum);

    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        pdst[i00] /= sum;
    }
}

kernel void kernel_soft_max_4(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    device const float4 * psrc4 = (device const float4 *)(src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    device       float4 * pdst4 = (device       float4 *)(dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    // parallel max
    float4 lmax4 = tpitg[0] < ne00/4 ? psrc4[tpitg[0]] : -INFINITY;
    for (int i00 = tpitg[0] + ntg[0]; i00 < ne00/4; i00 += ntg[0]) {
        lmax4 = fmax(lmax4, psrc4[i00]);
    }
    float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    const float max = simd_max(lmax);

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg[0]; i00 < ne00/4; i00 += ntg[0]) {
        const float4 exp_psrc4 = exp(psrc4[i00] - max);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }
    float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    const float sum = simd_sum(lsum);

    for (int i00 = tpitg[0]; i00 < ne00/4; i00 += ntg[0]) {
        pdst4[i00] /= sum;
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
    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 2 + il/2);
    float2 acc = 0.f;
    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (acc[0] + acc[1]) + sumy * m;
}

// putting them in the kernel cause a significant performance penalty
#define N_DST 4 // each SIMD group works on 4 rows
#define N_SIMDGROUP 2 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 32 // assuming SIMD group size is 32
//Note: This is a template, but strictly speaking it only applies to
//      quantizations where the block size is 32. It also does not
//      giard against the number of rows not being divisible by
//      N_DST, so this is another explicit assumption of the implementation.
template<typename block_q_type, int nr, int nsg, int nw>
void mul_vec_q_n_f32(device const void * src0, device const float * src1, device float * dst,
                    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne10, int64_t ne12, int64_t ne0, int64_t ne1, uint gqa,
                    uint3 tgpig, uint tiisg, uint sgitg) {
    const int nb = ne00/QK4_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * nsg + sgitg) * nr;
    const uint offset0 = first_row * nb + im/gqa*(nb*ne0);
    device const block_q_type * x = (device const block_q_type *) src0 + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;
    float yl[16];       // src1 vector cache
    float sumf[nr]={0.f};

    const int ix = tiisg/2;
    const int il = 8*(tiisg%2);

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
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
        }
    }
}

kernel void kernel_mul_mat_q4_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01[[buffer(4)]],
        constant   int64_t & ne02[[buffer(5)]],
        constant   int64_t & ne10[[buffer(9)]],
        constant   int64_t & ne12[[buffer(11)]],
        constant   int64_t & ne0[[buffer(15)]],
        constant   int64_t & ne1[[buffer(16)]],
        constant   uint    & gqa[[buffer(17)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,gqa,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mat_q4_1_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01[[buffer(4)]],
        constant   int64_t & ne02[[buffer(5)]],
        constant   int64_t & ne10[[buffer(9)]],
        constant   int64_t & ne12[[buffer(11)]],
        constant   int64_t & ne0[[buffer(15)]],
        constant   int64_t & ne1[[buffer(16)]],
        constant   uint    & gqa[[buffer(17)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,gqa,tgpig,tiisg,sgitg);
}

#define NB_Q8_0 8

kernel void kernel_mul_mat_q8_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01[[buffer(4)]],
        constant   int64_t & ne02[[buffer(5)]],
        constant   int64_t & ne10[[buffer(9)]],
        constant   int64_t & ne12[[buffer(11)]],
        constant   int64_t & ne0[[buffer(15)]],
        constant   int64_t & ne1[[buffer(16)]],
        constant   uint    & gqa[[buffer(17)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = ne00/QK8_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * nsg + sgitg) * nr;
    const uint offset0 = first_row * nb + im/gqa*(nb*ne0);
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

#define N_F32_F32 4

kernel void kernel_mul_mat_f32_f32(
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
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t rb = tgpig.y*N_F32_F32;
    const int64_t im = tgpig.z;

    device const float * x = (device const float *) (src0 + r0*nb01 + im/(ne12/ne02)*nb02);

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

kernel void kernel_mul_mat_f16_f32_1row(
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
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    device const half  * x = (device const half  *) (src0 + r0*nb01 + im/(ne12/ne02)*nb02);
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

#define N_F16_F32 4

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
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t rb = tgpig.y*N_F16_F32;
    const int64_t im = tgpig.z;

    device const half * x = (device const half *) (src0 + r0*nb01 + im/(ne12/ne02)*nb02);

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

// Assumes row size (ne00) is a multiple of 4
kernel void kernel_mul_mat_f16_f32_l4(
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
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]]) {

    const int nrows = ne11;
    const int64_t r0 = tgpig.x;
    const int64_t im = tgpig.z;

    device const half4 * x4 = (device const half4 *) (src0 + r0*nb01 + im/(ne12/ne02)*nb02);

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
        constant       float & freq_base,
        constant       float & freq_scale,
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
        constant       float & freq_base,
        constant       float & freq_scale,
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        uint3 tgpig[[threadgroup_position_in_grid]]) {
    const int64_t i3 = tgpig[2];
    const int64_t i2 = tgpig[1];
    const int64_t i1 = tgpig[0];

    const bool is_neox = mode & 2;

    device const int32_t * pos = src1;

    const int64_t p = pos[i2];

    const float theta_0 = freq_scale * (float)p;
    const float inv_ndims = -1.f/n_dims;

    if (!is_neox) {
        for (int64_t i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x) {

            const float theta = theta_0 * pow(freq_base, inv_ndims*i0);
            const float cos_theta = cos(theta);
            const float sin_theta = sin(theta);

            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            const T x0 = src[0];
            const T x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        }
    } else {
        for (int64_t ib = 0; ib < ne0/n_dims; ++ib) {
            for (int64_t ic = 2*tiitg; ic < n_dims; ic += 2*tptg.x) {

                const float theta = theta_0 * pow(freq_base, inv_ndims*ic - ib);
                const float cos_theta = cos(theta);
                const float sin_theta = sin(theta);

                const int64_t i0 = ib*n_dims + ic/2;

                device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                const float x0 = src[0];
                const float x1 = src[n_dims/2];

                dst_data[0]        = x0*cos_theta - x1*sin_theta;
                dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
            }
        }
    }
}

template [[host_name("kernel_rope_f32")]] kernel rope_t kernel_rope<float>;
template [[host_name("kernel_rope_f16")]] kernel rope_t kernel_rope<half>;

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

static inline uchar4 get_scale_min_k4(int j, device const uint8_t * q) {
    uchar4 r;
    if (j < 4) {
        r[0] = q[j+0] & 63;
        r[2] = q[j+1] & 63;
        r[1] = q[j+4] & 63;
        r[3] = q[j+5] & 63;
    } else {
        r[0] = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        r[2] = (q[j+5] & 0xF) | ((q[j-3] >> 6) << 4);
        r[1] = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
        r[3] = (q[j+5] >>  4) | ((q[j+1] >> 6) << 4);
    }
    return r;
}

//====================================== dot products =========================

kernel void kernel_mul_mat_q2_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01[[buffer(4)]],
        constant   int64_t & ne02[[buffer(5)]],
        constant   int64_t & ne10[[buffer(9)]],
        constant   int64_t & ne12[[buffer(11)]],
        constant   int64_t & ne0[[buffer(15)]],
        constant   int64_t & ne1[[buffer(16)]],
        constant   uint    & gqa[[buffer(17)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int r2 = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;
    const uint offset0 = r2/gqa*(nb*ne0);
    device const block_q2_K * x = (device const block_q2_K *) src0 + ib_row + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;
    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q2_K) * nb;

#if QK_K == 256
    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int im = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3
    const int is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * im + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*im + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * im + 4 * ir;
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
            dst[r1*ne0 + r2*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

#if QK_K == 256
kernel void kernel_mul_mat_q3_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01[[buffer(4)]],
        constant   int64_t & ne02[[buffer(5)]],
        constant   int64_t & ne10[[buffer(9)]],
        constant   int64_t & ne12[[buffer(11)]],
        constant   int64_t & ne0[[buffer(15)]],
        constant   int64_t & ne1[[buffer(16)]],
        constant   uint    & gqa[[buffer(17)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t r2 = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;
    const uint offset0 = r2/gqa*(nb*ne0);
    device const block_q3_K * x = (device const block_q3_K *) src0 + first_row*nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;

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
            dst[r1*ne0 + r2*ne0*ne1 + first_row + row] = sumf1[row];
        }
    }
}
#else
kernel void kernel_mul_mat_q3_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01[[buffer(4)]],
        constant   int64_t & ne02[[buffer(5)]],
        constant   int64_t & ne10[[buffer(9)]],
        constant   int64_t & ne12[[buffer(11)]],
        constant   int64_t & ne0[[buffer(15)]],
        constant   int64_t & ne1[[buffer(16)]],
        constant   uint    & gqa[[buffer(17)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t r2 = tgpig.z;

    const int row = 2 * r0 + sgitg;
    const uint offset0 = r2/gqa*(nb*ne0);
    device const block_q3_K * x = (device const block_q3_K *) src0 + row*nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;
    const int ix = tiisg/4;
    const int il = 4 * (tiisg%4);// 0, 4, 8, 12
    const int im = il/8;         // 0, 0, 1, 1
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
            const uint16_t hm = h[l/2] >> im;
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
        dst[r1*ne0 + r2*ne0*ne1 + row] = tot;
    }

}
#endif

#if QK_K == 256
kernel void kernel_mul_mat_q4_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01 [[buffer(4)]],
        constant   int64_t & ne02 [[buffer(5)]],
        constant   int64_t & ne10 [[buffer(9)]],
        constant   int64_t & ne12 [[buffer(11)]],
        constant   int64_t & ne0  [[buffer(15)]],
        constant   int64_t & ne1  [[buffer(16)]],
        constant   uint    & gqa  [[buffer(17)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int im = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int r2 = tgpig.z;
    //const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int first_row = r0 * N_DST;
    const int ib_row = first_row * nb;
    const uint offset0 = r2/gqa*(nb*ne0);
    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;
    float yl[16];
    float yh[16];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q4_K) * nb / 2;

    device const float * y4 = y + ix * QK_K + 64 * im + 8 * ir;

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

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + im;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * im + 4 * ir;
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
            dst[r1*ne0 + r2*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}
#else
kernel void kernel_mul_mat_q4_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01[[buffer(4)]],
        constant   int64_t & ne02[[buffer(5)]],
        constant   int64_t & ne10[[buffer(9)]],
        constant   int64_t & ne12[[buffer(11)]],
        constant   int64_t & ne0[[buffer(15)]],
        constant   int64_t & ne1[[buffer(16)]],
        constant   uint    & gqa[[buffer(17)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const int ix = tiisg/4;  // 0...7
    const int it = tiisg%4;  // 0...3

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int r2 = tgpig.z;
    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;
    const uint offset0 = r2/gqa*(nb*ne0);
    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;
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
            dst[r1*ne0+ r2*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}
#endif

kernel void kernel_mul_mat_q5_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01[[buffer(4)]],
        constant   int64_t & ne02[[buffer(5)]],
        constant   int64_t & ne10[[buffer(9)]],
        constant   int64_t & ne12[[buffer(11)]],
        constant   int64_t & ne0[[buffer(15)]],
        constant   int64_t & ne1[[buffer(16)]],
        constant   uint    & gqa[[buffer(17)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int r2 = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;
    const uint offset0 = r2/gqa*(nb*ne0);
    device const block_q5_K * x = (device const block_q5_K *) src0 + first_row*nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;

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
    const int im  = tid/4;
    const int ir  = tid%4;
    const int n   = 8;

    const int l0 = n*ir;
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1 = 1u << (2*im);
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
        device const uint16_t * a = (device const uint16_t *)x[i].scales + im;

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
    const int im = il/8;         // 0, 0, 1, 1
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
                const uint8_t hl = h[l] >> im;
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
            dst[r1*ne0 + r2*ne0*ne1 + first_row + row] = tot;
        }
    }

}

kernel void kernel_mul_mat_q6_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01[[buffer(4)]],
        constant   int64_t & ne02[[buffer(5)]],
        constant   int64_t & ne10[[buffer(9)]],
        constant   int64_t & ne12[[buffer(11)]],
        constant   int64_t & ne0[[buffer(15)]],
        constant   int64_t & ne1[[buffer(16)]],
        constant   uint    & gqa[[buffer(17)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int r2 = tgpig.z;

    const int row = 2 * r0 + sgitg;
    const uint offset0 = r2/gqa*(nb*ne0);
    device const block_q6_K * x = (device const block_q6_K *) src0 + row * nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + r2*ne00*ne1;

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
        dst[r1*ne0 + r2*ne0*ne1 + row] = tot;
    }
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
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const half d = xb->d;

    for (int i=0;i<16;i++) {
        reg[i/4][i%4] = (qs[i + 16*il] * d);
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const half d = xb->d;
    const half min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    half dl, ml;
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
    half dl = il<8 ? d_all * (dl_int - 32.h) : d_all * (dl_int / 16.h - 32.h);
    const half ml = 4.h * dl;

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
    const half d   = il < 2 ? xb->d : xb->d / 16.h;
    const half min = xb->dmin;
    const half dl = d * sc[0];
    const half ml = min * sc[1];
#else
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
    const half d = il < 2 ? xb->d : xb->d / 16.h;
    const half min = xb->dmin;
    const half dl = d * sc[0];
    const half ml = min * sc[1];

    const ushort mask = il<2 ? 0x0F : 0xF0;
    const half qh_val = il<2 ? 16.h : 256.h;
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
    half sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;
#else
    ql = ql + 16 * (il&1);
    half sc = scales[il];
#endif
    const uint16_t  kmask1 = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);
    const uint16_t  kmask2 = il>1 ? 0xF0              : 0x0F;
    const half        coef = il>1 ? 1.f/16.h          : 1.h;
    const half ml = d_all * sc * 32.h;
    const half dl = d_all * sc * coef;
    for (int i = 0; i < 16; ++i) {
        const half q = il&1 ? ((ql[i] & kmask2) | ((qh[i] & kmask1) << 2))
                            : ((ql[i] & kmask2) | ((qh[i] & kmask1) << 4));
        reg[i/4][i%4] = dl * q - ml;
    }
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
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
        dequantize_func(
            ((device const block_q *) ((device char *) src0 + r*nb01)) + ind/nl, ind%nl, temp);
        *(((device float4x4 *) ((device char *) dst + i*nb1)) + ind) = temp;
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
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm(device const  uchar * src0,
                          device const  uchar * src1,
                          device        float * dst,
                          constant    int64_t & ne00,
                          constant    int64_t & ne02,
                          constant    int64_t & nb01,
                          constant    int64_t & nb02,
                          constant    int64_t & ne12,
                          constant    int64_t & nb10,
                          constant    int64_t & nb11,
                          constant    int64_t & nb12,
                          constant    int64_t & ne0,
                          constant    int64_t & ne1,
                          constant       uint & gqa,
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

    uint   offset0 = im/gqa*nb02;
    ushort offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb12 * im
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        //load data and store to threadgroup memory
        half4x4 temp_a;
        dequantize_func(x, il, temp_a);
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

template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_t kernel_get_rows<float4x4,   1, dequantize_f32>;
template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_t kernel_get_rows<half4x4,    1, dequantize_f16>;
template [[host_name("kernel_get_rows_q4_0")]] kernel get_rows_t kernel_get_rows<block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_get_rows_q4_1")]] kernel get_rows_t kernel_get_rows<block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_get_rows_q8_0")]] kernel get_rows_t kernel_get_rows<block_q8_0, 2, dequantize_q8_0>;
template [[host_name("kernel_get_rows_q2_K")]] kernel get_rows_t kernel_get_rows<block_q2_K, QK_NL, dequantize_q2_K>;
template [[host_name("kernel_get_rows_q3_K")]] kernel get_rows_t kernel_get_rows<block_q3_K, QK_NL, dequantize_q3_K>;
template [[host_name("kernel_get_rows_q4_K")]] kernel get_rows_t kernel_get_rows<block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("kernel_get_rows_q5_K")]] kernel get_rows_t kernel_get_rows<block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("kernel_get_rows_q6_K")]] kernel get_rows_t kernel_get_rows<block_q6_K, QK_NL, dequantize_q6_K>;

typedef void (mat_mm_t)(
        device const  uchar * src0,
        device const  uchar * src1,
        device        float * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne02,
        constant    int64_t & nb01,
        constant    int64_t & nb02,
        constant    int64_t & ne12,
        constant    int64_t & nb10,
        constant    int64_t & nb11,
        constant    int64_t & nb12,
        constant    int64_t & ne0,
        constant    int64_t & ne1,
        constant       uint & gqa,
        threadgroup uchar *, uint3, uint, uint);

template [[host_name("kernel_mul_mm_f32_f32")]]  kernel mat_mm_t kernel_mul_mm<float4x4,   1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_f16_f32")]]  kernel mat_mm_t kernel_mul_mm<half4x4,    1,     dequantize_f16>;
template [[host_name("kernel_mul_mm_q4_0_f32")]] kernel mat_mm_t kernel_mul_mm<block_q4_0, 2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_q4_1_f32")]] kernel mat_mm_t kernel_mul_mm<block_q4_1, 2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_q8_0_f32")]] kernel mat_mm_t kernel_mul_mm<block_q8_0, 2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_q2_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q2_K, QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_q3_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q3_K, QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_q4_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_q5_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_q6_K_f32")]] kernel mat_mm_t kernel_mul_mm<block_q6_K, QK_NL, dequantize_q6_K>;
