#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <stdio.h>
#include <atomic>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "ggml-cuda.h"
#include "ggml.h"

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define CUDA_CHECK(err)                                                                 \
    do {                                                                                \
        cudaError_t err_ = (err);                                                       \
        if (err_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,   \
                cudaGetErrorString(err_));                                              \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#if CUDART_VERSION >= 12
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d: %s\n",                         \
                    err_, __FILE__, __LINE__, cublasGetStatusString(err_));             \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#else
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);  \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#endif // CUDART_VERSION >= 11

typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, float & v0, float & v1);
typedef void (*to_fp32_cuda_t)(const void * x, float * y, int k, cudaStream_t stream);
typedef void (*dot_kernel_k_t)(const void * vx, const int ib, const int iqs, const float * y, float & v);
typedef void (*ggml_cuda_func_t)(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
typedef void (*ggml_cuda_op_t)(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i, float * src0_ddf_i,
    float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main);

// QK = number of values after dequantization
// QR = QK / number of values before dequantization

#define QK4_0 32
#define QR4_0 2
typedef struct {
    half    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
#define QR4_1 2
typedef struct {
    half    d;              // delta
    half    m;              // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(ggml_fp16_t) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
#define QR5_0 2
typedef struct {
    half d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
#define QR5_1 2
typedef struct {
    half d;                 // delta
    half m;                 // min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
#define QR8_0 1
typedef struct {
    half    d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

//================================= k-quants

#define QK_K 256

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half d;                  // super-block scale for quantized scales
    half dmin;               // super-block scale for quantized mins
} block_q2_k;
static_assert(sizeof(block_q2_k) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4, "wrong q2_k block size/padding");

typedef struct {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4]; // nibbles / quants
    uint8_t scales[3*QK_K/64];
    half d;
} block_q3_k;
static_assert(sizeof(block_q3_k) == sizeof(ggml_fp16_t) + QK_K / 4 + 11 * QK_K / 64, "wrong q3_k block size/padding");

typedef struct {
    half d;                    // super-block scale for quantized scales
    half dmin;                 // super-block scale for quantized mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_k;
static_assert(sizeof(block_q4_k) == 2*sizeof(ggml_fp16_t) + 3*QK_K/64 + QK_K/2, "wrong q4_k block size/padding");

typedef struct {
    half    d;                   // super-block scale for quantized scales
    half    dmin;                // super-block scale for quantized mins
    uint8_t scales[3*QK_K/64];   // scales, quantized with 6 bits
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_k;
static_assert(sizeof(block_q5_k) == 2*sizeof(ggml_fp16_t) + 3*QK_K/64 + QK_K/2 + QK_K/8, "wrong q5_k block size/padding");

typedef struct {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    half    d;         // delta
} block_q6_k;
static_assert(sizeof(block_q6_k) == sizeof(ggml_fp16_t) + 13*QK_K/16, "wrong q6_k block size/padding");

#define WARP_SIZE 32

#define CUDA_ADD_BLOCK_SIZE 256
#define CUDA_MUL_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_ROPE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_CUDA_DMMV_X
#define GGML_CUDA_DMMV_X 32
#endif
#ifndef GGML_CUDA_DMMV_Y
#define GGML_CUDA_DMMV_Y 1
#endif

static __global__ void add_f32(const float * x, const float * y, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] + y[i];
}

static __global__ void mul_f32(const float * x, const float * y, float * dst, const int kx, const int ky) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= kx) {
        return;
    }
    dst[i] = x[i] * y[i%ky];
}

static __global__ void silu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (1.0f + expf(-x[i]));
}

static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const float eps = 1e-6;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int i = 0; i < ncols; i += WARP_SIZE) {
        const int col = i + tid;
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const float mean = tmp / ncols;
    const float scale = 1.0f / sqrtf(mean + eps);

    for (int i = 0; i < ncols; i += WARP_SIZE) {
        const int col = i + tid;
        dst[row*ncols + col] = scale * x[row*ncols + col];
    }
}

static __device__ void dequantize_q4_0(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const uint8_t vui = x[ib].qs[iqs];

    const int8_t vi0 = vui & 0xF;
    const int8_t vi1 = vui >> 4;

    v0 = (vi0 - 8)*d;
    v1 = (vi1 - 8)*d;
}

static __device__ void dequantize_q4_1(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float d = x[ib].d;
    const float m = x[ib].m;

    const uint8_t vui = x[ib].qs[iqs];

    const int8_t vi0 = vui & 0xF;
    const int8_t vi1 = vui >> 4;

    v0 = vi0*d + m;
    v1 = vi1*d + m;
}

static __device__ void dequantize_q5_0(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const uint8_t xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    const int32_t x0 = ((x[ib].qs[iqs] & 0xf) | xh_0) - 16;
    const int32_t x1 = ((x[ib].qs[iqs] >>  4) | xh_1) - 16;

    v0 = x0*d;
    v1 = x1*d;
}

static __device__ void dequantize_q5_1(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float d = x[ib].d;
    const float m = x[ib].m;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const uint8_t xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    const int32_t x0 = ((x[ib].qs[iqs] & 0xf) | xh_0);
    const int32_t x1 = ((x[ib].qs[iqs] >>  4) | xh_1);

    v0 = x0*d + m;
    v1 = x1*d + m;
}

static __device__ void dequantize_q8_0(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    const int8_t vi0 = x[ib].qs[iqs + 0];
    const int8_t vi1 = x[ib].qs[iqs + 1];

    v0 = vi0*d;
    v1 = vi1*d;
}

//================================== k-quants

static __global__ void dequantize_block_q2_k(const void * vx, float * yy) {

    const int i   = blockIdx.x;
    const int tid = threadIdx.x;
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const block_q2_k * x = (const block_q2_k *) vx;

    const uint8_t q = x[i].qs[32*n + l];
    float * y = yy + i*QK_K + 128*n;

    float dall = x[i].d;
    float dmin = x[i].dmin;
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);

}

static __device__ void vec_dot_q2_k(const void * vx, const int ib, const int iqs, const float * yy, float & result) {

    const block_q2_k * x = (const block_q2_k *) vx;

    // if n is 0, we want to do the lower 128, else the upper 128,
    // covering y[l+0],  y[l+32], y[l+64], y[l+96] and
    //          y[l+16], y[l+48], y[l+80], y[l+112]
    int n = iqs/128;                // 0 or 1
    int r = iqs - 128*n;            // 0...120 in steps of 8
    int l = r/8;                    // 0...15 in steps of 1

    const float   * y = yy + 128*n + l;
    const uint8_t * q = x[ib].qs + 32*n + l;
    const uint8_t * s = x[ib].scales + 8*n;

    const float dall = x[ib].d;
    const float dmin = x[ib].dmin;

    float sum = y[  0] * (dall * ((s[0] & 0xF) * ((q[ 0] >> 0) & 3)) - dmin * (s[0] >> 4))
              + y[ 32] * (dall * ((s[2] & 0xF) * ((q[ 0] >> 2) & 3)) - dmin * (s[2] >> 4))
              + y[ 64] * (dall * ((s[4] & 0xF) * ((q[ 0] >> 4) & 3)) - dmin * (s[4] >> 4))
              + y[ 96] * (dall * ((s[6] & 0xF) * ((q[ 0] >> 6) & 3)) - dmin * (s[6] >> 4))
              + y[ 16] * (dall * ((s[1] & 0xF) * ((q[16] >> 0) & 3)) - dmin * (s[1] >> 4))
              + y[ 48] * (dall * ((s[3] & 0xF) * ((q[16] >> 2) & 3)) - dmin * (s[3] >> 4))
              + y[ 80] * (dall * ((s[5] & 0xF) * ((q[16] >> 4) & 3)) - dmin * (s[5] >> 4))
              + y[112] * (dall * ((s[7] & 0xF) * ((q[16] >> 6) & 3)) - dmin * (s[7] >> 4));

    result = sum;

}

static __global__ void dequantize_block_q3_k(const void * vx, float * yy) {

    int r = threadIdx.x/4;
    int i = blockIdx.x;
    int tid = r/2;
    int is0 = r%2;
    int l0 = 16*is0 + 4*(threadIdx.x%4);
    int n = tid / 4;
    int j = tid - 4*n;

    const block_q3_k * x = (const block_q3_k *) vx;

    uint8_t m = 1 << (4*n + j);
    int is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    float * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));

}

static __device__ void vec_dot_q3_k(const void * vx, const int ib, const int iqs, const float * yy, float & result) {

    const block_q3_k * x = (const block_q3_k *) vx;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[3];
    uint32_t utmp[4];

    // if n is 0, we want to do the lower 128, else the upper 128,
    // covering y[l+0],  y[l+32], y[l+64], y[l+96] and
    //          y[l+16], y[l+48], y[l+80], y[l+112]
    int n = iqs/128;                // 0 or 1
    int r = iqs - 128*n;            // 0...120 in steps of 8
    int l = r/8;                    // 0...15 in steps of 1

    const float   * y = yy + 128*n + l;
    const uint8_t * q = x[ib].qs + 32*n + l;
    const uint8_t * hm = x[ib].hmask + l;
    const int8_t  * s = (const int8_t *)utmp + 8*n;

    memcpy(aux, x[ib].scales, 12);
    utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
    utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
    utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
    utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

    const float dall = x[ib].d;

    const uint8_t m = 1 << (4*n);

    float sum = y[  0] * (s[0] - 32) * (((q[ 0] >> 0) & 3) - (hm[ 0] & (m << 0) ? 0 : 4))
              + y[ 32] * (s[2] - 32) * (((q[ 0] >> 2) & 3) - (hm[ 0] & (m << 1) ? 0 : 4))
              + y[ 64] * (s[4] - 32) * (((q[ 0] >> 4) & 3) - (hm[ 0] & (m << 2) ? 0 : 4))
              + y[ 96] * (s[6] - 32) * (((q[ 0] >> 6) & 3) - (hm[ 0] & (m << 3) ? 0 : 4))
              + y[ 16] * (s[1] - 32) * (((q[16] >> 0) & 3) - (hm[16] & (m << 0) ? 0 : 4))
              + y[ 48] * (s[3] - 32) * (((q[16] >> 2) & 3) - (hm[16] & (m << 1) ? 0 : 4))
              + y[ 80] * (s[5] - 32) * (((q[16] >> 4) & 3) - (hm[16] & (m << 2) ? 0 : 4))
              + y[112] * (s[7] - 32) * (((q[16] >> 6) & 3) - (hm[16] & (m << 3) ? 0 : 4));

    result = sum * dall;

}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static __global__ void dequantize_block_q4_k(const void * vx, float * yy) {
    const block_q4_k * x = (const block_q4_k *) vx;

    const int i = blockIdx.x;

    //// assume 64 threads - this is very slightly better than the one below
    //const int tid = threadIdx.x;
    //const int il  = tid/16;
    //const int ir  = tid%16;
    //const int is  = 2*il;
    //const int n   = 2;

    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    float * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = x[i].d;
    const float dmin = x[i].dmin;

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
}

static __device__ void vec_dot_q4_k(const void * vx, const int ib, const int iqs, const float * yy, float & result) {

    const block_q4_k * x = (const block_q4_k *) vx;

                                    // iqs is in 0...248 in steps of 8 =>
    const int j  = iqs / 64;        // j  is in 0...3
    const int ir = (iqs - 64*j)/2;  // ir is in 0...28 in steps of 4
    const int is = 2*j;             // is is in 0...6 in steps of 2

    const float   * y = yy + 64*j + ir;
    const uint8_t * q = x[ib].qs + 32*j + ir;

    const float dall = x[ib].d;
    const float dmin = x[ib].dmin;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    float sum = 0;
    for (int k = 0; k < 4; ++k) {
        sum += y[k +  0] * (d1 * (q[k] & 0xF) - m1);
        sum += y[k + 32] * (d2 * (q[k] >>  4) - m2);
    }
    result = sum;

}

static __global__ void dequantize_block_q5_k(const void * vx, float * yy) {
    const block_q5_k * x = (const block_q5_k *) vx;

    const int i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    float * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = x[i].d;
    const float dmin = x[i].dmin;

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

static __device__ void vec_dot_q5_k(const void * vx, const int ib, const int iqs, const float * yy, float & result) {

    const block_q5_k * x = (const block_q5_k *) vx;

                                    // iqs is in 0...248 in steps of 8 =>
    const int j  = iqs / 64;        // j  is in 0...3
    const int ir = (iqs - 64*j)/2;  // ir is in 0...28 in steps of 4
    const int is = 2*j;             // is is in 0...6 in steps of 2

    const float   * y  = yy + 64*j + ir;
    const uint8_t * ql = x[ib].qs + 32*j + ir;
    const uint8_t * qh = x[ib].qh + ir;

    const float dall = x[ib].d;
    const float dmin = x[ib].dmin;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    uint8_t   hm  = 1 << is;
    float sum = 0;
    for (int k = 0; k < 4; ++k) {
        sum += y[k +  0] * (d1 * ((ql[k] & 0xF) + (qh[k] & hm ? 16 : 0)) - m1);
    }
    hm <<= 1;
    for (int k = 0; k < 4; ++k) {
        sum += y[k + 32] * (d2 * ((ql[k] >>  4) + (qh[k] & hm ? 16 : 0)) - m2);
    }
    result = sum;

}

static __global__ void dequantize_block_q6_k(const void * vx, float * yy) {
    const block_q6_k * x = (const block_q6_k *) vx;

    const int i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int ip  = tid/32;   // ip is 0 or 1
    const int il  = tid - 32*ip; // 0...32
    const int is  = 8*ip + il/16;

    float * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

static __device__ void vec_dot_q6_k(const void * vx, const int ib, const int iqs, const float * yy, float & result) {

    const block_q6_k * x = (const block_q6_k *) vx;

    const int ip = iqs / 128;        // 0 or 1
    const int il = (iqs - 128*ip)/8; // 0...15
    const int is = 8*ip;

    const float * y = yy + 128*ip + il;

    const float d = x[ib].d;

    const uint8_t * ql = x[ib].ql + 64*ip + il;
    const uint8_t * qh = x[ib].qh + 32*ip + il;
    const int8_t  * sc = x[ib].scales + is;

    result = y[  0] * d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh[ 0] >> 0) & 3) << 4)) - 32)
           + y[ 32] * d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh[ 0] >> 2) & 3) << 4)) - 32)
           + y[ 64] * d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh[ 0] >> 4) & 3) << 4)) - 32)
           + y[ 96] * d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh[ 0] >> 6) & 3) << 4)) - 32)
           + y[ 16] * d * sc[1] * ((int8_t)((ql[16] & 0xF) | (((qh[16] >> 0) & 3) << 4)) - 32)
           + y[ 48] * d * sc[3] * ((int8_t)((ql[48] & 0xF) | (((qh[16] >> 2) & 3) << 4)) - 32)
           + y[ 80] * d * sc[5] * ((int8_t)((ql[16]  >> 4) | (((qh[16] >> 4) & 3) << 4)) - 32)
           + y[112] * d * sc[7] * ((int8_t)((ql[48]  >> 4) | (((qh[16] >> 6) & 3) << 4)) - 32);

}

static __device__ void convert_f16(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const half * x = (const half *) vx;

    v0 = __half2float(x[ib + iqs + 0]);
    v1 = __half2float(x[ib + iqs + 1]);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void dequantize_block(const void * vx, float * y, const int k) {
    const int i = blockDim.x*blockIdx.x + 2*threadIdx.x;

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    float & v0 = y[iybs + iqs + 0];
    float & v1 = y[iybs + iqs + y_offset];
    dequantize_kernel(vx, ib, iqs, v0, v1);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void dequantize_mul_mat_vec(const void * vx, const float * y, float * dst, const int ncols) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            float v0, v1;
            dequantize_kernel(vx, ib, iqs + j/qr, v0, v1);
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val

            // matrix multiplication
            tmp += v0 * y[iybs + iqs + j/qr + 0];
            tmp += v1 * y[iybs + iqs + j/qr + y_offset];
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
        }
    }

    // sum up partial sums and write back result
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

template <int n_thread, dot_kernel_k_t dot_kernel>
static __global__ void dequantize_mul_mat_vec_k(const void * vx, const float * y, float * dst, const int ncols) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const int iter_stride = QK_K;
    const int vals_per_iter = iter_stride / n_thread;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    float tmp = 0; // partial sum for thread in warp

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = ib0 + col/QK_K; // x block index
        const int iqs = col%QK_K; // x quant index
        const int iybs = col - col%QK_K; // y block start index

        float v;
        dot_kernel(vx, ib, iqs, y + iybs, v);
        tmp += v;
    }

    // sum up partial sums and write back result
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static __global__ void rope_f32(const float * x, float * dst, const int ncols, const float p, const float theta_scale) {
    const int col = 2*(blockDim.x*blockIdx.x + threadIdx.x);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;

    const float theta = p*powf(theta_scale, col/2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

static void add_f32_cuda(const float * x, const float * y, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_ADD_BLOCK_SIZE - 1) / CUDA_ADD_BLOCK_SIZE;
    add_f32<<<num_blocks, CUDA_ADD_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
}

static void mul_f32_cuda(const float * x, const float * y, float * dst, const int kx, const int ky, cudaStream_t stream) {
    const int num_blocks = (kx + CUDA_MUL_BLOCK_SIZE - 1) / CUDA_MUL_BLOCK_SIZE;
    mul_f32<<<num_blocks, CUDA_MUL_BLOCK_SIZE, 0, stream>>>(x, y, dst, kx, ky);
}

static void silu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void rms_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    rms_norm_f32<<<nrows, block_dims, 0, stream>>>(x, dst, ncols);
}

static void dequantize_row_q4_0_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK4_0, QR4_0, dequantize_q4_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q4_1_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK4_1, QR4_1, dequantize_q4_1><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q5_0_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK5_0, QR5_0, dequantize_q5_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q5_1_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK5_1, QR5_1, dequantize_q5_1><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q8_0_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK8_0, QR8_0, dequantize_q8_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q2_k_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_k<<<nb, 64, 0, stream>>>(vx, y);
}

static void dequantize_row_q3_k_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_k<<<nb, 64, 0, stream>>>(vx, y);
}

static void dequantize_row_q4_k_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_k<<<nb, 32, 0, stream>>>(vx, y);
}

static void dequantize_row_q5_k_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_k<<<nb, 64, 0, stream>>>(vx, y);
}

static void dequantize_row_q6_k_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_k<<<nb, 64, 0, stream>>>(vx, y);
}

static void dequantize_mul_mat_vec_q4_0_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q4_1_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q5_0_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q5_1_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q8_0_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q2_k_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2;
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_k<32, vec_dot_q2_k><<<(nrows + ny - 1)/ny, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q3_k_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const dim3 block_dims(32, 2, 1);
    dequantize_mul_mat_vec_k<32, vec_dot_q3_k><<<nrows/2, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q4_k_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const dim3 block_dims(32, 2, 1);
    dequantize_mul_mat_vec_k<32, vec_dot_q4_k><<<nrows/2, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q5_k_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const dim3 block_dims(32, 2, 1);
    dequantize_mul_mat_vec_k<32, vec_dot_q5_k><<<nrows/2, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q6_k_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const dim3 block_dims(32, 2, 1);
    dequantize_mul_mat_vec_k<32, vec_dot_q6_k><<<nrows/2, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void convert_fp16_to_fp32_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<1, 1, convert_f16><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void convert_mul_mat_vec_f16_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<1, 1, convert_f16>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_row_q5_0_cuda;
        case GGML_TYPE_Q5_1:
            return dequantize_row_q5_1_cuda;
        case GGML_TYPE_Q8_0:
            return dequantize_row_q8_0_cuda;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_k_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_k_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_k_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_k_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_k_cuda;
        case GGML_TYPE_F16:
            return convert_fp16_to_fp32_cuda;
        default:
            return nullptr;
    }
}

static void rope_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float p, const float theta_scale, cudaStream_t stream) {
    GGML_ASSERT(nrows % 2 == 0);
    const dim3 block_dims(2*CUDA_ROPE_BLOCK_SIZE, 1, 1);
    const int num_blocks_x = (ncols + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(num_blocks_x, nrows, 1);
    rope_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, p, theta_scale);
}

// buffer pool for cuda
#define MAX_CUDA_BUFFERS 256

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

struct cuda_buffer {
    void * ptr = nullptr;
    size_t size = 0;
};

static cuda_buffer g_cuda_buffer_pool[GGML_CUDA_MAX_DEVICES][MAX_CUDA_BUFFERS];
static std::atomic_flag g_cuda_pool_lock = ATOMIC_FLAG_INIT;

static void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[id][i];
        if (b.size >= size && b.ptr != nullptr) {
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
    }
    void * ptr;
    CUDA_CHECK(cudaMalloc((void **) &ptr, size));
    *actual_size = size;
    return ptr;
}

static void ggml_cuda_pool_free(void * ptr, size_t size) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[id][i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
    CUDA_CHECK(cudaFree(ptr));
}


static void * g_scratch_buffer = nullptr;
static size_t g_scratch_size = 1024*1024*1024; // 1 GB by default
static size_t g_scratch_offset = 0;

#define GGML_CUDA_MAX_STREAMS 8 // Set this to 1 for reproducible matrix multiplication.
#define GGML_CUDA_MAX_EVENTS 64

static int g_device_count = -1;
static int g_main_device = 0;
static float g_tensor_split[GGML_CUDA_MAX_DEVICES] = {0};

static cublasHandle_t g_cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

static cudaStream_t g_cudaStreams_main[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { nullptr };

static cudaStream_t g_cudaStreams_memcpy_src1[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { nullptr };
static cudaEvent_t g_cudaEvents_memcpy_src1[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_EVENTS] = { nullptr };

void ggml_init_cublas() {
    static bool initialized = false;

    if (!initialized) {
        CUDA_CHECK(cudaGetDeviceCount(&g_device_count));
        GGML_ASSERT(g_device_count <= GGML_CUDA_MAX_DEVICES);
        int64_t total_vram = 0;
        fprintf(stderr, "%s: found %d CUDA devices:\n", __func__, g_device_count);
        for (int id = 0; id < g_device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            fprintf(stderr, "  Device %d: %s\n", id, prop.name);
            g_tensor_split[id] = total_vram;
            total_vram += prop.totalGlobalMem;
        }
        for (int id = 0; id < g_device_count; ++id) {
            g_tensor_split[id] /= total_vram;
        }

        for (int id = 0; id < g_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));

            // create streams
            for (int i = 0; i < GGML_CUDA_MAX_STREAMS; ++i) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams_main[id][i], cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams_memcpy_src1[id][i], cudaStreamNonBlocking));
            }
            // create events
            for (int i = 0; i < GGML_CUDA_MAX_EVENTS; ++i) {
                CUDA_CHECK(cudaEventCreateWithFlags(&g_cudaEvents_memcpy_src1[id][i], cudaEventDisableTiming));
            }

            // create cublas handle
            CUBLAS_CHECK(cublasCreate(&g_cublas_handles[id]));
            CUBLAS_CHECK(cublasSetMathMode(g_cublas_handles[id], CUBLAS_TF32_TENSOR_OP_MATH));
        }

        // configure logging to stdout
        // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

        initialized = true;
    }
}

void ggml_cuda_set_tensor_split(const float * tensor_split) {
    bool all_zero = true;
    for (int i = 0; i < g_device_count; ++i) {
        if (tensor_split[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return;
    }
    float split_sum = 0.0f;
    for (int i = 0; i < g_device_count; ++i) {
        g_tensor_split[i] = split_sum;
        split_sum += tensor_split[i];
    }
    for (int i = 0; i < g_device_count; ++i) {
        g_tensor_split[i] /= split_sum;
    }
}

void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory: %s\n",
            size/1024.0/1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

void ggml_cuda_host_free(void * ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

static cudaError_t ggml_cuda_h2d_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    char * dst_char = (char *) dst;
    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const void * x = (const void *) ((const char *) src->data + i1_low*nb1 + i2*nb2 + i3*nb3);
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_char, x, i1_diff*nb1, cudaMemcpyHostToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_char, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyHostToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_char + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyHostToDevice, stream);
            if (r != cudaSuccess) return r;
        }
        return cudaSuccess;
    }
}

inline void ggml_cuda_op_add(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne0 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    add_f32_cuda(src0_ddf_i, src1_ddf_i, dst_ddf_i, ne0*i01_diff, cudaStream_main);
    CUDA_CHECK(cudaGetLastError());

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_mul(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    for (int64_t i01 = i01_low; i01 < i01_high; i01++) {
        const int64_t i11 = i1*ne11 + i01%ne11; // broadcast src1 across src0

        float * src0_ddf_i01 = src0_ddf_i + i01*ne00;
        float * src1_ddf_i01 = src1_ddf_i + i11*ne10;
        float * dst_ddf_i01 = dst_ddf_i + i01*ne00;

        // compute
        mul_f32_cuda(src0_ddf_i01, src1_ddf_i01, dst_ddf_i01, ne00, ne10, cudaStream_main);
        CUDA_CHECK(cudaGetLastError());
    }

    (void) dst;
    (void) src0_ddq_i;
    (void) i02;
}

inline void ggml_cuda_op_silu(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    silu_f32_cuda(src0_ddf_i, dst_ddf_i, ne00*i01_diff, cudaStream_main);
    CUDA_CHECK(cudaGetLastError());

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_rms_norm(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    rms_norm_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, cudaStream_main);
    CUDA_CHECK(cudaGetLastError());

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_dequantize_mul_mat_vec(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddq_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = i01_high - i01_low;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            dequantize_mul_mat_vec_q4_0_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_mul_mat_vec_q8_0_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_q2_k_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_q3_k_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_q4_k_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_q5_k_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_q6_k_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_F16:
            convert_mul_mat_vec_f16_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
    CUDA_CHECK(cudaGetLastError());

    (void) src1;
    (void) dst;
    (void) src0_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_mul_mat_cublas(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int64_t ne0 = dst->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = dst->backend == GGML_BACKEND_GPU && id == g_main_device ? ne0 : i01_diff;

    CUBLAS_CHECK(cublasSetStream(g_cublas_handles[id], cudaStream_main));
    CUBLAS_CHECK(
        cublasSgemm(g_cublas_handles[id], CUBLAS_OP_T, CUBLAS_OP_N,
                i01_diff, ne11, ne10,
                &alpha, src0_ddf_i, ne00,
                        src1_ddf_i, ne10,
                &beta,  dst_ddf_i,  ldc));

    (void) dst;
    (void) src0_ddq_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_rope(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    const int n_past = ((int32_t *) src1->data)[0];
    const int n_dims = ((int32_t *) src1->data)[1];
    const int mode   = ((int32_t *) src1->data)[2];
    GGML_ASSERT(mode == 0);

    const float theta_scale = powf(10000.0, -2.0f/n_dims);
    const float p = ((mode & 1) == 0 ? n_past + i02 : i02);

    // compute
    rope_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, p, theta_scale, cudaStream_main);
    CUDA_CHECK(cudaGetLastError());

    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i1;
}

static void ggml_cuda_op(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                         ggml_cuda_op_t op, bool src0_needs_f32) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t nrows0 = ggml_nrows(src0);

    const bool use_src1 = src1 != nullptr;
    const int64_t ne10 = use_src1 ? src1->ne[0] : 1;
    const int64_t ne11 = use_src1 ? src1->ne[1] : 1;
    const int64_t ne12 = use_src1 ? src1->ne[2] : 1;
    const int64_t ne13 = use_src1 ? src1->ne[3] : 1;

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    GGML_ASSERT(dst->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(!use_src1 || src1->backend != GGML_BACKEND_GPU_SPLIT);

    // strides for iteration over dims 3 and 2
    const int64_t src0_stride = ne00 * ne01;
    const int64_t src1_stride = ne10 * ne11;
    const int64_t dst_stride = ne0 * ne1;
    const int64_t num_iters = ne02 * ne03;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);

    struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    struct ggml_tensor_extra_gpu * src1_extra = use_src1 ? (ggml_tensor_extra_gpu *) src1->extra : nullptr;
    struct ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;

    const bool src0_on_device = src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT;
    const bool src0_is_f32 = src0->type == GGML_TYPE_F32;

    const bool split = src0->backend == GGML_BACKEND_GPU_SPLIT;

    const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);

    // dd = data device
    char  * src0_ddq[GGML_CUDA_MAX_DEVICES] = {nullptr}; // quantized
    float * src0_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr}; // float
    float * src1_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr};
    float * dst_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr};

    // asq = actual size quantized, asf = actual size float
    size_t src0_asq[GGML_CUDA_MAX_DEVICES] = {0};
    size_t src0_asf[GGML_CUDA_MAX_DEVICES] = {0};
    size_t src1_asf[GGML_CUDA_MAX_DEVICES] = {0};
    size_t dst_asf[GGML_CUDA_MAX_DEVICES] = {0};

    for (int id = 0; id < g_device_count; ++id) {
        if (!split && id != g_main_device) {
            continue;
        }

        const bool src1_on_device = use_src1 && src1->backend == GGML_BACKEND_GPU && id == g_main_device;
        const bool dst_on_device = dst->backend == GGML_BACKEND_GPU && id == g_main_device;

        int64_t row_low, row_high;
        if (split) {
            row_low = id == 0 ? 0 : nrows0*g_tensor_split[id];
            row_low -= row_low % GGML_CUDA_DMMV_Y;
            row_high = id == g_device_count - 1 ? nrows0 : nrows0*g_tensor_split[id + 1];
            row_high -= row_high % GGML_CUDA_DMMV_Y;
        } else {
            row_low = 0;
            row_high = nrows0;
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t row_diff = row_high - row_low;

        cudaSetDevice(id);

        if (src0_on_device) {
            if (src0_is_f32) {
                src0_ddf[id] = (float *) src0_extra->data_device[id];
            } else {
                src0_ddq[id] = (char *) src0_extra->data_device[id];
            }
        } else {
            if (src0_is_f32) {
                src0_ddf[id] = (float *) ggml_cuda_pool_malloc(row_diff*ne00 * sizeof(float), &src0_asf[id]);
            } else {
                src0_ddq[id] = (char *) ggml_cuda_pool_malloc(row_diff*ne00 * src0_ts/src0_bs, &src0_asq[id]);
            }
        }

        if (src0_needs_f32 && !src0_is_f32) {
            src0_ddf[id] = (float *) ggml_cuda_pool_malloc(row_diff*ne00 * sizeof(float), &src0_asf[id]);
        }

        if (use_src1) {
            if (src1_on_device) {
                src1_ddf[id] = (float *) src1_extra->data_device[id];
            } else {
                src1_ddf[id] = (float *) ggml_cuda_pool_malloc(num_iters*src1_stride * sizeof(float), &src1_asf[id]);
            }
        }
        if (dst_on_device) {
            dst_ddf[id] = (float *) dst_extra->data_device[id];
        } else {
            size_t size_dst_ddf = split ? row_diff*ne1 * sizeof(float) : num_iters*dst_stride * sizeof(float);
            dst_ddf[id] = (float *) ggml_cuda_pool_malloc(size_dst_ddf, &dst_asf[id]);
        }

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            const int64_t i13 = i03 % ne13;
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const int64_t i12 = i02 % ne12;

                const int64_t i0 = i03*ne02 + i02;
                const int64_t i0_offset_low = row_low/ne01;
                const int64_t i0_offset_high = row_high/ne01;

                int64_t i01_low = 0;
                int64_t i01_high = ne01;
                if (split) {
                    if (i0 < i0_offset_low || i0 > i0_offset_high) {
                        continue;
                    }
                    if (i0 == i0_offset_low) {
                        i01_low = row_low % ne01;
                    }
                    if (i0 == i0_offset_high) {
                        i01_high = row_high % ne01;
                    }
                }
                const int64_t i01_diff = i01_high - i01_low;
                if (i01_diff == 0) {
                    continue;
                }
                const int64_t i11 = i13*ne12 + i12;

                cudaStream_t cudaStream_main = g_cudaStreams_main[id][i0 % GGML_CUDA_MAX_STREAMS];
                cudaStream_t cudaStream_memcpy_src1 = g_cudaStreams_memcpy_src1[id][i0 % GGML_CUDA_MAX_STREAMS];
                cudaEvent_t  cudaEvent_memcpy_src1 = g_cudaEvents_memcpy_src1[id][i0 % GGML_CUDA_MAX_EVENTS];

                // for split tensors the data begins at i0 == i0_offset_low
                char  * src0_ddq_i = src0_ddq[id] + (i0 - i0_offset_low)*src0_stride*src0_ts/src0_bs;
                float * src0_ddf_i = src0_ddf[id] + (i0 - i0_offset_low)*src0_stride;
                float * src1_ddf_i = src1_ddf[id] + i11*src1_stride;
                float * dst_ddf_i = dst_ddf[id] + (i0 - i0_offset_low)*dst_stride;

                // for split tensors the data pointer needs to be rounded down
                // to the bin edge for i03, i02 bins beyond the first
                if (i0 - i0_offset_low > 0) {
                    src0_ddq_i -= (row_low % ne01)*ne00 * src0_ts/src0_bs;
                    src0_ddf_i -= (row_low % ne01)*ne00;
                }
                if (i0 - i0_offset_low > 0) {
                    dst_ddf_i -= (row_low % ne0)*ne1;
                }

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (dst->backend == GGML_BACKEND_GPU && id == g_main_device) {
                    dst_ddf_i += i01_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (use_src1) {
                    if (src1->backend == GGML_BACKEND_CPU) {
                        CUDA_CHECK(ggml_cuda_h2d_tensor_2d(src1_ddf_i, src1, i03, i02, 0, ne11, cudaStream_memcpy_src1));
                    } else if (src1->backend == GGML_BACKEND_GPU) {
                        if (id != g_main_device) {
                            float * src1_ddf_i_source = (float *) src1_extra->data_device[g_main_device];
                            src1_ddf_i_source += i11*src1_stride;
                            CUDA_CHECK(cudaMemcpyAsync(src1_ddf_i, src1_ddf_i_source, src1_stride*sizeof(float),
                                                    cudaMemcpyDeviceToDevice, cudaStream_memcpy_src1));
                        }
                    } else {
                        GGML_ASSERT(false);
                    }
                }
                CUDA_CHECK(cudaEventRecord(cudaEvent_memcpy_src1, cudaStream_memcpy_src1));
                if (!src0_on_device) {
                    if (src0_is_f32) {
                        CUDA_CHECK(ggml_cuda_h2d_tensor_2d(src0_ddf_i, src0, i03, i02, i01_low, i01_high, cudaStream_main));
                    } else {
                        CUDA_CHECK(ggml_cuda_h2d_tensor_2d(src0_ddq_i, src0, i03, i02, i01_low, i01_high, cudaStream_main));
                    }
                }

                // convert src0 to f32 if it's necessary for the ggml_cuda_op
                if (src0_needs_f32 && !src0_is_f32) {
                    to_fp32_cuda(src0_ddq_i, src0_ddf_i, i01_diff*ne00, cudaStream_main);
                    CUDA_CHECK(cudaGetLastError());
                }

                // wait with main stream until src1 memcpy is done
                CUDA_CHECK(cudaStreamWaitEvent(cudaStream_main, cudaEvent_memcpy_src1, 0));

                // do the computation
                op(src0, src1, dst, src0_ddq_i, src0_ddf_i, src1_ddf_i, dst_ddf_i, i02, i01_low, i01_high, i11, cudaStream_main);

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device;
                    cudaMemcpyKind kind;
                    if (dst->backend == GGML_BACKEND_CPU) {
                        dst_off_device = dst->data;
                        kind = cudaMemcpyDeviceToHost;
                    } else if (dst->backend == GGML_BACKEND_GPU) {
                        dst_off_device = dst_extra->data_device[g_main_device];
                        kind = cudaMemcpyDeviceToDevice;
                    } else {
                        GGML_ASSERT(false);
                    }
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of cuBLAS matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        for (int64_t j = 0; j < ne1; ++j) {
                            float * dhf_dst_i = (float *) ((char *) dst_off_device + (j*ne0 + i01_low)*sizeof(float) + i02*nb2 + i03*nb3);
                            CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_ddf_i + j*i01_diff, i01_diff*sizeof(float), kind, cudaStream_main));
                        }
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_ddf_i, dst_stride*sizeof(float), kind, cudaStream_main));
                    }
                }
            }
        }
    }

    // wait until each device is finished, then free their buffers
    for (int id = 0; id < g_device_count; ++id) {
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (src0_asq[id] > 0) {
            ggml_cuda_pool_free(src0_ddq[id], src0_asq[id]);
        }
        if (src0_asf[id] > 0) {
            ggml_cuda_pool_free(src0_ddf[id], src0_asf[id]);
        }
        if (src1_asf[id] > 0) {
            ggml_cuda_pool_free(src1_ddf[id], src1_asf[id]);
        }
        if (dst_asf[id] > 0) {
            ggml_cuda_pool_free(dst_ddf[id], dst_asf[id]);
        }
    }
}

void ggml_cuda_add(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_add, true);
}

void ggml_cuda_mul(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul, true);
}

void ggml_cuda_silu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_silu, true);
}

void ggml_cuda_rms_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_rms_norm, true);
}

bool ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    GGML_ASSERT(src0->backend != GGML_BACKEND_GPU);
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // if (strcmp(dst->name, "KQ") == 0 || strcmp(dst->name, "KQV") == 0) {
        // fprintf(stderr, "(%ld, %ld, %ld, %ld) + (%ld, %ld, %ld, %ld) -> (%ld, %ld, %ld, %ld)\n",
        //         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        //         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        //         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    //     return false;
    // }

    // TODO: find the optimal values for these
    if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        (ne0 >= 32 && ne1 >= 32 && ne10 >= 32)) {
        return true;
    }

    return false;
}

void ggml_cuda_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (src0->type == GGML_TYPE_F32) {
        ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul_mat_cublas, true);
    } else if (ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F16) {
        if (src1->ne[1] == 1) {
            ggml_cuda_op(src0, src1, dst, ggml_cuda_op_dequantize_mul_mat_vec, false);
        } else {
            ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul_mat_cublas, true);
        }
    } else {
        GGML_ASSERT(false);
    }
}

void ggml_cuda_rope(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_rope, true);
}

void ggml_cuda_nop(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    (void) src0;
    (void) src1;
    (void) dst;
}

void ggml_cuda_load_data(const char * fname, struct ggml_tensor * tensor, const size_t offset) {
    FILE * fp = fopen(fname, "rb");
    int nrows = ggml_nrows(tensor);
    const size_t nb1 = tensor->nb[1];
    ggml_backend backend = tensor->backend;
    struct ggml_tensor_extra_gpu * extra = new struct ggml_tensor_extra_gpu;

    for (int id = 0; id < g_device_count; ++id) {
        extra->data_device[id] = nullptr;

        if (backend == GGML_BACKEND_GPU && id != g_main_device) {
            continue;
        }

        cudaSetDevice(id);

        int row_low, row_high;
        if (backend == GGML_BACKEND_GPU) {
            row_low = 0;
            row_high = nrows;
        } else if (backend == GGML_BACKEND_GPU_SPLIT) {
            row_low = id == 0 ? 0 : nrows*g_tensor_split[id];
            row_low -= row_low % GGML_CUDA_DMMV_Y;
            row_high = id == g_device_count - 1 ? nrows : nrows*g_tensor_split[id + 1];
            row_high -= row_high % GGML_CUDA_DMMV_Y;
        } else {
            GGML_ASSERT(false);
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t nrows_split = row_high - row_low;

        const size_t offset_split = offset + row_low*nb1;
        const size_t size = ggml_nbytes_split(tensor, nrows_split);

        void * buf;
        CUDA_CHECK(cudaMalloc(&buf, size));
        void * buf_host = malloc(size);

#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset_split, SEEK_SET);
#else
        int ret = fseek(fp, (long) offset_split, SEEK_SET);
#endif
        GGML_ASSERT(ret == 0); // same

        size_t ret2 = fread(buf_host, size, 1, fp);
        if (ret2 != 1) {
            fprintf(stderr, "unexpectedly reached end of file");
            exit(1);
        }

        cudaMemcpy(buf, buf_host, size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        free(buf_host);
        extra->data_device[id] = buf;
    }

    tensor->extra = extra;
    fclose(fp);
}

void ggml_cuda_free_data(struct ggml_tensor * tensor) {
    if (tensor->backend != GGML_BACKEND_GPU && tensor->backend != GGML_BACKEND_GPU_SPLIT) {
        return;
    }

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int id = 0; id < g_device_count; ++id) {
        if (extra->data_device[id] == nullptr) {
            continue;
        }

        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaFree(extra->data_device[id]));
    }

    delete extra;
}

void ggml_cuda_assign_buffers(struct ggml_tensor * tensor) {
    if (tensor->src0 != nullptr && tensor->src0->op == GGML_OP_RESHAPE) {
        ggml_cuda_assign_buffers(tensor);
    }

    const size_t size = ggml_nbytes(tensor);
    GGML_ASSERT(size <= g_scratch_size);
    if (g_scratch_offset + size > g_scratch_size) {
        g_scratch_offset = 0;
    }

    tensor->backend = GGML_BACKEND_GPU;
    struct ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu;

    bool inplace = tensor->src0 != nullptr && tensor->src0->data == tensor->data;

    CUDA_CHECK(cudaSetDevice(g_main_device));
    if (inplace && tensor->src0->backend == GGML_BACKEND_GPU) {
        struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu * ) tensor->src0->extra;
        extra->data_device[g_main_device] = src0_extra->data_device;
        GGML_ASSERT(false);
    } else {
        char * data = (char *) g_scratch_buffer;
        if (data == nullptr) {
            CUDA_CHECK(cudaMalloc(&data, g_scratch_size));
            g_scratch_buffer = data;
        }
        extra->data_device[g_main_device] = data + g_scratch_offset;
    }

    // fprintf(stderr, "data=%p offset=%ld data_device=%p\n", data, g_scratch_offset, extra->data_device[0]);
    g_scratch_offset += size;
    // fprintf(stderr, "%s: scratch %d, %p - %p\n",
    //         tensor->name, g_scratch_index, data + g_scratch_offset, data + g_scratch_offset + size);

    GGML_ASSERT(g_scratch_offset <= g_scratch_size);
    tensor->extra = extra;
}

void ggml_cuda_set_main_device(int main_device) {
    if (main_device > g_device_count) {
        fprintf(stderr, "warning: cannot set main_device=%d because there are only %d devices. Using device %d instead.\n",
                main_device, g_device_count, g_main_device);
        return;
    }
    g_main_device = main_device;
    if (g_device_count > 1) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, g_main_device));
        fprintf(stderr, "%s: using device %d (%s) as main device\n", __func__, g_main_device, prop.name);
    }
}

void ggml_cuda_set_scratch_size(size_t scratch_size) {
    g_scratch_size = scratch_size;
}

bool ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor){
    ggml_cuda_func_t func;
    const bool any_on_device = tensor->backend == GGML_BACKEND_GPU
        || tensor->src0->backend == GGML_BACKEND_GPU || tensor->src0->backend == GGML_BACKEND_GPU_SPLIT
        || (tensor->src1 != nullptr && tensor->src1->backend == GGML_BACKEND_GPU);

    switch (tensor->op) {
        case GGML_OP_ADD:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_add;
            break;
        case GGML_OP_MUL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_mul;
            break;
        case GGML_OP_SILU:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_silu;
            break;
        case GGML_OP_RMS_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_rms_norm;
            break;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_cuda_can_mul_mat(tensor->src0, tensor->src1, tensor)) {
                return false;
            }
            func = ggml_cuda_mul_mat;
            break;
        case GGML_OP_RESHAPE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_nop;
            break;
        case GGML_OP_ROPE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_rope;
            break;
        default:
            return false;
    }

    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }
    func(tensor->src0, tensor->src1, tensor);
    return true;
}
