#pragma once

// BF16X batched GEMM, phase 1: fused-decode tiled FFMA baseline.
//
// Weights stay compressed in VRAM; each thread block decodes its BF16X tile
// (128 x 32 weights, one 16-weight half per thread) into shared memory and
// consumes it with plain FFMAs. No decompressed weight ever reaches HBM.
//
// Phase 2 (planned) replaces the FFMA stage with half-precision MMA fragments
// (Marlin-style dequantize-into-fragments); the decode stage is unchanged.

#include "common.cuh"

#include <cstdint>

static __device__ __forceinline__ uint32_t bf16x_mm_load_u32(const uint8_t * p) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t * pa = (const uint32_t *) (a & ~(uintptr_t) 3);
    const int sh = (int) (a & 3) * 8;
    return sh == 0 ? pa[0] : __funnelshift_r(pa[0], pa[1], sh);
}

// decode one 16-weight half of a block_bf16x into float dst[16]
static __device__ __forceinline__ void bf16x_decode_half_to_f32(
        const block_bf16x * GGML_RESTRICT bx, const int iqs, float * GGML_RESTRICT dst) {

    const uint8_t * mb = bx->mant  + iqs * 14;
    const uint8_t * db = bx->delta + iqs * 6;
    const uint8_t * sb = bx->sgn   + iqs * 2;

    const uint32_t mw0 = bf16x_mm_load_u32(mb +  0);
    const uint32_t mw1 = bf16x_mm_load_u32(mb +  4);
    const uint32_t mw2 = bf16x_mm_load_u32(mb +  8);
    const uint32_t mw3 = bf16x_mm_load_u32(mb + 12);
    const uint32_t dw0 = bf16x_mm_load_u32(db +  0);
    const uint32_t dw1 = bf16x_mm_load_u32(db +  4);
    const uint32_t sw  = bf16x_mm_load_u32(sb);

    // quantize floors the half-block exponent at 7: pure add/sub assembly
    const uint32_t em7 = (uint32_t) bx->emax[iqs] << 7;

#pragma unroll
    for (int j = 0; j < 16; j += 2) {
        const int b7 = 7 * j;
        const int b3 = 3 * j;

        const uint32_t mw_sel[4] = {mw0, mw1, mw2, mw3};
        const uint32_t dw_sel[2] = {dw0, dw1};

        const uint32_t v14 = ((b7 % 32 + 14 <= 32)
            ? (mw_sel[b7 / 32] >> (b7 % 32))
            : __funnelshift_r(mw_sel[b7 / 32], mw_sel[b7 / 32 + 1], b7 % 32)) & 0x3FFF;
        const uint32_t v6 = ((b3 % 32 + 6 <= 32)
            ? (dw_sel[b3 / 32] >> (b3 % 32))
            : __funnelshift_r(dw_sel[b3 / 32], dw_sel[b3 / 32 + 1], b3 % 32)) & 0x3F;

        const uint32_t s0 = (sw >> j) & 1u;
        const uint32_t s1 = (sw >> (j + 1)) & 1u;
        const uint32_t w0 = ((em7 + (v14 & 0x7Fu) - ((v6 & 0x7u) << 7)) | (s0 << 15)) << 16;
        const uint32_t w1 = ((em7 + (v14 >> 7)    - ((v6 >> 3)    << 7)) | (s1 << 15)) << 16;

        dst[j + 0] = __uint_as_float(w0);
        dst[j + 1] = __uint_as_float(w1);
    }
}

// C[m = BM tile rows][n = BN tile cols] += W(m, k) * X(k, n), X in F32
template <int BM, int BN, int BK, int NTHREADS>
static __global__ void k_mul_mm_bf16x_f32(
        const char * __restrict__ src0,            // BF16X weights, [ne00=k, ne01=m]
        const float * __restrict__ src1,           // F32 activations, [ne10=k, ne11=n]
        float * __restrict__ dst,                  // F32 out,       [ne01=m, ne11=n]
        const int64_t ne00, const int64_t ne01,
        const int64_t ne10, const int64_t ne11,
        const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb11, const size_t nb12, const size_t nb13,
        const size_t nb1,  const size_t nb2,  const size_t nb3,
        const int r2, const int r3) {

    constexpr int HALVES_PER_ROW = BK / 16;
    constexpr int N_HALVES       = BM * HALVES_PER_ROW;   // == NTHREADS

    __shared__ float sW[BM][BK];
    __shared__ float sX[BK][BN];

    const int64_t m0 = (int64_t) blockIdx.x * BM;
    const int64_t n0 = (int64_t) blockIdx.y * BN;
    const int64_t im = blockIdx.z;

    const uint32_t i12 = (uint32_t) im % (uint32_t) ne12;
    const uint32_t i13 = (uint32_t) im / (uint32_t) ne12;
    const uint32_t i02 = i12 / r2;
    const uint32_t i03 = i13 / r3;

    const int tid = threadIdx.x;

    // rows beyond ne01 decode to nothing: their smem rows stay zero forever
    for (int i = tid; i < BM * BK; i += NTHREADS) {
        sW[i / BK][i % BK] = 0.0f;
    }

    const int nn = tid % BN;
    const int m_grp = tid / BN;

    constexpr int ROWS_PER_THREAD = BM / (NTHREADS / BN);
    float acc[ROWS_PER_THREAD];
#pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        acc[q] = 0.0f;
    }

    const int kblocks = (int) (ne00 / BK);

    // ---- main loop: decode one block-column of weights, consume, repeat ----
    for (int kb = 0; kb < kblocks; ++kb) {
        // decode: one 16-weight half per thread (BM x 2 halves per block column)
        if (tid < N_HALVES) {
            const int m_local = tid / HALVES_PER_ROW;
            const int hb      = tid % HALVES_PER_ROW;   // half index in block
            const int64_t i01 = m0 + m_local;

            if (i01 < ne01) {
                const block_bf16x * bx = (const block_bf16x *)
                    ((const char *) src0 + i01*nb01 + i02*nb02 + i03*nb03) + kb;

                float f[16];
                bf16x_decode_half_to_f32(bx, hb, f);

                const int kbase = hb * 16;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    sW[m_local][kbase + j] = f[j];
                }
            }
        }

        // ---- load X tile (F32 activations, contiguous rows asserted) ----
        for (int i = tid; i < BK * BN; i += NTHREADS) {
            const int kk = i / BN;
            const int nn_l = i % BN;
            const int64_t n_abs = n0 + nn_l;
            const int64_t k_abs = (int64_t) kb * BK + kk;
            const size_t row = (size_t) n_abs * nb11;
            sX[kk][nn_l] = (n_abs < ne11)
                ? *(const float *) ((const char *) src1 + k_abs * 4 + row
                                    + (size_t) i12 * nb12 + (size_t) i13 * nb13)
                : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float xv = sX[kk][nn];
#pragma unroll
            for (int q = 0; q < ROWS_PER_THREAD; ++q) {
                acc[q] = fmaf(sW[m_grp + q * (NTHREADS / BN)][kk], xv, acc[q]);
            }
        }

        __syncthreads();
    }

    // ---- store ----
#pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int64_t m_abs = m0 + m_grp + q * (NTHREADS / BN);
        const int64_t n_abs = n0 + nn;
        if (m_abs < ne01 && n_abs < ne11) {
            float * d = (float *) ((char *) dst + m_abs * 4 + n_abs * nb1
                                   + (size_t) i12 * nb2 + (size_t) i13 * nb3);
            *d = acc[q];
        }
    }
}

static bool ggml_cuda_mul_mat_bf16x_v1(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {

    GGML_TENSOR_BINARY_OP_LOCALS

    if (src0->type != GGML_TYPE_BF16X) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // element strides must be tight; row/batch strides are handled generically
    if (nb00 != sizeof(block_bf16x) || nb10 != 4 || nb0 != 4) {
        return false;
    }

    constexpr int BM = 128;
    constexpr int BN = 32;
    constexpr int BK = 32;   // exactly one BF16X block per row per k-step
    constexpr int NTHREADS = 256;

    GGML_ASSERT(nb00 == sizeof(block_bf16x));
    GGML_ASSERT(nb10 == 4);
    GGML_ASSERT(nb0  == 4);

    const int64_t ne00l = ne00, ne01l = ne01;
    const int64_t ne10l = ne10, ne11l = ne11;

    const int r2 = ne12 / ne02;
    const int r3 = ne13 / ne03;

    const dim3 grid((unsigned) ((ne01 + BM - 1) / BM),
                    (unsigned) ((ne11 + BN - 1) / BN),
                    (unsigned) (ne12 * ne13));

    cudaStream_t stream = ctx.stream();

    k_mul_mm_bf16x_f32<BM, BN, BK, NTHREADS><<<grid, NTHREADS, 0, stream>>>(
        (const char *) src0->data, (const float *) src1->data, (float *) dst->data,
        ne00l, ne01l, ne10l, ne11l, ne12,
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        nb1, nb2, nb3,
        r2, r3);
    return true;
}
