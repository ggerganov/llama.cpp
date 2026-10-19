#pragma once

// BF16X batched GEMM, phase 2: fused decode + half-MMA (m16n8k16).
//
// Weights stay compressed in VRAM. Each warp owns a 16-row m-tile; per lane,
// the 8 A-fragment weights are decoded directly out of the bit stream into
// mma fragment registers (funnel-shift extraction, no weight smem at all).
// Activations are staged through shared memory as half and packed into
// B-fragments. Accumulators are FP32.
//
//   warp layout : warp w -> m-tile w (16 rows), n-tiles 0..BN/8-1
//   k loop      : BK=32 per block column = two k16 MMA steps (one bf16x half each)
//   decode/lane : 4 pair-extractions (2 rows x 2 adjacent-k pairs) per k16 step

#include "common.cuh"

#include <cstdint>

static __device__ __forceinline__ uint32_t bf16x_mm_load_u32(const uint8_t * p) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t * pa = (const uint32_t *) (a & ~(uintptr_t) 3);
    const int sh = (int) (a & 3) * 8;
    return sh == 0 ? pa[0] : __funnelshift_r(pa[0], pa[1], sh);
}

// decode two adjacent bf16xl/BF16X weights (k = 2*pair_idx and +1) of one
// 16-weight half into a packed half2 register (w_lo | w_hi << 16)
static __device__ __forceinline__ uint32_t bf16x_decode_pair(
        const block_bf16x * GGML_RESTRICT row,   // row stream base (block start)
        const int iqs, const int pair_idx) {

    const uint8_t * mb = row->mant  + iqs * 14;
    const uint8_t * db = row->delta + iqs * 6;
    const uint8_t * sb = row->sgn   + iqs * 2;

    const uint32_t mw0 = bf16x_mm_load_u32(mb +  0);
    const uint32_t mw1 = bf16x_mm_load_u32(mb +  4);
    const uint32_t mw2 = bf16x_mm_load_u32(mb +  8);
    const uint32_t mw3 = bf16x_mm_load_u32(mb + 12);
    const uint32_t dw0 = bf16x_mm_load_u32(db +  0);
    const uint32_t dw1 = bf16x_mm_load_u32(db +  4);
    const uint32_t sw  = bf16x_mm_load_u32(sb);

    const uint32_t mw_sel[4] = {mw0, mw1, mw2, mw3};
    const uint32_t dw_sel[2] = {dw0, dw1};

    const int b7 = 7 * (2 * pair_idx);
    const int b3 = 3 * (2 * pair_idx);

    const uint32_t v14 = ((b7 % 32 + 14 <= 32)
        ? (mw_sel[b7 / 32] >> (b7 % 32))
        : __funnelshift_r(mw_sel[b7 / 32], mw_sel[b7 / 32 + 1], b7 % 32)) & 0x3FFF;
    const uint32_t v6 = ((b3 % 32 + 6 <= 32)
        ? (dw_sel[b3 / 32] >> (b3 % 32))
        : __funnelshift_r(dw_sel[b3 / 32], dw_sel[b3 / 32 + 1], b3 % 32)) & 0x3F;

    // emax floor of 7 at quantize time: add/sub assembly, no clamp
    const uint32_t em7 = (uint32_t) row->emax[iqs] << 7;

    const uint32_t s01 = (sw >> (2 * pair_idx)) & 3u;
    const uint32_t w_lo = ((em7 + (v14 & 0x7Fu) - ((v6 & 0x7u) << 7)) | ((s01 & 1u) << 15));
    const uint32_t w_hi = ((em7 + (v14 >> 7)    - ((v6 >> 3)    << 7)) | ((s01 >> 1) << 15));

    return w_lo | (w_hi << 16);
}

// C[m, n] = sum_k W(m, k) * X(k, n); W = BF16X (decoded in-register), X = F32
template <int BM, int BN, int BK, int NTHREADS>
static __global__ void k_mul_mm_bf16x_mma(
        const char * __restrict__ src0,            // BF16X weights [ne00=k, ne01=m]
        const float * __restrict__ src1,           // F32 activations [ne10=k, ne11=n]
        float * __restrict__ dst,                  // F32 out [ne01=m, ne11=n]
        const int64_t ne00, const int64_t ne01,
        const int64_t ne10, const int64_t ne11,
        const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb11, const size_t nb12, const size_t nb13,
        const size_t nb1,  const size_t nb2,  const size_t nb3,
        const int r2, const int r3) {

    static_assert(BM % 128 == 0);            // one 16-row m-tile per warp
    static_assert(BN % 8 == 0);
    constexpr int NWARP  = NTHREADS / 32;
    constexpr int MTILES = BM / 16;          // m-tiles per block == NWARP
    constexpr int NTILES = BN / 8;           // n-tiles per warp

    __shared__ __nv_bfloat16 sX[BK][BN];

    const int64_t m0 = (int64_t) blockIdx.x * BM;
    const int64_t n0 = (int64_t) blockIdx.y * BN;
    const int64_t im = blockIdx.z;

    const uint32_t i12 = (uint32_t) im % (uint32_t) ne12;
    const uint32_t i13 = (uint32_t) im / (uint32_t) ne12;
    const uint32_t i02 = i12 / r2;
    const uint32_t i03 = i13 / r3;

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g    = lane >> 2;              // fragment row group
    const int tig  = lane & 3;               // fragment column group

    const int64_t row_base_m = m0 + warp * 16;   // this warp's 16 m-rows

    // per-lane accumulators: NTILES x 4
    float c[NTILES][4];
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
        c[t][0] = c[t][1] = c[t][2] = c[t][3] = 0.0f;
    }

    // row pointers for the two fragment rows (g, g+8) of this lane
    const char * rowp[2];

    const int kblocks = (int) (ne00 / BK);

    for (int kb = 0; kb < kblocks; ++kb) {
        // ---- load X slice for this block column, as half ----
        for (int i = tid; i < BK * BN; i += NTHREADS) {
            const int kk = i / BN;
            const int nn = i % BN;
            const int64_t n_abs = n0 + nn;
            const int64_t k_abs = (int64_t) kb * BK + kk;
            const size_t rowb = (size_t) n_abs * nb11;
            const float v = (n_abs < ne11)
                ? *(const float *) ((const char *) src1 + k_abs * 4 + rowb
                                    + (size_t) i12 * nb12 + (size_t) i13 * nb13)
                : 0.0f;
            sX[kk][nn] = __float2bfloat16(v);
        }
        __syncthreads();

        // ---- two k16 MMA steps, one per 16-weight half (iqs = 0, 1) ----
#pragma unroll
        for (int iqs = 0; iqs < 2; ++iqs) {
            // ---- A fragments: decode this lane's 8 weights ----
            uint32_t a[4];

            const int64_t r0 = row_base_m + g;
            const int64_t r1 = r0 + 8;
            rowp[0] = (const char *) src0 + (r0 < ne01 ? r0 : 0) * nb01
                    + (size_t) i02 * nb02 + (size_t) i03 * nb03 + kb * sizeof(block_bf16x);
            rowp[1] = (const char *) src0 + (r1 < ne01 ? r1 : 0) * nb01
                    + (size_t) i02 * nb02 + (size_t) i03 * nb03 + kb * sizeof(block_bf16x);
            const bool ok0 = r0 < ne01;
            const bool ok1 = r1 < ne01;

            const int kp0 = tig;              // pair p covers k = 2p, 2p+1
            const int kp1 = tig + 4;          // k = tig*2 + 8, +9

            const uint32_t p00 = bf16x_decode_pair((const block_bf16x *) rowp[0], iqs, kp0);
            const uint32_t p01 = bf16x_decode_pair((const block_bf16x *) rowp[0], iqs, kp1);
            const uint32_t p10 = bf16x_decode_pair((const block_bf16x *) rowp[1], iqs, kp0);
            const uint32_t p11 = bf16x_decode_pair((const block_bf16x *) rowp[1], iqs, kp1);

            a[0] = ok0 ? p00 : 0u;
            a[1] = ok1 ? p10 : 0u;
            a[2] = ok0 ? p01 : 0u;
            a[3] = ok1 ? p11 : 0u;

            // ---- B fragments from smem (k-adjacent half2 pairs) ----
#pragma unroll
            for (int t = 0; t < NTILES; ++t) {
                const int n_col = t * 8 + g;
                const __nv_bfloat16 x00 = sX[iqs * 16 + tig * 2    ][n_col];
                const __nv_bfloat16 x01 = sX[iqs * 16 + tig * 2 + 1][n_col];
                const __nv_bfloat16 x10 = sX[iqs * 16 + tig * 2 + 8][n_col];
                const __nv_bfloat16 x11 = sX[iqs * 16 + tig * 2 + 9][n_col];
                const uint32_t b0 = (__bfloat16_as_ushort(x00)) | (__bfloat16_as_ushort(x01) << 16);
                const uint32_t b1 = (__bfloat16_as_ushort(x10)) | (__bfloat16_as_ushort(x11) << 16);

                float d[4] = { c[t][0], c[t][1], c[t][2], c[t][3] };
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
                c[t][0] = d[0]; c[t][1] = d[1]; c[t][2] = d[2]; c[t][3] = d[3];
            }
        }
        __syncthreads();
    }

    // ---- store: C fragment layout c0,c1 -> row g, cols tig*2,+1; c2,c3 -> row g+8 ----
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
        const int64_t n0t = n0 + t * 8 + tig * 2;
        const int64_t m0r = row_base_m + g;
        const int64_t m8  = m0r + 8;
        if (n0t < ne11) {
            if (m0r < ne01) {
                *((float *) ((char *) dst + m0r * 4 + n0t * nb1 + (size_t) i12 * nb2 + (size_t) i13 * nb3)) = c[t][0];
                *((float *) ((char *) dst + m0r * 4 + (n0t + 1) * nb1 + (size_t) i12 * nb2 + (size_t) i13 * nb3)) = c[t][1];
            }
            if (m8 < ne01) {
                *((float *) ((char *) dst + m8 * 4 + n0t * nb1 + (size_t) i12 * nb2 + (size_t) i13 * nb3)) = c[t][2];
                *((float *) ((char *) dst + m8 * 4 + (n0t + 1) * nb1 + (size_t) i12 * nb2 + (size_t) i13 * nb3)) = c[t][3];
            }
        }
    }
}

static bool ggml_cuda_mul_mat_bf16x_mma(
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

    constexpr int BM = 128;   // 8 warps x 16 rows
    constexpr int BN = 32;    // 4 n-tiles of 8 per warp
    constexpr int BK = 32;    // one BF16X block column
    constexpr int NTHREADS = 256;

    const int r2 = ne12 / ne02;
    const int r3 = ne13 / ne03;

    const dim3 grid((unsigned) ((ne01 + BM - 1) / BM),
                    (unsigned) ((ne11 + BN - 1) / BN),
                    (unsigned) (ne12 * ne13));

    cudaStream_t stream = ctx.stream();

    k_mul_mm_bf16x_mma<BM, BN, BK, NTHREADS><<<grid, NTHREADS, 0, stream>>>(
        (const char *) src0->data, (const float *) src1->data, (float *) dst->data,
        ne00, ne01, ne10, ne11, ne12,
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        nb1, nb2, nb3,
        r2, r3);
    return true;
}
