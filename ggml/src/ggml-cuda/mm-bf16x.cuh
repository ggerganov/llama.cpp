#pragma once

// BF16X batched GEMM, phase 2b: fused decode + half-MMA (m16n8k16), double-buffered.
//
// Weights stay compressed in VRAM. Each warp owns a 16-row m-tile; per lane,
// the 8 A-fragment weights are decoded directly out of the bit stream into
// mma fragment registers (funnel-shift extraction, no weight smem at all).
// Activations are staged through a double-buffered shared-memory tile as
// bf16 B-fragments. Accumulators are FP32.
//
//   warp layout : warp w -> m-tile w (16 rows), n-tiles 0..BN/8-1
//   k loop      : BK=32 per block column = two k16 MMA steps (one bf16x half each)
//   amortization: A decode once per k-block, reused across all BN/8 n-tiles

#include "common.cuh"

#include <cstdint>

static __device__ __forceinline__ uint32_t bf16x_mm_load_u32(const uint8_t * p) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t * pa = (const uint32_t *) (a & ~(uintptr_t) 3);
    const int sh = (int) (a & 3) * 8;
    return sh == 0 ? __ldg(pa) : __funnelshift_r(__ldg(pa), __ldg(pa + 1), sh);
}

// decode two adjacent BF16X weights (k = 2*pair_idx and +1) of one 16-weight
// half into a packed bf16x2 register (w_lo | w_hi << 16)
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

    // quantize floors the half-block exponent at 7: pure add/sub assembly;
    // d == 7 is the zero marker -> scale the assembled word to zero
    const uint32_t em7 = (uint32_t) row->emax[iqs] << 7;

    const uint32_t s01 = (sw >> (2 * pair_idx)) & 3u;
    const uint32_t d0 = v6 & 0x7u;
    const uint32_t d1 = (v6 >> 3) & 0x7u;
    const uint32_t w_lo = (((em7 + (v14 & 0x7Fu) - (d0 << 7)) | ((s01 & 1u) << 15)) & ((d0 != 7u) ? 0xFFFFFFFFu : 0u));
    const uint32_t w_hi = (((em7 + (v14 >> 7)    - (d1 << 7)) | ((s01 >> 1) << 15)) & ((d1 != 7u) ? 0xFFFFFFFFu : 0u));

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
    constexpr int NTILES = BN / 8;           // n-tiles per warp

    __shared__ __nv_bfloat16 sX[2][BK][BN];

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

    const int kblocks = (int) (ne00 / BK);

    auto load_x = [&](int kb, int buf) {
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
            sX[buf][kk][nn] = __float2bfloat16(v);
        }
    };

    load_x(0, 0);
    __syncthreads();

    for (int kb = 0; kb < kblocks; ++kb) {
        const int cur = kb & 1;

        // prefetch the next block column into the other smem buffer; the
        // trailing __syncthreads() of the previous iteration ordered this
        // after all reads of the buffer being overwritten
        if (kb + 1 < kblocks) {
            load_x(kb + 1, cur ^ 1);
        }

        // ---- A fragments: 4 regs per k16 step x 2 steps = 8 regs ----
        // per PTX m16n8k16: reg0={row g, pair tig}, reg1={row g+8, pair tig},
        //                   reg2={row g, pair tig+4}, reg3={row g+8, pair tig+4}
        const int64_t roff = (size_t) i02 * nb02 + (size_t) i03 * nb03
                           + kb * sizeof(block_bf16x);
        const int64_t r0 = row_base_m + g;
        const int64_t r1 = r0 + 8;
        const char * rp0 = (const char *) src0 + (r0 < ne01 ? r0 : 0) * nb01 + roff;
        const char * rp1 = (const char *) src0 + (r1 < ne01 ? r1 : 0) * nb01 + roff;
        const bool ok0 = r0 < ne01;
        const bool ok1 = r1 < ne01;

        uint32_t af0[4] = {0u, 0u, 0u, 0u};   // iqs=0 A-fragment
        uint32_t af1[4] = {0u, 0u, 0u, 0u};   // iqs=1 A-fragment
        if (ok0 || ok1) {
            // reg0 = row g,   pair tig     {A[g][tig*2], A[g][tig*2+1]}
            af0[0] = ok0 ? bf16x_decode_pair((const block_bf16x *) rp0, 0, tig)     : 0u;
            // reg1 = row g+8, pair tig     {A[g+8][tig*2], A[g+8][tig*2+1]}
            af0[1] = ok1 ? bf16x_decode_pair((const block_bf16x *) rp1, 0, tig)     : 0u;
            // reg2 = row g,   pair tig+4   {A[g][tig*2+8], A[g][tig*2+9]}
            af0[2] = ok0 ? bf16x_decode_pair((const block_bf16x *) rp0, 0, tig + 4) : 0u;
            // reg3 = row g+8, pair tig+4   {A[g+8][tig*2+8], A[g+8][tig*2+9]}
            af0[3] = ok1 ? bf16x_decode_pair((const block_bf16x *) rp1, 0, tig + 4) : 0u;
            // same layout for iqs=1
            af1[0] = ok0 ? bf16x_decode_pair((const block_bf16x *) rp0, 1, tig)     : 0u;
            af1[1] = ok1 ? bf16x_decode_pair((const block_bf16x *) rp1, 1, tig)     : 0u;
            af1[2] = ok0 ? bf16x_decode_pair((const block_bf16x *) rp0, 1, tig + 4) : 0u;
            af1[3] = ok1 ? bf16x_decode_pair((const block_bf16x *) rp1, 1, tig + 4) : 0u;
        }

        // ---- 2 k16 steps x NTILES n-tiles of mma on sX[cur] ----
#pragma unroll
        for (int iqs = 0; iqs < 2; ++iqs) {
            const uint32_t * af = (iqs == 0) ? af0 : af1;
#pragma unroll
            for (int t = 0; t < NTILES; ++t) {
                const int n_col = t * 8 + g;
                const __nv_bfloat16 x00 = sX[cur][iqs * 16 + tig * 2    ][n_col];
                const __nv_bfloat16 x01 = sX[cur][iqs * 16 + tig * 2 + 1][n_col];
                const __nv_bfloat16 x10 = sX[cur][iqs * 16 + tig * 2 + 8][n_col];
                const __nv_bfloat16 x11 = sX[cur][iqs * 16 + tig * 2 + 9][n_col];
                const uint32_t b0 = (__bfloat16_as_ushort(x00)) | (__bfloat16_as_ushort(x01) << 16);
                const uint32_t b1 = (__bfloat16_as_ushort(x10)) | (__bfloat16_as_ushort(x11) << 16);

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[t][0]), "+f"(c[t][1]), "+f"(c[t][2]), "+f"(c[t][3])
                    : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(b0), "r"(b1));
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
    constexpr int BN = 64;    // 8 n-tiles of 8 per warp
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
