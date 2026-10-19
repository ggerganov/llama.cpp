#pragma once

// BF16X batched GEMM: fused decode + bf16 MMA (m16n8k16), double-buffered.
//
// A-fragment layout for mma.m16n8k16.row.col (per lane, groupID=g, tig=lane&3):
//   reg0 = {A[g][tig*2],   A[g][tig*2+1]}     ← pair from row g,   half iqs
//   reg1 = {A[g+8][tig*2], A[g+8][tig*2+1]}   ← same pair, row g+8
//   reg2 = {A[g][tig*2+8], A[g][tig*2+9]}     ← pair tig+4, row g
//   reg3 = {A[g+8][tig*2+8],A[g+8][tig*2+9]}  ← pair tig+4, row g+8
//
// For BK=32 (one BF16X block), step iqs covers k_local = iqs*16..+15.
// B-fragment: reg0 = {X[tig*2][nc], X[tig*2+1][nc]},
//             reg1 = {X[tig*2+8][nc], X[tig*2+9][nc]}
// where nc = n-tile col base + g.
//
// d==7 = zero marker (quantize floors emax at 7, so d==7 -> zero weight).

#include "common.cuh"
#include <cstdint>

__device__ __forceinline__ uint32_t bfxld(const uint8_t * p) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t * pa = (const uint32_t *) (a & ~(uintptr_t) 3);
    const int sh = (int) (a & 3) * 8;
    return sh == 0 ? __ldg(pa) : __funnelshift_r(__ldg(pa), __ldg(pa + 1), sh);
}

// decode 2 adjacent bf16 elements (positions 2p and 2p+1 within the half)
// returns packed bf16x2: lo = element 2p, hi = element 2p+1
static __device__ __forceinline__ uint32_t bfx_pair(
        const block_bf16x * GGML_RESTRICT blk, const int iqs, const int p) {
    const uint8_t * mbase = blk->mant  + iqs * 14;
    const uint8_t * dbase = blk->delta + iqs * 6;
    const uint8_t * sbase = blk->sgn   + iqs * 2;

    const int b7 = 14 * p;   // bit offset for first mantissa of the pair
    const int b3 = 6 * p;    // bit offset for first delta of the pair

    const uint32_t mw[4] = {bfxld(mbase), bfxld(mbase+4), bfxld(mbase+8), bfxld(mbase+12)};
    const uint32_t dw[2] = {bfxld(dbase), bfxld(dbase+4)};
    const uint32_t sw = bfxld(sbase);

    uint32_t v14 = ((b7 % 32 + 14 <= 32)
        ? (mw[b7/32] >> (b7%32))
        : __funnelshift_r(mw[b7/32], mw[b7/32+1], b7%32)) & 0x3FFF;
    uint32_t v6 = ((b3 % 32 + 6 <= 32)
        ? (dw[b3/32] >> (b3%32))
        : __funnelshift_r(dw[b3/32], dw[b3/32+1], b3%32)) & 0x3F;

    const uint32_t em7 = (uint32_t) blk->emax[iqs] << 7;
    const uint32_t s01 = (sw >> (2 * p)) & 3u;
    const uint32_t d0 = v6 & 7u, d1 = (v6 >> 3) & 7u;

    uint32_t w_lo = (em7 + (v14 & 0x7F) - (d0 << 7)) | ((s01 & 1) << 15);
    uint32_t w_hi = (em7 + (v14 >> 7) - (d1 << 7)) | ((s01 >> 1) << 15);
    if (d0 == 7) w_lo = 0;
    if (d1 == 7) w_hi = 0;
    return w_lo | (w_hi << 16);
}

// decode 4 bf16 elements forming one lane's A-fragment half:
// af[0] = row g,   pair tig     (k = tig*2, tig*2+1)      ← PTX reg0
// af[1] = row g,   pair tig+4   (k = tig*2+8, tig*2+9)    ← PTX reg1 (SAME ROW)
// af[2] = row g+8, pair tig     (k = tig*2, tig*2+1)      ← PTX reg2 (SAME K-PAIR, row g+8)
// af[3] = row g+8, pair tig+4   (k = tig*2+8, tig*2+9)    ← PTX reg3
// (uses iqs to select the half within the block)
__device__ __forceinline__ void bfx_af16(
        const block_bf16x * GGML_RESTRICT w0,   // row g stream
        const block_bf16x * GGML_RESTRICT w1,   // row g+8 stream
        const int iqs, const int tig,
        uint32_t * GGML_RESTRICT af /* [4] */) {
    // PTX ISA m16n8k16 A-fragment order (groupID = g, threadID_in_group = tig):
    //   reg0 (a0,a1): row g,   k-pair tig        (k = 2*tig,   2*tig+1)
    //   reg1 (a2,a3): row g+8, k-pair tig        (same k-pair, next row group)
    //   reg2 (a4,a5): row g,   k-pair tig+4      (same row,    k = 2*tig+8, 2*tig+9)
    //   reg3 (a6,a7): row g+8, k-pair tig+4
    af[0] = bfx_pair(w0, iqs, tig);
    af[1] = bfx_pair(w1, iqs, tig);
    af[2] = bfx_pair(w0, iqs, tig + 4);
    af[3] = bfx_pair(w1, iqs, tig + 4);
}

__global__ void k_mul_mm_bf16x_mma(
        const char * __restrict__ src0,
        const float * __restrict__ src1,
        float * __restrict__ dst,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne10, const int64_t ne11,
        const int64_t ne12,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb11, const size_t nb12, const size_t nb13,
        const size_t nb1,  const size_t nb2,  const size_t nb3,
        const int r2, const int r3) {

    constexpr int BM = 128, BN = 64, BK = 32;
    constexpr int NTILES = BN / 8;

    __shared__ __nv_bfloat16 sX[2][BK][BN];

    const int64_t m0 = (int64_t) blockIdx.x * BM;
    const int64_t n0 = (int64_t) blockIdx.y * BN;
    const int64_t im = blockIdx.z;
    const uint32_t i12 = (uint32_t) im % (uint32_t) ne12;
    const uint32_t i13 = (uint32_t) im / (uint32_t) ne12;
    const uint32_t i02 = i12 / r2;
    const uint32_t i03 = i13 / r3;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g = lane >> 2, tig = lane & 3;
    const int64_t row_base = m0 + warp * 16;

    float c[NTILES][4];
#pragma unroll
    for (int t = 0; t < NTILES; ++t) c[t][0] = c[t][1] = c[t][2] = c[t][3] = 0.f;

    const int64_t row_g  = row_base + g;
    const int64_t row_g8 = row_base + 8 + g;
    const bool ok0 = row_g  < ne01;
    const bool ok1 = row_g8 < ne01;
    const block_bf16x * wrow0 = (const block_bf16x *)((const char *) src0
        + (ok0 ? row_g : 0) * nb01 + i02 * nb02 + i03 * nb03);
    const block_bf16x * wrow1 = (const block_bf16x *)((const char *) src0
        + (ok1 ? row_g8 : 0) * nb01 + i02 * nb02 + i03 * nb03);

    auto load_x = [&](int kb, int buf) {
        for (int i = tid; i < BK * BN; i += 256) {
            const int kk = i / BN, nn = i % BN;
            const int64_t n_abs = n0 + nn;
            const int64_t k_abs = (int64_t) kb * BK + kk;
            const float v = (n_abs < ne11)
                ? *(const float *) ((const char *) src1 + k_abs * 4
                    + (size_t) n_abs * nb11 + i12 * nb12 + i13 * nb13)
                : 0.0f;
            sX[buf][kk][nn] = __float2bfloat16(v);
        }
    };

    load_x(0, 0);
    __syncthreads();

    const int kblocks = (int) (ne00 / BK);
    for (int kb = 0; kb < kblocks; ++kb) {
        const int cur = kb & 1;
        if (kb + 1 < kblocks) load_x(kb + 1, cur ^ 1);

        // decode A for both k16 steps: 4 pair-decodes per step, 8 total
        uint32_t A0[4], A1[4];
        bfx_af16(wrow0 + kb, wrow1 + kb, 0, tig, A0);   // k16 step 0
        bfx_af16(wrow0 + kb, wrow1 + kb, 1, tig, A1);   // k16 step 1

        // mma step 0 (iqs=0, k = kb*32 + 0..15)
        {
#pragma unroll
            for (int t = 0; t < NTILES; ++t) {
                // per-tile accumulator: each n-tile owns its own C fragment
                float d[4] = {c[t][0], c[t][1], c[t][2], c[t][3]};
                const int nc = t * 8 + g;
                const uint32_t b0 = __bfloat16_as_ushort(sX[cur][tig * 2][nc])
                                  | (__bfloat16_as_ushort(sX[cur][tig * 2 + 1][nc]) << 16);
                const uint32_t b1 = __bfloat16_as_ushort(sX[cur][tig * 2 + 8][nc])
                                  | (__bfloat16_as_ushort(sX[cur][tig * 2 + 9][nc]) << 16);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                    : "r"(A0[0]), "r"(A0[1]), "r"(A0[2]), "r"(A0[3]), "r"(b0), "r"(b1));
                c[t][0] = d[0]; c[t][1] = d[1]; c[t][2] = d[2]; c[t][3] = d[3];
            }
        }
        // mma step 1 (iqs=1, k = kb*32 + 16..31)
        {
#pragma unroll
            for (int t = 0; t < NTILES; ++t) {
                float d[4] = {c[t][0], c[t][1], c[t][2], c[t][3]};
                const int nc = t * 8 + g;
                const uint32_t b0 = __bfloat16_as_ushort(sX[cur][16 + tig * 2][nc])
                                  | (__bfloat16_as_ushort(sX[cur][16 + tig * 2 + 1][nc]) << 16);
                const uint32_t b1 = __bfloat16_as_ushort(sX[cur][16 + tig * 2 + 8][nc])
                                  | (__bfloat16_as_ushort(sX[cur][16 + tig * 2 + 9][nc]) << 16);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                    : "r"(A1[0]), "r"(A1[1]), "r"(A1[2]), "r"(A1[3]), "r"(b0), "r"(b1));
                c[t][0] = d[0]; c[t][1] = d[1]; c[t][2] = d[2]; c[t][3] = d[3];
            }
        }
        __syncthreads();
    }

    // store
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
        const int64_t n0t = n0 + t * 8 + tig * 2;
        const int64_t mr = row_base + g;
        const int64_t m8 = mr + 8;
        // per-element guards: an odd ne11 must still store its last column
        if (mr < ne01) {
            if (n0t < ne11) {
                *((float *) ((char *) dst + mr * 4 + n0t * nb1 + i12 * nb2 + i13 * nb3)) = c[t][0];
            }
            if (n0t + 1 < ne11) {
                *((float *) ((char *) dst + mr * 4 + (n0t + 1) * nb1 + i12 * nb2 + i13 * nb3)) = c[t][1];
            }
        }
        if (m8 < ne01) {
            if (n0t < ne11) {
                *((float *) ((char *) dst + m8 * 4 + n0t * nb1 + i12 * nb2 + i13 * nb3)) = c[t][2];
            }
            if (n0t + 1 < ne11) {
                *((float *) ((char *) dst + m8 * 4 + (n0t + 1) * nb1 + i12 * nb2 + i13 * nb3)) = c[t][3];
            }
        }
    }
}

static bool ggml_cuda_mul_mat_bf16x_mma(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS
    if (src0->type != GGML_TYPE_BF16X) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (nb00 != sizeof(block_bf16x) || nb10 != 4 || nb0 != 4) return false;
    if (ne00 % 32 != 0) return false;   // BK tail is not handled; fall back to cuBLAS
    const int r2 = ne12 / ne02, r3 = ne13 / ne03;
    const dim3 grid((unsigned) ((ne01 + 127) / 128), (unsigned) ((ne11 + 63) / 64), (unsigned) (ne12 * ne13));
    cudaStream_t stream = ctx.stream();
    k_mul_mm_bf16x_mma<<<grid, 256, 0, stream>>>(
        (const char *) src0->data, (const float *) src1->data, (float *) dst->data,
        ne00, ne01, ne10, ne11, ne12,
        nb01, nb02, nb03, nb11, nb12, nb13, nb1, nb2, nb3, r2, r3);
    return true;
}
