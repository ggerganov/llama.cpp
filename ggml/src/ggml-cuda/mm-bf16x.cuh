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
//
// Decode strategy: preload both fragment rows' full bit-streams into
// registers once per k-block (11 words per row), then extract all
// A-fragment pairs from registers. This cuts global-load instructions
// per lane per k-block from 56 (8 pairs x 7 words) to 22.

#include "common.cuh"
#include <cstdint>

__device__ __forceinline__ uint32_t bfxld(const uint8_t * p) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t * pa = (const uint32_t *) (a & ~(uintptr_t) 3);
    const int sh = (int) (a & 3) * 8;
    return sh == 0 ? __ldg(pa) : __funnelshift_r(__ldg(pa), __ldg(pa + 1), sh);
}

// full bit-streams of one BF16X block as SCALARS. Dynamic array indexing
// (pair index depends on the runtime lane id) would force these to local
// memory; scalars + ternary selects keep everything in registers.
struct bfx_streams {
    uint32_t m0, m1, m2, m3, m4, m5, m6;
    uint32_t d0, d1, d2;
    uint32_t sw;
    uint32_t e0, e1;
};

__device__ __forceinline__ void bfx_load_row(const block_bf16x * GGML_RESTRICT blk, bfx_streams & s) {
    s.m0 = bfxld(blk->mant);      s.m1 = bfxld(blk->mant + 4);
    s.m2 = bfxld(blk->mant + 8);  s.m3 = bfxld(blk->mant + 12);
    s.m4 = bfxld(blk->mant + 16); s.m5 = bfxld(blk->mant + 20);
    s.m6 = bfxld(blk->mant + 24);
    s.d0 = bfxld(blk->delta);
    s.d1 = bfxld(blk->delta + 4);
    s.d2 = bfxld(blk->delta + 8);
    s.sw = bfxld(blk->sgn);
    s.e0 = (uint32_t) blk->emax[0] << 7;
    s.e1 = (uint32_t) blk->emax[1] << 7;
}

// assemble a packed bf16x2 pair from exponent-delta form; d==7 -> zero marker
// v14: 14 bits = mant(2p) | mant(2p+1)<<7; d6: 6 bits = delta(2p) | delta(2p+1)<<3
// s01: 2 bits = sign(2p) | sign(2p+1)<<1
__device__ __forceinline__ uint32_t d7_make2(uint32_t em7, uint32_t v14, uint32_t d6, uint32_t s01) {
    uint32_t w_lo = (em7 + (v14 & 0x7Fu) - ((d6 & 7u) << 7)) | ((s01 & 1u) << 15);
    uint32_t w_hi = (em7 + (v14 >> 7)    - ((d6 >> 3) << 7)) | ((s01 >> 1) << 15);
    if ((d6 & 7u) == 7u) w_lo = 0;
    if ((d6 >> 3) == 7u) w_hi = 0;
    return w_lo | (w_hi << 16);
}

// decode pair p = tig and p = tig+4 of half iqs from one row's streams.
// iqs is a compile-time constant at every call site (0/1 literal), so the
// branch below is resolved at compile time. Word selection uses ternaries
// (SEL) — dynamic array indexing would force the streams to local memory.
//
// half 0: mant bits [0,112)  = words m0..m3.5 | delta [0,48)  = d0,d1 | sign bits 0..15
// half 1: mant bits [112,224)= words m3.5..m6| delta [48,96)  = d1,d2 | sign bits 16..31
__device__ __forceinline__ void bfx_two_pairs(
        const bfx_streams & GGML_RESTRICT s, const int iqs, const int tig,
        uint32_t & GGML_RESTRICT out_tig, uint32_t & GGML_RESTRICT out_tig4) {
    const uint32_t em7 = iqs ? s.e1 : s.e0;
    if (iqs == 0) {
        // ---- p = tig: mant bits [14*tig, +14) ----
        // tig 0,1,2 -> (m0,m1); tig 3 -> (m1,m2)
        {
            const uint32_t a = (tig < 3) ? s.m0 : s.m1;
            const uint32_t b = (tig < 3) ? s.m1 : s.m2;
            const uint32_t v14 = __funnelshift_r(a, b, (14 * tig) & 31) & 0x3FFF;
            const uint32_t s01 = (s.sw >> (2 * tig)) & 3u;
            const uint32_t d6 = (s.d0 >> (6 * tig)) & 0x3Fu;   // 6*tig+6 <= 24 <= 32
            out_tig = d7_make2(em7, v14, d6, s01);
        }
        // ---- p = tig+4: mant bits [56+14*tig, +14) ----
        // tig 0 -> (m1,m2) sh24; 1 -> (m2,m3) sh6; 2 -> (m2,m3) sh20; 3 -> (m3,m4) sh2
        {
            const uint32_t a = (tig < 1) ? s.m1 : (tig < 3) ? s.m2 : s.m3;
            const uint32_t b = (tig < 1) ? s.m2 : (tig < 3) ? s.m3 : s.m4;
            const uint32_t v14 = __funnelshift_r(a, b, (56 + 14 * tig) & 31) & 0x3FFF;
            const uint32_t s01 = (s.sw >> (2 * tig + 8)) & 3u;
            // delta p = tig+4: bits [24+6*tig, +6): tig 0,1 -> (d0,d1); tig 2,3 -> (d1,d2)
            const uint32_t da = (tig < 2) ? s.d0 : s.d1;
            const uint32_t db = (tig < 2) ? s.d1 : s.d2;
            const uint32_t d6 = __funnelshift_r(da, db, (24 + 6 * tig) & 31) & 0x3F;
            out_tig4 = d7_make2(em7, v14, d6, s01);
        }
    } else {
        // ---- p = tig: mant bits [112 + 14*tig, +14) ----
        // tig 0,1 -> (m3,m4); tig 2,3 -> (m4,m5)
        {
            const uint32_t a = (tig < 2) ? s.m3 : s.m4;
            const uint32_t b = (tig < 2) ? s.m4 : s.m5;
            const uint32_t v14 = __funnelshift_r(a, b, (16 + 14 * tig) & 31) & 0x3FFF;
            const uint32_t s01 = (s.sw >> (16 + 2 * tig)) & 3u;
            // delta p = tig: bits [48+6*tig, +6): tig 0,1,2 -> (d1,d2); tig 3 -> d2
            const uint32_t da = (tig < 3) ? s.d1 : s.d2;
            const uint32_t d6 = __funnelshift_r(da, s.d2, (16 + 6 * tig) & 31) & 0x3F;
            out_tig = d7_make2(em7, v14, d6, s01);
        }
        // ---- p = tig+4: mant bits [168 + 14*tig, +14) ----
        // tig 0,1 -> (m5,m6); tig 2,3 -> m6 alone (last word of the stream)
        {
            const uint32_t a = (tig < 2) ? s.m5 : s.m6;
            const uint32_t v14 = __funnelshift_r(a, s.m6, (8 + 14 * tig) & 31) & 0x3FFF;
            const uint32_t s01 = (s.sw >> (16 + 2 * tig + 8)) & 3u;
            // delta p = tig+4: bits [72+6*tig, +6): all within d2
            const uint32_t d6 = (s.d2 >> ((8 + 6 * tig) & 31)) & 0x3Fu;
            out_tig4 = d7_make2(em7, v14, d6, s01);
        }
    }
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

    constexpr int BM = 128, BN = 32, BK = 32;
    constexpr int NTILES = BN / 8;

    // X tile is stored n-major (sX[n][k]) so that B fragments can be loaded
    // with ldmatrix (8 k-values contiguous = one 16-byte ldmatrix row).
    // +8 halfs of row padding makes the 8 ldmatrix rows hit distinct smem
    // bank groups (80B stride spreads 8 16B rows across all 32 banks).
    __shared__ __nv_bfloat16 sX[2][BN][BK + 8];

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
            sX[buf][nn][kk] = __float2bfloat16(v);   // n-major for ldmatrix
        }
    };

    load_x(0, 0);
    __syncthreads();

    const int kblocks = (int) (ne00 / BK);
    for (int kb = 0; kb < kblocks; ++kb) {
        const int cur = kb & 1;
        if (kb + 1 < kblocks) load_x(kb + 1, cur ^ 1);

        // preload both fragment rows' bit-streams (26 words), then decode
        // all 8 A-fragment pairs from registers (both k16 steps at once)
        bfx_streams st0, st1;
        bfx_load_row(wrow0 + kb, st0);
        bfx_load_row(wrow1 + kb, st1);

        uint32_t A0[4], A1[4];
        bfx_two_pairs(st0, 0, tig, A0[0], A0[2]);
        bfx_two_pairs(st1, 0, tig, A0[1], A0[3]);
        bfx_two_pairs(st0, 1, tig, A1[0], A1[2]);
        bfx_two_pairs(st1, 1, tig, A1[1], A1[3]);

        // ldmatrix: load one 16(k) x 8(n) B fragment from sX[buf][n][k].
        // lanes 0-7 supply row addresses for the k = k0..k0+7 matrix,
        // lanes 8-15 for the k = k0+8..k0+15 matrix; the resulting
        // register pair matches the mma B-fragment layout exactly.
        auto load_b = [&](int buf, int nc, int k0, uint32_t & b0, uint32_t & b1) {
            const int row8 = lane & 7;
            const int mat  = (lane >> 3) & 1;
            const __nv_bfloat16 * a = &sX[buf][nc + row8][k0 + 8 * mat];
            const unsigned sa = (unsigned) __cvta_generic_to_shared(a);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                : "=r"(b0), "=r"(b1) : "r"(sa));
        };

        // mma step 0 (iqs=0, k = kb*32 + 0..15)
        {
#pragma unroll
            for (int t = 0; t < NTILES; ++t) {
                // per-tile accumulator: each n-tile owns its own C fragment
                float d[4] = {c[t][0], c[t][1], c[t][2], c[t][3]};
                uint32_t b0, b1;
                load_b(cur, t * 8, 0, b0, b1);
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
                uint32_t b0, b1;
                load_b(cur, t * 8, 16, b0, b1);
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
    const dim3 grid((unsigned) ((ne01 + 127) / 128), (unsigned) ((ne11 + 31) / 32), (unsigned) (ne12 * ne13));
    cudaStream_t stream = ctx.stream();
    k_mul_mm_bf16x_mma<<<grid, 256, 0, stream>>>(
        (const char *) src0->data, (const float *) src1->data, (float *) dst->data,
        ne00, ne01, ne10, ne11, ne12,
        nb01, nb02, nb03, nb11, nb12, nb13, nb1, nb2, nb3, r2, r3);
    return true;
}
