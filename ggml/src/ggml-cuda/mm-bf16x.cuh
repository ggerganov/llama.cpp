#pragma once

// BF16X batched GEMM phase 2b: fused decode + bf16 MMA (m16n8k16).
// Simple tiled kernel: weights decode into A-fragments per lane,
// X staged through smem, double-buffered. Correctness-first baseline.

#include "common.cuh"
#include <cstdint>

__device__ __forceinline__ uint32_t bf16x_ld_u32_v2(const uint8_t * p) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t * pa = (const uint32_t *) (a & ~(uintptr_t) 3);
    const int sh = (int) (a & 3) * 8;
    return sh == 0 ? __ldg(pa) : __funnelshift_r(__ldg(pa), __ldg(pa + 1), sh);
}

// decode 2 adjacent bf16 weights (k=2p, 2p+1) of one 16-weight half
__device__ __forceinline__ uint32_t bf16x_dp_v2(
        const block_bf16x * GGML_RESTRICT row, const int iqs, const int p) {
    const uint8_t * mb = row->mant  + iqs * 14;
    const uint8_t * db = row->delta + iqs * 6;
    const uint8_t * sb = row->sgn   + iqs * 2;
    const uint32_t mw[4] = {
        bf16x_ld_u32_v2(mb), bf16x_ld_u32_v2(mb+4),
        bf16x_ld_u32_v2(mb+8), bf16x_ld_u32_v2(mb+12)};
    const uint32_t dw[2] = {bf16x_ld_u32_v2(db), bf16x_ld_u32_v2(db+4)};
    const uint32_t sw = bf16x_ld_u32_v2(sb);
    const int b7 = 14 * p, b3 = 6 * p;
    uint32_t v14 = ((b7 % 32 + 14 <= 32)
        ? (mw[b7/32] >> (b7%32))
        : __funnelshift_r(mw[b7/32], mw[b7/32+1], b7%32)) & 0x3FFF;
    uint32_t v6 = ((b3 % 32 + 6 <= 32)
        ? (dw[b3/32] >> (b3%32))
        : __funnelshift_r(dw[b3/32], dw[b3/32+1], b3%32)) & 0x3F;
    const uint32_t em7 = (uint32_t) row->emax[iqs] << 7;
    const uint32_t s01 = (sw >> (2 * p)) & 3u;
    const uint32_t d0 = v6 & 7u, d1 = (v6 >> 3) & 7u;
    uint32_t wlo = (em7 + (v14 & 0x7F) - (d0 << 7)) | ((s01 & 1) << 15);
    uint32_t whi = (em7 + (v14 >> 7)   - (d1 << 7)) | ((s01 >> 1) << 15);
    if (d0 == 7) wlo = 0;
    if (d1 == 7) whi = 0;
    return wlo | (whi << 16);
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
        for (int i = tid; i < BK * BN; i += NTHREADS) {
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

        // decode A for both k16 steps (8 u32 regs total)
        uint32_t A0[4] = {}, A1[4] = {};
        if (ok0) {
            A0[0] = bf16x_dp_v2(wrow0 + kb, 0, tig);
            A0[1] = bf16x_dp_v2(wrow0 + kb, 0, tig + 4);
            A1[0] = bf16x_dp_v2(wrow0 + kb, 1, tig);
            A1[1] = bf16x_dp_v2(wrow0 + kb, 1, tig + 4);
        }
        if (ok1) {
            A0[2] = bf16x_dp_v2(wrow1 + kb, 0, tig);
            A0[3] = bf16x_dp_v2(wrow1 + kb, 0, tig + 4);
            A1[2] = bf16x_dp_v2(wrow1 + kb, 1, tig);
            A1[3] = bf16x_dp_v2(wrow1 + kb, 1, tig + 4);
        }

        // mma step 0 (k16 step 0, half 0)
        {
            float d0[4] = {c[0][0], c[0][1], c[0][2], c[0][3]};
            for (int t = 0; t < NTILES; ++t) {
                const int nc = t * 8 + g;
                const __nv_bfloat16 x0 = sX[cur][tig * 2][nc];
                const __nv_bfloat16 x1 = sX[cur][tig * 2 + 1][nc];
                const __nv_bfloat16 x2 = sX[cur][tig * 2 + 8][nc];
                const __nv_bfloat16 x3 = sX[cur][tig * 2 + 9][nc];
                const uint32_t bb0 = __bfloat16_as_ushort(x0) | (__bfloat16_as_ushort(x1) << 16);
                const uint32_t bb1 = __bfloat16_as_ushort(x2) | (__bfloat16_as_ushort(x3) << 16);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3])
                    : "r"(A0[0]), "r"(A0[1]), "r"(bb0), "r"(bb1));
                c[t][0] = d0[0]; c[t][1] = d0[1]; c[t][2] = d0[2]; c[t][3] = d0[3];
            }
        }
        // mma step 1 (k16 step 1, half 1)
        {
            float d1[4] = {c[0][0], c[0][1], c[0][2], c[0][3]};
            for (int t = 0; t < NTILES; ++t) {
                const int nc = t * 8 + g;
                const __nv_bfloat16 x0 = sX[cur][16 + tig * 2][nc];
                const __nv_bfloat16 x1 = sX[cur][16 + tig * 2 + 1][nc];
                const __nv_bfloat16 x2 = sX[cur][16 + tig * 2 + 8][nc];
                const __nv_bfloat16 x3 = sX[cur][16 + tig * 2 + 9][nc];
                const uint32_t bb0 = __bfloat16_as_ushort(x0) | (__bfloat16_as_ushort(x1) << 16);
                const uint32_t bb1 = __bfloat16_as_ushort(x2) | (__bfloat16_as_ushort(x3) << 16);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3])
                    : "r"(A1[0]), "r"(A1[1]), "r"(bb0), "r"(bb1));
                c[t][0] = d1[0]; c[t][1] = d1[1]; c[t][2] = d1[2]; c[t][3] = d1[3];
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
        if (n0t + 1 < ne11) {
            if (mr < ne01) {
                *((float *) ((char *) dst + mr * 4 + n0t * nb1 + i12 * nb2 + i13 * nb3)) = c[t][0];
                *((float *) ((char *) dst + mr * 4 + (n0t + 1) * nb1 + i12 * nb2 + i13 * nb3)) = c[t][1];
            }
            if (m8 < ne01) {
                *((float *) ((char *) dst + m8 * 4 + n0t * nb1 + i12 * nb2 + i13 * nb3)) = c[t][2];
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
    const int r2 = ne12 / ne02, r3 = ne13 / ne03;
    const dim3 grid((unsigned) ((ne01 + 127) / 128), (unsigned) ((ne11 + 63) / 64), (unsigned) (ne12 * ne13));
    k_mul_mm_bf16x_mma<<<grid, 256, 0, ctx.stream>>>(
        (const char *) src0->data, (const float *) src1->data, (float *) dst->data,
        ne00, ne01, ne10, ne11, ne12,
        nb01, nb02, nb03, nb11, nb12, nb13, nb1, nb2, nb3, r2, r3);
    return true;
}
