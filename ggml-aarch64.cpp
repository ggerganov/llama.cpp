// SPDX-FileCopyrightText: Copyright 2024 Arm Ltd.
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#include "ggml-aarch64.h"

#define UNUSED GGML_UNUSED

size_t quantize_q4_0_aarch64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        int nrows_interleaved = 1;
        int blocklen_per_row;

#if defined(__ARM_FEATURE_SVE)
        if (svcntw() == 8) {
            nrows_interleaved = 8;
            blocklen_per_row = 8;
        }
        else if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            nrows_interleaved = 4;
            blocklen_per_row = 8;
        }
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
        nrows_interleaved = 4;
        blocklen_per_row = 8;
#elif defined(__ARM_NEON)
        nrows_interleaved = 4;
        blocklen_per_row = 4;
#endif

        assert(n_per_row % QK4_0 == 0);
        const int nb = n_per_row / QK4_0;

        void * out_ptr_B = NULL;
        void * out_ptr_B_start = NULL;
        if (nrows_interleaved == 8) {
            out_ptr_B = (block_q4_0x8 *) malloc(sizeof(block_q4_0x8) * nb);
            out_ptr_B_start = out_ptr_B;
        }
        else if (nrows_interleaved == 4) {
            out_ptr_B = (block_q4_0x4 *) malloc(sizeof(block_q4_0x4) * nb);
            out_ptr_B_start = out_ptr_B;
        }

        for (int b = 0; b < (nrow * n_per_row); b += nrows_interleaved * n_per_row) {
            block_q4_0 ** in_ptrs = new block_q4_0 * [nrows_interleaved];

            for (int i  = 0; i < nrows_interleaved; i++ ) {
                in_ptrs[i] = (block_q4_0 *) dst + (b + i * n_per_row) / QK4_0;
                quantize_row_q4_0_reference(src + b + i * n_per_row, (block_q4_0 *) in_ptrs[i], n_per_row);
            }

            for (int64_t x = 0; x < nb; x++) {
                if (nrows_interleaved == 8) {
                    *(block_q4_0x8 *) out_ptr_B = make_block_q4_0x8(in_ptrs, blocklen_per_row, 0x88);
                    out_ptr_B = (block_q4_0x8 *) out_ptr_B + 1;
                }
                else if (nrows_interleaved == 4) {
                    *(block_q4_0x4 *) out_ptr_B = make_block_q4_0x4(in_ptrs, blocklen_per_row, 0x88);
                    out_ptr_B = (block_q4_0x4 *) out_ptr_B + 1;
                }

                for (int i = 0; i < nrows_interleaved; i++) {
                    in_ptrs[i]++;
                }
            }
            delete [] in_ptrs;
            out_ptr_B = out_ptr_B_start;
            if (nrows_interleaved == 8) memcpy ((block_q4_0 *) dst + b / QK4_0, out_ptr_B_start, sizeof(block_q4_0x8) * nb);
            else if (nrows_interleaved == 4) memcpy ((block_q4_0 *) dst + b / QK4_0, out_ptr_B_start, sizeof(block_q4_0x4) * nb);
        }
        if (out_ptr_B_start) free(out_ptr_B_start);

        return ((nrow * n_per_row) / QK4_0 * sizeof(block_q4_0));
    }
    else {
        assert(false);
        return 0;
    }
}

void quantize_q8_0_aarch64(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k, int nrows_interleaved, int blocklen_per_row) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float * id = new float[nrows_interleaved];
    auto srcv = new float32x4_t[nrows_interleaved][8];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < nrows_interleaved; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        if (blocklen_per_row == 8) {
            for (int j = 0; j < 4; j++) {
                float32x4_t v = vmulq_n_f32(srcv[0][2 * j], id[0]);
                int32x4_t vi = vcvtnq_s32_f32(v);
                y[i].qs[32 * j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[32 * j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[32 * j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[32 * j + 3] = vgetq_lane_s32(vi, 3);
                v = vmulq_n_f32(srcv[0][2 * j + 1], id[0]);
                vi = vcvtnq_s32_f32(v);
                y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
                y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
                y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
                y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

                v = vmulq_n_f32(srcv[1][2 * j], id[1]);
                vi = vcvtnq_s32_f32(v);
                y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
                y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
                y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
                y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
                v = vmulq_n_f32(srcv[1][2 * j + 1], id[1]);
                vi = vcvtnq_s32_f32(v);
                y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
                y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
                y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
                y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

                v = vmulq_n_f32(srcv[2][2 * j], id[2]);
                vi = vcvtnq_s32_f32(v);
                y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
                y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
                y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
                y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
                v = vmulq_n_f32(srcv[2][2 * j + 1], id[2]);
                vi = vcvtnq_s32_f32(v);
                y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
                y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
                y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
                y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

                v = vmulq_n_f32(srcv[3][2 * j], id[3]);
                vi = vcvtnq_s32_f32(v);
                y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
                y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
                y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
                y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
                v = vmulq_n_f32(srcv[3][2 * j + 1], id[3]);
                vi = vcvtnq_s32_f32(v);
                y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
                y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
                y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
                y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
            }
        }
        else if (blocklen_per_row == 4) {
            for (int j = 0; j < 8; j++) {
                float32x4_t v = vmulq_n_f32(srcv[0][j], id[0]);
                int32x4_t vi = vcvtnq_s32_f32(v);
                y[i].qs[16 * j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[16 * j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[16 * j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[16 * j + 3] = vgetq_lane_s32(vi, 3);

                v = vmulq_n_f32(srcv[1][j], id[1]);
                vi = vcvtnq_s32_f32(v);
                y[i].qs[16 * j + 4] = vgetq_lane_s32(vi, 0);
                y[i].qs[16 * j + 5] = vgetq_lane_s32(vi, 1);
                y[i].qs[16 * j + 6] = vgetq_lane_s32(vi, 2);
                y[i].qs[16 * j + 7] = vgetq_lane_s32(vi, 3);

                v = vmulq_n_f32(srcv[2][j], id[2]);
                vi = vcvtnq_s32_f32(v);
                y[i].qs[16 * j + 8] = vgetq_lane_s32(vi, 0);
                y[i].qs[16 * j + 9] = vgetq_lane_s32(vi, 1);
                y[i].qs[16 * j + 10] = vgetq_lane_s32(vi, 2);
                y[i].qs[16 * j + 11] = vgetq_lane_s32(vi, 3);

                v = vmulq_n_f32(srcv[3][j], id[3]);
                vi = vcvtnq_s32_f32(v);
                y[i].qs[16 * j + 12] = vgetq_lane_s32(vi, 0);
                y[i].qs[16 * j + 13] = vgetq_lane_s32(vi, 1);
                y[i].qs[16 * j + 14] = vgetq_lane_s32(vi, 2);
                y[i].qs[16 * j + 15] = vgetq_lane_s32(vi, 3);
            }
        }
    }
    delete [] id;
    delete [] srcv;
#endif
}

// Routines to create the blocked formats
// Note input is array of pointers.
// The exact interleaving format needed is different for GEMM (using SMMLA)
// and GEMV (using SDOT) cases.  For GEMM, we interleave 8 pairs of values
// at a time (with the two nibbles separated at runtime to give 2x2x8
// matrices).  For GEMV, we need to interleave 4 pairs of values instead.
block_q4_0x4 make_block_q4_0x4(const block_q4_0 * const in[4], unsigned int block_len, unsigned int xor_mask) {
    block_q4_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i]->d;
    }

    for (int i = 0; i < QK4_0 * 2; i++) {
        // We are interleaving 4 rows in blocks of 8, making a total of 32
        // output bytes per block (2 MMLA input vectors).  This repeats
        // until we have processed the whole block.
        //
        // Per the comment above, for GEMV cases a similar process is used
        // but with blocks of 4 instead, giving a single DOT input vector.
        //
        // In the case of q4, we add on 128 to convert the top nibble from
        // "bias offset" form to pure sign form (this saves a subtract when
        // we unpack it).
        int src_offset = (i / (4 * block_len)) * block_len;
        int src_id = (i % (4 * block_len)) / block_len;
        src_offset += (i % block_len);

        out.qs[i] = in[src_id]->qs[src_offset] ^ xor_mask;
    }

    return out;
}

// 8-block version - see comments in code above
block_q4_0x8 make_block_q4_0x8(const block_q4_0 * const in[8], unsigned int block_len, unsigned int xor_mask) {
    block_q4_0x8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i]->d;
    }

    for (int i = 0; i < QK4_0 * 4; i++) {
        int src_offset = (i / (8 * block_len)) * block_len;
        int src_id = (i % (8 * block_len)) / block_len;
        src_offset += (i % block_len);

        out.qs[i] = in[src_id]->qs[src_offset] ^ xor_mask;
    }

    return out;
}

block_q8_0x4 make_block_q8_0x4(const block_q8_0 * const in[4], unsigned int block_len) {
    block_q8_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i]->d;
    }

    for (int i = 0; i < QK8_0 * 4; i++) {
        int src_offset = (i / (4 * block_len)) * block_len;
        int src_id = (i % (4 * block_len)) / block_len;
        src_offset += (i % block_len);

        out.qs[i] = in[src_id]->qs[src_offset];
    }

    return out;
}

// 8-block version - see comments in code above
block_q8_0x8 make_block_q8_0x8(const block_q8_0 * const in[8], unsigned int block_len) {
    block_q8_0x8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i]->d;
    }

    for (int i = 0; i < QK8_0 * 8; i++) {
        int src_offset = (i / (8 * block_len)) * block_len;
        int src_id = (i % (8 * block_len)) / block_len;
        src_offset += (i % block_len);

        out.qs[i] = in[src_id]->qs[src_offset];
    }

    return out;
}

inline int64_t roundup(const int64_t a, const int64_t b) {
    int64_t rem = a % b;

    if (rem) {
        return a + b - rem;
    } else {
        return a;
    }
}

void ggml_gemv_q4_0_q8_0_aarch64_sve256(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth) {
#if defined(__ARM_FEATURE_SVE)
    if (svcntw() != 8) {
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) ggml_gemv_q4_0_q8_0_aarch64_neon(n, s, vx, vy, nr, nc, ith, nth);
        return;
    }
    int64_t x0 = roundup((ith * nc) / nth, (int64_t)8);
    int64_t xend = roundup(((ith + 1) * nc) / nth, (int64_t)8);
    size_t width = xend - x0;

    int64_t nb = n / QK4_0;
    const void * b_ptr = (const void *)((const block_q4_0x8 *) vx + ((x0 / 8) * nb));
    const void * a_ptr = vy;
    float * res_ptr = s + x0;

    assert(n % 32 == 0);
    assert(width % 8 == 0);

    size_t num_blocks = n / 32;

    __asm__ __volatile__(
        "ptrue p0.b\n"
        "add %x[b_ptr], %x[b_ptr], #0x10\n"
        "1:"  // Column loop
        "add x22, %x[a_ptr], #0x2\n"
        "mov z31.b, #0x0\n"
        "mov x21, %x[num_blocks]\n"
        "2:"  // Block loop
        "ld1b { z30.b }, p0/Z, [%x[b_ptr]]\n"
        "ld1b { z29.b }, p0/Z, [%x[b_ptr], #1, MUL VL]\n"
        "mov z28.s, #0x0\n"
        "mov z27.s, #0x0\n"
        "ld1rd { z26.d }, p0/Z, [x22]\n"
        "ld1b { z25.b }, p0/Z, [%x[b_ptr], #2, MUL VL]\n"
        "sub x20, x22, #0x2\n"
        "sub x21, x21, #0x1\n"
        "ld1b { z24.b }, p0/Z, [%x[b_ptr], #3, MUL VL]\n"
        "ld1rd { z23.d }, p0/Z, [x22, #8]\n"
        "lsl z22.b, z30.b, #0x4\n"
        "lsl z16.b, z29.b, #0x4\n"
        "and z30.b, z30.b, #0xf0\n"
        "and z29.b, z29.b, #0xf0\n"
        "ld1rd { z21.d }, p0/Z, [x22, #16]\n"
        "ld1rd { z20.d }, p0/Z, [x22, #24]\n"
        "lsl z19.b, z25.b, #0x4\n"
        "and z25.b, z25.b, #0xf0\n"
        "ld1rh { z17.h }, p0/Z, [x20]\n"
        "ld1h { z18.s }, p0/Z, [%x[b_ptr], #-1, MUL VL]\n"
        "sdot z28.s, z22.b, z26.b\n"
        "sdot z27.s, z16.b, z26.b\n"
        "lsl z16.b, z24.b, #0x4\n"
        "add x22, x22, #0x22\n"
        "and z24.b, z24.b, #0xf0\n"
        "add %x[b_ptr], %x[b_ptr], #0x90\n"
        "fcvt z17.s, p0/m, z17.h\n"
        "fcvt z18.s, p0/m, z18.h\n"
        "sdot z28.s, z19.b, z23.b\n"
        "sdot z27.s, z16.b, z23.b\n"
        "fmul z18.s, z18.s, z17.s\n"
        "sdot z28.s, z30.b, z21.b\n"
        "sdot z27.s, z29.b, z21.b\n"
        "sdot z28.s, z25.b, z20.b\n"
        "sdot z27.s, z24.b, z20.b\n"
        "uzp1 z17.s, z28.s, z27.s\n"
        "uzp2 z16.s, z28.s, z27.s\n"
        "add z17.s, z17.s, z16.s\n"
        "asr z17.s, z17.s, #0x4\n"
        "scvtf z17.s, p0/m, z17.s\n"
        "fmla z31.s, p0/M, z17.s, z18.s\n"
        "cbnz x21, 2b\n"
        "sub %x[width], %x[width], #0x8\n"
        "st1w { z31.s }, p0, [%x[res_ptr]]\n"
        "add %x[res_ptr], %x[res_ptr], #0x20\n"
        "cbnz %x[width], 1b\n"
        : [b_ptr] "+&r" (b_ptr), [res_ptr] "+&r" (res_ptr), [width] "+&r" (width)
        : [a_ptr] "r" (a_ptr), [num_blocks] "r" (num_blocks)
        : "memory", "p0", "x20", "x21", "x22", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
#endif
}

void ggml_gemv_q4_0_q8_0_aarch64_neon(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth) {
    UNUSED(nr);
#if defined(__ARM_NEON)
    int64_t x0 = roundup((ith * nc) / nth, (int64_t)4);
    int64_t xend = roundup(((ith + 1) * nc) / nth, (int64_t)4);
    size_t width = xend - x0;

    int64_t nb = n / QK4_0;
    const void * b_ptr = (const void *)((const block_q4_0x4 *) vx + ((x0 / 4) * nb));
    const void * a_ptr = vy;
    float * res_ptr = s + x0;

    assert(n % 32 == 0);
    assert(width % 4 == 0);

    size_t num_blocks = n / 32;

    __asm__ __volatile__(
        "movi v2.16b, #0x4\n"
        "movi v1.16b, #0xf0\n"
        "add %x[b_ptr], %x[b_ptr], #0x8\n"
        "1:"  // Column loop
        "add x23, %x[a_ptr], #0x2\n"
        "movi v0.16b, #0x0\n"
        "mov x22, %x[num_blocks]\n"
        "2:"  // Block loop
        "ldr q31, [%x[b_ptr], #0x0]\n"
        "ldr q30, [%x[b_ptr], #0x10]\n"
        "mov x21, x23\n"
        "movi v29.4s, #0x0\n"
        "ldr q28, [%x[b_ptr], #0x20]\n"
        "ldr q27, [%x[b_ptr], #0x30]\n"
        "movi v26.4s, #0x0\n"
        "sub x20, x23, #0x2\n"
        "ld1r { v25.8h }, [x20]\n"
        "ldr q24, [%x[b_ptr], #-0x8]\n"
        "sub x22, x22, #0x1\n"
        "add x23, x23, #0x22\n"
        "ld1r { v23.2d }, [x21], #0x8\n"
        "sshl v22.16b, v31.16b, v2.16b\n"
        "sshl v16.16b, v30.16b, v2.16b\n"
        "add %x[b_ptr], %x[b_ptr], #0x48\n"
        "ld1r { v21.2d }, [x21], #0x8\n"
        "sshl v20.16b, v28.16b, v2.16b\n"
        "sshl v19.16b, v27.16b, v2.16b\n"
        "ld1r { v18.2d }, [x21], #0x8\n"
        "ld1r { v17.2d }, [x21], #0x8\n"
        "and v31.16b, v31.16b, v1.16b\n"
        "and v30.16b, v30.16b, v1.16b\n"
        ".inst 0x4e9796dd  // sdot v29.4s, v22.16b, v23.16b\n"
        ".inst 0x4e97961a  // sdot v26.4s, v16.16b, v23.16b\n"
        "and v28.16b, v28.16b, v1.16b\n"
        "and v27.16b, v27.16b, v1.16b\n"
        "fcvtl v25.4s, v25.4h\n"
        "fcvtl v16.4s, v24.4h\n"
        ".inst 0x4e95969d  // sdot v29.4s, v20.16b, v21.16b\n"
        ".inst 0x4e95967a  // sdot v26.4s, v19.16b, v21.16b\n"
        "fmul v16.4s, v16.4s, v25.4s\n"
        ".inst 0x4e9297fd  // sdot v29.4s, v31.16b, v18.16b\n"
        ".inst 0x4e9297da  // sdot v26.4s, v30.16b, v18.16b\n"
        ".inst 0x4e91979d  // sdot v29.4s, v28.16b, v17.16b\n"
        ".inst 0x4e91977a  // sdot v26.4s, v27.16b, v17.16b\n"
        "addp v29.4s, v29.4s, v26.4s\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "fmla v0.4s, v29.4s, v16.4s\n"
        "cbnz x22, 2b\n"
        "sub %x[width], %x[width], #0x4\n"
        "str q0, [%x[res_ptr], #0x0]\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "cbnz %x[width], 1b\n"
        : [b_ptr] "+&r" (b_ptr), [res_ptr] "+&r" (res_ptr), [width] "+&r" (width)
        : [a_ptr] "r" (a_ptr), [num_blocks] "r" (num_blocks)
        : "memory", "v0", "v1", "v2", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23"
    );
#endif
}

void ggml_gemv_q4_0_q8_0_aarch64_neon_noi8mm(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth) {
    UNUSED(nr);
#if defined(__ARM_NEON)
    int64_t x0 = roundup((ith * nc) / nth, (int64_t)4);
    int64_t xend = roundup(((ith + 1) * nc) / nth, (int64_t)4);
    size_t width = xend - x0;

    int64_t nb = n / QK4_0;
    const void * b_ptr = (const void *)((const block_q4_0x4 *) vx + ((x0 / 4) * nb));
    const void * a_ptr = vy;
    float * res_ptr = s + x0;

    assert(n % 32 == 0);
    assert(width % 4 == 0);

    size_t num_blocks = n / 32;

    __asm__ __volatile__(
        "movi v31.16b, #0x4\n"
        "movi v30.16b, #0xf0\n"
        "add %x[b_ptr], %x[b_ptr], #0x8\n"
        "1:"  // Column loop
        "add x22, %x[a_ptr], #0x2\n"
        "movi v29.16b, #0x0\n"
        "mov x21, %x[num_blocks]\n"
        "2:"  // Block loop
        "ldr q28, [%x[b_ptr], #0x0]\n"
        "ldr q27, [x22, #0x0]\n"
        "movi v26.4s, #0x0\n"
        "sub x20, x22, #0x2\n"
        "ldr q25, [x22, #0x10]\n"
        "ldr q24, [%x[b_ptr], #0x10]\n"
        "sub x21, x21, #0x1\n"
        "add x22, x22, #0x22\n"
        "ldr q23, [%x[b_ptr], #0x20]\n"
        "ldr q22, [%x[b_ptr], #0x30]\n"
        "ld1r { v21.8h }, [x20]\n"
        "ldr q20, [%x[b_ptr], #-0x8]\n"
        "sshl v16.16b, v28.16b, v31.16b\n"
        "and v28.16b, v28.16b, v30.16b\n"
        "sshl v19.16b, v24.16b, v31.16b\n"
        "and v24.16b, v24.16b, v30.16b\n"
        "add %x[b_ptr], %x[b_ptr], #0x48\n"
        "sshl v18.16b, v23.16b, v31.16b\n"
        "and v23.16b, v23.16b, v30.16b\n"
        ".inst 0x4f9be21a  // sdot v26.4s, v16.16b, v27.4b[0]\n"
        "sshl v17.16b, v22.16b, v31.16b\n"
        "and v22.16b, v22.16b, v30.16b\n"
        "fcvtl v21.4s, v21.4h\n"
        "fcvtl v16.4s, v20.4h\n"
        ".inst 0x4f99e39a  // sdot v26.4s, v28.16b, v25.4b[0]\n"
        "fmul v16.4s, v16.4s, v21.4s\n"
        ".inst 0x4fbbe27a  // sdot v26.4s, v19.16b, v27.4b[1]\n"
        ".inst 0x4fb9e31a  // sdot v26.4s, v24.16b, v25.4b[1]\n"
        ".inst 0x4f9bea5a  // sdot v26.4s, v18.16b, v27.4b[2]\n"
        ".inst 0x4f99eafa  // sdot v26.4s, v23.16b, v25.4b[2]\n"
        ".inst 0x4fbbea3a  // sdot v26.4s, v17.16b, v27.4b[3]\n"
        ".inst 0x4fb9eada  // sdot v26.4s, v22.16b, v25.4b[3]\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "fmla v29.4s, v26.4s, v16.4s\n"
        "cbnz x21, 2b\n"
        "sub %x[width], %x[width], #0x4\n"
        "str q29, [%x[res_ptr], #0x0]\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "cbnz %x[width], 1b\n"
        : [b_ptr] "+&r" (b_ptr), [res_ptr] "+&r" (res_ptr), [width] "+&r" (width)
        : [a_ptr] "r" (a_ptr), [num_blocks] "r" (num_blocks)
        : "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22"
    );
#endif
}

void ggml_gemv_q8_0_q8_0_aarch64_sve256(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth) {
#if defined(__ARM_FEATURE_SVE)
    int64_t x0 = roundup((ith * nc) / nth, (int64_t)8);
    int64_t xend = roundup(((ith + 1) * nc) / nth, (int64_t)8);

    int64_t nb = n / QK8_0;
    int64_t a_nb = n / QK8_0;

    const svbool_t ptrue = svptrue_b8();

    const block_q8_0x8 * b_ptr_start = (const block_q8_0x8 *) vx;
    const block_q8_0 * a_ptr_start = (const block_q8_0 *) vy;

    for (int64_t y = 0; y < nr; y++) {
        for (int64_t x = x0 / 8; x < xend / 8; x++) {
            // Pointers to LHS blocks
            const block_q8_0 * a_ptr = a_ptr_start + (y * a_nb);
            // Pointers to RHS blocks
            const block_q8_0x8 * b_ptr = b_ptr_start + (x * nb);

            // Master FP accumulator
            svfloat32_t acc_row = svdup_f32(0.0f);

            for (int64_t b = 0; b < nb; b++) {
                // Set up RHS - we need rhs_mat_* and col_scale_f32 (9 registers)
                const svint8_t rhs_vec_0_0_0 = svld1_s8(ptrue, b_ptr[b].qs);
                const svint8_t rhs_vec_0_1_0 = svld1_vnum_s8(ptrue, b_ptr[b].qs, 1);
                const svint8_t rhs_vec_0_2_0 = svld1_vnum_s8(ptrue, b_ptr[b].qs, 2);
                const svint8_t rhs_vec_0_3_0 = svld1_vnum_s8(ptrue, b_ptr[b].qs, 3);
                const svint8_t rhs_vec_0_0_1 = svld1_vnum_s8(ptrue, b_ptr[b].qs, 4);
                const svint8_t rhs_vec_0_1_1 = svld1_vnum_s8(ptrue, b_ptr[b].qs, 5);
                const svint8_t rhs_vec_0_2_1 = svld1_vnum_s8(ptrue, b_ptr[b].qs, 6);
                const svint8_t rhs_vec_0_3_1 = svld1_vnum_s8(ptrue, b_ptr[b].qs, 7);

                // Scale values
                const svfloat16_t col_scale_f16 = svreinterpret_f16_u32(svld1uh_u32(ptrue, (const uint16_t *) b_ptr[b].d));
                const svfloat32_t col_scale_f32 = svcvt_f32_f16_x(ptrue, col_scale_f16);

                const svfloat16_t row_scale_f16 = svdup_f16(a_ptr[b].d);
                const svfloat32_t row_scale_f32 = svcvt_f32_f16_x(ptrue, row_scale_f16);

                const svint8_t lhs_vec_0 = svld1rq_s8(ptrue, a_ptr[b].qs);
                const svint8_t lhs_vec_1 = svld1rq_s8(ptrue, a_ptr[b].qs + 16);

                svint32_t iacc = svdup_s32(0);

                iacc = svdot_lane(iacc, rhs_vec_0_0_0, lhs_vec_0, 0);
                iacc = svdot_lane(iacc, rhs_vec_0_0_1, lhs_vec_1, 0);

                iacc = svdot_lane(iacc, rhs_vec_0_1_0, lhs_vec_0, 1);
                iacc = svdot_lane(iacc, rhs_vec_0_1_1, lhs_vec_1, 1);

                iacc = svdot_lane(iacc, rhs_vec_0_2_0, lhs_vec_0, 2);
                iacc = svdot_lane(iacc, rhs_vec_0_2_1, lhs_vec_1, 2);

                iacc = svdot_lane(iacc, rhs_vec_0_3_0, lhs_vec_0, 3);
                iacc = svdot_lane(iacc, rhs_vec_0_3_1, lhs_vec_1, 3);

                acc_row = svmla_x(ptrue, acc_row, svcvt_f32_s32_x(ptrue, iacc), svmul_x(ptrue, col_scale_f32, row_scale_f32));
            }

            svst1(ptrue, s + (y * nc + x * 8), acc_row);
        }
    }
#endif
}

void ggml_gemv_q8_0_q8_0_aarch64_neon(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth) {
#if defined(__ARM_NEON)
    int64_t x0 = roundup((ith * nc) / nth, (int64_t)8);
    int64_t xend = roundup(((ith + 1) * nc) / nth, (int64_t)8);

    int64_t nb = n / QK8_0;
    int64_t a_nb = n / QK8_0;

    const block_q8_0x8 * b_ptr_start = (const block_q8_0x8 *) vx;
    const block_q8_0 * a_ptr_start = (const block_q8_0 *) vy;

    for (int64_t y = 0; y < nr; y++) {
        for (int64_t x = x0 / 8; x < xend / 8; x++) {
            // Pointers to LHS blocks
            const block_q8_0 * a_ptr = a_ptr_start + (y * a_nb);
            // Pointers to RHS blocks
            const block_q8_0x8 * b_ptr = b_ptr_start + (x * nb);
            // Master FP accumulator
            float32x4_t acc_row[2];
            acc_row[0] = acc_row[1] = vdupq_n_f32(0.0f);

            for (int64_t b = 0; b < nb; b++) {
                // Set up RHS - we need rhs_mat_* and col_scale_f32 (9 registers)
                const int8x16_t rhs_vec_0_0_0 = vld1q_s8(b_ptr[b].qs);
                const int8x16_t rhs_vec_1_0_0 = vld1q_s8(b_ptr[b].qs + 16);
                const int8x16_t rhs_vec_0_1_0 = vld1q_s8(b_ptr[b].qs + 32);
                const int8x16_t rhs_vec_1_1_0 = vld1q_s8(b_ptr[b].qs + 48);
                const int8x16_t rhs_vec_0_2_0 = vld1q_s8(b_ptr[b].qs + 64);
                const int8x16_t rhs_vec_1_2_0 = vld1q_s8(b_ptr[b].qs + 80);
                const int8x16_t rhs_vec_0_3_0 = vld1q_s8(b_ptr[b].qs + 96);
                const int8x16_t rhs_vec_1_3_0 = vld1q_s8(b_ptr[b].qs + 112);
                const int8x16_t rhs_vec_0_0_1 = vld1q_s8(b_ptr[b].qs + 128);
                const int8x16_t rhs_vec_1_0_1 = vld1q_s8(b_ptr[b].qs + 144);
                const int8x16_t rhs_vec_0_1_1 = vld1q_s8(b_ptr[b].qs + 160);
                const int8x16_t rhs_vec_1_1_1 = vld1q_s8(b_ptr[b].qs + 176);
                const int8x16_t rhs_vec_0_2_1 = vld1q_s8(b_ptr[b].qs + 192);
                const int8x16_t rhs_vec_1_2_1 = vld1q_s8(b_ptr[b].qs + 208);
                const int8x16_t rhs_vec_0_3_1 = vld1q_s8(b_ptr[b].qs + 224);
                const int8x16_t rhs_vec_1_3_1 = vld1q_s8(b_ptr[b].qs + 240);

                // Scale values - assemble the four row/column scales into a (64-bit) vector, then expand to FP32
                const float16x8_t col_scale_f16 = vld1q_f16((const ggml_fp16_internal_t *)(b_ptr[b].d));
                const float32x4_t col_scale_f32_0 = vcvt_f32_f16(vget_low_f16(col_scale_f16));
                const float32x4_t col_scale_f32_1 = vcvt_f32_f16(vget_high_f16(col_scale_f16));

                const float16x4_t row_scale_f16 = vld1_dup_f16((const ggml_fp16_internal_t *)(&(a_ptr[b].d)));
                const float32x4_t row_scale_f32 = vcvt_f32_f16(row_scale_f16);

                const int8x16_t lhs_vec_0 = vld1q_s8(a_ptr[b].qs);
                const int8x16_t lhs_vec_1 = vld1q_s8(a_ptr[b].qs + 16);

                int32x4_t iacc0 = vdupq_n_s32(0);
                int32x4_t iacc1 = vdupq_n_s32(0);

                iacc0 = vdotq_laneq_s32(iacc0, rhs_vec_0_0_0, lhs_vec_0, 0);
                iacc0 = vdotq_laneq_s32(iacc0, rhs_vec_0_0_1, lhs_vec_1, 0);

                iacc1 = vdotq_laneq_s32(iacc1, rhs_vec_1_0_0, lhs_vec_0, 0);
                iacc1 = vdotq_laneq_s32(iacc1, rhs_vec_1_0_1, lhs_vec_1, 0);

                iacc0 = vdotq_laneq_s32(iacc0, rhs_vec_0_1_0, lhs_vec_0, 1);
                iacc0 = vdotq_laneq_s32(iacc0, rhs_vec_0_1_1, lhs_vec_1, 1);

                iacc1 = vdotq_laneq_s32(iacc1, rhs_vec_1_1_0, lhs_vec_0, 1);
                iacc1 = vdotq_laneq_s32(iacc1, rhs_vec_1_1_1, lhs_vec_1, 1);

                iacc0 = vdotq_laneq_s32(iacc0, rhs_vec_0_2_0, lhs_vec_0, 2);
                iacc0 = vdotq_laneq_s32(iacc0, rhs_vec_0_2_1, lhs_vec_1, 2);

                iacc1 = vdotq_laneq_s32(iacc1, rhs_vec_1_2_0, lhs_vec_0, 2);
                iacc1 = vdotq_laneq_s32(iacc1, rhs_vec_1_2_1, lhs_vec_1, 2);

                iacc0 = vdotq_laneq_s32(iacc0, rhs_vec_0_3_0, lhs_vec_0, 3);
                iacc0 = vdotq_laneq_s32(iacc0, rhs_vec_0_3_1, lhs_vec_1, 3);

                iacc1 = vdotq_laneq_s32(iacc1, rhs_vec_1_3_0, lhs_vec_0, 3);
                iacc1 = vdotq_laneq_s32(iacc1, rhs_vec_1_3_1, lhs_vec_1, 3);

                acc_row[0] = vfmaq_f32(acc_row[0], vcvtq_f32_s32(iacc0), vmulq_f32(col_scale_f32_0, row_scale_f32));
                acc_row[1] = vfmaq_f32(acc_row[1], vcvtq_f32_s32(iacc1), vmulq_f32(col_scale_f32_1, row_scale_f32));
            }

            vst1q_f32(s + (y * nc + x * 8), acc_row[0]);
            vst1q_f32(s + (y * nc + x * 8 + 4), acc_row[1]);
        }
    }
#endif
}

void ggml_gemm_q4_0_q8_0_aarch64_sve256(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth) {
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (svcntw() != 8) {
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) ggml_gemm_q4_0_q8_0_aarch64_neon(n, s, vx, vy, nr, nc, ith, nth);
        return;
    }
    int64_t x0 = roundup((ith * nc) / nth, (int64_t)8);
    int64_t xend = roundup(((ith + 1) * nc) / nth, (int64_t)8);
    size_t width = xend - x0;

    int64_t nb = n / QK4_0;
    const void * b_ptr = (const void *)((const block_q4_0x8 *) vx + ((x0 / 8) * nb));
    const void * a_ptr = vy;
    float * res_ptr = s + x0;
    size_t res_stride = nc * sizeof(float);

    assert(n % 32 == 0);
    assert(width % 8 == 0);

    size_t num_blocks = n / 32;

    __asm__ __volatile__(
        "mov x20, #0x4\n"
        "mov x13, %x[nr]\n"
        "mov z28.s, #-0x4\n"
        "mov x12, #0x88\n"
        "ptrue p1.b\n"
        "whilelt p0.s, XZR, x20\n"
        "cmp x13, #0x10\n"
        "mul x12, %x[num_blocks], x12\n"
        "blt 4f\n"
        "1:"  // Row loop
        "add x11, %x[b_ptr], #0x10\n"
        "mov x10, %x[width]\n"
        "add x9, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:"  // Column loop
        "add x28, %x[a_ptr], #0x8\n"
        "mov z24.b, #0x0\n"
        "mov z15.b, #0x0\n"
        "mov x27, %x[num_blocks]\n"
        "add x26, x28, x12\n"
        "mov z12.b, #0x0\n"
        "mov z0.b, #0x0\n"
        "add x25, x26, x12\n"
        "mov z13.b, #0x0\n"
        "mov z1.b, #0x0\n"
        "add x24, x25, x12\n"
        "mov z20.b, #0x0\n"
        "mov z25.b, #0x0\n"
        "mov z11.b, #0x0\n"
        "mov z16.b, #0x0\n"
        "mov z19.b, #0x0\n"
        "mov z26.b, #0x0\n"
        "mov z8.b, #0x0\n"
        "mov z29.b, #0x0\n"
        "mov z27.b, #0x0\n"
        "mov z10.b, #0x0\n"
        "3:"  // Block loop
        "ld1b { z30.b }, p1/Z, [x11]\n"
        "ld1b { z21.b }, p1/Z, [x11, #1, MUL VL]\n"
        "mov z18.s, #0x0\n"
        "mov z7.s, #0x0\n"
        "ld1rqb { z3.b }, p1/Z, [x28]\n"
        "ld1rqb { z5.b }, p1/Z, [x28, #16]\n"
        "mov z9.s, #0x0\n"
        "mov z22.s, #0x0\n"
        "ld1b { z4.b }, p1/Z, [x11, #2, MUL VL]\n"
        "ld1b { z17.b }, p1/Z, [x11, #3, MUL VL]\n"
        "sub x20, x11, #0x10\n"
        "sub x23, x28, #0x8\n"
        "lsl z31.b, z30.b, #0x4\n"
        "lsl z6.b, z21.b, #0x4\n"
        "ld1h { z23.s }, p1/Z, [x20]\n"
        "sub x22, x26, #0x8\n"
        "and z30.b, z30.b, #0xf0\n"
        "and z21.b, z21.b, #0xf0\n"
        "sub x21, x25, #0x8\n"
        "sub x20, x24, #0x8\n"
        "lsl z14.b, z4.b, #0x4\n"
        "lsl z2.b, z17.b, #0x4\n"
        "subs x27, x27, #0x1\n"
        "add x11, x11, #0x90\n"
        ".inst 0x451f9872  // smmla z18.s, z3.b, z31.b\n"
        ".inst 0x45069867  // smmla z7.s, z3.b, z6.b\n"
        "ld1rqb { z3.b }, p1/Z, [x28, #32]\n"
        "and z4.b, z4.b, #0xf0\n"
        ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
        ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
        "ld1rqb { z5.b }, p1/Z, [x28, #48]\n"
        "and z17.b, z17.b, #0xf0\n"
        "fcvt z23.s, p1/m, z23.h\n"
        ".inst 0x450e9872  // smmla z18.s, z3.b, z14.b\n"
        ".inst 0x45029867  // smmla z7.s, z3.b, z2.b\n"
        "ld1rqb { z3.b }, p1/Z, [x28, #64]\n"
        ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
        ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
        "ld1rqb { z5.b }, p1/Z, [x28, #80]\n"
        "fscale z23.s, p1/m, z23.s, z28.s\n"
        ".inst 0x451e9872  // smmla z18.s, z3.b, z30.b\n"
        ".inst 0x45159867  // smmla z7.s, z3.b, z21.b\n"
        "ld1rqb { z3.b }, p1/Z, [x28, #96]\n"
        ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
        ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
        "ld1rqb { z5.b }, p1/Z, [x28, #112]\n"
        "add x28, x28, #0x88\n"
        ".inst 0x45049872  // smmla z18.s, z3.b, z4.b\n"
        ".inst 0x45119867  // smmla z7.s, z3.b, z17.b\n"
        "ld1h { z3.s }, p0/Z, [x23]\n"
        ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
        ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
        "fcvt z3.s, p1/m, z3.h\n"
        "uzp1 z5.d, z18.d, z7.d\n"
        "uzp2 z18.d, z18.d, z7.d\n"
        "mov z3.q, z3.q[0]\n"
        "uzp1 z7.d, z9.d, z22.d\n"
        "uzp2 z22.d, z9.d, z22.d\n"
        "fmul z9.s, z23.s, z3.s[0]\n"
        "scvtf z5.s, p1/m, z5.s\n"
        "scvtf z18.s, p1/m, z18.s\n"
        "scvtf z7.s, p1/m, z7.s\n"
        "scvtf z22.s, p1/m, z22.s\n"
        "fmla z24.s, p1/M, z5.s, z9.s\n"
        "ld1rqb { z5.b }, p1/Z, [x26]\n"
        "fmul z9.s, z23.s, z3.s[1]\n"
        "fmla z15.s, p1/M, z18.s, z9.s\n"
        "ld1rqb { z18.b }, p1/Z, [x26, #16]\n"
        "fmul z9.s, z23.s, z3.s[2]\n"
        "fmul z3.s, z23.s, z3.s[3]\n"
        "fmla z12.s, p1/M, z7.s, z9.s\n"
        "mov z9.s, #0x0\n"
        "ld1h { z7.s }, p0/Z, [x22]\n"
        ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
        "fmla z0.s, p1/M, z22.s, z3.s\n"
        "mov z22.s, #0x0\n"
        "ld1h { z3.s }, p0/Z, [x21]\n"
        ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
        "ld1rqb { z5.b }, p1/Z, [x26, #32]\n"
        "fcvt z7.s, p1/m, z7.h\n"
        "fcvt z3.s, p1/m, z3.h\n"
        ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
        ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
        "ld1rqb { z5.b }, p1/Z, [x26, #64]\n"
        "mov z7.q, z7.q[0]\n"
        "mov z3.q, z3.q[0]\n"
        ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
        ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
        "ld1rqb { z5.b }, p1/Z, [x26, #96]\n"
        ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
        ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
        "uzp1 z5.d, z9.d, z22.d\n"
        "scvtf z5.s, p1/m, z5.s\n"
        "uzp2 z22.d, z9.d, z22.d\n"
        "fmul z9.s, z23.s, z7.s[0]\n"
        "scvtf z22.s, p1/m, z22.s\n"
        "fmla z13.s, p1/M, z5.s, z9.s\n"
        "ld1rqb { z9.b }, p1/Z, [x25]\n"
        "fmul z5.s, z23.s, z7.s[1]\n"
        "fmla z1.s, p1/M, z22.s, z5.s\n"
        "mov z5.s, #0x0\n"
        "mov z22.s, #0x0\n"
        ".inst 0x451f9a45  // smmla z5.s, z18.b, z31.b\n"
        ".inst 0x45069a56  // smmla z22.s, z18.b, z6.b\n"
        "ld1rqb { z18.b }, p1/Z, [x26, #48]\n"
        ".inst 0x450e9a45  // smmla z5.s, z18.b, z14.b\n"
        ".inst 0x45029a56  // smmla z22.s, z18.b, z2.b\n"
        "ld1rqb { z18.b }, p1/Z, [x26, #80]\n"
        ".inst 0x451e9a45  // smmla z5.s, z18.b, z30.b\n"
        ".inst 0x45159a56  // smmla z22.s, z18.b, z21.b\n"
        "ld1rqb { z18.b }, p1/Z, [x26, #112]\n"
        "add x26, x26, #0x88\n"
        ".inst 0x45049a45  // smmla z5.s, z18.b, z4.b\n"
        ".inst 0x45119a56  // smmla z22.s, z18.b, z17.b\n"
        "uzp1 z18.d, z5.d, z22.d\n"
        "scvtf z18.s, p1/m, z18.s\n"
        "uzp2 z22.d, z5.d, z22.d\n"
        "fmul z5.s, z23.s, z7.s[2]\n"
        "fmul z7.s, z23.s, z7.s[3]\n"
        "scvtf z22.s, p1/m, z22.s\n"
        "fmla z20.s, p1/M, z18.s, z5.s\n"
        "ld1rqb { z18.b }, p1/Z, [x25, #16]\n"
        "ld1h { z5.s }, p0/Z, [x20]\n"
        "fcvt z5.s, p1/m, z5.h\n"
        "fmla z25.s, p1/M, z22.s, z7.s\n"
        "mov z22.s, #0x0\n"
        "mov z7.s, #0x0\n"
        ".inst 0x451f9936  // smmla z22.s, z9.b, z31.b\n"
        ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
        "ld1rqb { z9.b }, p1/Z, [x25, #32]\n"
        "mov z5.q, z5.q[0]\n"
        ".inst 0x450e9936  // smmla z22.s, z9.b, z14.b\n"
        ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
        "ld1rqb { z9.b }, p1/Z, [x25, #64]\n"
        ".inst 0x451e9936  // smmla z22.s, z9.b, z30.b\n"
        ".inst 0x45159927  // smmla z7.s, z9.b, z21.b\n"
        "ld1rqb { z9.b }, p1/Z, [x25, #96]\n"
        ".inst 0x45049936  // smmla z22.s, z9.b, z4.b\n"
        ".inst 0x45119927  // smmla z7.s, z9.b, z17.b\n"
        "uzp1 z9.d, z22.d, z7.d\n"
        "scvtf z9.s, p1/m, z9.s\n"
        "uzp2 z22.d, z22.d, z7.d\n"
        "fmul z7.s, z23.s, z3.s[0]\n"
        "scvtf z22.s, p1/m, z22.s\n"
        "fmla z11.s, p1/M, z9.s, z7.s\n"
        "ld1rqb { z9.b }, p1/Z, [x24]\n"
        "fmul z7.s, z23.s, z3.s[1]\n"
        "fmla z16.s, p1/M, z22.s, z7.s\n"
        "mov z22.s, #0x0\n"
        "mov z7.s, #0x0\n"
        ".inst 0x451f9a56  // smmla z22.s, z18.b, z31.b\n"
        ".inst 0x45069a47  // smmla z7.s, z18.b, z6.b\n"
        "ld1rqb { z18.b }, p1/Z, [x25, #48]\n"
        ".inst 0x450e9a56  // smmla z22.s, z18.b, z14.b\n"
        ".inst 0x45029a47  // smmla z7.s, z18.b, z2.b\n"
        "ld1rqb { z18.b }, p1/Z, [x25, #80]\n"
        ".inst 0x451e9a56  // smmla z22.s, z18.b, z30.b\n"
        ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
        "ld1rqb { z18.b }, p1/Z, [x25, #112]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x45049a56  // smmla z22.s, z18.b, z4.b\n"
        ".inst 0x45119a47  // smmla z7.s, z18.b, z17.b\n"
        "uzp1 z18.d, z22.d, z7.d\n"
        "scvtf z18.s, p1/m, z18.s\n"
        "uzp2 z7.d, z22.d, z7.d\n"
        "fmul z22.s, z23.s, z3.s[2]\n"
        "fmul z3.s, z23.s, z3.s[3]\n"
        "scvtf z7.s, p1/m, z7.s\n"
        "fmla z19.s, p1/M, z18.s, z22.s\n"
        "ld1rqb { z18.b }, p1/Z, [x24, #16]\n"
        "fmul z22.s, z23.s, z5.s[0]\n"
        "fmla z26.s, p1/M, z7.s, z3.s\n"
        "mov z3.s, #0x0\n"
        "mov z7.s, #0x0\n"
        ".inst 0x451f9923  // smmla z3.s, z9.b, z31.b\n"
        ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
        "ld1rqb { z9.b }, p1/Z, [x24, #32]\n"
        ".inst 0x450e9923  // smmla z3.s, z9.b, z14.b\n"
        ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
        "mov z9.s, #0x0\n"
        ".inst 0x451f9a49  // smmla z9.s, z18.b, z31.b\n"
        "mov z31.s, #0x0\n"
        ".inst 0x45069a5f  // smmla z31.s, z18.b, z6.b\n"
        "ld1rqb { z6.b }, p1/Z, [x24, #48]\n"
        "ld1rqb { z18.b }, p1/Z, [x24, #64]\n"
        ".inst 0x450e98c9  // smmla z9.s, z6.b, z14.b\n"
        "fmul z14.s, z23.s, z5.s[1]\n"
        ".inst 0x450298df  // smmla z31.s, z6.b, z2.b\n"
        "ld1rqb { z6.b }, p1/Z, [x24, #80]\n"
        "fmul z2.s, z23.s, z5.s[2]\n"
        "fmul z23.s, z23.s, z5.s[3]\n"
        ".inst 0x451e9a43  // smmla z3.s, z18.b, z30.b\n"
        ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
        "ld1rqb { z5.b }, p1/Z, [x24, #96]\n"
        ".inst 0x451e98c9  // smmla z9.s, z6.b, z30.b\n"
        ".inst 0x451598df  // smmla z31.s, z6.b, z21.b\n"
        "ld1rqb { z18.b }, p1/Z, [x24, #112]\n"
        "add x24, x24, #0x88\n"
        ".inst 0x450498a3  // smmla z3.s, z5.b, z4.b\n"
        ".inst 0x451198a7  // smmla z7.s, z5.b, z17.b\n"
        ".inst 0x45049a49  // smmla z9.s, z18.b, z4.b\n"
        ".inst 0x45119a5f  // smmla z31.s, z18.b, z17.b\n"
        "uzp1 z18.d, z3.d, z7.d\n"
        "uzp2 z5.d, z3.d, z7.d\n"
        "scvtf z18.s, p1/m, z18.s\n"
        "uzp1 z6.d, z9.d, z31.d\n"
        "uzp2 z9.d, z9.d, z31.d\n"
        "scvtf z5.s, p1/m, z5.s\n"
        "fmla z8.s, p1/M, z18.s, z22.s\n"
        "scvtf z6.s, p1/m, z6.s\n"
        "scvtf z9.s, p1/m, z9.s\n"
        "fmla z29.s, p1/M, z5.s, z14.s\n"
        "fmla z27.s, p1/M, z6.s, z2.s\n"
        "fmla z10.s, p1/M, z9.s, z23.s\n"
        "bgt 3b\n"
        "mov x20, %x[res_ptr]\n"
        "subs x10, x10, #0x8\n"
        "add %x[res_ptr], %x[res_ptr], #0x20\n"
        "st1w { z24.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z15.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z12.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z0.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z13.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z1.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z20.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z25.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z11.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z16.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z19.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z26.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z8.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z29.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z27.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "st1w { z10.s }, p1, [x20]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x13, x13, #0x10\n"
        "cmp x13, #0x10\n"
        "mov %x[res_ptr], x9\n"
        "madd %x[a_ptr], x20, x12, %x[a_ptr]\n"
        "bge 1b\n"
        "4:"  // Row loop skip
        "cbz x13, 9f\n"
        "5:"  // Row tail: Row loop
        "add x25, %x[b_ptr], #0x10\n"
        "mov x24, %x[width]\n"
        "add x23, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:"  // Row tail: Column loop
        "mov z24.b, #0x0\n"
        "mov z15.b, #0x0\n"
        "add x28, %x[a_ptr], #0x8\n"
        "mov x22, %x[num_blocks]\n"
        "mov z12.b, #0x0\n"
        "mov z0.b, #0x0\n"
        "7:"  // Row tail: Block loop
        "ld1b { z3.b }, p1/Z, [x25]\n"
        "ld1b { z6.b }, p1/Z, [x25, #1, MUL VL]\n"
        "mov z2.s, #0x0\n"
        "mov z25.s, #0x0\n"
        "ld1rqb { z26.b }, p1/Z, [x28]\n"
        "ld1rqb { z21.b }, p1/Z, [x28, #16]\n"
        "mov z27.s, #0x0\n"
        "mov z19.s, #0x0\n"
        "ld1b { z29.b }, p1/Z, [x25, #2, MUL VL]\n"
        "ld1b { z16.b }, p1/Z, [x25, #3, MUL VL]\n"
        "sub x21, x25, #0x10\n"
        "sub x20, x28, #0x8\n"
        "lsl z20.b, z3.b, #0x4\n"
        "lsl z4.b, z6.b, #0x4\n"
        "ld1rqb { z10.b }, p1/Z, [x28, #32]\n"
        "ld1rqb { z23.b }, p1/Z, [x28, #48]\n"
        "and z3.b, z3.b, #0xf0\n"
        "and z6.b, z6.b, #0xf0\n"
        "ld1rqb { z11.b }, p1/Z, [x28, #64]\n"
        "ld1rqb { z7.b }, p1/Z, [x28, #80]\n"
        "lsl z8.b, z29.b, #0x4\n"
        "lsl z14.b, z16.b, #0x4\n"
        "ld1rqb { z18.b }, p1/Z, [x28, #96]\n"
        "ld1rqb { z30.b }, p1/Z, [x28, #112]\n"
        ".inst 0x45149b42  // smmla z2.s, z26.b, z20.b\n"
        ".inst 0x45049b59  // smmla z25.s, z26.b, z4.b\n"
        "and z29.b, z29.b, #0xf0\n"
        "ld1h { z17.s }, p1/Z, [x21]\n"
        ".inst 0x45149abb  // smmla z27.s, z21.b, z20.b\n"
        ".inst 0x45049ab3  // smmla z19.s, z21.b, z4.b\n"
        "and z16.b, z16.b, #0xf0\n"
        "ld1h { z4.s }, p0/Z, [x20]\n"
        "subs x22, x22, #0x1\n"
        "add x28, x28, #0x88\n"
        "fcvt z17.s, p1/m, z17.h\n"
        "add x25, x25, #0x90\n"
        ".inst 0x45089942  // smmla z2.s, z10.b, z8.b\n"
        ".inst 0x450e9959  // smmla z25.s, z10.b, z14.b\n"
        "fcvt z4.s, p1/m, z4.h\n"
        ".inst 0x45089afb  // smmla z27.s, z23.b, z8.b\n"
        ".inst 0x450e9af3  // smmla z19.s, z23.b, z14.b\n"
        "fscale z17.s, p1/m, z17.s, z28.s\n"
        "mov z4.q, z4.q[0]\n"
        ".inst 0x45039962  // smmla z2.s, z11.b, z3.b\n"
        ".inst 0x45069979  // smmla z25.s, z11.b, z6.b\n"
        "fmul z23.s, z17.s, z4.s[0]\n"
        "fmul z9.s, z17.s, z4.s[1]\n"
        "fmul z21.s, z17.s, z4.s[2]\n"
        "fmul z4.s, z17.s, z4.s[3]\n"
        ".inst 0x450398fb  // smmla z27.s, z7.b, z3.b\n"
        ".inst 0x450698f3  // smmla z19.s, z7.b, z6.b\n"
        ".inst 0x451d9a42  // smmla z2.s, z18.b, z29.b\n"
        ".inst 0x45109a59  // smmla z25.s, z18.b, z16.b\n"
        ".inst 0x451d9bdb  // smmla z27.s, z30.b, z29.b\n"
        ".inst 0x45109bd3  // smmla z19.s, z30.b, z16.b\n"
        "uzp1 z31.d, z2.d, z25.d\n"
        "uzp2 z13.d, z2.d, z25.d\n"
        "scvtf z31.s, p1/m, z31.s\n"
        "uzp1 z17.d, z27.d, z19.d\n"
        "uzp2 z18.d, z27.d, z19.d\n"
        "scvtf z13.s, p1/m, z13.s\n"
        "fmla z24.s, p1/M, z31.s, z23.s\n"
        "scvtf z17.s, p1/m, z17.s\n"
        "scvtf z18.s, p1/m, z18.s\n"
        "fmla z15.s, p1/M, z13.s, z9.s\n"
        "fmla z12.s, p1/M, z17.s, z21.s\n"
        "fmla z0.s, p1/M, z18.s, z4.s\n"
        "bgt 7b\n"
        "mov x20, %x[res_ptr]\n"
        "cmp x13, #0x1\n"
        "st1w { z24.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x13, #0x2\n"
        "st1w { z15.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x13, #0x3\n"
        "st1w { z12.s }, p1, [x20]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "st1w { z0.s }, p1, [x20]\n"
        "8:"  // Row tail: Accumulator store skip
        "subs x24, x24, #0x8\n"
        "add %x[res_ptr], %x[res_ptr], #0x20\n"
        "bne 6b\n"
        "subs x13, x13, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x12\n"
        "mov %x[res_ptr], x23\n"
        "bgt 5b\n"
        "9:"  // Row tail: Row loop skip
        : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
        : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [num_blocks] "r" (num_blocks), [res_stride] "r" (res_stride), [width] "r" (width)
        : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
#endif
}

void ggml_gemm_q4_0_q8_0_aarch64_neon(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    int64_t x0 = roundup((ith * nc) / nth, (int64_t)4);
    int64_t xend = roundup(((ith + 1) * nc) / nth, (int64_t)4);
    size_t width = xend - x0;

    int64_t nb = n / QK4_0;
    const void * b_ptr = (const void *)((const block_q4_0x4 *) vx + ((x0 / 4) * nb));
    const void * a_ptr = vy;
    float * res_ptr = s + x0;
    size_t res_stride = nc * sizeof(float);

    assert(n % 32 == 0);
    assert(width % 4 == 0);

    size_t num_blocks = n / 32;

    __asm__ __volatile__(
        "mov x10, %x[nr]\n"
        "mov x9, #0x88\n"
        "cmp x10, #0x10\n"
        "mul x9, %x[num_blocks], x9\n"
        "blt 4f\n"
        "1:"  // Row loop
        "add x28, %x[b_ptr], #0x8\n"
        "mov x27, %x[width]\n"
        "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:"  // Column loop
        "add x25, %x[a_ptr], #0x8\n"
        "movi v2.16b, #0x0\n"
        "movi v10.16b, #0x0\n"
        "mov x24, %x[num_blocks]\n"
        "add x23, x25, x9\n"
        "movi v12.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "add x22, x23, x9\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "add x21, x22, x9\n"
        "movi v22.16b, #0x0\n"
        "movi v23.16b, #0x0\n"
        "movi v25.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v7.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "movi v6.16b, #0x0\n"
        "movi v30.16b, #0x0\n"
        "movi v24.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "3:"  // Block loop
        "ldr q21, [x28, #0x0]\n"
        "ldr q16, [x28, #0x10]\n"
        "movi v1.16b, #0x4\n"
        "movi v19.4s, #0x0\n"
        "ldr q27, [x25, #0x0]\n"
        "ldr q15, [x25, #0x10]\n"
        "movi v26.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        "ldr q29, [x28, #0x20]\n"
        "ldr q3, [x28, #0x30]\n"
        "movi v17.4s, #0x0\n"
        "movi v0.16b, #0xf0\n"
        "ldr d20, [x25, #-0x8]\n"
        "ldr d9, [x23, #-0x8]\n"
        "sshl v8.16b, v21.16b, v1.16b\n"
        "sshl v31.16b, v16.16b, v1.16b\n"
        "and v21.16b, v21.16b, v0.16b\n"
        "and v16.16b, v16.16b, v0.16b\n"
        "sub x20, x28, #0x8\n"
        "subs x24, x24, #0x1\n"
        "add x28, x28, #0x48\n"
        ".inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b\n"
        ".inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b\n"
        "ldr q27, [x25, #0x20]\n"
        ".inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b\n"
        ".inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b\n"
        "sshl v15.16b, v29.16b, v1.16b\n"
        "sshl v1.16b, v3.16b, v1.16b\n"
        "and v29.16b, v29.16b, v0.16b\n"
        "and v3.16b, v3.16b, v0.16b\n"
        "ldr q0, [x25, #0x30]\n"
        "fcvtl v20.4s, v20.4h\n"
        ".inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b\n"
        "fcvtl v9.4s, v9.4h\n"
        ".inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b\n"
        "ldr q27, [x25, #0x40]\n"
        ".inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b\n"
        ".inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b\n"
        "ldr q0, [x25, #0x50]\n"
        ".inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b\n"
        ".inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b\n"
        "ldr q27, [x25, #0x60]\n"
        ".inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b\n"
        ".inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b\n"
        "ldr q0, [x25, #0x70]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b\n"
        ".inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b\n"
        "ldr d27, [x20, #0x0]\n"
        ".inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b\n"
        ".inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b\n"
        "fcvtl v27.4s, v27.4h\n"
        "uzp1 v0.2d, v19.2d, v26.2d\n"
        "uzp2 v26.2d, v19.2d, v26.2d\n"
        "fmul v19.4s, v27.4s, v20.s[0]\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "fmla v2.4s, v0.4s, v19.4s\n"
        "ldr q19, [x23, #0x0]\n"
        "uzp1 v0.2d, v18.2d, v17.2d\n"
        "uzp2 v18.2d, v18.2d, v17.2d\n"
        "fmul v17.4s, v27.4s, v20.s[1]\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v10.4s, v26.4s, v17.4s\n"
        "ldr q17, [x23, #0x10]\n"
        "fmul v26.4s, v27.4s, v20.s[2]\n"
        "fmul v20.4s, v27.4s, v20.s[3]\n"
        "fmla v12.4s, v0.4s, v26.4s\n"
        "ldr d0, [x22, #-0x8]\n"
        "ldr d26, [x21, #-0x8]\n"
        "fcvtl v0.4s, v0.4h\n"
        "fmla v28.4s, v18.4s, v20.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
        ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
        "ldr q19, [x23, #0x20]\n"
        "fcvtl v26.4s, v26.4h\n"
        ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
        ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
        "ldr q19, [x23, #0x40]\n"
        ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
        ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
        "ldr q19, [x23, #0x60]\n"
        ".inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b\n"
        ".inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b\n"
        "uzp1 v19.2d, v20.2d, v18.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp2 v20.2d, v20.2d, v18.2d\n"
        "fmul v18.4s, v27.4s, v9.s[0]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v11.4s, v19.4s, v18.4s\n"
        "ldr q18, [x22, #0x0]\n"
        "fmul v19.4s, v27.4s, v9.s[1]\n"
        "fmla v13.4s, v20.4s, v19.4s\n"
        "movi v19.4s, #0x0\n"
        "movi v20.4s, #0x0\n"
        ".inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b\n"
        ".inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b\n"
        "ldr q17, [x23, #0x30]\n"
        ".inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b\n"
        ".inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b\n"
        "ldr q17, [x23, #0x50]\n"
        ".inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b\n"
        ".inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b\n"
        "ldr q17, [x23, #0x70]\n"
        "add x23, x23, #0x88\n"
        ".inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b\n"
        ".inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b\n"
        "uzp1 v17.2d, v19.2d, v20.2d\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "uzp2 v20.2d, v19.2d, v20.2d\n"
        "fmul v19.4s, v27.4s, v9.s[2]\n"
        "fmul v9.4s, v27.4s, v9.s[3]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v22.4s, v17.4s, v19.4s\n"
        "ldr q17, [x22, #0x10]\n"
        "movi v19.4s, #0x0\n"
        ".inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b\n"
        "fmla v23.4s, v20.4s, v9.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        ".inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b\n"
        "ldr q18, [x22, #0x20]\n"
        ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
        ".inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b\n"
        ".inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b\n"
        "ldr q18, [x22, #0x40]\n"
        ".inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b\n"
        ".inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b\n"
        "ldr q18, [x22, #0x60]\n"
        ".inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b\n"
        ".inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b\n"
        "ldr q17, [x22, #0x30]\n"
        ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
        ".inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b\n"
        "ldr q17, [x22, #0x50]\n"
        ".inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b\n"
        ".inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b\n"
        "ldr q17, [x22, #0x70]\n"
        "add x22, x22, #0x88\n"
        ".inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b\n"
        ".inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b\n"
        "uzp1 v17.2d, v19.2d, v20.2d\n"
        "uzp2 v20.2d, v19.2d, v20.2d\n"
        "fmul v19.4s, v27.4s, v0.s[0]\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v25.4s, v17.4s, v19.4s\n"
        "ldr q19, [x21, #0x0]\n"
        "fmul v17.4s, v27.4s, v0.s[1]\n"
        "fmla v5.4s, v20.4s, v17.4s\n"
        "ldr q17, [x21, #0x10]\n"
        "uzp1 v20.2d, v9.2d, v18.2d\n"
        "uzp2 v9.2d, v9.2d, v18.2d\n"
        "fmul v18.4s, v27.4s, v0.s[2]\n"
        "fmul v0.4s, v27.4s, v0.s[3]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "fmla v7.4s, v20.4s, v18.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
        ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
        "ldr q19, [x21, #0x20]\n"
        "fmla v4.4s, v9.4s, v0.4s\n"
        "movi v9.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
        "fmul v8.4s, v27.4s, v26.s[0]\n"
        ".inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b\n"
        "ldr q17, [x21, #0x30]\n"
        ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
        "fmul v31.4s, v27.4s, v26.s[1]\n"
        ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
        "ldr q19, [x21, #0x40]\n"
        ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
        "fmul v15.4s, v27.4s, v26.s[2]\n"
        "fmul v27.4s, v27.4s, v26.s[3]\n"
        ".inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b\n"
        "ldr q1, [x21, #0x50]\n"
        ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
        ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
        "ldr q26, [x21, #0x60]\n"
        ".inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b\n"
        ".inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b\n"
        "ldr q21, [x21, #0x70]\n"
        "add x21, x21, #0x88\n"
        ".inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b\n"
        ".inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b\n"
        ".inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b\n"
        ".inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b\n"
        "uzp1 v29.2d, v20.2d, v18.2d\n"
        "uzp2 v21.2d, v20.2d, v18.2d\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "uzp1 v18.2d, v9.2d, v0.2d\n"
        "uzp2 v16.2d, v9.2d, v0.2d\n"
        "scvtf v21.4s, v21.4s, #0x4\n"
        "fmla v6.4s, v29.4s, v8.4s\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v30.4s, v21.4s, v31.4s\n"
        "fmla v24.4s, v18.4s, v15.4s\n"
        "fmla v14.4s, v16.4s, v27.4s\n"
        "bgt 3b\n"
        "mov x20, %x[res_ptr]\n"
        "subs x27, x27, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q10, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q28, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q11, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q7, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q4, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q6, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q24, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q14, [x20, #0x0]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x10, x10, #0x10\n"
        "cmp x10, #0x10\n"
        "mov %x[res_ptr], x26\n"
        "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
        "bge 1b\n"
        "4:"  // Row loop skip
        "cbz x10, 9f\n"
        "5:"  // Row tail: Row loop
        "add x24, %x[b_ptr], #0x8\n"
        "mov x23, %x[width]\n"
        "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:"  // Row tail: Column loop
        "movi v2.16b, #0x0\n"
        "movi v10.16b, #0x0\n"
        "add x25, %x[a_ptr], #0x8\n"
        "mov x21, %x[num_blocks]\n"
        "movi v12.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "7:"  // Row tail: Block loop
        "ldr q6, [x24, #0x0]\n"
        "ldr q5, [x24, #0x10]\n"
        "movi v17.16b, #0x4\n"
        "movi v8.4s, #0x0\n"
        "ldr q4, [x25, #0x0]\n"
        "ldr q13, [x25, #0x10]\n"
        "movi v27.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "ldr q31, [x24, #0x20]\n"
        "ldr q14, [x24, #0x30]\n"
        "movi v29.4s, #0x0\n"
        "movi v22.16b, #0xf0\n"
        "ldr q11, [x25, #0x20]\n"
        "ldr q23, [x25, #0x30]\n"
        "sshl v21.16b, v6.16b, v17.16b\n"
        "sshl v16.16b, v5.16b, v17.16b\n"
        "ldr q20, [x25, #0x40]\n"
        "ldr q26, [x25, #0x50]\n"
        "and v6.16b, v6.16b, v22.16b\n"
        "and v5.16b, v5.16b, v22.16b\n"
        "ldr q25, [x25, #0x60]\n"
        "ldr q3, [x25, #0x70]\n"
        "sshl v19.16b, v31.16b, v17.16b\n"
        "sshl v18.16b, v14.16b, v17.16b\n"
        "ldr d17, [x25, #-0x8]\n"
        ".inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b\n"
        ".inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b\n"
        "and v31.16b, v31.16b, v22.16b\n"
        ".inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b\n"
        ".inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b\n"
        "and v14.16b, v14.16b, v22.16b\n"
        "sub x20, x24, #0x8\n"
        "ldr d16, [x20, #0x0]\n"
        "subs x21, x21, #0x1\n"
        "add x25, x25, #0x88\n"
        "fcvtl v17.4s, v17.4h\n"
        "add x24, x24, #0x48\n"
        ".inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b\n"
        ".inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b\n"
        ".inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b\n"
        ".inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b\n"
        "fcvtl v16.4s, v16.4h\n"
        ".inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b\n"
        ".inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b\n"
        "fmul v23.4s, v16.4s, v17.s[0]\n"
        "fmul v21.4s, v16.4s, v17.s[1]\n"
        "fmul v1.4s, v16.4s, v17.s[2]\n"
        "fmul v20.4s, v16.4s, v17.s[3]\n"
        ".inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b\n"
        ".inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b\n"
        ".inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b\n"
        ".inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b\n"
        ".inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b\n"
        ".inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b\n"
        "uzp1 v19.2d, v8.2d, v27.2d\n"
        "uzp2 v18.2d, v8.2d, v27.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp1 v17.2d, v0.2d, v29.2d\n"
        "uzp2 v16.2d, v0.2d, v29.2d\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v2.4s, v19.4s, v23.4s\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v10.4s, v18.4s, v21.4s\n"
        "fmla v12.4s, v17.4s, v1.4s\n"
        "fmla v28.4s, v16.4s, v20.4s\n"
        "bgt 7b\n"
        "mov x20, %x[res_ptr]\n"
        "cmp x10, #0x1\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x2\n"
        "str q10, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x3\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "str q28, [x20, #0x0]\n"
        "8:"  // Row tail: Accumulator store skip
        "subs x23, x23, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "bne 6b\n"
        "subs x10, x10, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x9\n"
        "mov %x[res_ptr], x22\n"
        "bgt 5b\n"
        "9:"  // Row tail: Row loop skip
        : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
        : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [num_blocks] "r" (num_blocks), [res_stride] "r" (res_stride), [width] "r" (width)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
#endif
}

void ggml_gemm_q4_0_q8_0_aarch64_neon_noi8mm(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth) {
#if defined(__ARM_NEON)
    int64_t x0 = roundup((ith * nc) / nth, (int64_t)4);
    int64_t xend = roundup(((ith + 1) * nc) / nth, (int64_t)4);
    size_t width = xend - x0;

    int64_t nb = n / QK4_0;
    const void * b_ptr = (const void *)((const block_q4_0x4 *) vx + ((x0/4) * nb));
    const void * a_ptr = vy;
    float * res_ptr = s + x0;
    size_t res_stride = nc * sizeof(float);

    assert(n % 32 == 0);
    assert(width % 4 == 0);

    size_t num_blocks = n / 32;

    __asm__ __volatile__(
        "mov x10, %x[nr]\n"
        "mov x9, #0x88\n"
        "cmp x10, #0x10\n"
        "mul x9, %x[num_blocks], x9\n"
        "blt 4f\n"
        "1:"  // Row loop
        "add x28, %x[b_ptr], #0x8\n"
        "mov x27, %x[width]\n"
        "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:"  // Column loop
        "add x25, %x[a_ptr], #0x8\n"
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "mov x24, %x[num_blocks]\n"
        "add x23, x25, x9\n"
        "movi v18.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "add x22, x23, x9\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "add x21, x22, x9\n"
        "movi v23.16b, #0x0\n"
        "movi v16.16b, #0x0\n"
        "movi v25.16b, #0x0\n"
        "movi v7.16b, #0x0\n"
        "movi v0.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v21.16b, #0x0\n"
        "movi v8.16b, #0x0\n"
        "movi v1.16b, #0x0\n"
        "3:"  // Block loop
        "ldr q3, [x28, #0x0]\n"
        "ldr q31, [x25, #0x0]\n"
        "movi v28.16b, #0x4\n"
        "movi v10.4s, #0x0\n"
        "ldr q22, [x28, #0x10]\n"
        "ldr q6, [x25, #0x10]\n"
        "movi v29.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "ldr q27, [x28, #0x20]\n"
        "ldr q30, [x28, #0x30]\n"
        "movi v20.4s, #0x0\n"
        "movi v24.16b, #0xf0\n"
        "ldr d2, [x25, #-0x8]\n"
        "ldr d26, [x23, #-0x8]\n"
        "sshl v12.16b, v3.16b, v28.16b\n"
        "sub x20, x28, #0x8\n"
        "ldr d17, [x20, #0x0]\n"
        "and v3.16b, v3.16b, v24.16b\n"
        "subs x24, x24, #0x1\n"
        "add x28, x28, #0x48\n"
        ".inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]\n"
        ".inst 0x4fbfe19d  // sdot v29.4s, v12.16b, v31.4b[1]\n"
        ".inst 0x4f9fe989  // sdot v9.4s, v12.16b, v31.4b[2]\n"
        ".inst 0x4fbfe994  // sdot v20.4s, v12.16b, v31.4b[3]\n"
        "sshl v31.16b, v22.16b, v28.16b\n"
        "and v22.16b, v22.16b, v24.16b\n"
        "fcvtl v17.4s, v17.4h\n"
        "fcvtl v2.4s, v2.4h\n"
        "fcvtl v26.4s, v26.4h\n"
        ".inst 0x4f86e3ea  // sdot v10.4s, v31.16b, v6.4b[0]\n"
        ".inst 0x4fa6e3fd  // sdot v29.4s, v31.16b, v6.4b[1]\n"
        ".inst 0x4f86ebe9  // sdot v9.4s, v31.16b, v6.4b[2]\n"
        ".inst 0x4fa6ebf4  // sdot v20.4s, v31.16b, v6.4b[3]\n"
        "sshl v6.16b, v27.16b, v28.16b\n"
        "sshl v28.16b, v30.16b, v28.16b\n"
        "and v27.16b, v27.16b, v24.16b\n"
        "and v30.16b, v30.16b, v24.16b\n"
        "ldr q24, [x25, #0x20]\n"
        ".inst 0x4f98e0ca  // sdot v10.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8c9  // sdot v9.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8d4  // sdot v20.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x30]\n"
        ".inst 0x4f98e38a  // sdot v10.4s, v28.16b, v24.4b[0]\n"
        ".inst 0x4fb8e39d  // sdot v29.4s, v28.16b, v24.4b[1]\n"
        ".inst 0x4f98eb89  // sdot v9.4s, v28.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb94  // sdot v20.4s, v28.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x40]\n"
        ".inst 0x4f98e06a  // sdot v10.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e869  // sdot v9.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e874  // sdot v20.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x50]\n"
        ".inst 0x4f98e2ca  // sdot v10.4s, v22.16b, v24.4b[0]\n"
        ".inst 0x4fb8e2dd  // sdot v29.4s, v22.16b, v24.4b[1]\n"
        ".inst 0x4f98eac9  // sdot v9.4s, v22.16b, v24.4b[2]\n"
        ".inst 0x4fb8ead4  // sdot v20.4s, v22.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x60]\n"
        ".inst 0x4f98e36a  // sdot v10.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb69  // sdot v9.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb74  // sdot v20.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x70]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x4f98e3ca  // sdot v10.4s, v30.16b, v24.4b[0]\n"
        ".inst 0x4fb8e3dd  // sdot v29.4s, v30.16b, v24.4b[1]\n"
        ".inst 0x4f98ebc9  // sdot v9.4s, v30.16b, v24.4b[2]\n"
        ".inst 0x4fb8ebd4  // sdot v20.4s, v30.16b, v24.4b[3]\n"
        "fmul v24.4s, v17.4s, v2.s[0]\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v15.4s, v10.4s, v24.4s\n"
        "ldr q24, [x23, #0x0]\n"
        "fmul v10.4s, v17.4s, v2.s[1]\n"
        "fmla v19.4s, v29.4s, v10.4s\n"
        "ldr q10, [x23, #0x10]\n"
        "fmul v29.4s, v17.4s, v2.s[2]\n"
        "fmul v2.4s, v17.4s, v2.s[3]\n"
        "fmla v18.4s, v9.4s, v29.4s\n"
        "movi v9.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        ".inst 0x4f98e189  // sdot v9.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e19d  // sdot v29.4s, v12.16b, v24.4b[1]\n"
        "fmla v14.4s, v20.4s, v2.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        ".inst 0x4f98e994  // sdot v20.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x20]\n"
        ".inst 0x4f8ae3e9  // sdot v9.4s, v31.16b, v10.4b[0]\n"
        ".inst 0x4faae3fd  // sdot v29.4s, v31.16b, v10.4b[1]\n"
        ".inst 0x4f8aebf4  // sdot v20.4s, v31.16b, v10.4b[2]\n"
        ".inst 0x4faaebe2  // sdot v2.4s, v31.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x30]\n"
        ".inst 0x4f98e0c9  // sdot v9.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8d4  // sdot v20.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x40]\n"
        ".inst 0x4f8ae389  // sdot v9.4s, v28.16b, v10.4b[0]\n"
        ".inst 0x4faae39d  // sdot v29.4s, v28.16b, v10.4b[1]\n"
        ".inst 0x4f8aeb94  // sdot v20.4s, v28.16b, v10.4b[2]\n"
        ".inst 0x4faaeb82  // sdot v2.4s, v28.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x50]\n"
        ".inst 0x4f98e069  // sdot v9.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e874  // sdot v20.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x60]\n"
        ".inst 0x4f8ae2c9  // sdot v9.4s, v22.16b, v10.4b[0]\n"
        ".inst 0x4faae2dd  // sdot v29.4s, v22.16b, v10.4b[1]\n"
        ".inst 0x4f8aead4  // sdot v20.4s, v22.16b, v10.4b[2]\n"
        ".inst 0x4faaeac2  // sdot v2.4s, v22.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x70]\n"
        "add x23, x23, #0x88\n"
        ".inst 0x4f98e369  // sdot v9.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb74  // sdot v20.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x0]\n"
        ".inst 0x4f8ae3c9  // sdot v9.4s, v30.16b, v10.4b[0]\n"
        ".inst 0x4faae3dd  // sdot v29.4s, v30.16b, v10.4b[1]\n"
        ".inst 0x4f8aebd4  // sdot v20.4s, v30.16b, v10.4b[2]\n"
        ".inst 0x4faaebc2  // sdot v2.4s, v30.16b, v10.4b[3]\n"
        "fmul v10.4s, v17.4s, v26.s[0]\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v11.4s, v9.4s, v10.4s\n"
        "ldr q9, [x22, #0x10]\n"
        "fmul v10.4s, v17.4s, v26.s[1]\n"
        "fmla v13.4s, v29.4s, v10.4s\n"
        "ldr d29, [x22, #-0x8]\n"
        "fmul v10.4s, v17.4s, v26.s[2]\n"
        "fmul v26.4s, v17.4s, v26.s[3]\n"
        "fcvtl v29.4s, v29.4h\n"
        "fmla v23.4s, v20.4s, v10.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "fmla v16.4s, v2.4s, v26.4s\n"
        "movi v26.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        ".inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
        ".inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x20]\n"
        ".inst 0x4f89e3f4  // sdot v20.4s, v31.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
        ".inst 0x4f89ebfa  // sdot v26.4s, v31.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebe2  // sdot v2.4s, v31.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x30]\n"
        ".inst 0x4f98e0d4  // sdot v20.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0ca  // sdot v10.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8da  // sdot v26.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x40]\n"
        ".inst 0x4f89e394  // sdot v20.4s, v28.16b, v9.4b[0]\n"
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
        ".inst 0x4f89eb9a  // sdot v26.4s, v28.16b, v9.4b[2]\n"
        ".inst 0x4fa9eb82  // sdot v2.4s, v28.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x50]\n"
        ".inst 0x4f98e074  // sdot v20.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e06a  // sdot v10.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e87a  // sdot v26.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x60]\n"
        ".inst 0x4f89e2d4  // sdot v20.4s, v22.16b, v9.4b[0]\n"
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
        ".inst 0x4f89eada  // sdot v26.4s, v22.16b, v9.4b[2]\n"
        ".inst 0x4fa9eac2  // sdot v2.4s, v22.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x70]\n"
        "add x22, x22, #0x88\n"
        ".inst 0x4f98e374  // sdot v20.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e36a  // sdot v10.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb7a  // sdot v26.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x21, #0x0]\n"
        ".inst 0x4f89e3d4  // sdot v20.4s, v30.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ca  // sdot v10.4s, v30.16b, v9.4b[1]\n"
        ".inst 0x4f89ebda  // sdot v26.4s, v30.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebc2  // sdot v2.4s, v30.16b, v9.4b[3]\n"
        "fmul v9.4s, v17.4s, v29.s[0]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v25.4s, v20.4s, v9.4s\n"
        "ldr q9, [x21, #0x10]\n"
        "fmul v20.4s, v17.4s, v29.s[1]\n"
        "fmla v7.4s, v10.4s, v20.4s\n"
        "ldr d20, [x21, #-0x8]\n"
        "fmul v10.4s, v17.4s, v29.s[2]\n"
        "fmul v29.4s, v17.4s, v29.s[3]\n"
        "fcvtl v20.4s, v20.4h\n"
        "fmla v0.4s, v26.4s, v10.4s\n"
        "movi v26.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "fmla v4.4s, v2.4s, v29.4s\n"
        "movi v2.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        ".inst 0x4f98e19a  // sdot v26.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
        ".inst 0x4f98e982  // sdot v2.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e99d  // sdot v29.4s, v12.16b, v24.4b[3]\n"
        "ldr q12, [x21, #0x20]\n"
        "fmul v24.4s, v17.4s, v20.s[0]\n"
        ".inst 0x4f89e3fa  // sdot v26.4s, v31.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
        ".inst 0x4f89ebe2  // sdot v2.4s, v31.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebfd  // sdot v29.4s, v31.16b, v9.4b[3]\n"
        "ldr q9, [x21, #0x30]\n"
        "fmul v31.4s, v17.4s, v20.s[1]\n"
        ".inst 0x4f8ce0da  // sdot v26.4s, v6.16b, v12.4b[0]\n"
        ".inst 0x4face0ca  // sdot v10.4s, v6.16b, v12.4b[1]\n"
        ".inst 0x4f8ce8c2  // sdot v2.4s, v6.16b, v12.4b[2]\n"
        ".inst 0x4face8dd  // sdot v29.4s, v6.16b, v12.4b[3]\n"
        "ldr q12, [x21, #0x40]\n"
        "fmul v6.4s, v17.4s, v20.s[2]\n"
        "fmul v20.4s, v17.4s, v20.s[3]\n"
        ".inst 0x4f89e39a  // sdot v26.4s, v28.16b, v9.4b[0]\n"
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
        ".inst 0x4f89eb82  // sdot v2.4s, v28.16b, v9.4b[2]\n"
        ".inst 0x4fa9eb9d  // sdot v29.4s, v28.16b, v9.4b[3]\n"
        "ldr q9, [x21, #0x50]\n"
        ".inst 0x4f8ce07a  // sdot v26.4s, v3.16b, v12.4b[0]\n"
        ".inst 0x4face06a  // sdot v10.4s, v3.16b, v12.4b[1]\n"
        ".inst 0x4f8ce862  // sdot v2.4s, v3.16b, v12.4b[2]\n"
        ".inst 0x4face87d  // sdot v29.4s, v3.16b, v12.4b[3]\n"
        "ldr q12, [x21, #0x60]\n"
        ".inst 0x4f89e2da  // sdot v26.4s, v22.16b, v9.4b[0]\n"
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
        ".inst 0x4f89eac2  // sdot v2.4s, v22.16b, v9.4b[2]\n"
        ".inst 0x4fa9eadd  // sdot v29.4s, v22.16b, v9.4b[3]\n"
        "ldr q17, [x21, #0x70]\n"
        "add x21, x21, #0x88\n"
        ".inst 0x4f8ce37a  // sdot v26.4s, v27.16b, v12.4b[0]\n"
        ".inst 0x4face36a  // sdot v10.4s, v27.16b, v12.4b[1]\n"
        ".inst 0x4f8ceb62  // sdot v2.4s, v27.16b, v12.4b[2]\n"
        ".inst 0x4faceb7d  // sdot v29.4s, v27.16b, v12.4b[3]\n"
        ".inst 0x4f91e3da  // sdot v26.4s, v30.16b, v17.4b[0]\n"
        ".inst 0x4fb1e3ca  // sdot v10.4s, v30.16b, v17.4b[1]\n"
        ".inst 0x4f91ebc2  // sdot v2.4s, v30.16b, v17.4b[2]\n"
        ".inst 0x4fb1ebdd  // sdot v29.4s, v30.16b, v17.4b[3]\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "fmla v5.4s, v26.4s, v24.4s\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "fmla v21.4s, v10.4s, v31.4s\n"
        "fmla v8.4s, v2.4s, v6.4s\n"
        "fmla v1.4s, v29.4s, v20.4s\n"
        "bgt 3b\n"
        "mov x20, %x[res_ptr]\n"
        "subs x27, x27, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q14, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q11, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q16, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q7, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q0, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q4, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q21, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q8, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q1, [x20, #0x0]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x10, x10, #0x10\n"
        "cmp x10, #0x10\n"
        "mov %x[res_ptr], x26\n"
        "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
        "bge 1b\n"
        "4:"  // Row loop skip
        "cbz x10, 9f\n"
        "5:"  // Row tail: Row loop
        "add x24, %x[b_ptr], #0x8\n"
        "mov x23, %x[width]\n"
        "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:"  // Row tail: Column loop
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "add x25, %x[a_ptr], #0x8\n"
        "mov x21, %x[num_blocks]\n"
        "movi v18.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "7:"  // Row tail: Block loop
        "ldr q7, [x24, #0x0]\n"
        "ldr q5, [x25, #0x0]\n"
        "movi v9.16b, #0x4\n"
        "movi v4.4s, #0x0\n"
        "ldr q3, [x24, #0x10]\n"
        "ldr q2, [x25, #0x10]\n"
        "movi v1.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "ldr q13, [x24, #0x20]\n"
        "ldr q31, [x25, #0x20]\n"
        "movi v30.4s, #0x0\n"
        "movi v29.16b, #0xf0\n"
        "ldr q28, [x24, #0x30]\n"
        "ldr q27, [x25, #0x30]\n"
        "sshl v20.16b, v7.16b, v9.16b\n"
        "sub x20, x24, #0x8\n"
        "ldr q26, [x25, #0x40]\n"
        "ldr q25, [x25, #0x50]\n"
        "sshl v17.16b, v3.16b, v9.16b\n"
        "and v7.16b, v7.16b, v29.16b\n"
        "ldr q24, [x25, #0x60]\n"
        "ldr q16, [x25, #0x70]\n"
        "sshl v22.16b, v13.16b, v9.16b\n"
        "and v3.16b, v3.16b, v29.16b\n"
        "ldr d21, [x20, #0x0]\n"
        "ldr d12, [x25, #-0x8]\n"
        ".inst 0x4f85e284  // sdot v4.4s, v20.16b, v5.4b[0]\n"
        ".inst 0x4fa5e281  // sdot v1.4s, v20.16b, v5.4b[1]\n"
        ".inst 0x4f85ea80  // sdot v0.4s, v20.16b, v5.4b[2]\n"
        ".inst 0x4fa5ea9e  // sdot v30.4s, v20.16b, v5.4b[3]\n"
        "sshl v9.16b, v28.16b, v9.16b\n"
        "subs x21, x21, #0x1\n"
        "and v13.16b, v13.16b, v29.16b\n"
        "and v28.16b, v28.16b, v29.16b\n"
        "add x25, x25, #0x88\n"
        "add x24, x24, #0x48\n"
        "fcvtl v21.4s, v21.4h\n"
        "fcvtl v12.4s, v12.4h\n"
        ".inst 0x4f82e224  // sdot v4.4s, v17.16b, v2.4b[0]\n"
        ".inst 0x4fa2e221  // sdot v1.4s, v17.16b, v2.4b[1]\n"
        ".inst 0x4f82ea20  // sdot v0.4s, v17.16b, v2.4b[2]\n"
        ".inst 0x4fa2ea3e  // sdot v30.4s, v17.16b, v2.4b[3]\n"
        "fmul v11.4s, v21.4s, v12.s[0]\n"
        "fmul v23.4s, v21.4s, v12.s[1]\n"
        "fmul v17.4s, v21.4s, v12.s[2]\n"
        ".inst 0x4f9fe2c4  // sdot v4.4s, v22.16b, v31.4b[0]\n"
        "fmul v6.4s, v21.4s, v12.s[3]\n"
        ".inst 0x4fbfe2c1  // sdot v1.4s, v22.16b, v31.4b[1]\n"
        ".inst 0x4f9feac0  // sdot v0.4s, v22.16b, v31.4b[2]\n"
        ".inst 0x4fbfeade  // sdot v30.4s, v22.16b, v31.4b[3]\n"
        ".inst 0x4f9be124  // sdot v4.4s, v9.16b, v27.4b[0]\n"
        ".inst 0x4fbbe121  // sdot v1.4s, v9.16b, v27.4b[1]\n"
        ".inst 0x4f9be920  // sdot v0.4s, v9.16b, v27.4b[2]\n"
        ".inst 0x4fbbe93e  // sdot v30.4s, v9.16b, v27.4b[3]\n"
        ".inst 0x4f9ae0e4  // sdot v4.4s, v7.16b, v26.4b[0]\n"
        ".inst 0x4fbae0e1  // sdot v1.4s, v7.16b, v26.4b[1]\n"
        ".inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]\n"
        ".inst 0x4fbae8fe  // sdot v30.4s, v7.16b, v26.4b[3]\n"
        ".inst 0x4f99e064  // sdot v4.4s, v3.16b, v25.4b[0]\n"
        ".inst 0x4fb9e061  // sdot v1.4s, v3.16b, v25.4b[1]\n"
        ".inst 0x4f99e860  // sdot v0.4s, v3.16b, v25.4b[2]\n"
        ".inst 0x4fb9e87e  // sdot v30.4s, v3.16b, v25.4b[3]\n"
        ".inst 0x4f98e1a4  // sdot v4.4s, v13.16b, v24.4b[0]\n"
        ".inst 0x4fb8e1a1  // sdot v1.4s, v13.16b, v24.4b[1]\n"
        ".inst 0x4f98e9a0  // sdot v0.4s, v13.16b, v24.4b[2]\n"
        ".inst 0x4fb8e9be  // sdot v30.4s, v13.16b, v24.4b[3]\n"
        ".inst 0x4f90e384  // sdot v4.4s, v28.16b, v16.4b[0]\n"
        ".inst 0x4fb0e381  // sdot v1.4s, v28.16b, v16.4b[1]\n"
        ".inst 0x4f90eb80  // sdot v0.4s, v28.16b, v16.4b[2]\n"
        ".inst 0x4fb0eb9e  // sdot v30.4s, v28.16b, v16.4b[3]\n"
        "scvtf v4.4s, v4.4s, #0x4\n"
        "scvtf v1.4s, v1.4s, #0x4\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "fmla v15.4s, v4.4s, v11.4s\n"
        "scvtf v30.4s, v30.4s, #0x4\n"
        "fmla v19.4s, v1.4s, v23.4s\n"
        "fmla v18.4s, v0.4s, v17.4s\n"
        "fmla v14.4s, v30.4s, v6.4s\n"
        "bgt 7b\n"
        "mov x20, %x[res_ptr]\n"
        "cmp x10, #0x1\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x2\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x3\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "str q14, [x20, #0x0]\n"
        "8:"  // Row tail: Accumulator store skip
        "subs x23, x23, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "bne 6b\n"
        "subs x10, x10, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x9\n"
        "mov %x[res_ptr], x22\n"
        "bgt 5b\n"
        "9:"  // Row tail: Row loop skip
        : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
        : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [num_blocks] "r" (num_blocks), [res_stride] "r" (res_stride), [width] "r" (width)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
#endif
}

void ggml_gemm_q8_0_q8_0_aarch64(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    int64_t x0 = roundup((ith * nc) / nth, (int64_t)4);
    int64_t xend = roundup(((ith + 1) * nc) / nth, (int64_t)4);

    int64_t nb = n / QK8_0;
    int64_t a_nb = n / QK8_0;

    const block_q8_0x4 * b_ptr_start = (const block_q8_0x4 *) vx;
    const block_q8_0x4 * a_ptr_start = (const block_q8_0x4 *) vy;

    for (int64_t y = 0; y < nr / 4; y += nr / 4) {
        for (int64_t x = x0 / 4; x < xend / 4; x++) {
            const block_q8_0x4 ** a_ptrs = new const block_q8_0x4 * [nr / 4];

            a_ptrs[0] = a_ptr_start + (y * a_nb);
            for (int i = 0; i < (nr / 4) - 1; i++) {
                a_ptrs[i + 1] = a_ptrs[i] + a_nb;
            }

            const block_q8_0x4 * b_ptr = b_ptr_start + (x * nb);

            // Master FP accumulators
            float32x4_t * acc_rows = new float32x4_t[nr];
            for (int i = 0; i < nr; i++) {
                acc_rows[i] = vdupq_n_f32(0.0f);
            }

            for (int64_t b = 0; b < nb; b++) {
                // Set up RHS - we need rhs_mat_* and col_scale_f32 (9 registers)
                const int8x16_t rhs_mat_01_0 = vld1q_s8(b_ptr[b].qs);
                const int8x16_t rhs_mat_23_0 = vld1q_s8(b_ptr[b].qs + 16);
                const int8x16_t rhs_mat_01_1 = vld1q_s8(b_ptr[b].qs + 32);
                const int8x16_t rhs_mat_23_1 = vld1q_s8(b_ptr[b].qs + 48);
                const int8x16_t rhs_mat_01_2 = vld1q_s8(b_ptr[b].qs + 64);
                const int8x16_t rhs_mat_23_2 = vld1q_s8(b_ptr[b].qs + 80);
                const int8x16_t rhs_mat_01_3 = vld1q_s8(b_ptr[b].qs + 96);
                const int8x16_t rhs_mat_23_3 = vld1q_s8(b_ptr[b].qs + 112);

                // Scale values - assemble the four row/column scales into a (64-bit) vector, then expand to FP32
                const float16x4_t col_scale_f16 = vld1_f16((const ggml_fp16_internal_t *)(b_ptr[b].d));
                const float32x4_t col_scale_f32 = vcvt_f32_f16(col_scale_f16);

                // Process LHS in pairs of rows
                for (int rp = 0; rp < nr / 4; rp++) {
                    const int8x16_t lhs_mat_01_0 = vld1q_s8(a_ptrs[rp][b].qs);
                    const int8x16_t lhs_mat_23_0 = vld1q_s8(a_ptrs[rp][b].qs + 16);
                    const int8x16_t lhs_mat_01_1 = vld1q_s8(a_ptrs[rp][b].qs + 32);
                    const int8x16_t lhs_mat_23_1 = vld1q_s8(a_ptrs[rp][b].qs + 48);

                    const int8x16_t lhs_mat_01_2 = vld1q_s8(a_ptrs[rp][b].qs + 64);
                    const int8x16_t lhs_mat_23_2 = vld1q_s8(a_ptrs[rp][b].qs + 80);
                    const int8x16_t lhs_mat_01_3 = vld1q_s8(a_ptrs[rp][b].qs + 96);
                    const int8x16_t lhs_mat_23_3 = vld1q_s8(a_ptrs[rp][b].qs + 112);

                    // Do the MMLAs into 2x2 matrices
                    const int32x4_t iacc_mat_00 =
                        vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), lhs_mat_01_0, rhs_mat_01_0), lhs_mat_01_1, rhs_mat_01_1), lhs_mat_01_2, rhs_mat_01_2), lhs_mat_01_3, rhs_mat_01_3);
                    const int32x4_t iacc_mat_01 =
                        vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), lhs_mat_01_0, rhs_mat_23_0), lhs_mat_01_1, rhs_mat_23_1), lhs_mat_01_2, rhs_mat_23_2), lhs_mat_01_3, rhs_mat_23_3);
                    const int32x4_t iacc_mat_10 =
                        vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), lhs_mat_23_0, rhs_mat_01_0), lhs_mat_23_1, rhs_mat_01_1), lhs_mat_23_2, rhs_mat_01_2), lhs_mat_23_3, rhs_mat_01_3);
                    const int32x4_t iacc_mat_11 =
                        vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), lhs_mat_23_0, rhs_mat_23_0), lhs_mat_23_1, rhs_mat_23_1), lhs_mat_23_2, rhs_mat_23_2), lhs_mat_23_3, rhs_mat_23_3);

                    // Straighten out to make 4 row vectors
                    const int32x4_t iacc_row_0 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(iacc_mat_00), vreinterpretq_u64_s32(iacc_mat_01)));
                    const int32x4_t iacc_row_1 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(iacc_mat_00), vreinterpretq_u64_s32(iacc_mat_01)));
                    const int32x4_t iacc_row_2 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(iacc_mat_10), vreinterpretq_u64_s32(iacc_mat_11)));
                    const int32x4_t iacc_row_3 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(iacc_mat_10), vreinterpretq_u64_s32(iacc_mat_11)));

                    const float16x4_t row_scale_f16 = vld1_f16((const ggml_fp16_internal_t *)(a_ptrs[rp][b].d));
                    const float32x4_t row_scale_f32 = vcvt_f32_f16(row_scale_f16);

                    acc_rows[rp * 4] = vfmaq_f32(acc_rows[rp * 4], vcvtq_f32_s32(iacc_row_0), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 0));
                    acc_rows[rp * 4 + 1] = vfmaq_f32(acc_rows[rp * 4 + 1], vcvtq_f32_s32(iacc_row_1), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 1));
                    acc_rows[rp * 4 + 2] = vfmaq_f32(acc_rows[rp * 4 + 2], vcvtq_f32_s32(iacc_row_2), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 2));
                    acc_rows[rp * 4 + 3] = vfmaq_f32(acc_rows[rp * 4 + 3], vcvtq_f32_s32(iacc_row_3), vmulq_laneq_f32(col_scale_f32, row_scale_f32, 3));
                }
            }

            for (int i = 0; i < nr; i++) {
                vst1q_f32(s + ((y * 4 + i) * nc + x * 4), acc_rows[i]);
            }
            delete [] acc_rows;
            delete [] a_ptrs;
        }
    }
#endif
}
