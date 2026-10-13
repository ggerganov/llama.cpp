/* SPDX-License-Identifier: MIT
 * GCC/K1OM glue for the private IMCI assembly loops. All scratch is fixed
 * size and the public wrapper proves alignment and writable disjointness.
 */
#include "ggml-knc-internal.h"
#include <string.h>

#define KNC_ASSUME_ALIGNED(p) ((p) = __builtin_assume_aligned((p), 64))

#if (defined(TARGET_KNC_3120A) + defined(TARGET_KNC_5110P) + defined(TARGET_KNC_7120P)) != 1
#error "Define exactly one of TARGET_KNC_3120A, TARGET_KNC_5110P or TARGET_KNC_7120P"
#endif

/* GCC 5.1.1 K1OM folds memcpy(float, packed_bytes, 4) into VBROADCASTSS,
 * which faults unless its source is four-byte aligned. Packed Q8 metadata
 * is allowed at any byte alignment. Force a four-byte integer GPR load;
 * its memory operand has byte alignment and exactly four bytes of extent.
 * Subsequent FP loads use compiler-owned naturally aligned local storage.
 * Host tests retain instrumented memcpy so ASAN observes every source byte.
 */
static float knc_load_f32_unaligned(const uint8_t *p) {
    uint32_t bits;
    float value;
#if defined(GGML_KNC_TEST_EMULATION)
    memcpy(&bits, p, sizeof(bits));
#else
    __asm__("movl %1, %0" : "=r"(bits)
            : "m"(*(const unsigned char (*)[4])p));
#endif
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Scalar metadata only: no F16C dependency and no alignment/aliasing pun. */
static float knc_half_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t) h >> 10) & 31u;
    uint32_t mant = (uint32_t) h & 1023u;
    uint32_t bits;
    float out;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int unbiased = -14;
            while ((mant & 1024u) == 0) { mant <<= 1; --unbiased; }
            bits = sign | ((uint32_t) (unbiased + 127) << 23) | ((mant & 1023u) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    memcpy(&out, &bits, sizeof(out));
    return out;
}

float ggml_knc_dot_packed_unchecked(size_t blocks, const uint8_t *x,
                                    const uint8_t *y, ggml_knc_workspace *w) {
    size_t b;
    float result = 0.0f;
    /* One fixed final-reduction buffer; live accumulators stay in ZMMs. */
    int32_t lanes[16] __attribute__((aligned(64)));
    uint8_t *q4 = w->internal_q4;
    uint8_t *qh = w->internal_qh;
    int8_t *q8 = w->internal_q8;
    KNC_ASSUME_ALIGNED(w);
    KNC_ASSUME_ALIGNED(q4);
    KNC_ASSUME_ALIGNED(qh);
    KNC_ASSUME_ALIGNED(q8);

    for (b = 0; b < blocks; ++b) {
        int j;
        int32_t correction = 0;
        int32_t dot = 0;
        uint16_t hd, hm;
        float yd, d, dmin;

        /* Packed fields cannot inherit arena alignment. Copy only live
         * bytes to the existing 576-byte per-worker aligned workspace. */
        memcpy(q4, x + 48, 128);
        memcpy(qh, x + 16, 32);
        memcpy(q8, y + 4, 256);
        memcpy(w->internal_bsums, y + 260, 32);
        memcpy(&hd, x, 2);
        memcpy(&hm, x + 2, 2);
        yd = knc_load_f32_unaligned(y);

        for (j = 0; j < 4; ++j) {
            w->internal_scales[j] = x[4 + j] & 63;
            w->internal_mins[j] = x[8 + j] & 63;
            w->internal_scales[j + 4] = (x[12 + j] & 15) | ((x[4 + j] >> 6) << 4);
            w->internal_mins[j + 4] = (x[12 + j] >> 4) | ((x[8 + j] >> 6) << 4);
        }
        for (j = 0; j < 8; ++j) {
            correction += w->internal_mins[j] *
                          ((int32_t) w->internal_bsums[2*j] + w->internal_bsums[2*j + 1]);
        }

        /* Four physical ZMM FMA chains never spill. Each merged lane is an
         * exact integer below 2^24; the assembly converts once to int32.
         * Horizontal integer addition preserves the exact Q5 block sum. */
        ggml_knc_q5_block_native(q4, qh, q8, w->internal_scales,
                                x, y, blocks - b, lanes);
        for (j = 0; j < 16; ++j) dot += lanes[j];
        d = knc_half_to_float(hd) * yd;
        dmin = knc_half_to_float(hm) * yd;
        result += d * (float) dot - dmin * (float) correction;
        x += GGML_KNC_Q5_K_BYTES;
        y += GGML_KNC_Q8_K_BYTES;
    }
    return result;
}

float ggml_knc_dot_f32_unchecked(size_t n, const float *x, const float *y) {
    size_t i;
    float result = 0.0f;
    /* One final vector store, never an accumulator spill/reload. */
    float lanes[16] __attribute__((aligned(64)));
    KNC_ASSUME_ALIGNED(x);
    KNC_ASSUME_ALIGNED(y);
    ggml_knc_f32_vectors_native(n, x, y, lanes);
    for (i = 0; i < 16; ++i) result += lanes[i];
    for (i = n - n % 16; i < n; ++i) result += x[i] * y[i];
    return result;
}
