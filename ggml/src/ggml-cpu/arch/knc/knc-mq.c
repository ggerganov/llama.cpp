/* SPDX-License-Identifier: MIT
 * Packed layouts and block arithmetic follow GGML and the MIT KNC kernels.
 * Reuse one aligned Q8 block across a row tile; never expand model weights. */
#include "knc-mq.h"
#include <string.h>

#if !defined(__k1om__) && !defined(GGML_KNC_TEST_EMULATION)
#    error Native K1OM required; host semantics must be explicitly enabled
#endif

#define DECLARE_BLOCK(q)                                                                                        \
    extern void ggml_knc_q##q##_block_native(const uint8_t *, const uint8_t *, const int8_t *, const int32_t *, \
                                             const uint8_t *, const uint8_t *, size_t, int32_t *)
DECLARE_BLOCK(3);
DECLARE_BLOCK(4);
DECLARE_BLOCK(5);
DECLARE_BLOCK(6);

struct ggml_knc_mq_scratch {
    uint8_t low[128] __attribute__((aligned(64)));
    uint8_t high[64] __attribute__((aligned(64)));
    int8_t  y[256] __attribute__((aligned(64)));
    int32_t scales[16] __attribute__((aligned(64)));
    int32_t lanes[16] __attribute__((aligned(64)));
    int32_t bsums[16];
    float   yd;
};

static float load_half(const uint8_t * p) {
    const uint32_t h    = p[0] | ((uint32_t) p[1] << 8);
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t       exp = (h >> 10) & 31u, mant = h & 1023u, bits;
    float          value;
    if (!exp) {
        int shift = 0;
        if (!mant) {
            bits = sign;
        } else {
            while (!(mant & 1024u)) {
                mant <<= 1;
                ++shift;
            }
            bits = sign | ((uint32_t) (113 - shift) << 23) | ((mant & 1023u) << 13);
        }
    } else {
        bits = sign | ((exp == 31 ? 255 : exp + 112) << 23) | (mant << 13);
    }
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void stage_activation(struct ggml_knc_mq_scratch * s, const uint8_t * y) {
    uint32_t bits;
    memcpy(s->y, y + 4, sizeof(s->y));
    for (int j = 0; j < 16; ++j) {
        const uint32_t v = y[260 + 2 * j] | ((uint32_t) y[261 + 2 * j] << 8);
        s->bsums[j]      = v < 32768 ? (int32_t) v : (int32_t) v - 65536;
    }
#ifdef __k1om__
    /* GCC 5 can turn memcpy into an alignment-requiring VBROADCASTSS. */
    __asm__("movl %1, %0" : "=r"(bits) : "m"(*(const unsigned char(*)[4]) y));
#else
    memcpy(&bits, y, sizeof(bits));
#endif
    memcpy(&s->yd, &bits, sizeof(bits));
#ifdef GGML_KNC_MQ_TEST_STAGING
    extern void ggml_knc_mq_test_staged(void);
    ggml_knc_mq_test_staged();
#endif
}

static int32_t reduce(const struct ggml_knc_mq_scratch * s) {
    int32_t dot = 0;
    for (int j = 0; j < 16; ++j) {
        dot += s->lanes[j];
    }
    return dot;
}

/* KNC scalar metadata uses x87. Preserve its extended block expression until
 * the float row accumulator is updated; a float function return rounds early. */
static inline long double block_q3(struct ggml_knc_mq_scratch * s,
                                   const uint8_t *              x,
                                   const uint8_t *              y,
                                   size_t                       remaining) {
    memcpy(s->low, x + 32, 64);
    memcpy(s->high, x, 32);
    for (int j = 0; j < 16; ++j) {
        s->scales[j] = (((x[96 + j % 8] >> (j / 8 * 4)) & 15) | (((x[104 + j % 4] >> (j / 4 * 2)) & 3) << 4)) - 32;
    }
    ggml_knc_q3_block_native(s->low, s->high, s->y, s->scales, x, y, remaining, s->lanes);
    return ((long double) load_half(x + 108) * s->yd) * reduce(s);
}

static inline long double block_q4(struct ggml_knc_mq_scratch * s,
                                   const uint8_t *              x,
                                   const uint8_t *              y,
                                   size_t                       remaining) {
    int32_t correction = 0;
    memcpy(s->low, x + 16, 128);
    for (int j = 0; j < 8; ++j) {
        const int sc     = j < 4 ? x[4 + j] & 63 : (x[8 + j] & 15) | ((x[j] >> 6) << 4);
        const int mn     = j < 4 ? x[8 + j] & 63 : (x[8 + j] >> 4) | ((x[4 + j] >> 6) << 4);
        s->scales[2 * j] = s->scales[2 * j + 1] = sc;
        correction += mn * (s->bsums[2 * j] + s->bsums[2 * j + 1]);
    }
    ggml_knc_q4_block_native(s->low, s->high, s->y, s->scales, x, y, remaining, s->lanes);
    return ((long double) load_half(x) * s->yd) * reduce(s) - ((long double) load_half(x + 2) * s->yd) * correction;
}

static inline long double block_q5(struct ggml_knc_mq_scratch * s,
                                   const uint8_t *              x,
                                   const uint8_t *              y,
                                   size_t                       remaining) {
    int32_t correction = 0;
    memcpy(s->low, x + 48, 128);
    memcpy(s->high, x + 16, 32);
    for (int j = 0; j < 8; ++j) {
        const int sc = j < 4 ? x[4 + j] & 63 : (x[8 + j] & 15) | ((x[j] >> 6) << 4);
        const int mn = j < 4 ? x[8 + j] & 63 : (x[8 + j] >> 4) | ((x[4 + j] >> 6) << 4);
        s->scales[j] = sc;
        correction += mn * (s->bsums[2 * j] + s->bsums[2 * j + 1]);
    }
    ggml_knc_q5_block_native(s->low, s->high, s->y, s->scales, x, y, remaining, s->lanes);
    /* The established Q5 glue uses x87 extended products and rounds only
     * after adding a complete block to its float accumulator. Rounding a
     * returned block to float first changes later Q8 quantization decisions.
     * Keep that native instruction order explicitly, without fast-math. */
    return ((long double) reduce(s) * s->yd) * load_half(x) - ((long double) correction * s->yd) * load_half(x + 2);
}

static inline long double block_q6(struct ggml_knc_mq_scratch * s,
                                   const uint8_t *              x,
                                   const uint8_t *              y,
                                   size_t                       remaining) {
    memcpy(s->low, x, 128);
    memcpy(s->high, x + 128, 64);
    for (int j = 0; j < 16; ++j) {
        s->scales[j] = x[192 + j] < 128 ? x[192 + j] : (int) x[192 + j] - 256;
    }
    ggml_knc_q6_block_native(s->low, s->high, s->y, s->scales, x, y, remaining, s->lanes);
    return ((long double) load_half(x + 208) * s->yd) * reduce(s);
}

#define DEFINE_TILE(q, stride)                                                                         \
    static void tile_q##q(size_t blocks, size_t rows, const uint8_t * weights, size_t row_stride,      \
                          const uint8_t * activations, float * out) {                                  \
        struct ggml_knc_mq_scratch scratch;                                                            \
        float                      sums[GGML_KNC_MQ_MAX_ROWS] = { 0 };                                 \
        for (size_t b = 0; b < blocks; ++b) {                                                          \
            const uint8_t * y = activations + b * 292;                                                 \
            stage_activation(&scratch, y);                                                             \
            for (size_t r = 0; r < rows; ++r) {                                                        \
                sums[r] += block_q##q(&scratch, weights + r * row_stride + b * stride, y, blocks - b); \
            }                                                                                          \
        }                                                                                              \
        for (size_t r = 0; r < rows; ++r) {                                                            \
            out[r] = sums[r];                                                                          \
        }                                                                                              \
    }
DEFINE_TILE(3, 110)
DEFINE_TILE(4, 144)
DEFINE_TILE(5, 176)
DEFINE_TILE(6, 210)

ggml_knc_mq_tile_fn ggml_knc_mq_kernel(int bits) {
    switch (bits) {
        case 3:
            return tile_q3;
        case 4:
            return tile_q4;
        case 5:
            return tile_q5;
        case 6:
            return tile_q6;
        default:
            return NULL;
    }
}
