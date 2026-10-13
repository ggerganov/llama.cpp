/* SPDX-License-Identifier: MIT
 * Independent packed-value oracle and actual staging-call counts. */
#include "knc-mq.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t         staged, prefetches;
static const uint8_t *weights, *activations;
static size_t         weight_bytes, activation_bytes;

void ggml_knc_mq_test_staged(void) {
    ++staged;
}

void ggml_knc_test_prefetch(const void * p, int hint) {
    const uintptr_t a = (uintptr_t) p, w = (uintptr_t) weights, y = (uintptr_t) activations;
    assert((a >= w && a - w < weight_bytes) || (a >= y && a - y < activation_bytes));
    assert(hint == 5 || hint == 6);
    ++prefetches;
}

static uint32_t rng = 104729;

static uint8_t random_byte(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static int signed_byte(uint8_t x) {
    return x < 128 ? x : (int) x - 256;
}

static void half_one(uint8_t * p) {
    p[0] = 0;
    p[1] = 0x3c;
}

static int value(const uint8_t * x, int q, int i) {
    const int g = i / 16, lane = i % 32;
    int       raw, scale;
    if (q == 3) {
        raw   = ((x[32 + i / 128 * 32 + lane] >> (i % 128 / 32 * 2)) & 3) - ((x[lane] & (1 << (i / 32))) ? 0 : 4);
        scale = ((x[96 + g % 8] >> (g / 8 * 4)) & 15) | (((x[104 + g % 4] >> (g / 4 * 2)) & 3) << 4);
        return raw * (scale - 32);
    }
    if (q == 4 || q == 5) {
        const int j = i / 32, low = q == 4 ? 16 : 48;
        scale        = j < 4 ? x[4 + j] & 63 : (x[8 + j] & 15) | ((x[j] >> 6) << 4);
        const int mn = j < 4 ? x[8 + j] & 63 : (x[8 + j] >> 4) | ((x[4 + j] >> 6) << 4);
        raw          = (x[low + i / 64 * 32 + lane] >> (i % 64 / 32 * 4)) & 15;
        if (q == 5) {
            raw |= ((x[16 + lane] >> (i / 32)) & 1) << 4;
        }
        return scale * raw - mn;
    }
    raw = ((x[i / 128 * 64 + i % 128 / 32 % 2 * 32 + lane] >> (i % 128 / 64 * 4)) & 15) |
          (((x[128 + i / 128 * 32 + lane] >> (i % 128 / 32 * 2)) & 3) << 4);
    return (raw - 32) * signed_byte(x[192 + g]);
}

int main(void) {
    size_t    checks    = 0;
    const int strides[] = { 0, 0, 0, 110, 144, 176, 210 };
    assert(ggml_knc_mq_kernel(2) == NULL && ggml_knc_mq_kernel(7) == NULL);
    for (int q = 3; q <= 6; ++q) {
        for (int off = 0; off < 64; ++off) {
            for (int nb = 1; nb <= 5; nb += 2) {
                for (int nr = 1; nr <= 8; ++nr) {
                    const size_t stride = nb * strides[q] + 7;
                    weight_bytes        = (nr - 1) * stride + nb * strides[q];
                    activation_bytes    = nb * 292;
                    uint8_t *u = malloc(weight_bytes + off), *v = malloc(activation_bytes + 63 - off);
                    assert(u && v);
                    uint8_t *x = u + off, *y = v + 63 - off;
                    weights     = x;
                    activations = y;
                    for (int b = 0; b < nb; ++b) {
                        uint8_t *   c   = y + b * 292;
                        const float one = 1.0f;
                        memcpy(c, &one, 4);
                        for (int j = 0; j < 256; ++j) {
                            c[4 + j] = random_byte();
                        }
                        for (int g = 0; g < 16; ++g) {
                            int sum = 0;
                            for (int j = 0; j < 16; ++j) {
                                sum += signed_byte(c[4 + 16 * g + j]);
                            }
                            const uint16_t s = (uint16_t) sum;
                            c[260 + 2 * g]   = s & 255;
                            c[261 + 2 * g]   = s >> 8;
                        }
                    }
                    float expected[8] = { 0 }, actual[10];
                    for (int r = 0; r < nr; ++r) {
                        for (int b = 0; b < nb; ++b) {
                            uint8_t * a = x + r * stride + b * strides[q];
                            for (int j = 0; j < strides[q]; ++j) {
                                a[j] = random_byte();
                            }
                            half_one(a + (q == 3 ? 108 : q == 6 ? 208 : 0));
                            if (q == 4 || q == 5) {
                                half_one(a + 2);
                            }
                            int32_t dot = 0;
                            for (int i = 0; i < 256; ++i) {
                                dot += value(a, q, i) * signed_byte(y[b * 292 + 4 + i]);
                            }
                            expected[r] += (float) dot;
                        }
                    }
                    actual[0] = actual[nr + 1] = 12345.0f;
                    staged                     = 0;
                    ggml_knc_mq_kernel(q)(nb, nr, x, stride, y, actual + 1);
                    assert(staged == (size_t) nb);
                    assert(actual[0] == 12345.0f && actual[nr + 1] == 12345.0f);
                    for (int r = 0; r < nr; ++r) {
                        if (actual[r + 1] != expected[r]) {
                            fprintf(stderr, "FAIL q%d offset=%d blocks=%d rows=%d row=%d actual=%g expected=%g\n", q,
                                    off, nb, nr, r, actual[r + 1], expected[r]);
                            return 1;
                        }
                        ++checks;
                    }
                    free(u);
                    free(v);
                }
            }
        }
    }
    printf(
        "PASS %zu independent row checks, staging once per block per tile, %zu bounded prefetches; host semantics only\n",
        checks, prefetches);
    return 0;
}
