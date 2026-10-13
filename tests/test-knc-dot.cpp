#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include "ggml-cpu/vec.h"
#include "ggml-cpu.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "ggml-knc.h"
extern "C" void ggml_vec_dot_q5_K_q8_K_generic(int, float *, size_t, const void *, size_t, const void *, size_t, int);
extern "C" void ggml_knc_vec_dot_f32_generic(int, float *, size_t, const float *, size_t, const float *, size_t, int);
extern "C" void ggml_knc_optimized_enable(int);
static int      checks;

static void equal(float got, float expected) {
    ++checks;
    if (!std::isfinite(got) || fabs(got - expected) > 0.005f + 0.0005f * fabs(expected)) {
        fprintf(stderr, "FAIL check=%d got=%.9g expected=%.9g\n", checks, got, expected);
        exit(1);
    }
}

int main() {
    ggml_cpu_init();
    ggml_init_params ip{ 1 << 20, nullptr, true };
    auto             ctx = ggml_init(ip);
    if (!ctx) {
        return 2;
    }
    alignas(64) float x[4096 + 64], y[4096 + 64];
    for (int i = 0; i < 4160; ++i) {
        x[i] = sinf(i * 0.17f) * 3;
        y[i] = cosf(i * 0.07f) * 2;
    }
    for (int n : { 0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 4096 }) {
        float a = 0, b = 0;
        ggml_vec_dot_f32(n, &a, 0, x, 0, y, 0, 1);
        ggml_knc_vec_dot_f32_generic(n, &b, 0, x, 0, y, 0, 1);
        equal(a, b);
    }
    for (int off = 0; off < 16; ++off) {
        for (int n : { 0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 4096 }) {
            float a = 0, b = 0;
            ggml_vec_dot_f32(n, &a, 0, x + off, 0, y + (15 - off), 0, 1);
            ggml_knc_vec_dot_f32_generic(n, &b, 0, x + off, 0, y + (15 - off), 0, 1);
            equal(a, b);
        }
    }
    for (int n : { 0, 256, 512, 768, 1024, 4096 }) {
        std::vector<unsigned char> q5(n / 256 * sizeof(block_q5_K) + 64), q8(n / 256 * sizeof(block_q8_K) + 64);
        quantize_row_q5_K_ref(x, reinterpret_cast<block_q5_K *>(q5.data()), n);
        quantize_row_q8_K_ref(y, reinterpret_cast<block_q8_K *>(q8.data()), n);
        float ref = 0;
        ggml_vec_dot_q5_K_q8_K_generic(n, &ref, 0, q5.data(), 0, q8.data(), 0, 1);
        for (int a = 0; a < 64; ++a) {
            for (int b = 0; b < 64; ++b) {
                std::vector<unsigned char> u(q5.size() + 64), v(q8.size() + 64);
                memcpy(u.data() + a, q5.data(), n / 256 * sizeof(block_q5_K));
                memcpy(v.data() + b, q8.data(), n / 256 * sizeof(block_q8_K));
                float got = 0;
                ggml_vec_dot_q5_K_q8_K(n, &got, 0, u.data() + a, 0, v.data() + b, 0, 1);
                equal(got, ref);
            }
        }
    }
    // All byte offsets are valid inputs. Generic oracle reads only aligned vectors.
    for (auto type : { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        auto traits = ggml_get_type_traits_cpu(type);
        for (int n : { 0, 256, 512, 768, 4096 }) {
            std::vector<unsigned char> q(n / 256 * ggml_type_size(type) + 64), v(n / 256 * sizeof(block_q8_K) + 64);
            ggml_get_type_traits(type)->from_float_ref(x, q.data(), n);
            quantize_row_q8_K_ref(y, (block_q8_K *) v.data(), n);
            float ref = 0;
            ggml_knc_optimized_enable(0);
            traits->vec_dot(n, &ref, 0, q.data(), 0, v.data(), 0, 1);
            ggml_knc_optimized_enable(1);
            for (int a = 0; a < 64; ++a) {
                for (int b = 0; b < 64; ++b) {
                    std::vector<unsigned char> u(q.size() + 64), w(v.size() + 64);
                    memcpy(u.data() + a, q.data(), n / 256 * ggml_type_size(type));
                    memcpy(w.data() + b, v.data(), n / 256 * sizeof(block_q8_K));
                    float got = 0;
                    traits->vec_dot(n, &got, 0, u.data() + a, 0, w.data() + b, 0, 1);
                    equal(got, ref);
                }
            }
        }
    }
    // Exercise the complete packed bit/scaling domain, not only a sine quantization.
    uint32_t rng = 1234567;
    for (auto type : { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        auto      traits = ggml_get_type_traits_cpu(type);
        const int size   = ggml_type_size(type);
        for (int trial = 0; trial < 1024; ++trial) {
            alignas(64) unsigned char packed[256];
            alignas(64) block_q8_K    q8;
            for (int i = 0; i < size; ++i) {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                packed[i] = rng;
            }
            for (int i = 0; i < 256; ++i) {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                q8.qs[i] = (int8_t) rng;
            }
            q8.d = 1;
            for (int j = 0; j < 16; ++j) {
                int sum = 0;
                for (int k = 0; k < 16; ++k) {
                    sum += q8.qs[j * 16 + k];
                }
                q8.bsums[j] = sum;
            }
            int d     = type == GGML_TYPE_Q2_K ? 80 : type == GGML_TYPE_Q3_K ? 108 : type == GGML_TYPE_Q6_K ? 208 : 0;
            packed[d] = 0;
            packed[d + 1] = 0x3c;  // exactly 1 in binary16
            if (type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q4_K) {
                packed[d + 2] = 0;
                packed[d + 3] = 0x38;
            }
            if (trial == 0 && type == GGML_TYPE_Q6_K) {
                memset(packed, 0, 192);
                memset(packed + 192, 128, 16);
                memset(q8.qs, 128, 256);
            }
            float ref, got;
            ggml_knc_optimized_enable(0);
            traits->vec_dot(256, &ref, 0, packed, 0, &q8, 0, 1);
            ggml_knc_optimized_enable(1);
            traits->vec_dot(256, &got, 0, packed, 0, &q8, 0, 1);
            equal(got, ref);
            int good = 1;
#pragma omp parallel for num_threads(8) reduction(& : good)
            for (int j = 0; j < 8; ++j) {
                float v;
                traits->vec_dot(256, &v, 0, packed, 0, &q8, 0, 1);
                good &= v == got;
            }
            if (!good) {
                fputs("TLS worker contamination\n", stderr);
                return 1;
            }
        }
    }
    ggml_free(ctx);
    printf("PASS %d integrated GGML dispatch/packed-boundary checks; no model acceptance\n", checks);
    return 0;
}
