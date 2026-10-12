// Parallel kernels of the emulated analog array, see analog-kernels.h

#include "analog-kernels.h"
#include "analog-parallel.h"
#include "analog-shader.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ANALOG_BLOCK 32

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

static std::atomic<int>  g_n_threads{1};
static std::atomic<bool> g_enabled{true};
static std::atomic<bool> g_env_loaded{false};
static std::atomic<int>  g_dispatch{GGML_ANALOG_DISPATCH_CPU};
static std::atomic<int>  g_env_n_threads{0};

const float * ggml_analog_kernel_noise_table(void) {
    static std::mutex         mutex;
    static std::vector<float> table;

    std::lock_guard<std::mutex> lock(mutex);

    if (table.empty()) {
        table.resize(GGML_ANALOG_NOISE_TABLE);

        for (int i = 0; i < GGML_ANALOG_NOISE_TABLE; ++i) {
            uint64_t h = (uint64_t) i*0x9E3779B97F4A7C15ull + 12345u;

            h += 0x9E3779B97F4A7C15ull;
            h = (h ^ (h >> 30))*0xBF58476D1CE4E5B9ull;
            h = (h ^ (h >> 27))*0x94D049BB133111EBull;
            h = h ^ (h >> 31);

            const float u = (float)  (h        & 0xFFFFFF)/(float) (1 << 24);
            const float v = (float) ((h >> 24) & 0xFFFFFF)/(float) (1 << 24);

            // u + v - 1 is triangular with variance 1/6
            table[i] = 2.4494898f*(u + v - 1.0f);
        }
    }

    return table.data();
}

// the environment is applied at first use and again from ggml_analog_kernels_env_reload(), when
// the command line options have written the GGML_ANALOG_* variables
static void analog_kernels_env_apply(void) {
    const char * val = getenv("GGML_ANALOG_KERNELS");
    if (val != nullptr && val[0] == '0') {
        g_enabled.store(false, std::memory_order_relaxed);
    }

    val = getenv("GGML_ANALOG_N_THREADS");
    if (val != nullptr) {
        g_env_n_threads.store(atoi(val), std::memory_order_relaxed);
    }

    val = getenv("GGML_ANALOG_DISPATCH");

    if (val != nullptr) {
        const std::string s(val);

        if (s == "shader" || s == "mirror") {
            g_dispatch.store(GGML_ANALOG_DISPATCH_SHADER, std::memory_order_relaxed);
        } else if (s == "vulkan" || s == "gpu") {
            g_dispatch.store(GGML_ANALOG_DISPATCH_VULKAN, std::memory_order_relaxed);
        } else {
            g_dispatch.store(GGML_ANALOG_DISPATCH_CPU, std::memory_order_relaxed);
        }
    }
}

static void analog_kernels_env_load(void) {
    if (g_env_loaded.exchange(true)) {
        return;
    }

    analog_kernels_env_apply();
}

void ggml_analog_kernels_env_reload(void) {
    analog_kernels_env_apply();

    g_env_loaded.store(true, std::memory_order_relaxed);
}

void ggml_analog_kernels_set_dispatch(enum ggml_analog_dispatch dispatch) {
    analog_kernels_env_load();

    g_dispatch.store((int) dispatch, std::memory_order_relaxed);
}

enum ggml_analog_dispatch ggml_analog_kernels_get_dispatch(void) {
    analog_kernels_env_load();

    return (enum ggml_analog_dispatch) g_dispatch.load(std::memory_order_relaxed);
}

const char * ggml_analog_kernels_dispatch_name(void) {
    switch (ggml_analog_kernels_get_dispatch()) {
        case GGML_ANALOG_DISPATCH_SHADER: return "shader mirror";
        case GGML_ANALOG_DISPATCH_VULKAN: return "vulkan";
        default:                          return "cpu kernels";
    }
}

void ggml_analog_kernels_set_n_threads(int n_threads) {
    analog_kernels_env_load();

    g_n_threads.store(std::max(1, n_threads), std::memory_order_relaxed);
}

int ggml_analog_kernels_get_n_threads(void) {
    return g_n_threads.load(std::memory_order_relaxed);
}

// the value of GGML_ANALOG_N_THREADS, 0 when it was not set
int ggml_analog_kernels_get_env_n_threads(void) {
    analog_kernels_env_load();

    return g_env_n_threads.load(std::memory_order_relaxed);
}

void ggml_analog_kernels_set_enabled(bool enabled) {
    analog_kernels_env_load();

    g_enabled.store(enabled, std::memory_order_relaxed);
}

bool ggml_analog_kernels_get_enabled(void) {
    analog_kernels_env_load();

    return g_enabled.load(std::memory_order_relaxed);
}

const char * ggml_analog_kernels_name(void) {
    static const std::string name = []() {
        std::string s;

#if defined(__AVX2__)
        s += "avx2";
#elif defined(__ARM_NEON)
        s += "neon";
#else
        s += "scalar";
#endif

#if defined(_OPENMP)
        s += "+openmp";
#else
        s += "+threads";
#endif

        if (ggml_analog_kernels_get_dispatch() != GGML_ANALOG_DISPATCH_CPU) {
            s += ", via " + std::string(ggml_analog_kernels_dispatch_name());
        }

        if (!g_enabled.load(std::memory_order_relaxed)) {
            s += ", disabled";
        }

        return s;
    }();

    return name.c_str();
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static int analog_kernels_n_threads(void) {
    return g_enabled.load(std::memory_order_relaxed) ? g_n_threads.load(std::memory_order_relaxed) : 1;
}

// the passes of the array model, run either by the CPU kernels of this file or by the compute
// shaders; the shader path has the same signature and the same result
static bool analog_kernels_use_shaders(void) {
    return ggml_analog_kernels_get_dispatch() != GGML_ANALOG_DISPATCH_CPU;
}

static inline uint64_t analog_hash(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30))*0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27))*0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// unit variance normal sample of a cell, the same four fold hash that the reference uses
static inline float analog_gauss(uint64_t h) {
    uint32_t u = 0;
    for (int i = 0; i < 4; ++i) {
        h = analog_hash(h);
        u += (uint32_t) (h >> 40);
    }

    const float s = (float) u/(float) (1u << 24);

    return (s - 2.0f)*1.7320508f;
}

static inline uint32_t analog_noise_index(int64_t row, int64_t col) {
    return (((uint32_t) row)*0x9E3779B1u + (uint32_t) (col + 1)*0x85EBCA6Bu) & (GGML_ANALOG_NOISE_TABLE - 1);
}

// ---------------------------------------------------------------------------
// vector helpers
// ---------------------------------------------------------------------------

#if defined(__AVX2__)

// round half away from zero, the same result as std::round for every finite input
static inline __m256 analog_round_ps(__m256 v) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 half = _mm256_set1_ps(0.5f);

    const __m256 a = _mm256_andnot_ps(sign, v);
    const __m256 f = _mm256_floor_ps(_mm256_add_ps(a, half));

    return _mm256_or_ps(f, _mm256_and_ps(v, sign));
}

static inline float analog_hmax_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);

    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));

    return _mm_cvtss_f32(lo);
}

#endif

// ---------------------------------------------------------------------------
// input DAC
// ---------------------------------------------------------------------------

static void analog_dac_rows(
        const float * xc,
              float * qc,
        int64_t k,
        float step, float inv_step,
        bool vectorize) {
    for (int64_t k0 = 0; k0 < k; k0 += ANALOG_BLOCK) {
        const int64_t kn = std::min<int64_t>(ANALOG_BLOCK, k - k0);

        // the full scale of the block, a max is exact in any order
        float d = 0.0f;

        int64_t i = 0;

#if defined(__AVX2__)
        if (vectorize) {
            const __m256 abs_mask = _mm256_set1_ps(-0.0f);

            for (; i + 8 <= kn; i += 8) {
                d = std::max(d, analog_hmax_ps(_mm256_andnot_ps(abs_mask, _mm256_loadu_ps(xc + k0 + i))));
            }
        }
#else
        GGML_UNUSED(vectorize);
#endif

        for (; i < kn; ++i) {
            d = std::max(d, std::fabs(xc[k0 + i]));
        }

        if (d == 0.0f) {
            for (int64_t i = 0; i < kn; ++i) {
                qc[k0 + i] = 0.0f;
            }
            continue;
        }

        i = 0;

#if defined(__AVX2__)
        if (vectorize) {
            const __m256 vd        = _mm256_set1_ps(d);
            const __m256 vstep     = _mm256_set1_ps(step);
            const __m256 vinv_step = _mm256_set1_ps(inv_step);
            const __m256 vone      = _mm256_set1_ps(1.0f);
            const __m256 vminus    = _mm256_set1_ps(-1.0f);

            for (; i + 8 <= kn; i += 8) {
                __m256 v = _mm256_div_ps(_mm256_loadu_ps(xc + k0 + i), vd);
                v = _mm256_max_ps(_mm256_min_ps(v, vone), vminus);
                v = _mm256_mul_ps(analog_round_ps(_mm256_mul_ps(v, vinv_step)), vstep);

                _mm256_storeu_ps(qc + k0 + i, _mm256_mul_ps(v, vd));
            }
        }
#endif

        for (; i < kn; ++i) {
            const float v = std::min(std::max(xc[k0 + i]/d, -1.0f), 1.0f);
            qc[k0 + i] = std::round(v*inv_step)*step*d;
        }
    }
}

void ggml_analog_ref_dac(
        const float * src,
              float * dst,
        int64_t k, int64_t ncols,
        const ggml_analog_config & cfg) {
    const int   n_levels = 1 << cfg.bits_x;
    const float step     = 2.0f/(n_levels - 1);
    const float inv_step = 1.0f/step;

    for (int64_t c = 0; c < ncols; ++c) {
        analog_dac_rows(src + c*k, dst + c*k, k, step, inv_step, false);
    }
}

void ggml_analog_kernel_dac(
        const float * src,
              float * dst,
        int64_t k, int64_t ncols,
        const ggml_analog_config & cfg) {
    if (!g_enabled.load(std::memory_order_relaxed)) {
        ggml_analog_ref_dac(src, dst, k, ncols, cfg);
        return;
    }

    if (analog_kernels_use_shaders()) {
        ggml_analog_shader_dac(src, dst, k, ncols, cfg);
        return;
    }

    const int   n_levels = 1 << cfg.bits_x;
    const float step     = 2.0f/(n_levels - 1);
    const float inv_step = 1.0f/step;

    // the columns are independent, whether a column is vectorized or scalar does not matter
    ggml_analog_parallel(ncols, analog_kernels_n_threads(), [&](int64_t c0, int64_t c1) {
        for (int64_t c = c0; c < c1; ++c) {
            analog_dac_rows(src + c*k, dst + c*k, k, step, inv_step, true);
        }
    });
}

// ---------------------------------------------------------------------------
// readout
// ---------------------------------------------------------------------------

static void analog_readout_impl(
        const float * x,
              float * y,
        const float * row_norm,
        const float * row_inv_norm,
        int64_t k, int64_t m,
        int64_t n1, int64_t n2, int64_t n3,
        int64_t ne2, int64_t ne3,
        const ggml_analog_config & cfg,
        int n_threads) {
    const int64_t ncols = n1*n2*n3;

    // the reduction dimension is split over the columns of the array, the partial sums of the
    // individual passes are accumulated by the core and averaged down here
    const int64_t n_passes = (k + cfg.tile_cols - 1)/cfg.tile_cols;

    const float sigma     = cfg.sigma_therm/std::sqrt((float) n_passes);
    const float adc_step  = 2.0f/(1 << cfg.bits_adc);
    const float adc_rstep = 1.0f/adc_step;

    const bool    has_noise = sigma > 0.0f;
    const float * const noise = ggml_analog_kernel_noise_table();

    // the dynamic range of the readout is set per row and per input vector, the same kind of
    // scaling that the per-block quantization of the formats implies
    std::vector<float> x_rms(ncols);

    ggml_analog_parallel(ncols, n_threads, [&](int64_t c0, int64_t c1) {
        for (int64_t c = c0; c < c1; ++c) {
            const float * xc = x + c*k;

            float sq = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sq += xc[i]*xc[i];
            }

            x_rms[c] = std::sqrt(sq/k);
        }
    });

    // broadcast factors of the weight matrix over the input planes
    const int64_t r2 = n2/ne2;
    const int64_t r3 = n3/ne3;

    ggml_analog_parallel(ncols, n_threads, [&](int64_t c0, int64_t c1) {
        for (int64_t c = c0; c < c1; ++c) {
            const int64_t i2 = (c/n1) % n2;
            const int64_t i3 = (c/n1)/n2;

            const int64_t row0 = ((i3/r3)*ne2 + i2/r2)*m;

            const float scale     = cfg.adc_sigma*x_rms[c];
            const float scale_inv = scale > 0.0f ? 1.0f/scale : 0.0f;

            float * dc = y + c*m;

            int64_t i0 = 0;

#if defined(__AVX2__)
            const __m256 vsigma     = _mm256_set1_ps(sigma);
            const __m256 vscale_inv = _mm256_set1_ps(scale_inv);
            const __m256 vadc_rstep = _mm256_set1_ps(adc_rstep);
            const __m256 vadc_step  = _mm256_set1_ps(adc_step);
            const __m256 vzero      = _mm256_setzero_ps();

            const __m256i vcol = _mm256_set1_epi32((int) ((uint32_t) (c + 1)*0x85EBCA6Bu));
            const __m256i vk1  = _mm256_set1_epi32((int) 0x9E3779B1u);
            const __m256i vmsk = _mm256_set1_epi32(GGML_ANALOG_NOISE_TABLE - 1);

            for (; i0 + 8 <= m; i0 += 8) {
                const __m256 rn = _mm256_loadu_ps(row_norm     + row0 + i0);
                const __m256 ri = _mm256_loadu_ps(row_inv_norm + row0 + i0);

                const __m256 full_scale = _mm256_mul_ps(_mm256_set1_ps(scale), rn);

                __m256 v = _mm256_loadu_ps(dc + i0);

                if (has_noise) {
                    // one noise sample per (row, column) pair, from the shared table
                    const __m256i rows = _mm256_setr_epi32(
                            (int) (row0 + i0 + 0), (int) (row0 + i0 + 1), (int) (row0 + i0 + 2), (int) (row0 + i0 + 3),
                            (int) (row0 + i0 + 4), (int) (row0 + i0 + 5), (int) (row0 + i0 + 6), (int) (row0 + i0 + 7));

                    const __m256i idx = _mm256_and_si256(_mm256_add_epi32(_mm256_mullo_epi32(rows, vk1), vcol), vmsk);

                    const __m256 nz = _mm256_i32gather_ps(noise, idx, 4);

                    v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_mul_ps(vsigma, full_scale), nz));
                }

                // saturating ADC over the full scale of the readout
                v = _mm256_min_ps(_mm256_max_ps(v, _mm256_sub_ps(vzero, full_scale)), full_scale);

                const __m256 norm = _mm256_mul_ps(_mm256_mul_ps(v, vscale_inv), ri);

                v = _mm256_mul_ps(_mm256_mul_ps(analog_round_ps(_mm256_mul_ps(norm, vadc_rstep)), vadc_step), full_scale);

                // rows whose full scale is zero read out as an exact zero
                const __m256 ok = _mm256_cmp_ps(full_scale, vzero, _CMP_GT_OQ);

                _mm256_storeu_ps(dc + i0, _mm256_blendv_ps(vzero, v, ok));
            }
#endif

            for (; i0 < m; ++i0) {
                const float full_scale = scale*row_norm[row0 + i0];

                if (full_scale <= 0.0f) {
                    dc[i0] = 0.0f;
                    continue;
                }

                float v = dc[i0];

                if (has_noise) {
                    v += sigma*full_scale*noise[analog_noise_index(row0 + i0, c)];
                }

                v = std::min(std::max(v, -full_scale), full_scale);

                const float norm = v*scale_inv*row_inv_norm[row0 + i0];

                dc[i0] = std::round(norm*adc_rstep)*adc_step*full_scale;
            }
        }
    });
}

void ggml_analog_ref_readout(
        const float * x,
              float * y,
        const float * row_norm,
        const float * row_inv_norm,
        int64_t k, int64_t m,
        int64_t n1, int64_t n2, int64_t n3,
        int64_t ne2, int64_t ne3,
        const ggml_analog_config & cfg) {
    analog_readout_impl(x, y, row_norm, row_inv_norm, k, m, n1, n2, n3, ne2, ne3, cfg, 1);
}

void ggml_analog_kernel_readout(
        const float * x,
              float * y,
        const float * row_norm,
        const float * row_inv_norm,
        int64_t k, int64_t m,
        int64_t n1, int64_t n2, int64_t n3,
        int64_t ne2, int64_t ne3,
        const ggml_analog_config & cfg) {
    if (!g_enabled.load(std::memory_order_relaxed)) {
        ggml_analog_ref_readout(x, y, row_norm, row_inv_norm, k, m, n1, n2, n3, ne2, ne3, cfg);
        return;
    }

    if (analog_kernels_use_shaders()) {
        ggml_analog_shader_readout(x, y, row_norm, row_inv_norm, k, m, n1, n2, n3, ne2, ne3, cfg);
        return;
    }

    analog_readout_impl(x, y, row_norm, row_inv_norm, k, m, n1, n2, n3, ne2, ne3, cfg, analog_kernels_n_threads());
}

// ---------------------------------------------------------------------------
// programming
// ---------------------------------------------------------------------------

// the two row conversions of the container types, the kernels only deal with float rows
void ggml_analog_dequant_row(const struct ggml_tensor * w, const void * src, float * dst, int64_t k) {
    if (w->type == GGML_TYPE_F32) {
        memcpy(dst, src, k*sizeof(float));
    } else if (w->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, k);
    } else {
        ggml_get_type_traits(w->type)->to_float(src, dst, k);
    }
}

void ggml_analog_quant_row(const struct ggml_tensor * w, void * dst, const float * src, int64_t k) {
    if (w->type == GGML_TYPE_F32) {
        memcpy(dst, src, k*sizeof(float));
    } else if (w->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) dst, k);
    } else {
        ggml_quantize_chunk(w->type, src, dst, 0, 1, k, nullptr);
    }
}

static void analog_program_rows(
        const struct ggml_tensor * w,
              struct ggml_tensor * snap,
        const ggml_analog_config & cfg,
              float * row_norm,
              float * row_inv_norm,
        int64_t r0, int64_t r1) {
    const int64_t k     = w->ne[0];
    const size_t  row_size_src = ggml_row_size(w->type,  k);
    const size_t  row_size_dst = ggml_row_size(snap->type, k);

    const int   n_levels  = 1 << cfg.bits_w;
    const float step      = 2.0f/(n_levels - 1);
    const float inv_step  = 1.0f/step;

    // hoisted: the inner loop of the programming pass runs over every cell of the array
    const float sigma_prog = cfg.sigma_prog;
    const float drift      = 1.0f - cfg.drift;
    const float ir_drop    = cfg.ir_drop;
    const int64_t tile_cols = cfg.tile_cols;

    std::vector<float> buf(k);

    for (int64_t r = r0; r < r1; ++r) {
        const char * src_row = (const char *) w->data    + r*row_size_src;
              char * dst_row = (      char *) snap->data + r*row_size_dst;

        ggml_analog_dequant_row(w, src_row, buf.data(), k);

        const uint64_t row_hash = analog_hash(analog_hash(cfg.seed) ^ (uint64_t) (r + 1)*0x2545F4914F6CDD1Dull);

        float sq_sum = 0.0f;

        for (int64_t k0 = 0; k0 < k; k0 += ANALOG_BLOCK) {
            const int64_t kn = std::min<int64_t>(ANALOG_BLOCK, k - k0);

            // full scale of the block, the same scaling that the quantization formats use
            float d = 0.0f;

#if defined(__AVX2__)
            {
                const __m256 abs_mask = _mm256_set1_ps(-0.0f);

                int64_t i = 0;
                for (; i + 8 <= kn; i += 8) {
                    d = std::max(d, analog_hmax_ps(_mm256_andnot_ps(abs_mask, _mm256_loadu_ps(buf.data() + k0 + i))));
                }
                for (; i < kn; ++i) {
                    d = std::max(d, std::fabs(buf[k0 + i]));
                }
            }
#else
            for (int64_t i = 0; i < kn; ++i) {
                d = std::max(d, std::fabs(buf[k0 + i]));
            }
#endif

            if (d == 0.0f) {
                continue;
            }

            const float inv_d = 1.0f/d;

            // the position of the cell along the column of the tile, advanced instead of divided
            int64_t col = k0 % cfg.tile_cols;

            for (int64_t i = 0; i < kn; ++i) {
                const uint64_t h = analog_hash(row_hash ^ (uint64_t) (k0 + i)*0x9E3779B97F4A7C15ull);

                float v = buf[k0 + i]*inv_d; // normalized cell conductance
                v = std::min(std::max(v, -1.0f), 1.0f);

                // finite number of conductance levels
                v = std::round(v*inv_step)*step;

                // programming error of the write operation
                if (sigma_prog > 0.0f) {
                    v += sigma_prog*analog_gauss(h);
                }

                // conductance drift, applied once at the end of programming
                if (drift > 0.0f) {
                    v *= drift;
                }

                // IR drop: the drive voltage decays along the column of the array
                if (ir_drop > 0.0f) {
                    v *= 1.0f - ir_drop*(float) col/(float) tile_cols;
                }

                if (++col == tile_cols) {
                    col = 0;
                }

                buf[k0 + i] = v*d;
                sq_sum += (v*d)*(v*d);
            }
        }

        row_norm[r]     = std::sqrt(sq_sum);
        row_inv_norm[r] = row_norm[r] > 0.0f ? 1.0f/row_norm[r] : 0.0f;

        ggml_analog_quant_row(snap, dst_row, buf.data(), k);
    }
}

void ggml_analog_ref_program(
        const struct ggml_tensor * w,
              struct ggml_tensor * snap,
        const ggml_analog_config & cfg,
              float * row_norm,
              float * row_inv_norm) {
    analog_program_rows(w, snap, cfg, row_norm, row_inv_norm, 0, ggml_nelements(w)/w->ne[0]);
}

void ggml_analog_kernel_program(
        const struct ggml_tensor * w,
              struct ggml_tensor * snap,
        const ggml_analog_config & cfg,
              float * row_norm,
              float * row_inv_norm) {
    if (!g_enabled.load(std::memory_order_relaxed)) {
        ggml_analog_ref_program(w, snap, cfg, row_norm, row_inv_norm);
        return;
    }

    if (analog_kernels_use_shaders()) {
        ggml_analog_shader_program(w, snap, cfg, row_norm, row_inv_norm);
        return;
    }

    // the rows are independent, the order inside a row is unchanged
    ggml_analog_parallel(ggml_nelements(w)/w->ne[0], analog_kernels_n_threads(), [&](int64_t r0, int64_t r1) {
        analog_program_rows(w, snap, cfg, row_norm, row_inv_norm, r0, r1);
    });
}
