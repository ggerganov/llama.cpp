// Execution of the array kernels through the compute shaders of vulkan/.
//
// Every kernel of the shaders has a line by line mirror in this file: the same index math, the same
// random numbers, the same order of the floating point operations, only with the work items of a
// dispatch turned into a loop. That makes the mirror the reference of the shaders on a machine
// without a Vulkan device, and it is what tests/test-analog-kernels.cpp compares against the CPU
// kernels of analog-kernels.cpp.

#include "analog-shader.h"
#include "analog-kernels.h"
#include "analog-parallel.h"
#include "ggml-impl.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(GGML_ANALOG_VULKAN)
#include "analog-vulkan.h"
#endif

#define ANALOG_BLOCK 32

#define ANALOG_NOISE_TABLE 8192
#define ANALOG_NOISE_A 0x9E3779B1u
#define ANALOG_NOISE_B 0x85EBCA6Bu

static std::atomic<bool> g_prefer_vulkan{false};

// ---------------------------------------------------------------------------
// random numbers, identical to the ones of the shaders and of the CPU kernels
// ---------------------------------------------------------------------------

static inline uint64_t analog_hash(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30))*0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27))*0x94D049BB133111EBull;

    return x ^ (x >> 31);
}

static inline float analog_gauss(uint64_t h) {
    uint32_t u = 0;

    for (int i = 0; i < 4; ++i) {
        h = analog_hash(h);
        u += (uint32_t) (h >> 40);
    }

    const float s = (float) u/(float) (1u << 24);

    return (s - 2.0f)*1.7320508f;
}

// the value of the readout noise table for an index, computed the same way as the CPU table
static inline float analog_noise_at(uint32_t idx) {
    const uint64_t h = analog_hash((uint64_t) idx*0x9E3779B97F4A7C15ull + 12345u);

    const float u = (float)  (h        & 0xFFFFFF)/(float) (1u << 24);
    const float v = (float) ((h >> 24) & 0xFFFFFF)/(float) (1u << 24);

    return 2.4494898f*(u + v - 1.0f);
}

static inline float analog_noise_of(uint32_t row, uint32_t col) {
    const uint32_t idx = (row*ANALOG_NOISE_A + (col + 1u)*ANALOG_NOISE_B) & (ANALOG_NOISE_TABLE - 1u);

    return analog_noise_at(idx);
}

// ---------------------------------------------------------------------------
// mirror of the shaders, see vulkan/*.comp for the definition of the kernels
// ---------------------------------------------------------------------------

// analog_dac.comp
static void analog_mirror_dac(
        const float * x,
              float * q,
        uint32_t k, uint32_t ncols,
        float step, float inv_step) {
    const uint32_t nb = (k + ANALOG_BLOCK - 1)/ANALOG_BLOCK;

    ggml_analog_parallel((int64_t) nb*ncols, ggml_analog_kernels_get_n_threads(), [&](int64_t b0, int64_t b1) {
        for (int64_t blk = b0; blk < b1; ++blk) {
            const uint32_t col = (uint32_t) blk/nb;
            const uint32_t b   = (uint32_t) blk%nb;

            const uint32_t k0 = b*ANALOG_BLOCK;
            const uint32_t kn = k0 + ANALOG_BLOCK <= k ? ANALOG_BLOCK : k - k0;

            const uint32_t base = col*k + k0;

            // the full scale of the block, a max is exact in any order
            float d = 0.0f;
            for (uint32_t i = 0; i < kn; ++i) {
                d = std::max(d, std::fabs(x[base + i]));
            }

            for (uint32_t i = 0; i < kn; ++i) {
                if (d == 0.0f) {
                    q[base + i] = 0.0f;
                    continue;
                }

                const float v = std::min(std::max(x[base + i]/d, -1.0f), 1.0f);

                q[base + i] = std::round(v*inv_step)*step*d;
            }
        }
    });
}

// analog_program.comp
static void analog_mirror_program(
        const float * w,
              float * snap,
        uint32_t k, uint32_t nrows,
        const ggml_analog_config & cfg,
              float * row_norm,
              float * row_inv_norm,
        float step, float inv_step) {
    const uint64_t seed = cfg.seed;

    ggml_analog_parallel(nrows, ggml_analog_kernels_get_n_threads(), [&](int64_t r0, int64_t r1) {
        for (int64_t rr = r0; rr < r1; ++rr) {
            const uint32_t r = (uint32_t) rr;

            const uint64_t row_hash = analog_hash(analog_hash(seed) ^ ((uint64_t) r + 1)*0x2545F4914F6CDD1Dull);

            const uint32_t base_row = r*k;

            float sq_sum = 0.0f;

            for (uint32_t k0 = 0; k0 < k; k0 += ANALOG_BLOCK) {
                const uint32_t kn = k0 + ANALOG_BLOCK <= k ? ANALOG_BLOCK : k - k0;
                const uint32_t base = base_row + k0;

                float d = 0.0f;
                for (uint32_t i = 0; i < kn; ++i) {
                    d = std::max(d, std::fabs(w[base + i]));
                }

                if (d == 0.0f) {
                    // an all-zero block is not programmed, the cell keeps the value it was read with
                    for (uint32_t i = 0; i < kn; ++i) {
                        snap[base + i] = w[base + i];
                    }
                    continue;
                }

                const float inv_d = 1.0f/d;

                for (uint32_t i = 0; i < kn; ++i) {
                    const uint64_t h = analog_hash(row_hash ^ (uint64_t) (k0 + i)*0x9E3779B97F4A7C15ull);

                    float v = w[base + i]*inv_d; // normalized cell conductance
                    v = std::min(std::max(v, -1.0f), 1.0f);

                    // finite number of conductance levels
                    v = std::round(v*inv_step)*step;

                    // programming error of the write operation
                    if (cfg.sigma_prog > 0.0f) {
                        v += cfg.sigma_prog*analog_gauss(h);
                    }

                    // conductance drift, applied once at the end of programming
                    if (cfg.drift > 0.0f) {
                        v *= 1.0f - cfg.drift;
                    }

                    // IR drop: the drive voltage decays along the column of the array
                    if (cfg.ir_drop > 0.0f) {
                        const uint32_t col = (k0 + i) % (uint32_t) cfg.tile_cols;
                        v *= 1.0f - cfg.ir_drop*(float) col/(float) cfg.tile_cols;
                    }

                    const float sv = v*d;

                    snap[base + i] = sv;
                    sq_sum += sv*sv;
                }
            }

            const float norm = std::sqrt(sq_sum);

            row_norm[r]     = norm;
            row_inv_norm[r] = norm > 0.0f ? 1.0f/norm : 0.0f;
        }
    });
}

// analog_xrms.comp
static void analog_mirror_xrms(
        const float * x,
              float * x_rms,
        uint32_t k, uint32_t ncols) {
    ggml_analog_parallel(ncols, ggml_analog_kernels_get_n_threads(), [&](int64_t c0, int64_t c1) {
        for (int64_t c = c0; c < c1; ++c) {
            const uint32_t base = (uint32_t) c*k;

            float sq = 0.0f;
            for (uint32_t i = 0; i < k; ++i) {
                sq += x[base + i]*x[base + i];
            }

            x_rms[c] = std::sqrt(sq/(float) k);
        }
    });
}

// analog_readout.comp
static void analog_mirror_readout(
              float * y,
        const float * x_rms,
        const float * row_norm,
        const float * row_inv_norm,
        uint32_t m,
        uint32_t n1, uint32_t n2, uint32_t n3,
        uint32_t ne2, uint32_t ne3,
        float adc_sigma, float sigma, float adc_step, float adc_rstep) {
    const uint32_t ncols = n1*n2*n3;

    ggml_analog_parallel(ncols, ggml_analog_kernels_get_n_threads(), [&](int64_t c0, int64_t c1) {
        for (int64_t cc = c0; cc < c1; ++cc) {
            const uint32_t c = (uint32_t) cc;

            const uint32_t i2 = (c/n1)%n2;
            const uint32_t i3 = (c/n1)/n2;

            const uint32_t r2 = n2/ne2;
            const uint32_t r3 = n3/ne3;

            const uint32_t row0 = ((i3/r3)*ne2 + i2/r2)*m;

            const float scale     = adc_sigma*x_rms[c];
            const float scale_inv = scale > 0.0f ? 1.0f/scale : 0.0f;

            const uint32_t base = c*m;

            for (uint32_t i = 0; i < m; ++i) {
                const uint32_t idx = base + i;

                const float full_scale = scale*row_norm[row0 + i];

                if (full_scale <= 0.0f) {
                    y[idx] = 0.0f;
                    continue;
                }

                float v = y[idx];

                // thermal / shot noise of the readout, relative to the full scale of an array pass
                if (sigma > 0.0f) {
                    v += sigma*full_scale*analog_noise_of(row0 + i, c);
                }

                // saturating ADC over the full scale of the readout
                v = std::min(std::max(v, -full_scale), full_scale);

                const float norm = v*scale_inv*row_inv_norm[row0 + i];

                y[idx] = std::round(norm*adc_rstep)*adc_step*full_scale;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

void ggml_analog_shader_set_prefer_vulkan(bool prefer) {
    g_prefer_vulkan.store(prefer, std::memory_order_relaxed);
}

// whether the GLSL compiler was part of the build, the mirror does not need the modules
bool ggml_analog_shader_available(void) {
    return ggml_analog_shader_spirv_count() > 0;
}

// the Vulkan device, when the build has the dispatch and one can be initialized
static bool analog_shader_vulkan_ready(void) {
#if defined(GGML_ANALOG_VULKAN)
    return g_prefer_vulkan.load(std::memory_order_relaxed) && ggml_analog_vulkan_available();
#else
    return false;
#endif
}

const char * ggml_analog_shader_status(void) {
#if defined(GGML_ANALOG_VULKAN)
    if (analog_shader_vulkan_ready()) {
        return ggml_analog_vulkan_status();
    }
#endif

    if (g_prefer_vulkan.load(std::memory_order_relaxed)) {
        return "mirror (no Vulkan device)";
    }

    return "mirror";
}

// the number of threads of the mirror is the one of the kernels, the dispatches are parallel in the
// same way a workgroup grid is
void ggml_analog_shader_dac(
        const float * src,
              float * dst,
        int64_t k, int64_t ncols,
        const ggml_analog_config & cfg) {
    const int   n_levels = 1 << cfg.bits_x;
    const float step     = 2.0f/(n_levels - 1);
    const float inv_step = 1.0f/step;

    GGML_ASSERT(k > 0 && k <= (int64_t) 0xFFFFFFFF);
    GGML_ASSERT(ncols > 0 && (int64_t) ((uint32_t) ncols)*k <= (int64_t) 0xFFFFFFFF);

#if defined(GGML_ANALOG_VULKAN)
    if (analog_shader_vulkan_ready() &&
        ggml_analog_vulkan_dac(src, dst, (uint32_t) k, (uint32_t) ncols, step, inv_step)) {
        return;
    }
#endif

    analog_mirror_dac(src, dst, (uint32_t) k, (uint32_t) ncols, step, inv_step);
}

void ggml_analog_shader_readout(
        const float * x,
              float * y,
        const float * row_norm,
        const float * row_inv_norm,
        int64_t k, int64_t m,
        int64_t n1, int64_t n2, int64_t n3,
        int64_t ne2, int64_t ne3,
        const ggml_analog_config & cfg) {
    const int64_t n_passes = (k + cfg.tile_cols - 1)/cfg.tile_cols;

    const float sigma     = cfg.sigma_therm/std::sqrt((float) n_passes);
    const float adc_step  = 2.0f/(1 << cfg.bits_adc);
    const float adc_rstep = 1.0f/adc_step;

    const int64_t ncols = n1*n2*n3;

    GGML_ASSERT(k > 0 && k <= (int64_t) 0xFFFFFFFF);
    GGML_ASSERT(m > 0 && m <= (int64_t) 0xFFFFFFFF);
    GGML_ASSERT(ncols > 0 && (int64_t) ((uint32_t) ncols)*m <= (int64_t) 0xFFFFFFFF);

#if defined(GGML_ANALOG_VULKAN)
    if (analog_shader_vulkan_ready() &&
        ggml_analog_vulkan_readout(x, y, row_norm, row_inv_norm, (uint32_t) k, (uint32_t) m,
                (uint32_t) n1, (uint32_t) n2, (uint32_t) n3, (uint32_t) ne2, (uint32_t) ne3,
                cfg.adc_sigma, sigma, adc_step, adc_rstep)) {
        return;
    }
#endif

    // the dynamic range of the readout of every column, the buffer is per call and per thread
    static thread_local std::vector<float> x_rms;

    x_rms.resize(ncols);

    analog_mirror_xrms(x, x_rms.data(), (uint32_t) k, (uint32_t) ncols);

    analog_mirror_readout(y, x_rms.data(), row_norm, row_inv_norm, (uint32_t) m,
            (uint32_t) n1, (uint32_t) n2, (uint32_t) n3, (uint32_t) ne2, (uint32_t) ne3,
            cfg.adc_sigma, sigma, adc_step, adc_rstep);
}

void ggml_analog_shader_program(
        const struct ggml_tensor * w,
              struct ggml_tensor * snap,
        const ggml_analog_config & cfg,
              float * row_norm,
              float * row_inv_norm) {
    const int64_t k     = w->ne[0];
    const int64_t nrows = ggml_nelements(w)/k;

    const int   n_levels = 1 << cfg.bits_w;
    const float step     = 2.0f/(n_levels - 1);
    const float inv_step = 1.0f/step;

    GGML_ASSERT(k > 0 && k <= (int64_t) 0xFFFFFFFF);
    GGML_ASSERT(nrows > 0 && (int64_t) ((uint32_t) nrows)*k <= (int64_t) 0xFFFFFFFF);

    // the shaders program F32 rows, the conversion of the container types happens here
    std::vector<float> wf((size_t) k*nrows);
    std::vector<float> sf((size_t) k*nrows);

    for (int64_t r = 0; r < nrows; ++r) {
        ggml_analog_dequant_row(w, (const char *) w->data + r*ggml_row_size(w->type, k), wf.data() + r*k, k);
    }

    bool done = false;

#if defined(GGML_ANALOG_VULKAN)
    if (analog_shader_vulkan_ready()) {
        done = ggml_analog_vulkan_program(wf.data(), sf.data(), (uint32_t) k, (uint32_t) nrows, cfg,
                row_norm, row_inv_norm, step, inv_step);
    }
#endif

    if (!done) {
        analog_mirror_program(wf.data(), sf.data(), (uint32_t) k, (uint32_t) nrows, cfg, row_norm, row_inv_norm, step, inv_step);
    }

    for (int64_t r = 0; r < nrows; ++r) {
        ggml_analog_quant_row(snap, (char *) snap->data + r*ggml_row_size(snap->type, k), sf.data() + r*k, k);
    }
}

// ---------------------------------------------------------------------------
// SPIR-V modules of the shaders
// ---------------------------------------------------------------------------

int ggml_analog_shader_spirv_count(void) {
#if defined(GGML_ANALOG_SPIRV)
    return 4;
#else
    return 0;
#endif
}

#if defined(GGML_ANALOG_SPIRV)
#include "analog-shaders.gen.h"

struct analog_spirv_entry {
    const char * name;
    const uint32_t * words;
    size_t n_words;
};

static const analog_spirv_entry g_spirv[] = {
    { "analog_dac",     g_analog_shader_analog_dac,     sizeof(g_analog_shader_analog_dac)/4     },
    { "analog_program", g_analog_shader_analog_program, sizeof(g_analog_shader_analog_program)/4 },
    { "analog_xrms",    g_analog_shader_analog_xrms,    sizeof(g_analog_shader_analog_xrms)/4    },
    { "analog_readout", g_analog_shader_analog_readout, sizeof(g_analog_shader_analog_readout)/4 },
};
#endif

const char * ggml_analog_shader_spirv_name(int index) {
#if defined(GGML_ANALOG_SPIRV)
    if (index < 0 || index >= (int) (sizeof(g_spirv)/sizeof(g_spirv[0]))) {
        return nullptr;
    }

    return g_spirv[index].name;
#else
    GGML_UNUSED(index);

    return nullptr;
#endif
}

const uint32_t * ggml_analog_shader_spirv_words(int index, size_t * n_words) {
#if defined(GGML_ANALOG_SPIRV)
    if (index < 0 || index >= (int) (sizeof(g_spirv)/sizeof(g_spirv[0]))) {
        return nullptr;
    }

    *n_words = g_spirv[index].n_words;

    return g_spirv[index].words;
#else
    GGML_UNUSED(index);

    *n_words = 0;

    return nullptr;
#endif
}
