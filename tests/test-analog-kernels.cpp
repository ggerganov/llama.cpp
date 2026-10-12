// Test of the parallel kernels and the compute shaders of the analog backend.
//
// The three passes of the array model (input DAC, programming of a tile, readout) are implemented
// three times:
//
//   - analog-kernels.cpp  : the scalar reference and the parallel/vectorized CPU kernels
//   - analog-shader.cpp   : the mirror of the compute shaders of vulkan/
//   - vulkan/*.comp       : the shaders themselves, compiled to the embedded SPIR-V modules
//
// This test checks that the parallel kernels and the shader mirror produce the same values as the
// scalar reference, that the embedded SPIR-V modules are the ones the dispatch expects, and how
// much the passes speed up with threads.

#include "ggml.h"
#include "ggml-analog.h"

#include "analog-array.h"
#include "analog-kernels.h"
#include "analog-shader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define ANALOG_BLOCK 32

static int g_failures = 0;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static uint32_t g_rng = 1234;

static float frand(void) {
    g_rng = g_rng*1664525u + 1013904223u;

    return ((float) (g_rng >> 8)/(float) (1 << 24))*2.0f - 1.0f;
}

static void fill_random(float * data, int64_t n, float scale) {
    for (int64_t i = 0; i < n; ++i) {
        data[i] = scale*frand();
    }
}

static double now_us(void) {
    return (double) ggml_time_us();
}

// median of n_rep runs of fn
template <typename F>
static double time_us(int n_rep, F && fn) {
    std::vector<double> vals;

    for (int i = 0; i < n_rep; ++i) {
        const double t0 = now_us();
        fn();
        vals.push_back(now_us() - t0);
    }

    std::sort(vals.begin(), vals.end());

    return vals[vals.size()/2];
}

struct diff_stats {
    double max_abs = 0.0;
    double rms     = 0.0;
    int64_t n      = 0;
};

static diff_stats compare(const float * a, const float * b, int64_t n) {
    diff_stats s;
    double sum = 0.0;

    for (int64_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);

        s.max_abs = std::max(s.max_abs, d);
        sum += d*d;
    }

    s.rms = n > 0 ? std::sqrt(sum/n) : 0.0;
    s.n   = n;

    return s;
}

static void check(const char * what, const diff_stats & s, bool exact) {
    const bool ok = exact ? s.max_abs == 0.0 : s.max_abs < 1e-12;

    if (!ok) {
        g_failures++;
    }

    printf("  %-46s max abs %.6e  rms %.6e  %s\n", what, s.max_abs, s.rms, ok ? "ok" : "MISMATCH");
}

static struct ggml_tensor * new_tensor_2d(struct ggml_context * ctx, enum ggml_type type, int64_t k, int64_t nrows) {
    return ggml_new_tensor_2d(ctx, type, k, nrows);
}

// ---------------------------------------------------------------------------
// passes
// ---------------------------------------------------------------------------

struct pass_case {
    const char * name;
    enum ggml_type type;
    int64_t k;
    int64_t nrows;
    ggml_analog_config cfg;
};

// run the programming pass with the reference, the CPU kernels and the shader mirror, returns the
// three snapshots
static void run_program_case(const pass_case & tc, struct ggml_context * ctx, int n_threads) {
    struct ggml_tensor * w    = new_tensor_2d(ctx, tc.type, tc.k, tc.nrows);
    struct ggml_tensor * snap = ggml_dup_tensor(ctx, w);

    if (tc.type == GGML_TYPE_F32) {
        fill_random((float *) w->data, tc.k*tc.nrows, 0.7f);
    } else {
        std::vector<float> f(tc.k*tc.nrows);
        fill_random(f.data(), tc.k*tc.nrows, 0.7f);

        for (int64_t r = 0; r < tc.nrows; ++r) {
            ggml_quantize_chunk(tc.type, f.data() + r*tc.k, (char *) w->data + r*ggml_row_size(tc.type, tc.k), 0, 1, tc.k, nullptr);
        }
    }

    const size_t cells = (size_t) tc.k*tc.nrows;

    std::vector<float> ref_snap(cells);
    std::vector<float> ref_norm(tc.nrows);
    std::vector<float> ref_inorm(tc.nrows);

    ggml_analog_kernels_set_enabled(false);
    ggml_analog_kernels_set_n_threads(1);
    ggml_analog_kernel_program(w, snap, tc.cfg, ref_norm.data(), ref_inorm.data());
    ggml_analog_kernels_set_enabled(true);

    // the reference converts the snapshot to the type of the tensor, read it back as float
    {
        std::vector<float> row(tc.k);

        for (int64_t r = 0; r < tc.nrows; ++r) {
            ggml_analog_dequant_row(snap, (const char *) snap->data + r*ggml_row_size(snap->type, tc.k), row.data(), tc.k);
            memcpy(ref_snap.data() + r*tc.k, row.data(), sizeof(float)*tc.k);
        }
    }

    ggml_analog_kernels_set_n_threads(n_threads);

    std::vector<float> ker_snap(cells);
    std::vector<float> ker_norm(tc.nrows);
    std::vector<float> ker_inorm(tc.nrows);

    ggml_analog_kernel_program(w, snap, tc.cfg, ker_norm.data(), ker_inorm.data());

    {
        std::vector<float> row(tc.k);

        for (int64_t r = 0; r < tc.nrows; ++r) {
            ggml_analog_dequant_row(snap, (const char *) snap->data + r*ggml_row_size(snap->type, tc.k), row.data(), tc.k);
            memcpy(ker_snap.data() + r*tc.k, row.data(), sizeof(float)*tc.k);
        }
    }

    std::vector<float> mir_snap(cells);
    std::vector<float> mir_norm(tc.nrows);
    std::vector<float> mir_inorm(tc.nrows);

    ggml_analog_shader_set_prefer_vulkan(false);
    ggml_analog_shader_program(w, snap, tc.cfg, mir_norm.data(), mir_inorm.data());

    {
        std::vector<float> row(tc.k);

        for (int64_t r = 0; r < tc.nrows; ++r) {
            ggml_analog_dequant_row(snap, (const char *) snap->data + r*ggml_row_size(snap->type, tc.k), row.data(), tc.k);
            memcpy(mir_snap.data() + r*tc.k, row.data(), sizeof(float)*tc.k);
        }
    }

    char label[128];

    snprintf(label, sizeof(label), "%s: kernel snapshot", tc.name);
    check(label, compare(ref_snap.data(), ker_snap.data(), cells), true);

    snprintf(label, sizeof(label), "%s: shader snapshot", tc.name);
    check(label, compare(ref_snap.data(), mir_snap.data(), cells), true);

    snprintf(label, sizeof(label), "%s: kernel row norm", tc.name);
    check(label, compare(ref_norm.data(), ker_norm.data(), tc.nrows), true);

    snprintf(label, sizeof(label), "%s: shader row norm", tc.name);
    check(label, compare(ref_norm.data(), mir_norm.data(), tc.nrows), true);
}

static void run_dac_case(const pass_case & tc, struct ggml_context * ctx, int n_threads) {
    GGML_UNUSED(ctx);

    const int64_t k     = tc.k;
    const int64_t ncols = 4;

    const size_t n = (size_t) k*ncols;

    std::vector<float> x(n);
    std::vector<float> ref(n);
    std::vector<float> ker(n);
    std::vector<float> mir(n);

    fill_random(x.data(), n, 1.0f);

    ggml_analog_kernels_set_enabled(false);
    ggml_analog_kernels_set_n_threads(1);
    ggml_analog_kernel_dac(x.data(), ref.data(), k, ncols, tc.cfg);
    ggml_analog_kernels_set_enabled(true);

    ggml_analog_kernels_set_n_threads(n_threads);
    ggml_analog_kernel_dac(x.data(), ker.data(), k, ncols, tc.cfg);

    ggml_analog_shader_set_prefer_vulkan(false);
    ggml_analog_shader_dac(x.data(), mir.data(), k, ncols, tc.cfg);

    char label[128];

    snprintf(label, sizeof(label), "%s: kernel DAC", tc.name);
    check(label, compare(ref.data(), ker.data(), n), true);

    snprintf(label, sizeof(label), "%s: shader DAC", tc.name);
    check(label, compare(ref.data(), mir.data(), n), true);
}

static void run_readout_case(const pass_case & tc, struct ggml_context * ctx, int n_threads) {
    const int64_t k  = tc.k;
    const int64_t m  = tc.nrows;
    const int64_t n1 = 3;
    const int64_t n2 = 2;
    const int64_t n3 = 2;

    const int64_t ncols = n1*n2*n3;

    // the weight matrix has one row per (m, ne2, ne3), the row norms are indexed by the output row
    const int64_t nrows = m*2*2;

    std::vector<float> x(k*ncols);
    std::vector<float> y(m*ncols);
    std::vector<float> norm(nrows);
    std::vector<float> inorm(nrows);

    fill_random(x.data(), k*ncols, 1.0f);
    fill_random(y.data(), m*ncols, 2.0f/sqrt((double) k));
    fill_random(norm.data(), nrows, 1.0f);

    for (int64_t i = 0; i < nrows; ++i) {
        norm[i] = std::fabs(norm[i]) + 0.1f;
        inorm[i] = 1.0f/norm[i];
    }

    std::vector<float> ref(y);
    std::vector<float> ker(y);
    std::vector<float> mir(y);

    GGML_UNUSED(ctx);

    ggml_analog_kernels_set_enabled(false);
    ggml_analog_kernels_set_n_threads(1);
    ggml_analog_kernel_readout(x.data(), ref.data(), norm.data(), inorm.data(), k, m, n1, n2, n3, 2, 2, tc.cfg);
    ggml_analog_kernels_set_enabled(true);

    ggml_analog_kernels_set_n_threads(n_threads);
    ggml_analog_kernel_readout(x.data(), ker.data(), norm.data(), inorm.data(), k, m, n1, n2, n3, 2, 2, tc.cfg);

    ggml_analog_shader_set_prefer_vulkan(false);
    ggml_analog_shader_readout(x.data(), mir.data(), norm.data(), inorm.data(), k, m, n1, n2, n3, 2, 2, tc.cfg);

    char label[128];

    snprintf(label, sizeof(label), "%s: kernel readout", tc.name);
    check(label, compare(ref.data(), ker.data(), m*ncols), true);

    snprintf(label, sizeof(label), "%s: shader readout", tc.name);
    check(label, compare(ref.data(), mir.data(), m*ncols), true);
}

// ---------------------------------------------------------------------------
// the embedded SPIR-V modules
// ---------------------------------------------------------------------------

// capabilities of the SPIR-V specification that the shaders are expected to declare
#define SPIRV_CAPABILITY_SHADER 1
#define SPIRV_CAPABILITY_INT64  11

#define SPIRV_OP_ENTRY_POINT 15
#define SPIRV_OP_CAPABILITY  17

#define SPIRV_EXECUTION_MODEL_GL_COMPUTE 5

static bool spirv_check(int index, bool expect_int64) {
    size_t n_words = 0;

    const uint32_t * words = ggml_analog_shader_spirv_words(index, &n_words);
    const char * name = ggml_analog_shader_spirv_name(index);

    if (words == nullptr || n_words < 5) {
        printf("  %-20s no module\n", name ? name : "?");

        return false;
    }

    if (words[0] != 0x07230203) {
        printf("  %-20s bad magic 0x%08x\n", name, words[0]);

        return false;
    }

    const uint32_t version = words[1];
    const uint32_t bound   = words[3];

    bool has_int64      = false;
    bool has_entry_main = false;

    std::string caps;

    for (size_t i = 5; i < n_words; ) {
        const uint32_t word    = words[i];
        const uint32_t opcode  = word & 0xFFFF;
        const uint32_t n_operands = word >> 16;

        if (n_operands == 0 || i + n_operands > n_words) {
            break;
        }

        if (opcode == SPIRV_OP_CAPABILITY) {
            const uint32_t capability = words[i + 1];

            if (capability == SPIRV_CAPABILITY_INT64) {
                has_int64 = true;
            }

            if (!caps.empty()) {
                caps += ", ";
            }

            caps += std::to_string(capability);
        } else if (opcode == SPIRV_OP_ENTRY_POINT) {
            const uint32_t model = words[i + 1];

            const char * entry = (const char *) &words[i + 3];

            if (model == SPIRV_EXECUTION_MODEL_GL_COMPUTE && strcmp(entry, "main") == 0) {
                has_entry_main = true;
            }
        }

        i += n_operands;
    }

    const bool ok = (!expect_int64 || has_int64) && has_entry_main && bound > 0 && version >= 0x00010300;

    printf("  %-20s %7zu words, SPIR-V %u.%u, bound %u, capabilities [%s]%s  %s\n",
            name, n_words, (version >> 16) & 0xFF, (version >> 8) & 0xFF, bound, caps.c_str(),
            has_entry_main ? ", entry point main (GLCompute)" : "", ok ? "ok" : "MISMATCH");

    // the shaders carry the constants of the device model: if one of them is not in the module, the
    // shader and the C++ model of the array (analog-kernels.cpp) have drifted apart
    const struct {
        uint32_t value;
        const char * what;
    } model_constants[] = {
        { 0x9E3779B9u, "splitmix 0x9E3779B97F4A7C15 (hi)"  },
        { 0x7F4A7C15u, "splitmix 0x9E3779B97F4A7C15 (lo)"  },
        { 0xBF58476Du, "splitmix 0xBF58476D1CE4E5B9 (hi)"  },
        { 0x1CE4E5B9u, "splitmix 0xBF58476D1CE4E5B9 (lo)"  },
        { 0x94D049BBu, "splitmix 0x94D049BB133111EB (hi)"  },
        { 0x133111EBu, "splitmix 0x94D049BB133111EB (lo)"  },
    };

    int n_found = 0;
    int n_constants = (int) (sizeof(model_constants)/sizeof(model_constants[0]));

    std::string missing;

    for (int c = 0; c < n_constants; ++c) {
        bool found = false;

        for (size_t i = 5; i < n_words && !found; ++i) {
            found = words[i] == model_constants[c].value;
        }

        if (found) {
            n_found++;
        } else {
            missing += missing.empty() ? model_constants[c].what : std::string(", ") + model_constants[c].what;
        }
    }

    printf("  %-20s %d/%d constants of the model are in the module%s\n",
            "", n_found, n_constants,
            missing.empty() || !expect_int64 ? "" : (" (missing " + missing + ")").c_str());

    // the two passes that draw the noise of the array have to carry the hash of the model; the DAC
    // and the RMS pass are pure quantization and have no constants to find
    const bool rng_ok = !expect_int64 || n_found > 0;

    if (!rng_ok) {
        printf("  %-20s the module does not carry the noise of the array\n", name);
    }

    return ok && rng_ok;
}

// ---------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------

static void time_program(const pass_case & tc, struct ggml_context * ctx, int n_threads, int n_rep) {
    struct ggml_tensor * w    = new_tensor_2d(ctx, tc.type, tc.k, tc.nrows);
    struct ggml_tensor * snap = ggml_dup_tensor(ctx, w);

    std::vector<float> f(tc.k*tc.nrows);
    fill_random(f.data(), f.size(), 0.7f);

    for (int64_t r = 0; r < tc.nrows; ++r) {
        ggml_quantize_chunk(tc.type, f.data() + r*tc.k, (char *) w->data + r*ggml_row_size(tc.type, tc.k), 0, 1, tc.k, nullptr);
    }

    std::vector<float> norm(tc.nrows);
    std::vector<float> inorm(tc.nrows);

    ggml_analog_kernels_set_n_threads(n_threads);

    const double t_ref = time_us(n_rep, [&]() {
        ggml_analog_kernels_set_enabled(false);
        ggml_analog_kernel_program(w, snap, tc.cfg, norm.data(), inorm.data());
        ggml_analog_kernels_set_enabled(true);
    });

    const double t_ker = time_us(n_rep, [&]() {
        ggml_analog_kernel_program(w, snap, tc.cfg, norm.data(), inorm.data());
    });

    ggml_analog_shader_set_prefer_vulkan(false);

    const double t_mir = time_us(n_rep, [&]() {
        ggml_analog_shader_program(w, snap, tc.cfg, norm.data(), inorm.data());
    });

    printf("  %-16s %2d tr: reference %8.3f us  kernels %8.3f us (%.2fx)  mirror %8.3f us (%.2fx)\n",
            tc.name, n_threads, t_ref, t_ker, t_ref/t_ker, t_mir, t_ref/t_mir);
}

static void time_readout(const pass_case & tc, int n_threads, int n_rep) {
    const int64_t k  = tc.k;
    const int64_t m  = tc.nrows;
    const int64_t n1 = 8;
    const int64_t n2 = 4;
    const int64_t n3 = 2;

    const int64_t ncols = n1*n2*n3;
    const int64_t nrows = m*n2*n3;

    std::vector<float> x(k*ncols);
    std::vector<float> y(m*ncols);
    std::vector<float> norm(nrows);
    std::vector<float> inorm(nrows);

    fill_random(x.data(), k*ncols, 1.0f);
    fill_random(y.data(), m*ncols, 2.0f/sqrt((double) k));
    fill_random(norm.data(), nrows, 1.0f);

    for (int64_t i = 0; i < nrows; ++i) {
        norm[i]   = std::fabs(norm[i]) + 0.1f;
        inorm[i]  = 1.0f/norm[i];
    }

    std::vector<float> out(y);

    ggml_analog_kernels_set_n_threads(n_threads);

    const double t_ref = time_us(n_rep, [&]() {
        ggml_analog_kernels_set_enabled(false);

        memcpy(out.data(), y.data(), sizeof(float)*y.size());

        ggml_analog_kernel_readout(x.data(), out.data(), norm.data(), inorm.data(), k, m, n1, n2, n3, n2, n3, tc.cfg);

        ggml_analog_kernels_set_enabled(true);
    });

    const double t_ker = time_us(n_rep, [&]() {
        memcpy(out.data(), y.data(), sizeof(float)*y.size());

        ggml_analog_kernel_readout(x.data(), out.data(), norm.data(), inorm.data(), k, m, n1, n2, n3, n2, n3, tc.cfg);
    });

    ggml_analog_shader_set_prefer_vulkan(false);

    const double t_mir = time_us(n_rep, [&]() {
        memcpy(out.data(), y.data(), sizeof(float)*y.size());

        ggml_analog_shader_readout(x.data(), out.data(), norm.data(), inorm.data(), k, m, n1, n2, n3, n2, n3, tc.cfg);
    });

    printf("  %-16s %2d tr: reference %8.3f us  kernels %8.3f us (%.2fx)  mirror %8.3f us (%.2fx)  [%lld rows x %lld cols]\n",
            tc.name, n_threads, t_ref, t_ker, t_ref/t_ker, t_mir, t_ref/t_mir,
            (long long) m, (long long) ncols);
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    int n_rep = 5;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("usage: %s [-r <n_rep>]\n\n"
                   "  -r <n_rep>   runs per timing sample (default 5), the median is reported\n\n"
                   "The passes run on the CPU kernels, or on the mirror of the compute shaders when\n"
                   "GGML_ANALOG_DISPATCH=shader, the scalar reference is GGML_ANALOG_KERNELS=0.\n", argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            n_rep = atoi(argv[++i]);
        }
    }

    const int n_threads = std::max(1, (int) std::thread::hardware_concurrency());

    printf("analog kernels: %s\n", ggml_analog_kernels_name());
    printf("analog shaders: %s, %d SPIR-V modules, %s\n", ggml_analog_shader_status(),
            ggml_analog_shader_spirv_count(), ggml_analog_shader_available() ? "built" : "not built");
    printf("dispatch: %s, %d thread(s) available\n", ggml_analog_kernels_dispatch_name(), n_threads);

    // -----------------------------------------------------------------------
    printf("\n=============== compute shaders ===============\n");

    if (ggml_analog_shader_spirv_count() == 0) {
        printf("  the build had no GLSL compiler, the SPIR-V modules are not available\n");
    } else {
        for (int i = 0; i < ggml_analog_shader_spirv_count(); ++i) {
            // the programming and the readout draw the noise of the array, they need 64 bit
            // arithmetic, the DAC and the RMS pass only do 32 bit work
            const std::string name = ggml_analog_shader_spirv_name(i);
            const bool needs_int64 = name == "analog_program" || name == "analog_readout";

            if (!spirv_check(i, needs_int64)) {
                g_failures++;
            }
        }
    }

    // -----------------------------------------------------------------------
    printf("\n=============== passes against the scalar reference ===============\n");

    ggml_analog_config cfg_ideal;
    cfg_ideal.ideal = true;

    ggml_analog_config cfg_default;

    ggml_analog_config cfg_nonideal;
    cfg_nonideal.sigma_prog  = 0.05f;
    cfg_nonideal.sigma_therm = 0.02f;
    cfg_nonideal.drift       = 0.01f;
    cfg_nonideal.ir_drop     = 0.05f;
    cfg_nonideal.bits_w      = 3;
    cfg_nonideal.bits_x      = 4;
    cfg_nonideal.bits_adc    = 5;
    cfg_nonideal.tile_cols   = 64;

    const struct {
        const char * name;
        ggml_analog_config cfg;
    } configs[] = {
        { "ideal",     cfg_ideal   },
        { "default",   cfg_default },
        { "non-ideal", cfg_nonideal },
    };

    struct ggml_init_params params = {
        /* .mem_size   = */ 64*1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    const struct {
        const char * name;
        enum ggml_type type;
        int64_t k;
        int64_t nrows;
    } shapes[] = {
        { "f32  64x8",    GGML_TYPE_F32,  64,    8 },
        { "f32  128x128", GGML_TYPE_F32,  128, 128 },
        { "f16  256x512", GGML_TYPE_F16,  256, 512 },
        { "q8_0 512x256", GGML_TYPE_Q8_0, 512, 256 },
        { "q4_0 1024x128", GGML_TYPE_Q4_0, 1024, 128 },
    };

    for (const auto & config : configs) {
        printf("  --- %s ---\n", config.name);

        for (const auto & shape : shapes) {
            pass_case tc;

            tc.name  = shape.name;
            tc.type  = shape.type;
            tc.k     = shape.k;
            tc.nrows = shape.nrows;
            tc.cfg   = config.cfg;

            run_dac_case(tc, ctx, n_threads);
            run_program_case(tc, ctx, n_threads);
            run_readout_case(tc, ctx, n_threads);
        }

        ggml_free(ctx);
        ctx = ggml_init(params);
    }

    // -----------------------------------------------------------------------
    printf("\n=============== speed of the passes ===============\n");

    pass_case tc;

    tc.type  = GGML_TYPE_F16;
    tc.k     = 1024;
    tc.nrows = 2048;
    tc.cfg   = cfg_nonideal;

    for (int t = 1; t <= n_threads; t *= 2) {
        tc.name = "program f16";
        time_program(tc, ctx, t, n_rep);
        tc.name = "readout f16";
        time_readout(tc, t, n_rep);
    }

    printf("\n");

    ggml_free(ctx);

    printf("main: %s\n", g_failures == 0 ? "all kernel tests passed" : "FAILED");

    return g_failures == 0 ? 0 : 1;
}
