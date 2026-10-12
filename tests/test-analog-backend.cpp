// Validates the analog backend against the CPU backend on GGML_OP_MUL_MAT and measures the
// emulated matrix multiplication. See docs/backend/ANALOG.md

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-analog.h"

// the passes of the array model and where they run (GGML_ANALOG_DISPATCH)
#include "analog-kernels.h"
#include "analog-shader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// setenv is posix, _putenv_s is the windows spelling
#if defined(_WIN32)
static int setenv(const char * name, const char * value, int overwrite) {
    (void) overwrite;

    return _putenv_s(name, value);
}

static int unsetenv(const char * name) {
    return _putenv_s(name, "");
}
#endif
#include <vector>

struct test_case {
    const char *    name;
    enum ggml_type  type;
    int64_t         k; // reduction
    int64_t         m; // rows of the weight matrix
    int64_t         n; // number of input vectors
};

static const std::vector<test_case> test_cases = {
    // name                 weight type      K     M     N
    { "f32   256x256  x1",  GGML_TYPE_F32,   256,  256,   1 },
    { "f32   512x512  x4",  GGML_TYPE_F32,   512,  512,   4 },
    { "f16   512x1024 x1",  GGML_TYPE_F16,   512, 1024,   1 },
    { "q8_0  512x512  x1",  GGML_TYPE_Q8_0,  512,  512,   1 },
    { "q8_0 1024x512  x8",  GGML_TYPE_Q8_0, 1024,  512,   8 },
    { "q4_0  512x512  x1",  GGML_TYPE_Q4_0,  512,  512,   1 },
    { "q4_0 1024x1024 x4",  GGML_TYPE_Q4_0, 1024, 1024,   4 },
    { "q4_k 1024x512  x1",  GGML_TYPE_Q4_K, 1024,  512,   1 },
    { "iq2 1024x512  x1",   GGML_TYPE_IQ2_XXS, 1024, 512,  1 }, // requires an imatrix format
};

struct result {
    double max_abs_err = 0.0;
    double rmse        = 0.0;
    double max_rel_err = 0.0;
    double cpu_us      = 0.0; // digital CPU kernel
    double analog_us   = 0.0; // emulated array, wall time
    double dev_us      = 0.0; // modeled array, from the device model
    double dev_uj      = 0.0; // modeled energy
    int64_t n_mvm      = 0;
    int64_t n_passes   = 0;
    double ref_err     = 0.0; // against a scalar reference implementation
    double dispatch_err = 0.0; // result of the passes against the CPU kernels, < 0 when not checked
};

struct graph_data {
    ggml_context *        ctx_w = nullptr;
    ggml_context *        ctx_g = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf_g = nullptr;
    ggml_tensor *         w     = nullptr;
    ggml_tensor *         x     = nullptr;
    ggml_tensor *         y     = nullptr;
    ggml_cgraph *         gf    = nullptr;

    std::vector<float> x_data;
};

static void fill_random(float * data, int64_t n, float scale, uint32_t seed) {
    uint32_t s = seed;

    for (int64_t i = 0; i < n; ++i) {
        s = s*1664525u + 1013904223u;
        const float u1 = (float) (s >> 8)/(float) (1 << 24);
        s = s*1664525u + 1013904223u;
        const float u2 = (float) (s >> 8)/(float) (1 << 24);

        data[i] = scale*(u1 + u2 - 1.0f); // triangle distribution
    }
}

static void quantize_weights(enum ggml_type type, const std::vector<float> & src, void * dst, int64_t n) {
    if (type == GGML_TYPE_F32) {
        memcpy(dst, src.data(), n*sizeof(float));
        return;
    }

    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t *) dst, n);
        return;
    }

    if (ggml_quantize_requires_imatrix(type)) {
        // the real quantizer uses importance weights; they do not matter here, the CPU backend is
        // given the very same bytes, so an imatrix of ones is enough
        const std::vector<float> imatrix(n, 1.0f);

        ggml_quantize_chunk(type, src.data(), dst, 0, 1, n, imatrix.data());
        return;
    }

    ggml_quantize_chunk(type, src.data(), dst, 0, 1, n, nullptr);
}

static bool build_graph(const test_case & tc, uint32_t seed, graph_data & gd) {
    const size_t ctx_size = 16*1024*1024;
    const int64_t k = tc.k;
    const int64_t m = tc.m;
    const int64_t n = tc.n;

    {
        struct ggml_init_params params = { ctx_size, nullptr, true };
        gd.ctx_w = ggml_init(params);
    }
    {
        struct ggml_init_params params = { ctx_size, nullptr, true };
        gd.ctx_g = ggml_init(params);
    }

    if (gd.ctx_w == nullptr || gd.ctx_g == nullptr) {
        return false;
    }

    gd.w = ggml_new_tensor_2d(gd.ctx_w, tc.type, k, m);
    strcpy(gd.w->name, "test weights");

    gd.x = ggml_new_tensor_2d(gd.ctx_g, GGML_TYPE_F32, k, n);
    strcpy(gd.x->name, "test activations");

    gd.y = ggml_mul_mat(gd.ctx_g, gd.w, gd.x);

    gd.gf = ggml_new_graph(gd.ctx_g);
    ggml_build_forward_expand(gd.gf, gd.y);

    // the weights live in a buffer that is marked as such, this is what makes the device
    // treat the tensors as arrays of stored conductances
    gd.buf_w = ggml_backend_alloc_ctx_tensors_from_buft(gd.ctx_w, ggml_backend_cpu_buffer_type());
    if (gd.buf_w == nullptr) {
        return false;
    }
    ggml_backend_buffer_set_usage(gd.buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    {
        std::vector<float> data(k*m);
        fill_random(data.data(), k*m, 0.5f, seed);

        std::vector<uint8_t> quantized(ggml_nbytes(gd.w));
        quantize_weights(tc.type, data, quantized.data(), k*m);

        ggml_backend_tensor_set(gd.w, quantized.data(), 0, ggml_nbytes(gd.w));
    }

    // the graph tensors (activations and outputs) are pre-allocated as well, so that both
    // backends can run the same graph without re-allocating it
    gd.buf_g = ggml_backend_alloc_ctx_tensors_from_buft(gd.ctx_g, ggml_backend_cpu_buffer_type());
    if (gd.buf_g == nullptr) {
        return false;
    }

    gd.x_data.resize(k*n);
    fill_random(gd.x_data.data(), k*n, 1.0f, seed + 1);

    return true;
}

static void free_graph(graph_data & gd) {
    if (gd.buf_w) {
        ggml_backend_buffer_free(gd.buf_w);
    }
    if (gd.buf_g) {
        ggml_backend_buffer_free(gd.buf_g);
    }
    if (gd.ctx_g) {
        ggml_free(gd.ctx_g);
    }
    if (gd.ctx_w) {
        ggml_free(gd.ctx_w);
    }
}

// run the graph on the given backends, returns the median time of n_rep runs in us
static double compute_graph(ggml_backend_t * backends, int n_backends, graph_data & gd, std::vector<float> & out, int n_rep) {
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, 512, false, true);

    ggml_backend_sched_reset(sched);

    if (!ggml_backend_sched_alloc_graph(sched, gd.gf)) {
        fprintf(stderr, "%s: failed to allocate the graph\n", __func__);
        exit(1);
    }

    ggml_backend_tensor_set(gd.x, gd.x_data.data(), 0, gd.x_data.size()*sizeof(float));

    // the first evaluation programs the arrays and initializes the threadpool
    ggml_backend_sched_graph_compute(sched, gd.gf);

    std::vector<double> times;

    for (int i = 0; i < n_rep; ++i) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_sched_graph_compute(sched, gd.gf);
        const int64_t t1 = ggml_time_us();

        times.push_back((double) (t1 - t0));
    }

    std::sort(times.begin(), times.end());

    out.resize(ggml_nelements(gd.y));
    ggml_backend_tensor_get(gd.y, out.data(), 0, ggml_nelements(gd.y)*sizeof(float));

    ggml_backend_sched_free(sched);

    return times[times.size()/2];
}

// scalar reference: the first element of the first column of the product
static double reference_mul_mat(const graph_data & gd) {
    const int64_t k = gd.w->ne[0];

    std::vector<uint8_t> w_raw(ggml_nbytes(gd.w));
    ggml_backend_tensor_get(gd.w, w_raw.data(), 0, w_raw.size());

    std::vector<float> w(k);

    if (gd.w->type == GGML_TYPE_F32) {
        memcpy(w.data(), w_raw.data(), k*sizeof(float));
    } else if (gd.w->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) w_raw.data(), w.data(), k);
    } else {
        ggml_get_type_traits(gd.w->type)->to_float(w_raw.data(), w.data(), k);
    }

    double sum = 0.0;
    for (int64_t i = 0; i < k; ++i) {
        sum += (double) w[i]*gd.x_data[i];
    }

    return sum;
}

static result run_case(const test_case & tc, ggml_backend_t cpu, ggml_backend_t analog) {
    result res;

    graph_data gd;

    if (!build_graph(tc, 1234, gd)) {
        fprintf(stderr, "%s: failed to build the graph\n", tc.name);
        exit(1);
    }

    std::vector<float> out_cpu;
    std::vector<float> out_analog;

    // enough repetitions for the median to be meaningful, but a bound on the total work
    const int n_rep = (int) std::max<int64_t>(3, std::min<int64_t>(64, 8*1000*1000/(tc.k*tc.m*tc.n)));

    // reference: the pure CPU backend
    {
        ggml_backend_t backends[1] = { cpu };
        res.cpu_us = compute_graph(backends, 1, gd, out_cpu, n_rep);
    }

    // the same graph with the analog device, which takes priority over the CPU for weight matrices
    {
        ggml_backend_analog_stats_reset();

        ggml_backend_t backends[2] = { analog, cpu };
        res.analog_us = compute_graph(backends, 2, gd, out_analog, n_rep);

        struct ggml_analog_stats stats;
        ggml_backend_analog_stats_get(&stats);

        // the statistics also cover the warm-up evaluation, which programs the arrays
        res.n_mvm    = stats.n_mvm/(n_rep + 1);
        res.n_passes = stats.n_array_passes/(n_rep + 1);
        res.dev_us   = stats.est_time_ms*1000.0/(n_rep + 1);
        res.dev_uj   = stats.est_energy_mj*1000.0/(n_rep + 1);
    }

    // when the passes run somewhere else than on the CPU kernels, the whole model must produce
    // the same result: same operation order, same random numbers
    if (ggml_analog_kernels_get_dispatch() != GGML_ANALOG_DISPATCH_CPU) {
        const enum ggml_analog_dispatch saved = ggml_analog_kernels_get_dispatch();

        std::vector<float> out_kernels;

        ggml_analog_kernels_set_dispatch(GGML_ANALOG_DISPATCH_CPU);
        {
            ggml_backend_t backends[2] = { analog, cpu };
            compute_graph(backends, 2, gd, out_kernels, 1);
        }
        ggml_analog_kernels_set_dispatch(saved);

        res.dispatch_err = 0.0;
        for (size_t i = 0; i < out_kernels.size(); ++i) {
            res.dispatch_err = std::max(res.dispatch_err, (double) std::fabs(out_kernels[i] - out_analog[i]));
        }
    } else {
        res.dispatch_err = -1.0;
    }

    {
        const double ref = reference_mul_mat(gd);
        res.ref_err = std::fabs((double) out_cpu[0] - ref)/std::max(std::fabs(ref), 1e-3);
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < out_cpu.size(); ++i) {
        const double err = std::fabs((double) out_cpu[i] - (double) out_analog[i]);
        const double scale = std::max(std::fabs((double) out_cpu[i]), 1e-3);

        res.max_abs_err = std::max(res.max_abs_err, err);
        res.max_rel_err = std::max(res.max_rel_err, err/scale);
        sum_sq += err*err;
    }
    res.rmse = std::sqrt(sum_sq/out_cpu.size());

    free_graph(gd);

    return res;
}

int main(int argc, char ** argv) {
    bool verbose = false;
    const char * filter = nullptr;
    const char * dispatch = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("usage: %s [-v] [-f <case>] [-d <dispatch>]\n\n"
                   "  -v                log every operation of the array\n"
                   "  -f <case>         only the cases whose name contains <case>\n"
                   "  -d <dispatch>     where the passes of the array model run: cpu kernels (default),\n"
                   "                    the mirror of the compute shaders, or a Vulkan device\n", argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            filter = argv[++i];
        }
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            dispatch = argv[++i];
        }
    }

    setenv("GGML_ANALOG_VERBOSE", verbose ? "1" : "0", 1);

    // where the three passes of the array model run: the CPU kernels (default), the mirror of the
    // compute shaders, or the shaders on a Vulkan device. The result must not change.
    if (dispatch != nullptr) {
        if (strcmp(dispatch, "cpu") != 0 && strcmp(dispatch, "shader") != 0 && strcmp(dispatch, "vulkan") != 0) {
            fprintf(stderr, "unknown dispatch '%s', expected cpu, shader or vulkan\n", dispatch);
            return 1;
        }

        setenv("GGML_ANALOG_DISPATCH", dispatch, 1);
    }

    ggml_time_init();
    ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    int n_fail = 0;

    for (int pass = 0; pass < 2; ++pass) {
        const bool ideal = pass == 0;

        setenv("GGML_ANALOG_ENABLE", "1", 1);

        if (ideal) {
            setenv("GGML_ANALOG_IDEAL", "1", 1);
        } else {
            unsetenv("GGML_ANALOG_IDEAL");
        }

        // the parameters are read when the device is initialized
        ggml_backend_analog_config_reload();
        ggml_backend_analog_set_enabled(true);

        printf("\n=============== %s array ===============\n", ideal ? "ideal" : "non-ideal");

        if (ggml_analog_kernels_get_dispatch() != GGML_ANALOG_DISPATCH_CPU) {
            printf("the passes of the array model run on the %s (%s)\n",
                    ggml_analog_kernels_dispatch_name(), ggml_analog_shader_status());
        }
        printf("%-20s %10s %10s %10s %10s %10s %10s %10s\n",
                "case", "cpu [us]", "array [us]", "device [us]", "max abs err", "rmse", "max rel err", "MAC/s");
        fflush(stdout);

        for (const auto & tc : test_cases) {
            if (filter != nullptr && strstr(tc.name, filter) == nullptr) {
                continue;
            }

            ggml_backend_t cpu = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(cpu, 2);

            ggml_backend_t analog = ggml_backend_init_by_name("Analog", nullptr);
            if (analog == nullptr) {
                fprintf(stderr, "the Analog backend is not available\n");
                return 1;
            }
            ggml_backend_analog_set_n_threads(analog, 2);

            const result res = run_case(tc, cpu, analog);

            const double macs = (double) tc.k*tc.m*tc.n;
            const double gmacs = macs/(res.analog_us*1e-6)/1e9;

            printf("%-20s %10.1f %10.1f %10.2f %10.6f %10.6f %10.6f %10.3f\n",
                    tc.name, res.cpu_us, res.analog_us, res.dev_us, res.max_abs_err, res.rmse, res.max_rel_err, gmacs);
            fflush(stdout);

            if (res.dispatch_err >= 0.0) {
                // a Vulkan device is allowed to contract a multiply and an add, the mirror is not
                const double tol = ggml_analog_kernels_get_dispatch() == GGML_ANALOG_DISPATCH_VULKAN ? 1e-6 : 0.0;

                printf("        dispatch: %s vs the CPU kernels, max abs diff %.9e  %s\n",
                        ggml_analog_kernels_dispatch_name(), res.dispatch_err,
                        res.dispatch_err <= tol ? "ok" : "MISMATCH");

                if (res.dispatch_err > tol) {
                    n_fail++;
                }
            }

            if (res.ref_err > 0.05) {
                printf("        ^^ FAIL: the CPU reference does not match the scalar reference (%.3e)\n", res.ref_err);
                n_fail++;
            }

            if (res.n_mvm == 0) {
                printf("        ^^ FAIL: the operation was not offloaded to the analog device\n");
                n_fail++;
            } else if (ideal) {
                // with an ideal array the emulation must reproduce the mixed-signal core exactly
                const double tol = 1e-12*std::sqrt(macs) + 1e-30;
                if (res.max_abs_err > tol) {
                    printf("        ^^ FAIL: ideal array error %.3e > tol %.3e\n", res.max_abs_err, tol);
                    n_fail++;
                }
            } else if (res.max_abs_err == 0.0) {
                printf("        ^^ FAIL: the non-ideal array must perturb the result\n");
                n_fail++;
            }

            if (verbose) {
                ggml_backend_analog_stats_print();
            }

            ggml_backend_free(analog);
            ggml_backend_free(cpu);
        }
    }

    ggml_backend_analog_set_enabled(false);

    if (n_fail > 0) {
        printf("\n%s: %d test case(s) failed\n", __func__, n_fail);
        return 1;
    }

    printf("\n%s: all test cases passed\n", __func__);

    if (ggml_analog_kernels_get_dispatch() != GGML_ANALOG_DISPATCH_CPU) {
        printf("%s: the passes ran on the %s, bit for bit against the CPU kernels\n",
                __func__, ggml_analog_kernels_dispatch_name());
    }

    return 0;
}
