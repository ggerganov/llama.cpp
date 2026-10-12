#include "analog-array.h"
#include "analog-kernels.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static int analog_env_int(const char * name, int fallback) {
    const char * val = getenv(name);
    if (val == nullptr || val[0] == '\0') {
        return fallback;
    }
    return atoi(val);
}

static float analog_env_float(const char * name, float fallback) {
    const char * val = getenv(name);
    if (val == nullptr || val[0] == '\0') {
        return fallback;
    }
    return (float) atof(val);
}

void ggml_analog_config_load_from_env(ggml_analog_config & cfg) {
    // start from the defaults, so that unset variables do not keep a previous value
    cfg = ggml_analog_config();

    cfg.bits_w   = analog_env_int  ("GGML_ANALOG_BITS_W",   cfg.bits_w);
    cfg.bits_x   = analog_env_int  ("GGML_ANALOG_BITS_X",   cfg.bits_x);
    cfg.bits_adc = analog_env_int  ("GGML_ANALOG_BITS_ADC", cfg.bits_adc);

    cfg.sigma_prog  = analog_env_float("GGML_ANALOG_SIGMA_PROG",  cfg.sigma_prog);
    cfg.sigma_therm = analog_env_float("GGML_ANALOG_SIGMA_THERM", cfg.sigma_therm);
    cfg.drift       = analog_env_float("GGML_ANALOG_DRIFT",       cfg.drift);
    cfg.ir_drop     = analog_env_float("GGML_ANALOG_IR_DROP",     cfg.ir_drop);

    cfg.tile_rows = analog_env_int("GGML_ANALOG_TILE_ROWS", cfg.tile_rows);
    cfg.tile_cols = analog_env_int("GGML_ANALOG_TILE_COLS", cfg.tile_cols);

    const int n_threads_env = analog_env_int("GGML_ANALOG_N_THREADS", 0);

    cfg.adc_sigma      = analog_env_float("GGML_ANALOG_ADC_SIGMA",      cfg.adc_sigma);
    cfg.pj_per_mac     = analog_env_float("GGML_ANALOG_PJ_PER_MAC",     cfg.pj_per_mac);
    cfg.mvm_latency_ns = analog_env_float("GGML_ANALOG_MVM_LATENCY_NS", cfg.mvm_latency_ns);

    cfg.seed       = (uint64_t) analog_env_int("GGML_ANALOG_SEED", (int) cfg.seed);
    cfg.ideal      = analog_env_int("GGML_ANALOG_IDEAL",      cfg.ideal ? 1 : 0) != 0;
    cfg.verbose    = analog_env_int("GGML_ANALOG_VERBOSE",    cfg.verbose ? 1 : 0) != 0;
    cfg.all_matmul = analog_env_int("GGML_ANALOG_ALL_MATMUL", cfg.all_matmul ? 1 : 0) != 0;

    // clamp to values the emulation can represent
    cfg.bits_w    = std::min(std::max(cfg.bits_w,   2), 16);
    cfg.bits_x    = std::min(std::max(cfg.bits_x,   2), 16);
    cfg.bits_adc  = std::min(std::max(cfg.bits_adc, 2), 16);
    cfg.tile_rows = std::max(cfg.tile_rows, 1);
    cfg.tile_cols = std::max(cfg.tile_cols, 1);
    cfg.n_threads = std::max(cfg.n_threads, 1);

    // the passes of the model run with as many threads as the backend was given
    ggml_analog_kernels_set_n_threads(n_threads_env);
}

void ggml_analog_config_print(const ggml_analog_config & cfg) {
    GGML_LOG_INFO("analog: %s: cells %d bit, input DAC %d bit, readout ADC %d bit\n",
            cfg.ideal ? "ideal" : "non-ideal", cfg.bits_w, cfg.bits_x, cfg.bits_adc);
    GGML_LOG_INFO("analog: non-idealities: sigma_prog %.4f, sigma_therm %.4f, drift %.4f, ir_drop %.4f, seed %d\n",
            cfg.sigma_prog, cfg.sigma_therm, cfg.drift, cfg.ir_drop, (int) cfg.seed);
    GGML_LOG_INFO("analog: readout range: %.2f sigma\n", cfg.adc_sigma);
    GGML_LOG_INFO("analog: array %d x %d, %.2f pJ/MAC, %.1f ns per array pass, %d threads\n",
            cfg.tile_rows, cfg.tile_cols, cfg.pj_per_mac, cfg.mvm_latency_ns, cfg.n_threads);
}

// true when neither the array model nor the readout modify the result, i.e. the emulated device
// behaves exactly like the mixed-signal core that evaluates the MVM
static bool analog_config_is_noop(const ggml_analog_config & cfg, const struct ggml_tensor * w) {
    const struct ggml_type_traits * tt = ggml_get_type_traits(w->type);

    // effective number of bits per cell that the container can hold
    const double container_bits = 8.0*tt->type_size/tt->blck_size;

    return cfg.ideal ||
           (cfg.sigma_prog  == 0.0f && cfg.drift == 0.0f && cfg.ir_drop == 0.0f &&
            cfg.sigma_therm == 0.0f && cfg.bits_w >= container_bits && cfg.bits_adc >= 16);
}

// ---------------------------------------------------------------------------
// device statistics
// ---------------------------------------------------------------------------

static struct ggml_analog_stats g_stats = {};
static std::mutex              g_stats_mutex;

static void analog_stats_add(const struct ggml_analog_stats & s) {
    std::lock_guard<std::mutex> lock(g_stats_mutex);

    g_stats.n_tensors       += s.n_tensors;
    g_stats.n_cells         += s.n_cells;
    g_stats.bytes           += s.bytes;
    g_stats.time_program_ms += s.time_program_ms;
    g_stats.n_mvm           += s.n_mvm;
    g_stats.n_macs          += s.n_macs;
    g_stats.n_column_tiles  += s.n_column_tiles;
    g_stats.n_array_passes  += s.n_array_passes;
    g_stats.n_adc_samples   += s.n_adc_samples;
    g_stats.time_compute_ms += s.time_compute_ms;
    g_stats.time_readout_ms += s.time_readout_ms;
    g_stats.est_time_ms     += s.est_time_ms;
    g_stats.est_energy_mj   += s.est_energy_mj;
}

void ggml_analog_stats_add_external(const struct ggml_analog_stats & s) {
    analog_stats_add(s);
}

void ggml_analog_stats_get_external(struct ggml_analog_stats * s) {
    std::lock_guard<std::mutex> lock(g_stats_mutex);

    *s = g_stats;
}

void ggml_analog_stats_reset_external(void) {
    std::lock_guard<std::mutex> lock(g_stats_mutex);

    g_stats = {};
}

// ---------------------------------------------------------------------------
// array programming
// ---------------------------------------------------------------------------

static struct ggml_tensor * analog_tensor_alloc_like(const struct ggml_tensor * w, size_t nbytes) {
    struct ggml_tensor * t = new ggml_tensor(*w);

    t->data      = ggml_aligned_malloc(nbytes);
    t->buffer    = nullptr;
    t->view_src  = nullptr;
    t->view_offs = 0;
    t->extra     = nullptr;
    t->flags     = 0;
    memset(t->src, 0, sizeof(t->src));

    GGML_ASSERT(t->data != nullptr);

    return t;
}

struct ggml_analog_array {
    struct ggml_tensor * w;            // programmed snapshot of the weights
    float *              row_norm;     // L2 norm of each row, used to set the readout range
    float *              row_inv_norm; // 1/row_norm, precomputed for the readout
    size_t               bytes;
    int64_t              cells;
    int64_t              nrows;
    double               program_ms;
};

ggml_analog_array * ggml_analog_array_new(const struct ggml_tensor * w, const ggml_analog_config & cfg) {
    const int64_t K     = w->ne[0];
    const int64_t nrows = ggml_nelements(w)/K;

    const size_t nbytes = ggml_nbytes(w);

    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_row_size(w->type, K)*(size_t) nrows == nbytes);

    const int64_t t_start = ggml_time_us();

    // formats that cannot be re-quantized without an importance matrix are stored as F16: the
    // cell precision limits them either way, and the copy of the weights is still exact
    const bool as_f16 = ggml_quantize_requires_imatrix(w->type) && !analog_config_is_noop(cfg, w);

    ggml_analog_array * arr = new ggml_analog_array;

    arr->w            = analog_tensor_alloc_like(w, as_f16 ? ggml_row_size(GGML_TYPE_F16, K)*nrows : nbytes);
    arr->row_norm     = nullptr;
    arr->row_inv_norm = nullptr;
    arr->cells      = ggml_nelements(w);
    arr->nrows      = nrows;
    arr->program_ms = 0.0;

    if (as_f16) {
        // the snapshot is of a different type than the tensor it is programmed from
        arr->w->type  = GGML_TYPE_F16;
        arr->w->nb[0] = ggml_type_size(GGML_TYPE_F16);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            arr->w->nb[i] = arr->w->nb[i - 1]*arr->w->ne[i - 1];
        }
    }

    arr->bytes = ggml_nbytes(arr->w);

    if (!as_f16 && analog_config_is_noop(cfg, w)) {
        // the array keeps an exact copy of the weights, no readout is applied
        GGML_ASSERT(ggml_nbytes(arr->w) == nbytes);
        memcpy(arr->w->data, w->data, nbytes);
    } else {
        arr->row_norm     = (float *) ggml_aligned_malloc(sizeof(float)*nrows);
        arr->row_inv_norm = (float *) ggml_aligned_malloc(sizeof(float)*nrows);
        GGML_ASSERT(arr->row_norm != nullptr && arr->row_inv_norm != nullptr);
        memset(arr->row_norm, 0, sizeof(float)*nrows);

        // the programming pass: weights -> array snapshot, run in parallel over the rows
        ggml_analog_kernel_program(w, arr->w, cfg, arr->row_norm, arr->row_inv_norm);
    }

    arr->program_ms = (ggml_time_us() - t_start)/1000.0;

    if (cfg.verbose) {
        GGML_LOG_INFO("analog: programmed %s (%s -> %s, %" PRId64 " rows x %" PRId64 ", %" PRId64 " cells) in %.2f ms\n",
                w->name, ggml_type_name(w->type), ggml_type_name(arr->w->type), nrows, K, arr->cells, arr->program_ms);
    }

    struct ggml_analog_stats s = {};
    s.n_tensors       = 1;
    s.n_cells         = arr->cells;
    s.bytes           = (double) arr->bytes;
    s.time_program_ms = arr->program_ms;
    analog_stats_add(s);

    return arr;
}

void ggml_analog_array_free(ggml_analog_array * arr) {
    if (arr == nullptr) {
        return;
    }

    if (arr->w != nullptr) {
        ggml_aligned_free(arr->w->data, arr->bytes);
        delete arr->w;
    }

    if (arr->row_norm != nullptr) {
        ggml_aligned_free(arr->row_norm, sizeof(float)*arr->nrows);
    }

    if (arr->row_inv_norm != nullptr) {
        ggml_aligned_free(arr->row_inv_norm, sizeof(float)*arr->nrows);
    }

    delete arr;
}

struct ggml_tensor * ggml_analog_array_weights(ggml_analog_array * arr) {
    return arr->w;
}

int64_t ggml_analog_array_cells(const ggml_analog_array * arr) {
    return arr->cells;
}

double ggml_analog_array_program_ms(const ggml_analog_array * arr) {
    return arr->program_ms;
}

// ---------------------------------------------------------------------------
// input DAC
// ---------------------------------------------------------------------------

struct ggml_tensor * ggml_analog_quantize_input(const struct ggml_tensor * x, const ggml_analog_config & cfg) {
    // the mixed-signal core already quantizes the activations to 8 bits per 32-element block
    if (cfg.ideal || cfg.bits_x >= 8) {
        return nullptr;
    }

    const int64_t K     = x->ne[0];
    const int64_t ncols = ggml_nelements(x)/K;

    struct ggml_tensor * q = analog_tensor_alloc_like(x, ggml_nbytes(x));

    // the input DAC quantizes the activations to the levels of the drivers of the array
    ggml_analog_kernel_dac((const float *) x->data, (float *) q->data, K, ncols, cfg);

    return q;
}

void ggml_analog_quantize_input_free(struct ggml_tensor * q) {
    if (q == nullptr) {
        return;
    }

    ggml_aligned_free(q->data, ggml_nbytes(q));
    delete q;
}

// ---------------------------------------------------------------------------
// readout
// ---------------------------------------------------------------------------

void ggml_analog_array_readout(
        const ggml_analog_array *  arr,
        const struct ggml_tensor * x,
              struct ggml_tensor * dst,
        const ggml_analog_config & cfg) {
    if (analog_config_is_noop(cfg, arr->w)) {
        return;
    }

    const int64_t K  = arr->w->ne[0];
    const int64_t M  = arr->w->ne[1];
    const int64_t n1 = dst->ne[1];
    const int64_t n2 = dst->ne[2];
    const int64_t n3 = dst->ne[3];

    GGML_ASSERT(M == dst->ne[0]);
    GGML_ASSERT(ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && x->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(x) == K*n1*n2*n3);

    const float * xp = (const float *) x->data;
        float * dp = (      float *) dst->data;

    // the readout of the array: ADC quantization and noise of the peripheral circuits
    ggml_analog_kernel_readout(xp, dp, arr->row_norm, arr->row_inv_norm, K, M, n1, n2, n3, arr->w->ne[2], arr->w->ne[3], cfg);
}

// ---------------------------------------------------------------------------
// modeled device cost
// ---------------------------------------------------------------------------

void ggml_analog_mvm_cost(
        const struct ggml_tensor * w,
        const struct ggml_tensor * dst,
        const ggml_analog_config & cfg,
              struct ggml_analog_stats * stats) {
    const int64_t K = w->ne[0];
    const int64_t M = w->ne[1];

    const int64_t ncols = ggml_nelements(dst)/M;

    const int64_t col_tiles = (K + cfg.tile_cols - 1)/cfg.tile_cols;
    const int64_t n_passes  = ncols*col_tiles;

    stats->n_mvm          += 1;
    stats->n_macs         += K*M*ncols;
    stats->n_column_tiles += col_tiles*ncols;
    stats->n_array_passes += n_passes;
    stats->n_adc_samples  += M*ncols;
    stats->est_time_ms    += (double) n_passes*cfg.mvm_latency_ns/1e6;
    stats->est_energy_mj  += (double) (K*M*ncols)*cfg.pj_per_mac/1e9;
}
