#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-analog.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include "analog-array.h"
#include "analog-kernels.h"
#include "analog-shader.h"

#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>

// The device is a weight-stationary accelerator: MUL_MAT with a weight first operand runs through
// the emulated array, with an embedded CPU core and the readout stage afterwards. See docs/backend/ANALOG.md

struct ggml_backend_analog_context {
    ggml_backend_t    cpu        = nullptr; // the emulated mixed-signal core
    ggml_threadpool_t threadpool = nullptr;
    int               n_threads  = GGML_DEFAULT_N_THREADS;

    ggml_analog_config cfg;

    struct analog_array_entry {
        ggml_analog_array * arr;
        uint64_t            signature; // of the tensor the array was programmed from
    };

    // the scheduler may compute splits from several threads when pipeline parallelism is enabled
    std::mutex mutex;

    std::map<const struct ggml_tensor *, analog_array_entry> arrays;

    std::set<std::string> warned; // names of the weights that are not programmed into an array
};

// ---------------------------------------------------------------------------
// global state
// ---------------------------------------------------------------------------

static std::mutex         g_mutex;
static std::atomic<bool>  g_env_loaded{false};
static std::atomic<bool>  g_enabled{false};
static std::atomic<bool>  g_all_matmul{false};
static ggml_analog_config g_cfg;

static int analog_env_int(const char * name, int fallback) {
    const char * val = getenv(name);
    if (val == nullptr || val[0] == '\0') {
        return fallback;
    }
    return atoi(val);
}

void ggml_backend_analog_stats_print(void) {
    struct ggml_analog_stats s;
    ggml_backend_analog_stats_get(&s);

    if (s.n_tensors == 0 && s.n_mvm == 0) {
        return;
    }

    GGML_LOG_INFO("analog: ===== array statistics =====\n");
    GGML_LOG_INFO("analog: programmed tensors: %8" PRId64 " (%.3f GiB, %.3f M cells) in %.1f ms\n",
            s.n_tensors, s.bytes/(1024.0*1024.0*1024.0), (double) s.n_cells/1e6, s.time_program_ms);
    GGML_LOG_INFO("analog: array operations  : %8" PRId64 " (%.3f G MAC, %.3f G ADC samples)\n",
            s.n_mvm, (double) s.n_macs/1e9, (double) s.n_adc_samples/1e9);
    GGML_LOG_INFO("analog: emulated compute  : %8.1f ms (readout %.1f ms, %" PRId64 " array passes)\n",
            s.time_compute_ms, s.time_readout_ms, s.n_array_passes);
    GGML_LOG_INFO("analog: modeled device    : %8.3f ms, %.3f mJ\n", s.est_time_ms, s.est_energy_mj);
}

static void analog_stats_print_atexit(void) {
    // the tools log asynchronously and do not flush at exit (see common_log_main), so the block is
    // written straight to stderr: a requested printout must not depend on the log worker
    struct ggml_analog_stats s;
    ggml_backend_analog_stats_get(&s);

    if (s.n_tensors == 0 && s.n_mvm == 0) {
        return;
    }

    fprintf(stderr, "analog: ===== array statistics =====\n");
    fprintf(stderr, "analog: programmed tensors: %8" PRId64 " (%.3f GiB, %.3f M cells) in %.1f ms\n",
            s.n_tensors, s.bytes/(1024.0*1024.0*1024.0), (double) s.n_cells/1e6, s.time_program_ms);
    fprintf(stderr, "analog: array operations  : %8" PRId64 " (%.3f G MAC, %.3f G ADC samples)\n",
            s.n_mvm, (double) s.n_macs/1e9, (double) s.n_adc_samples/1e9);
    fprintf(stderr, "analog: emulated compute  : %8.1f ms (readout %.1f ms, %" PRId64 " array passes)\n",
            s.time_compute_ms, s.time_readout_ms, s.n_array_passes);
    fprintf(stderr, "analog: modeled device    : %8.3f ms, %.3f mJ\n", s.est_time_ms, s.est_energy_mj);
    fflush(stderr);
}

static bool g_stats_atexit = false; // the exit handler is registered at most once

// re-read the environment: the tools write the GGML_ANALOG_* variables after this registry may
// already have been constructed, see ggml_backend_analog_config_reload
static void analog_env_apply_locked(void) {
    g_enabled = analog_env_int("GGML_ANALOG_ENABLE", 0) != 0;
    ggml_analog_config_load_from_env(g_cfg);
    g_all_matmul = g_cfg.all_matmul;

    // the kernels module caches its own variables, re-read them before they are queried below
    ggml_analog_kernels_env_reload();

    // cpu kernels, the mirror of the compute shaders, or the shaders on a Vulkan device
    const enum ggml_analog_dispatch dispatch = ggml_analog_kernels_get_dispatch();

    if (dispatch != GGML_ANALOG_DISPATCH_CPU) {
        ggml_analog_shader_set_prefer_vulkan(dispatch == GGML_ANALOG_DISPATCH_VULKAN);

        GGML_LOG_INFO("analog: array passes: %s (%s), %s, %s\n",
                ggml_analog_kernels_dispatch_name(), ggml_analog_shader_status(),
                ggml_analog_kernels_name(),
                ggml_analog_shader_available() ? "SPIR-V built" : "no SPIR-V");
    }

    if (analog_env_int("GGML_ANALOG_STATS", 0) != 0 && !g_stats_atexit) {
        atexit(analog_stats_print_atexit);
        g_stats_atexit = true;
    }
}

static void analog_env_load(void) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_env_loaded.load()) {
        return;
    }

    analog_env_apply_locked();

    g_env_loaded = true;
    GGML_LOG_INFO("analog: backend %s (set GGML_ANALOG_ENABLE=1 to enable)\n", g_enabled.load() ? "enabled" : "disabled");
}

void ggml_backend_analog_set_enabled(bool enabled) {
    analog_env_load();

    std::lock_guard<std::mutex> lock(g_mutex);
    g_enabled = enabled;
}

bool ggml_backend_analog_get_enabled(void) {
    if (!g_env_loaded.load(std::memory_order_relaxed)) {
        analog_env_load();
    }

    return g_enabled.load(std::memory_order_relaxed);
}

void ggml_backend_analog_config_reload(void) {
    std::lock_guard<std::mutex> lock(g_mutex);

    analog_env_apply_locked();

    g_env_loaded = true;
}

void ggml_backend_analog_stats_reset(void) {
    ggml_analog_stats_reset_external();
}

void ggml_backend_analog_stats_get(struct ggml_analog_stats * stats) {
    ggml_analog_stats_get_external(stats);
}

// ---------------------------------------------------------------------------
// stream interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_analog_get_name(ggml_backend_t backend) {
    return "Analog";

    GGML_UNUSED(backend);
}

static uint64_t analog_tensor_signature(const struct ggml_tensor * t) {
    uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t) t->type;

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        h = (h ^ (uint64_t) t->ne[i])*0xBF58476D1CE4E5B9ull;
        h = (h ^ (uint64_t) t->nb[i])*0x94D049BB133111EBull;
    }

    h = (h ^ (uint64_t) (uintptr_t) t->data)*0x2545F4914F6CDD1Dull;
    h = (h ^ ggml_nbytes(t))*0x9E3779B97F4A7C15ull;

    return h;
}

// the array holds the weights of weight-stationary operations, i.e. matrix multiplications whose
// first operand is a model weight; everything else is evaluated by the mixed-signal core alone
static bool analog_array_applies(const struct ggml_tensor * w) {
    if (!g_all_matmul.load(std::memory_order_relaxed) &&
        (w->buffer == nullptr || w->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
        return false;
    }

    // tensors in an extra buffer type are stored in a repacked layout
    if (w->extra != nullptr) {
        return false;
    }

    // the emulation iterates over the tensor in the standard row-major layout
    if (!ggml_is_contiguous(w)) {
        return false;
    }

    if (w->ne[0] <= 0 || w->ne[1] <= 0) {
        return false;
    }

    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_BF16 &&
        ggml_get_type_traits(w->type)->to_float == nullptr) {
        return false;
    }

    return true;
}

static ggml_analog_array * analog_get_array(ggml_backend_analog_context * ctx, const struct ggml_tensor * w) {
    const uint64_t signature = analog_tensor_signature(w);

    std::lock_guard<std::mutex> lock(ctx->mutex);

    auto it = ctx->arrays.find(w);
    if (it != ctx->arrays.end()) {
        // the address may have been reused by a different tensor after a model reload
        if (it->second.signature == signature) {
            return it->second.arr;
        }

        ggml_analog_array_free(it->second.arr);
        ctx->arrays.erase(it);
    }

    ggml_analog_array * arr = ggml_analog_array_new(w, ctx->cfg);
    if (arr == nullptr) {
        return nullptr;
    }

    ctx->arrays[w] = { arr, signature };

    return arr;
}

static void analog_ensure_threadpool(ggml_backend_analog_context * ctx) {
    if (ctx->threadpool != nullptr) {
        return;
    }

    struct ggml_threadpool_params params = ggml_threadpool_params_default(ctx->n_threads);
    struct ggml_threadpool * threadpool = ggml_threadpool_new(&params);

    if (threadpool == nullptr) {
        GGML_LOG_ERROR("%s: failed to create threadpool with %d threads\n", __func__, ctx->n_threads);
        return;
    }

    ggml_backend_cpu_set_threadpool(ctx->cpu, threadpool);

    ctx->threadpool = threadpool;
    ctx->cfg.n_threads = ctx->n_threads;
}

// a matrix multiplication that is evaluated with the weights programmed into an array
struct analog_mvm {
    struct ggml_tensor * node   = nullptr;
    struct ggml_tensor * w_orig = nullptr;
    struct ggml_tensor * x_orig = nullptr;
    struct ggml_tensor * x_dac  = nullptr; // quantized activations, may be null
    ggml_analog_array *  arr    = nullptr;
};

static void analog_program(ggml_backend_analog_context * ctx, struct ggml_cgraph * cgraph, std::vector<analog_mvm> & mvms) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }

        struct ggml_tensor * w = node->src[0];
        struct ggml_tensor * x = node->src[1];

        if (w == nullptr || x == nullptr || x->type != GGML_TYPE_F32) {
            continue;
        }

        // the readout overwrites the output of the operation in place
        if (node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) {
            continue;
        }

        // the Hadamard transform is not a matrix multiplication in the array sense
        if (ggml_get_op_params_i32(node, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
            continue;
        }

        if (!analog_array_applies(w)) {
            // weights that cannot be programmed into an array are evaluated by the core alone
            if (w->buffer != nullptr && w->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                std::lock_guard<std::mutex> lock(ctx->mutex);
                if (ctx->warned.insert(w->name).second) {
                    GGML_LOG_WARN("analog: %s: %s is not programmed into an array\n", __func__, w->name);
                }
            }
            continue;
        }

        ggml_analog_array * arr = analog_get_array(ctx, w);
        if (arr == nullptr) {
            continue;
        }

        analog_mvm mvm;

        mvm.node   = node;
        mvm.w_orig = w;
        mvm.x_orig = x;
        mvm.arr    = arr;

        if (ggml_is_contiguous(x)) {
            mvm.x_dac = ggml_analog_quantize_input(x, ctx->cfg);
        }

        node->src[0] = ggml_analog_array_weights(arr);
        if (mvm.x_dac != nullptr) {
            node->src[1] = mvm.x_dac;
        }

        mvms.push_back(mvm);
    }
}

static void analog_restore(std::vector<analog_mvm> & mvms) {
    for (auto & mvm : mvms) {
        mvm.node->src[0] = mvm.w_orig;
        mvm.node->src[1] = mvm.x_orig;

        ggml_analog_quantize_input_free(mvm.x_dac);
        mvm.x_dac = nullptr;
    }
}

static void analog_readout(ggml_backend_analog_context * ctx, std::vector<analog_mvm> & mvms, struct ggml_analog_stats * stats) {
    for (auto & mvm : mvms) {
        // the readout of the array, applied to the partial sums produced by the core
        const struct ggml_tensor * x = mvm.x_dac != nullptr ? mvm.x_dac : mvm.x_orig;

        const int64_t t0 = ggml_time_us();

        ggml_analog_array_readout(mvm.arr, x, mvm.node, ctx->cfg);
        ggml_analog_mvm_cost(mvm.w_orig, mvm.node, ctx->cfg, stats);

        const int64_t t1 = ggml_time_us();

        stats->time_readout_ms += (t1 - t0)/1000.0;

        if (ctx->cfg.verbose) {
            GGML_LOG_INFO("analog: %s: %" PRId64 " x %" PRId64 " x %" PRId64 " (%s): readout %.1f us\n",
                    mvm.w_orig->name, mvm.w_orig->ne[0], mvm.w_orig->ne[1],
                    ggml_nelements(mvm.node)/mvm.w_orig->ne[1], ggml_type_name(mvm.w_orig->type),
                    (t1 - t0)/1.0);
        }
    }
}

static enum ggml_status ggml_backend_analog_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_analog_context * ctx = (ggml_backend_analog_context *) backend->context;

    analog_ensure_threadpool(ctx);

    // program the weights of the matrix multiplications into the arrays, the remaining operations
    // are evaluated by the mixed-signal core of the device
    const int64_t t_start = ggml_time_us();

    std::vector<analog_mvm> mvms;

    analog_program(ctx, cgraph, mvms);

    // the core evaluates the whole split in a single pass, like the CPU backend would
    const enum ggml_status status = ggml_backend_graph_compute(ctx->cpu, cgraph);

    analog_restore(mvms);

    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }

    const int64_t t_core = ggml_time_us();

    struct ggml_analog_stats stats = {};

    stats.time_compute_ms = (double) (t_core - t_start)/1000.0;

    analog_readout(ctx, mvms, &stats);

    ggml_analog_stats_add_external(stats);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_analog_free(ggml_backend_t backend) {
    ggml_backend_analog_context * ctx = (ggml_backend_analog_context *) backend->context;

    for (auto & entry : ctx->arrays) {
        ggml_analog_array_free(entry.second.arr);
    }

    if (ctx->threadpool != nullptr) {
        ggml_threadpool_free(ctx->threadpool);
    }

    if (ctx->cpu != nullptr) {
        ggml_backend_free(ctx->cpu);
    }

    delete ctx;
    delete backend;
}

static void ggml_backend_analog_set_n_threads_impl(ggml_backend_t backend, int n_threads) {
    ggml_backend_analog_context * ctx = (ggml_backend_analog_context *) backend->context;

    if (n_threads == ctx->n_threads) {
        return;
    }

    ctx->n_threads = n_threads;
    ctx->cfg.n_threads = n_threads;

    ggml_analog_kernels_set_n_threads(n_threads);

    ggml_backend_cpu_set_n_threads(ctx->cpu, n_threads);

    if (ctx->threadpool != nullptr) {
        // the number of threads of the pool is fixed at creation
        ggml_threadpool_free(ctx->threadpool);
        ctx->threadpool = nullptr;
    }
}

static struct ggml_backend_i analog_backend_i = {
    /* .get_name                = */ ggml_backend_analog_get_name,
    /* .free                    = */ ggml_backend_analog_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_analog_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_analog_guid(void) {
    static ggml_guid guid = { 0x9f, 0x2c, 0x11, 0xe4, 0x7b, 0x3d, 0x4a, 0x18, 0xb2, 0x77, 0x6e, 0x05, 0x8c, 0xd1, 0x2a, 0xf3 };
    return &guid;
}

ggml_backend_t ggml_backend_analog_init(void) {
    if (!g_env_loaded.load(std::memory_order_relaxed)) {
        analog_env_load();
    }

    ggml_backend_analog_context * ctx = new ggml_backend_analog_context;

    ctx->cpu       = ggml_backend_cpu_init();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        ctx->cfg = g_cfg;
    }

    const int n_threads_env = ggml_analog_kernels_get_env_n_threads();

    ctx->n_threads     = n_threads_env > 0 ? n_threads_env : GGML_DEFAULT_N_THREADS;
    ctx->cfg.n_threads = ctx->n_threads;

    ggml_analog_kernels_set_n_threads(ctx->n_threads);

    if (ctx->cpu == nullptr) {
        GGML_LOG_ERROR("%s: failed to initialize the mixed-signal core\n", __func__);
        delete ctx;
        return nullptr;
    }

    ggml_backend_cpu_set_n_threads(ctx->cpu, ctx->n_threads);

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_analog_guid(),
        /* .iface   = */ analog_backend_i,
        /* .device  = */ ggml_backend_analog_reg()->iface.get_device(ggml_backend_analog_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_analog(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_analog_guid());
}

void ggml_backend_analog_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_analog(backend));

    ggml_backend_analog_set_n_threads_impl(backend, n_threads);
}

// ---------------------------------------------------------------------------
// device interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_analog_device_get_name(ggml_backend_dev_t dev) {
    return "Analog";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_analog_device_get_description(ggml_backend_dev_t dev) {
    return "Analog in-memory computing emulation";

    GGML_UNUSED(dev);
}

static void ggml_backend_analog_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // no dedicated memory to report
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_analog_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_analog_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_analog_device_get_name(dev);
    props->description = ggml_backend_analog_device_get_description(dev);
    props->type        = ggml_backend_analog_device_get_type(dev);
    ggml_backend_analog_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_t ggml_backend_analog_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_analog_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_analog_device_get_buffer_type(ggml_backend_dev_t dev) {
    // the arrays share the memory of the host, like BLAS
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

// fused ops are only enabled on the layer weights device (see resolve_fused_ops), which this
// device never is: they must stay on the CPU backend to remain enabled
static bool analog_is_fused_layer_op(const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_LIGHTNING_INDEXER:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_POST:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_analog_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (!ggml_backend_analog_get_enabled()) {
        return false;
    }

    if (analog_is_fused_layer_op(op)) {
        return false;
    }

    // the device evaluates the whole graph, either on the emulated arrays or on the embedded
    // mixed-signal core, so it can run whatever the CPU backend can run
    static const ggml_backend_dev_t cpu_dev = []() -> ggml_backend_dev_t {
        ggml_backend_reg_t reg = ggml_backend_cpu_reg();

        return reg != nullptr ? reg->iface.get_device(reg, 0) : nullptr;
    }();

    if (cpu_dev == nullptr) {
        return false;
    }

    return cpu_dev->iface.supports_op(cpu_dev, op);

    GGML_UNUSED(dev);
}

static bool ggml_backend_analog_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static bool ggml_backend_analog_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return ggml_backend_analog_device_supports_op(dev, op);
}

static const struct ggml_backend_device_i ggml_backend_analog_device_i = {
    /* .get_name             = */ ggml_backend_analog_device_get_name,
    /* .get_description      = */ ggml_backend_analog_device_get_description,
    /* .get_memory           = */ ggml_backend_analog_device_get_memory,
    /* .get_type             = */ ggml_backend_analog_device_get_type,
    /* .get_props            = */ ggml_backend_analog_device_get_props,
    /* .init_backend         = */ ggml_backend_analog_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_analog_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_analog_device_supports_op,
    /* .supports_buft        = */ ggml_backend_analog_device_supports_buft,
    /* .offload_op           = */ ggml_backend_analog_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---------------------------------------------------------------------------
// backend reg interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_analog_reg_get_name(ggml_backend_reg_t reg) {
    return "ANALOG";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_analog_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_analog_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_analog_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &device;
}

static void * ggml_backend_analog_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_analog_set_n_threads;
    }

    if (std::strcmp(name, "ggml_backend_analog_set_enabled") == 0) {
        return (void *) ggml_backend_analog_set_enabled;
    }

    if (std::strcmp(name, "ggml_backend_analog_config_reload") == 0) {
        return (void *) ggml_backend_analog_config_reload;
    }

    if (std::strcmp(name, "ggml_backend_analog_stats_print") == 0) {
        return (void *) ggml_backend_analog_stats_print;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_analog_reg_i = {
    /* .get_name         = */ ggml_backend_analog_reg_get_name,
    /* .get_device_count = */ ggml_backend_analog_reg_get_device_count,
    /* .get_device       = */ ggml_backend_analog_reg_get_device,
    /* .get_proc_address = */ ggml_backend_analog_get_proc_address,
};

ggml_backend_reg_t ggml_backend_analog_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_analog_reg_i,
        /* .context     = */ nullptr,
    };

    analog_env_load();

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_analog_reg)
