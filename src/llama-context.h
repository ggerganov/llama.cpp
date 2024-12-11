#pragma once

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-control-vector.h"

#include "ggml-cpp.h"

#include <map>
#include <unordered_map>
#include <vector>
#include <set>

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int      n_threads;       // number of threads to use for generation
    int      n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float defrag_thold;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool no_perf;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};

struct llama_context {
    llama_context(const llama_model & model)
        : model(model)
        , t_start_us(model.t_start_us)
        , t_load_us(model.t_load_us) {}

    const struct llama_model & model;

    struct llama_cparams        cparams;
    struct llama_sbatch         sbatch;
    struct llama_kv_cache       kv_self;
    struct llama_control_vector cvec;

    std::unordered_map<struct llama_lora_adapter *, float> lora_adapters;

    std::vector<ggml_backend_ptr> backends;
    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    ggml_backend_t backend_cpu = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    bool has_evaluated_once = false;

    mutable int64_t t_start_us;
    mutable int64_t t_load_us;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch

    bool logits_all = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // TODO: find a better way to accommodate mutli-dimension position encoding methods
    // number of position id each token get, 1 for each token in most cases.
    // when using m-rope, it will be 3 position ids per token to representing 3 dimension coordinate.
    int n_pos_per_token = 1;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_ptr sched;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    // input tensors
    struct ggml_tensor * inp_tokens;      // I32 [n_batch]
    struct ggml_tensor * inp_embd;        // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos;         // I32 [n_batch]
    struct ggml_tensor * inp_out_ids;     // I32 [n_outputs]
    struct ggml_tensor * inp_KQ_mask;     // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa; // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_K_shift;     // I32 [kv_size]
    struct ggml_tensor * inp_mean;        // F32 [n_batch, n_batch]
    struct ggml_tensor * inp_cls;         // I32 [n_batch]
    struct ggml_tensor * inp_s_copy;      // I32 [kv_size]
    struct ggml_tensor * inp_s_mask;      // F32 [1, n_kv]
    struct ggml_tensor * inp_s_seq;       // I32 [n_kv, n_batch]
    struct ggml_tensor * inp_pos_bucket;    // I32 [n_batch|n_kv, n_batch]
    struct ggml_tensor * inp_embd_enc;      // F32 [n_embd, n_outputs_enc]
    struct ggml_tensor * inp_KQ_mask_cross; // F32 [n_outputs_enc, n_batch]
};

static bool llama_kv_cache_init(
             struct llama_kv_cache & cache,
               const llama_context * ctx,
                         ggml_type   type_k,
                         ggml_type   type_v,
                          uint32_t   kv_size,
                              bool   offload) {
    const llama_model & model = ctx->model;
    const llama_cparams & cparams = ctx->cparams;

    const struct llama_hparams & hparams = model.hparams;

    const int32_t n_layer = hparams.n_layer;

    LLAMA_LOG_INFO("%s: kv_size = %d, offload = %d, type_k = '%s', type_v = '%s', n_layer = %d\n", __func__, kv_size, offload, ggml_type_name(type_k), ggml_type_name(type_v), n_layer);

    cache.has_shift = false;

    cache.recurrent = llama_model_is_recurrent(&model);
    cache.v_trans   = !cache.recurrent && !cparams.flash_attn;

    cache.head = 0;
    cache.size = kv_size;
    cache.used = 0;

    cache.type_k = type_k;
    cache.type_v = type_v;

    cache.cells.clear();
    cache.cells.resize(kv_size);

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }
            ctx_map[buft] = ctx;
            cache.ctxs.emplace_back(ctx);
            return ctx;
        }
        return it->second;
    };

    cache.k_l.reserve(n_layer);
    cache.v_l.reserve(n_layer);

    for (int i = 0; i < n_layer; i++) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i) + hparams.n_embd_k_s();
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i) + hparams.n_embd_v_s();

        LLAMA_LOG_DEBUG("%s: layer %d: n_embd_k_gqa = %d, n_embd_v_gqa = %d\n", __func__, i, n_embd_k_gqa, n_embd_v_gqa);

        ggml_backend_buffer_type_t buft;
        if (offload) {
            auto * dev = model.dev_layer.at(i).dev;
            buft = ggml_backend_dev_buffer_type(dev);
        } else {
            buft = ggml_backend_cpu_buffer_type();
        }
        ggml_context * ctx = ctx_for_buft(buft);

        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create ggml context for kv cache\n", __func__);
            return false;
        }

        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        auto * buft = it.first;
        auto * ctx  = it.second;

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for kv cache\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        cache.bufs.emplace_back(buf);
    }

    return true;
}

static uint32_t llama_kv_cache_get_padding(const struct llama_cparams & cparams) {
    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}

// Make sure enough space is available for outputs.
// Returns max number of outputs for which space was reserved.
static size_t llama_output_reserve(llama_context & lctx, size_t n_outputs) {
    const auto & cparams = lctx.cparams;
    const auto & hparams = lctx.model.hparams;

    const size_t n_outputs_max = std::max(n_outputs, (size_t) cparams.n_seq_max);

    const auto n_batch = cparams.n_batch;
    const auto n_vocab = hparams.n_vocab;
    const auto n_embd  = hparams.n_embd;

    // TODO: use a per-batch flag for logits presence instead
    const bool has_logits = !cparams.embeddings;
    const bool has_embd   =  cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE);

    const size_t logits_size = has_logits ? n_vocab*n_outputs_max : 0;
    const size_t embd_size   = has_embd   ?  n_embd*n_outputs_max : 0;

    if (lctx.output_ids.empty()) {
        // init, never resized afterwards
        lctx.output_ids.resize(n_batch);
    }

    const size_t prev_size = lctx.buf_output ? ggml_backend_buffer_get_size(lctx.buf_output.get()) : 0;
    const size_t new_size  = (logits_size + embd_size) * sizeof(float);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!lctx.buf_output || prev_size < new_size) {
        if (lctx.buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            lctx.buf_output = nullptr;
            lctx.logits = nullptr;
            lctx.embd = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = lctx.model.dev_output.dev;
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        lctx.buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (lctx.buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(lctx.buf_output.get());

    lctx.logits = has_logits ? output_base               : nullptr;
    lctx.embd   = has_embd   ? output_base + logits_size : nullptr;

    lctx.output_size = n_outputs_max;
    lctx.logits_size = logits_size;
    lctx.embd_size   = embd_size;

    // set all ids as invalid (negative)
    std::fill(lctx.output_ids.begin(), lctx.output_ids.end(), -1);

    ggml_backend_buffer_clear(lctx.buf_output.get(), 0);

    lctx.n_outputs = 0;

    return n_outputs_max;
}

// make the outputs have the same order they had in the user-provided batch
static void llama_output_reorder(struct llama_context * ctx) {
    std::vector<size_t> & out_ids = ctx->sbatch.out_ids;
    if (!out_ids.empty()) {
        uint32_t n_vocab = ctx->model.hparams.n_vocab;
        uint32_t n_embd  = ctx->model.hparams.n_embd;
        int32_t n_outputs = ctx->n_outputs;
        GGML_ASSERT((size_t) n_outputs == out_ids.size());
        // TODO: is there something more efficient which also minimizes swaps?
        // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
        for (int32_t i = 0; i < n_outputs - 1; ++i) {
            int32_t j_min = i;
            for (int32_t j = i + 1; j < n_outputs; ++j) {
                if (out_ids[j] < out_ids[j_min]) {
                    j_min = j;
                }
            }
            if (j_min == i) { continue; }
            std::swap(out_ids[i], out_ids[j_min]);
            if (ctx->logits_size > 0) {
                for (uint32_t k = 0; k < n_vocab; k++) {
                    std::swap(ctx->logits[i*n_vocab + k], ctx->logits[j_min*n_vocab + k]);
                }
            }
            if (ctx->embd_size > 0) {
                for (uint32_t k = 0; k < n_embd; k++) {
                    std::swap(ctx->embd[i*n_embd + k], ctx->embd[j_min*n_embd + k]);
                }
            }
        }
        std::fill(ctx->output_ids.begin(), ctx->output_ids.end(), -1);
        for (int32_t i = 0; i < n_outputs; ++i) {
            ctx->output_ids[out_ids[i]] = i;
        }
        out_ids.clear();
    }
}
