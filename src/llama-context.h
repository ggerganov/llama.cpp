#pragma once

#include "llama.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-adapter.h"

#include "ggml-cpp.h"

#include <map>
#include <unordered_map>
#include <vector>
#include <set>

using llama_loras = std::unordered_map<struct llama_adapter_lora *, float>;

struct llama_context {
    llama_context(const llama_model & model)
        : model(model)
        , t_start_us(model.t_start_us)
        , t_load_us (model.t_load_us) {}

    const struct llama_model & model;

    llama_cparams      cparams;
    llama_sbatch       sbatch;  // TODO: revisit if needed
    llama_adapter_cvec cvec;
    llama_loras        loras;

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
    bool need_reserve = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_ptr sched;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    enum ggml_status compute_graph(
                ggml_cgraph * graph,
                       bool   batched);

    // max token position across all sequences in the current context
    llama_pos pos_max() const;

    // certain implementations could require a padding for the context size
    uint32_t get_ctx_padding(const llama_cparams & cparams) const;

    void reset();

    void prepare_k_shift();
    void prepare_defrag();
    void prepare_decode(const llama_ubatch & ubatch);

    void set_inputs(const llama_ubatch & ubatch);

    ggml_tensor * build_lora_mm(
            ggml_context * ctx0,
             ggml_tensor * w,
             ggml_tensor * cur);

    ggml_tensor * build_lora_mm_id(
            ggml_context * ctx0,
             ggml_tensor * w,   // struct ggml_tensor * as
             ggml_tensor * cur, // struct ggml_tensor * b
             ggml_tensor * ids);

    // input tensors
    struct ggml_tensor * inp_tokens;        // I32 [n_batch]
    struct ggml_tensor * inp_embd;          // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos;           // I32 [n_batch]
    struct ggml_tensor * inp_out_ids;       // I32 [n_outputs]
    struct ggml_tensor * inp_mean;          // F32 [n_batch, n_batch]
    struct ggml_tensor * inp_cls;           // I32 [n_batch]

    // === encoder-decoder ===

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    struct ggml_tensor * inp_embd_enc;      // F32 [n_embd, n_outputs_enc]
    struct ggml_tensor * inp_pos_bucket;    // I32 [n_batch|n_kv, n_batch]

    // === unified KV cache ===

    llama_kv_cache     kv_self;

    struct ggml_tensor * inp_KQ_mask;         // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_cnv;     //     [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa;     // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa_cnv; //     [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_cross;   // F32 [n_outputs_enc, n_batch]
    struct ggml_tensor * inp_K_shift;         // I32 [kv_size]

    // return true if need to reserve new worst-case graph
    void kv_self_update();

    void build_attn_inp(
            ggml_context * ctx0,
                 int32_t   n_tokens,
                    bool   causal,
                    bool   swa,
                    bool   worst_case);

    void build_attn_kv_store(
            ggml_context * ctx0,
             ggml_cgraph * graph,
             ggml_tensor * k_cur,
             ggml_tensor * v_cur,
                 int32_t   n_tokens,
                 int64_t   il,
                 bool      worst_case);

    ggml_tensor * build_attn_qkv(
            ggml_context * ctx0,
             ggml_cgraph * graph,
             ggml_tensor * wo,
             ggml_tensor * wo_b,
             ggml_tensor * q_cur,
                 int32_t   n_tokens,
                 float     kq_scale,
                 int       il,
                 bool      worst_case);

    ggml_tensor * build_soft_max_ext(
            ggml_context * ctx0,
             ggml_tensor * kq,
                 float     kq_scale);

    ggml_tensor * get_rope_factors(int il);

    void build_k_shift(
            ggml_context * ctx0,
             ggml_cgraph * graph);

    // find holes from the beginning of the KV cache and fill them by moving data from the end of the cache
    void build_defrag(
            ggml_context * ctx0,
             ggml_cgraph * graph);

    // === recurrent ===

    // TODO: add recurrent cache
    // TODO: add mamba-specific llama_context

    // TODO: change these to build_mamba_inp and hide `state_copy` and `state_mask` inside the llama_context impl
    ggml_tensor * build_inp_s_copy(
            ggml_context * ctx0,
                    bool   worst_case);

    ggml_tensor * build_inp_s_mask(
            ggml_context * ctx0,
                    bool   worst_case);

    ggml_tensor * build_copy_mask_state(
            ggml_context * ctx0,
             ggml_cgraph * graph,
             ggml_tensor * s,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
                 int32_t   n_tokens,
                 int32_t   n_state,
                 int32_t   n_seqs,
                    bool   worst_case);

    ggml_tensor * build_mamba_layer(
            ggml_context * ctx0,
             ggml_cgraph * graph,
             ggml_tensor * cur,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il,
                    bool   worst_case);

    struct ggml_tensor * inp_s_copy;        // I32 [kv_size]
    struct ggml_tensor * inp_s_mask;        // F32 [1, n_kv]

    // === vision ===

    // TODO: find a better way to accommodate mutli-dimension position encoding methods
    // number of position id each token get, 1 for each token in most cases.
    // when using m-rope, it will be 3 position ids per token to representing 3 dimension coordinate.
    int n_pos_per_token = 1;
};

// Make sure enough space is available for outputs.
// Returns max number of outputs for which space was reserved.
size_t llama_output_reserve(struct llama_context & lctx, size_t n_outputs);

// make the outputs have the same order they had in the user-provided batch
void llama_output_reorder(struct llama_context & ctx);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(struct llama_context * ctx);
