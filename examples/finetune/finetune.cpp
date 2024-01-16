#include "ggml.h"
#include "ggml-alloc.h"
#include "llama.h"
#include "common.h"
#include "train.h"
#include <vector>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static const size_t tensor_alignment = 32;

struct my_llama_hparams {
    uint32_t n_vocab    = 32000;
    uint32_t n_ctx      = 512;
    uint32_t n_embd     = 4096;
    uint32_t n_ff       = 11008;
    uint32_t n_head     = 32;
    uint32_t n_head_kv  = 32;
    uint32_t n_layer    = 32;

    // float f_norm_eps     = 1e-5f; // falcon
    float f_norm_rms_eps = 1e-5f; // llama

    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;

    uint32_t n_gqa() const {
        return n_head/n_head_kv;
    }

    uint32_t n_embd_head() const {
        return n_embd/n_head;
    }

    uint32_t n_embd_gqa() const {
        return n_embd/n_gqa();
    }

    bool operator!=(const my_llama_hparams& other) const {
        return memcmp(this, &other, sizeof(other));
    }
};

struct my_llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct my_llama_model {
    struct my_llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<my_llama_layer> layers;
};

struct my_llama_lora_hparams {
    uint32_t lora_r = 1;
    uint32_t lora_alpha = 1;
    uint32_t n_rank_attention_norm = 1;
    uint32_t n_rank_wq = 4;
    uint32_t n_rank_wk = 4;
    uint32_t n_rank_wv = 4;
    uint32_t n_rank_wo = 4;
    uint32_t n_rank_ffn_norm = 1;
    uint32_t n_rank_w1 = 4;
    uint32_t n_rank_w2 = 4;
    uint32_t n_rank_w3 = 4;
    uint32_t n_rank_tok_embeddings = 4;
    uint32_t n_rank_norm = 1;
    uint32_t n_rank_output = 4;

    bool operator!=(const my_llama_lora_hparams& other) const {
        return memcmp(this, &other, sizeof(other));
    }
};

struct my_llama_lora_layer {
    // normalization
    struct ggml_tensor * attention_norm_a;
    struct ggml_tensor * attention_norm_b;

    // attention
    struct ggml_tensor * wq_a;
    struct ggml_tensor * wq_b;
    struct ggml_tensor * wk_a;
    struct ggml_tensor * wk_b;
    struct ggml_tensor * wv_a;
    struct ggml_tensor * wv_b;
    struct ggml_tensor * wo_a;
    struct ggml_tensor * wo_b;

    // normalization
    struct ggml_tensor * ffn_norm_a;
    struct ggml_tensor * ffn_norm_b;

    // ff
    struct ggml_tensor * w1_a;
    struct ggml_tensor * w1_b;
    struct ggml_tensor * w2_a;
    struct ggml_tensor * w2_b;
    struct ggml_tensor * w3_a;
    struct ggml_tensor * w3_b;
};

struct my_llama_lora {
    struct ggml_context * ctx = NULL;
    std::vector<uint8_t> data;

    my_llama_lora_hparams hparams;

    struct ggml_tensor * tok_embeddings_a;
    struct ggml_tensor * tok_embeddings_b;

    struct ggml_tensor * norm_a;
    struct ggml_tensor * norm_b;
    struct ggml_tensor * output_a;
    struct ggml_tensor * output_b;

    std::vector<my_llama_lora_layer> layers;
};

// gguf constants
static const char * LLM_KV_TRAINING_TYPE_FINETUNE_LORA   = "finetune_lora";
static const char * LLM_KV_TRAINING_TYPE                 = "training.type";

static const char * LLM_KV_TRAINING_LORA_RANK_TOKEN_EMBD  = "training.lora.rank.token_embd";
static const char * LLM_KV_TRAINING_LORA_RANK_OUTPUT_NORM = "training.lora.rank.output_norm";
static const char * LLM_KV_TRAINING_LORA_RANK_OUTPUT      = "training.lora.rank.output";
static const char * LLM_KV_TRAINING_LORA_RANK_ATTN_NORM   = "training.lora.rank.attn_norm";
static const char * LLM_KV_TRAINING_LORA_RANK_ATTN_Q      = "training.lora.rank.attn_q";
static const char * LLM_KV_TRAINING_LORA_RANK_ATTN_K      = "training.lora.rank.attn_k";
static const char * LLM_KV_TRAINING_LORA_RANK_ATTN_V      = "training.lora.rank.attn_v";
static const char * LLM_KV_TRAINING_LORA_RANK_ATTN_OUT    = "training.lora.rank.attn_output";
static const char * LLM_KV_TRAINING_LORA_RANK_FFN_NORM    = "training.lora.rank.ffn_norm";
static const char * LLM_KV_TRAINING_LORA_RANK_FFN_GATE    = "training.lora.rank.ffn_gate";
static const char * LLM_KV_TRAINING_LORA_RANK_FFN_DOWN    = "training.lora.rank.ffn_down";
static const char * LLM_KV_TRAINING_LORA_RANK_FFN_UP      = "training.lora.rank.ffn_up";

// gguf constants (sync with gguf.py)

static const char * LLM_KV_GENERAL_ARCHITECTURE        = "general.architecture";
static const char * LLM_KV_GENERAL_FILE_TYPE           = "general.file_type";

static const char * LLM_KV_CONTEXT_LENGTH              = "%s.context_length";
static const char * LLM_KV_EMBEDDING_LENGTH            = "%s.embedding_length";
static const char * LLM_KV_BLOCK_COUNT                 = "%s.block_count";
static const char * LLM_KV_FEED_FORWARD_LENGTH         = "%s.feed_forward_length";
static const char * LLM_KV_ATTENTION_HEAD_COUNT        = "%s.attention.head_count";
static const char * LLM_KV_ATTENTION_HEAD_COUNT_KV     = "%s.attention.head_count_kv";
static const char * LLM_KV_ATTENTION_LAYERNORM_RMS_EPS = "%s.attention.layer_norm_rms_epsilon";
static const char * LLM_KV_ROPE_DIMENSION_COUNT        = "%s.rope.dimension_count";
static const char * LLM_KV_ROPE_FREQ_BASE              = "%s.rope.freq_base"; // TODO load in llama.cpp
static const char * LLM_KV_ROPE_SCALE_LINEAR           = "%s.rope.scale_linear";

static const char * LLM_TENSOR_TOKEN_EMBD    = "token_embd";
static const char * LLM_TENSOR_OUTPUT_NORM   = "output_norm";
static const char * LLM_TENSOR_OUTPUT        = "output";
static const char * LLM_TENSOR_ATTN_NORM     = "blk.%d.attn_norm";
static const char * LLM_TENSOR_ATTN_Q        = "blk.%d.attn_q";
static const char * LLM_TENSOR_ATTN_K        = "blk.%d.attn_k";
static const char * LLM_TENSOR_ATTN_V        = "blk.%d.attn_v";
static const char * LLM_TENSOR_ATTN_OUT      = "blk.%d.attn_output";
static const char * LLM_TENSOR_FFN_NORM      = "blk.%d.ffn_norm";
static const char * LLM_TENSOR_FFN_GATE      = "blk.%d.ffn_gate";
static const char * LLM_TENSOR_FFN_DOWN      = "blk.%d.ffn_down";
static const char * LLM_TENSOR_FFN_UP        = "blk.%d.ffn_up";

static void print_params(struct my_llama_hparams * params) {
    printf("%s: n_vocab               : %u\n", __func__, params->n_vocab);
    printf("%s: n_ctx                 : %u\n", __func__, params->n_ctx);
    printf("%s: n_embd                : %u\n", __func__, params->n_embd);
    printf("%s: n_ff                  : %u\n", __func__, params->n_ff);
    printf("%s: n_head                : %u\n", __func__, params->n_head);
    printf("%s: n_head_kv             : %u\n", __func__, params->n_head_kv);
    printf("%s: n_layer               : %u\n", __func__, params->n_layer);
    printf("%s: norm_rms_eps          : %f\n", __func__, params->f_norm_rms_eps);
    printf("%s: rope_freq_base        : %f\n", __func__, params->rope_freq_base);
    printf("%s: rope_freq_scale       : %f\n", __func__, params->rope_freq_scale);
}

static void print_lora_params(struct my_llama_lora_hparams * params) {
    printf("%s: n_rank_attention_norm : %u\n", __func__, params->n_rank_attention_norm);
    printf("%s: n_rank_wq             : %u\n", __func__, params->n_rank_wq);
    printf("%s: n_rank_wk             : %u\n", __func__, params->n_rank_wk);
    printf("%s: n_rank_wv             : %u\n", __func__, params->n_rank_wv);
    printf("%s: n_rank_wo             : %u\n", __func__, params->n_rank_wo);
    printf("%s: n_rank_ffn_norm       : %u\n", __func__, params->n_rank_ffn_norm);
    printf("%s: n_rank_w1             : %u\n", __func__, params->n_rank_w1);
    printf("%s: n_rank_w2             : %u\n", __func__, params->n_rank_w2);
    printf("%s: n_rank_w3             : %u\n", __func__, params->n_rank_w3);
    printf("%s: n_rank_tok_embeddings : %u\n", __func__, params->n_rank_tok_embeddings);
    printf("%s: n_rank_norm           : %u\n", __func__, params->n_rank_norm);
    printf("%s: n_rank_output         : %u\n", __func__, params->n_rank_output);
}

#define GGUF_GET_KEY(ctx, dst, func, type, req, key) \
{ \
    const std::string skey(key); \
    const int kid = gguf_find_key(ctx, skey.c_str()); \
    if (kid >= 0) { \
        enum gguf_type ktype = gguf_get_kv_type(ctx, kid); \
        if (ktype != (type)) { \
            die_fmt("key %s has wrong type: %s", skey.c_str(), gguf_type_name(ktype)); \
        } \
        (dst) = func(ctx, kid); \
    } else if (req) { \
        die_fmt("key not found in model: %s", skey.c_str()); \
    } \
}

static void load_model_hparams_gguf(struct gguf_context * ctx, struct my_llama_hparams * hparams, const char * expected_arch) {
    std::string arch;

    GGUF_GET_KEY(ctx, arch, gguf_get_val_str, GGUF_TYPE_STRING, true, LLM_KV_GENERAL_ARCHITECTURE);
    if (expected_arch != NULL) {
        if (arch != expected_arch) {
            printf("%s: arch=%s expected_arch=%s\n", __func__, arch.c_str(), expected_arch);
        }
        GGML_ASSERT(arch == expected_arch);
    }

    std::vector<char> keybuf;
    keybuf.resize(512);
    auto kv = [&arch, &keybuf](const char * key) -> const char * {
        snprintf(keybuf.data(), keybuf.size(), key, arch.c_str());
        return keybuf.data();
    };

    GGUF_GET_KEY(ctx, hparams->n_embd,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_EMBEDDING_LENGTH));
    GGUF_GET_KEY(ctx, hparams->n_ctx,          gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_CONTEXT_LENGTH));
    GGUF_GET_KEY(ctx, hparams->n_ff,           gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_FEED_FORWARD_LENGTH));
    GGUF_GET_KEY(ctx, hparams->n_head,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_ATTENTION_HEAD_COUNT));
    GGUF_GET_KEY(ctx, hparams->n_layer,        gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_BLOCK_COUNT));

    // n_head_kv is optional, default to n_head
    hparams->n_head_kv = hparams->n_head;
    GGUF_GET_KEY(ctx, hparams->n_head_kv,      gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_ATTENTION_HEAD_COUNT_KV));

    float rope_freq_scale = 1.0f;
    GGUF_GET_KEY(ctx, hparams->f_norm_rms_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS));
    GGUF_GET_KEY(ctx, hparams->rope_freq_base, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_FREQ_BASE));
    GGUF_GET_KEY(ctx, rope_freq_scale,         gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_SCALE_LINEAR));
    if (rope_freq_scale != 1.0f) {
        hparams->rope_freq_scale = 1.0f / rope_freq_scale;
    }
}

static void init_model(struct llama_model * input, struct my_llama_model * model, const char * fn_model, uint32_t n_ctx) {
    auto & hparams = model->hparams;

    std::vector<char> tn_buf;
    tn_buf.resize(GGML_MAX_NAME);
    auto tn = [&tn_buf](const char * key) -> const char * {
        snprintf(tn_buf.data(), tn_buf.size(), "%s.weight", key);
        return tn_buf.data();
    };
    auto tni = [&tn_buf](const char * key, int bid) -> const char * {
        snprintf(tn_buf.data(), tn_buf.size(), key, bid);
        std::string s = tn_buf.data();
        snprintf(tn_buf.data(), tn_buf.size(), "%s.weight", s.c_str());
        return tn_buf.data();
    };


    // get parameters directly from gguf file
    {
        struct gguf_init_params params = {
            /*.no_alloc = */ false,
            /*.ctx      = */ NULL,
        };
        struct gguf_context * mctx = gguf_init_from_file(fn_model, params);

        load_model_hparams_gguf(mctx, &hparams, "llama");

        gguf_free(mctx);
    }
    hparams.n_vocab = llama_n_vocab(input);
    hparams.n_ctx = n_ctx;

    // get tensors from llama_model (possibly mmapped)
    model->tok_embeddings = llama_get_model_tensor(input, tn(LLM_TENSOR_TOKEN_EMBD));
    model->norm           = llama_get_model_tensor(input, tn(LLM_TENSOR_OUTPUT_NORM));
    model->output         = llama_get_model_tensor(input, tn(LLM_TENSOR_OUTPUT));

    assert_shape_2d(model->tok_embeddings, hparams.n_embd, hparams.n_vocab);
    assert_shape_1d(model->norm,           hparams.n_embd);
    assert_shape_2d(model->output,         hparams.n_embd, hparams.n_vocab);

    model->layers.resize(hparams.n_layer);
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        auto & layer = model->layers[i];

        layer.attention_norm = llama_get_model_tensor(input, tni(LLM_TENSOR_ATTN_NORM, i));
        layer.wq             = llama_get_model_tensor(input, tni(LLM_TENSOR_ATTN_Q, i));
        layer.wk             = llama_get_model_tensor(input, tni(LLM_TENSOR_ATTN_K, i));
        layer.wv             = llama_get_model_tensor(input, tni(LLM_TENSOR_ATTN_V, i));
        layer.wo             = llama_get_model_tensor(input, tni(LLM_TENSOR_ATTN_OUT, i));
        layer.ffn_norm       = llama_get_model_tensor(input, tni(LLM_TENSOR_FFN_NORM, i));
        layer.w1             = llama_get_model_tensor(input, tni(LLM_TENSOR_FFN_GATE, i));
        layer.w2             = llama_get_model_tensor(input, tni(LLM_TENSOR_FFN_DOWN, i));
        layer.w3             = llama_get_model_tensor(input, tni(LLM_TENSOR_FFN_UP, i));

        assert_shape_1d(layer.attention_norm, hparams.n_embd);
        assert_shape_2d(layer.wq,             hparams.n_embd, hparams.n_embd);
        assert_shape_2d(layer.wk,             hparams.n_embd, hparams.n_embd_gqa());
        assert_shape_2d(layer.wv,             hparams.n_embd, hparams.n_embd_gqa());
        assert_shape_2d(layer.wo,             hparams.n_embd, hparams.n_embd);
        assert_shape_1d(layer.ffn_norm,       hparams.n_embd);
        assert_shape_2d(layer.w1,             hparams.n_embd, hparams.n_ff);
        assert_shape_2d(layer.w2,             hparams.n_ff,   hparams.n_embd);
        assert_shape_2d(layer.w3,             hparams.n_embd, hparams.n_ff);
    }
}

static void set_param_lora(struct my_llama_lora * lora) {
    const uint32_t n_layer = lora->layers.size();

    struct ggml_context* ctx = lora->ctx;

    ggml_set_param(ctx, lora->tok_embeddings_a);
    ggml_set_param(ctx, lora->tok_embeddings_b);
    ggml_set_param(ctx, lora->norm_a);
    ggml_set_param(ctx, lora->norm_b);
    ggml_set_param(ctx, lora->output_a);
    ggml_set_param(ctx, lora->output_b);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = lora->layers[i];

        ggml_set_param(ctx, layer.attention_norm_a);
        ggml_set_param(ctx, layer.attention_norm_b);
        ggml_set_param(ctx, layer.wq_a);
        ggml_set_param(ctx, layer.wq_b);
        ggml_set_param(ctx, layer.wk_a);
        ggml_set_param(ctx, layer.wk_b);
        ggml_set_param(ctx, layer.wv_a);
        ggml_set_param(ctx, layer.wv_b);
        ggml_set_param(ctx, layer.wo_a);
        ggml_set_param(ctx, layer.wo_b);
        ggml_set_param(ctx, layer.ffn_norm_a);
        ggml_set_param(ctx, layer.ffn_norm_b);
        ggml_set_param(ctx, layer.w1_a);
        ggml_set_param(ctx, layer.w1_b);
        ggml_set_param(ctx, layer.w2_a);
        ggml_set_param(ctx, layer.w2_b);
        ggml_set_param(ctx, layer.w3_a);
        ggml_set_param(ctx, layer.w3_b);
    }
}

static void alloc_lora(struct ggml_allocr * alloc, struct my_llama_lora * lora) {
    ggml_allocr_alloc(alloc, lora->tok_embeddings_a);
    ggml_allocr_alloc(alloc, lora->tok_embeddings_b);
    ggml_allocr_alloc(alloc, lora->norm_a);
    ggml_allocr_alloc(alloc, lora->norm_b);
    ggml_allocr_alloc(alloc, lora->output_a);
    ggml_allocr_alloc(alloc, lora->output_b);
    for (uint32_t i = 0; i < lora->layers.size(); ++i) {
        auto & layer = lora->layers[i];
        ggml_allocr_alloc(alloc, layer.attention_norm_a);
        ggml_allocr_alloc(alloc, layer.attention_norm_b);
        ggml_allocr_alloc(alloc, layer.wq_a);
        ggml_allocr_alloc(alloc, layer.wq_b);
        ggml_allocr_alloc(alloc, layer.wk_a);
        ggml_allocr_alloc(alloc, layer.wk_b);
        ggml_allocr_alloc(alloc, layer.wv_a);
        ggml_allocr_alloc(alloc, layer.wv_b);
        ggml_allocr_alloc(alloc, layer.wo_a);
        ggml_allocr_alloc(alloc, layer.wo_b);
        ggml_allocr_alloc(alloc, layer.ffn_norm_a);
        ggml_allocr_alloc(alloc, layer.ffn_norm_b);
        ggml_allocr_alloc(alloc, layer.w1_a);
        ggml_allocr_alloc(alloc, layer.w1_b);
        ggml_allocr_alloc(alloc, layer.w2_a);
        ggml_allocr_alloc(alloc, layer.w2_b);
        ggml_allocr_alloc(alloc, layer.w3_a);
        ggml_allocr_alloc(alloc, layer.w3_b);
    }
    ggml_allocr_alloc(alloc, lora->tok_embeddings_a->grad);
    ggml_allocr_alloc(alloc, lora->tok_embeddings_b->grad);
    ggml_allocr_alloc(alloc, lora->norm_a->grad);
    ggml_allocr_alloc(alloc, lora->norm_b->grad);
    ggml_allocr_alloc(alloc, lora->output_a->grad);
    ggml_allocr_alloc(alloc, lora->output_b->grad);
    for (uint32_t i = 0; i < lora->layers.size(); ++i) {
        auto & layer = lora->layers[i];
        ggml_allocr_alloc(alloc, layer.attention_norm_a->grad);
        ggml_allocr_alloc(alloc, layer.attention_norm_b->grad);
        ggml_allocr_alloc(alloc, layer.wq_a->grad);
        ggml_allocr_alloc(alloc, layer.wq_b->grad);
        ggml_allocr_alloc(alloc, layer.wk_a->grad);
        ggml_allocr_alloc(alloc, layer.wk_b->grad);
        ggml_allocr_alloc(alloc, layer.wv_a->grad);
        ggml_allocr_alloc(alloc, layer.wv_b->grad);
        ggml_allocr_alloc(alloc, layer.wo_a->grad);
        ggml_allocr_alloc(alloc, layer.wo_b->grad);
        ggml_allocr_alloc(alloc, layer.ffn_norm_a->grad);
        ggml_allocr_alloc(alloc, layer.ffn_norm_b->grad);
        ggml_allocr_alloc(alloc, layer.w1_a->grad);
        ggml_allocr_alloc(alloc, layer.w1_b->grad);
        ggml_allocr_alloc(alloc, layer.w2_a->grad);
        ggml_allocr_alloc(alloc, layer.w2_b->grad);
        ggml_allocr_alloc(alloc, layer.w3_a->grad);
        ggml_allocr_alloc(alloc, layer.w3_b->grad);
    }
}

static void init_lora(const struct my_llama_model * model, struct my_llama_lora * lora) {
    const auto & lparams = lora->hparams;

    const uint32_t n_embd     = model->hparams.n_embd;
    const uint32_t n_embd_gqa = model->hparams.n_embd_gqa();
    const uint32_t n_layer    = model->hparams.n_layer;
    const uint32_t n_vocab    = model->hparams.n_vocab;
    const uint32_t n_ff       = model->hparams.n_ff;

    std::vector<char> tn_buf;
    tn_buf.resize(GGML_MAX_NAME);
    auto tn = [&tn_buf](const char * key, const char * suffix) -> const char * {
        snprintf(tn_buf.data(), tn_buf.size(), "%s%s", key, suffix);
        return tn_buf.data();
    };
    auto tni = [&tn_buf](const char * key, const char * suffix, int bid) -> const char * {
        snprintf(tn_buf.data(), tn_buf.size(), key, bid);
        std::string s = tn_buf.data();
        snprintf(tn_buf.data(), tn_buf.size(), "%s%s", s.c_str(), suffix);
        return tn_buf.data();
    };

    // context for lora tensors without their data
    struct ggml_init_params ctx_lora_params;
    ctx_lora_params.mem_size   = ggml_tensor_overhead()*2*(6 + n_layer*18);
    ctx_lora_params.mem_buffer = NULL;
    ctx_lora_params.no_alloc   = true;

    struct ggml_context * ctx = ggml_init(ctx_lora_params);
    lora->ctx = ctx;

    lora->tok_embeddings_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_tok_embeddings, n_embd);
    lora->tok_embeddings_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_tok_embeddings, n_vocab);
    lora->norm_a           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_norm, n_embd);
    lora->norm_b           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_norm, 1);
    lora->output_a         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_output, n_embd);
    lora->output_b         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_output, n_vocab);

    ggml_set_name(lora->tok_embeddings_a, tn(LLM_TENSOR_TOKEN_EMBD,  ".weight.lora_a"));
    ggml_set_name(lora->tok_embeddings_b, tn(LLM_TENSOR_TOKEN_EMBD,  ".weight.lora_b"));
    ggml_set_name(lora->norm_a,           tn(LLM_TENSOR_OUTPUT_NORM, ".weight.lora_a"));
    ggml_set_name(lora->norm_b,           tn(LLM_TENSOR_OUTPUT_NORM, ".weight.lora_b"));
    ggml_set_name(lora->output_a,         tn(LLM_TENSOR_OUTPUT,      ".weight.lora_a"));
    ggml_set_name(lora->output_b,         tn(LLM_TENSOR_OUTPUT,      ".weight.lora_b"));

    lora->layers.resize(n_layer);
    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = lora->layers[i];

        layer.attention_norm_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_attention_norm, n_embd);
        layer.attention_norm_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_attention_norm, 1);

        layer.wq_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wq, n_embd);
        layer.wq_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wq, n_embd);
        layer.wk_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wk, n_embd);
        layer.wk_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wk, n_embd_gqa);
        layer.wv_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wv, n_embd);
        layer.wv_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wv, n_embd_gqa);
        layer.wo_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wo, n_embd);
        layer.wo_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_wo, n_embd);

        layer.ffn_norm_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_ffn_norm, n_embd);
        layer.ffn_norm_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_ffn_norm, 1);

        layer.w1_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w1, n_embd);
        layer.w1_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w1, n_ff);
        layer.w2_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w2, n_ff);
        layer.w2_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w2, n_embd);
        layer.w3_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w3, n_embd);
        layer.w3_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lparams.n_rank_w3, n_ff);

        ggml_set_name(layer.attention_norm_a, tni(LLM_TENSOR_ATTN_NORM, ".weight.lora_a", i));
        ggml_set_name(layer.attention_norm_b, tni(LLM_TENSOR_ATTN_NORM, ".weight.lora_b", i));
        ggml_set_name(layer.wq_a,             tni(LLM_TENSOR_ATTN_Q,    ".weight.lora_a", i));
        ggml_set_name(layer.wq_b,             tni(LLM_TENSOR_ATTN_Q,    ".weight.lora_b", i));
        ggml_set_name(layer.wk_a,             tni(LLM_TENSOR_ATTN_K,    ".weight.lora_a", i));
        ggml_set_name(layer.wk_b,             tni(LLM_TENSOR_ATTN_K,    ".weight.lora_b", i));
        ggml_set_name(layer.wv_a,             tni(LLM_TENSOR_ATTN_V,    ".weight.lora_a", i));
        ggml_set_name(layer.wv_b,             tni(LLM_TENSOR_ATTN_V,    ".weight.lora_b", i));
        ggml_set_name(layer.wo_a,             tni(LLM_TENSOR_ATTN_OUT,  ".weight.lora_a", i));
        ggml_set_name(layer.wo_b,             tni(LLM_TENSOR_ATTN_OUT,  ".weight.lora_b", i));
        ggml_set_name(layer.ffn_norm_a,       tni(LLM_TENSOR_FFN_NORM,  ".weight.lora_a", i));
        ggml_set_name(layer.ffn_norm_b,       tni(LLM_TENSOR_FFN_NORM,  ".weight.lora_b", i));
        ggml_set_name(layer.w1_a,             tni(LLM_TENSOR_FFN_GATE,  ".weight.lora_a", i));
        ggml_set_name(layer.w1_b,             tni(LLM_TENSOR_FFN_GATE,  ".weight.lora_b", i));
        ggml_set_name(layer.w2_a,             tni(LLM_TENSOR_FFN_DOWN,  ".weight.lora_a", i));
        ggml_set_name(layer.w2_b,             tni(LLM_TENSOR_FFN_DOWN,  ".weight.lora_b", i));
        ggml_set_name(layer.w3_a,             tni(LLM_TENSOR_FFN_UP,    ".weight.lora_a", i));
        ggml_set_name(layer.w3_b,             tni(LLM_TENSOR_FFN_UP,    ".weight.lora_b", i));
    }

    set_param_lora(lora);

    // measure data size
    size_t size = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size += GGML_PAD(ggml_nbytes(t), tensor_alignment);
    }

    // allocate data
    struct ggml_allocr * alloc = NULL;
    lora->data.resize(size + tensor_alignment);
    alloc = ggml_allocr_new(lora->data.data(), lora->data.size(), tensor_alignment);
    alloc_lora(alloc, lora);
    ggml_allocr_free(alloc);
}

static void randomize_lora(struct my_llama_lora * lora, int seed, float mean, float std, float min, float max) {
    const uint32_t n_layer = lora->layers.size();

    struct random_normal_distribution * rnd = init_random_normal_distribution(seed, mean, std, min, max);

    randomize_tensor_normal(lora->tok_embeddings_a, rnd);
    ggml_set_zero(lora->tok_embeddings_b);
    randomize_tensor_normal(lora->norm_a,           rnd);
    ggml_set_zero(lora->norm_b);
    randomize_tensor_normal(lora->output_a,         rnd);
    ggml_set_zero(lora->output_b);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = lora->layers[i];
        randomize_tensor_normal(layer.attention_norm_a, rnd);
        ggml_set_zero(layer.attention_norm_b);

        randomize_tensor_normal(layer.wq_a, rnd);
        ggml_set_zero(layer.wq_b);
        randomize_tensor_normal(layer.wk_a, rnd);
        ggml_set_zero(layer.wk_b);
        randomize_tensor_normal(layer.wv_a, rnd);
        ggml_set_zero(layer.wv_b);
        randomize_tensor_normal(layer.wo_a, rnd);
        ggml_set_zero(layer.wo_b);

        randomize_tensor_normal(layer.ffn_norm_a, rnd);
        ggml_set_zero(layer.ffn_norm_b);

        randomize_tensor_normal(layer.w1_a, rnd);
        ggml_set_zero(layer.w1_b);
        randomize_tensor_normal(layer.w2_a, rnd);
        ggml_set_zero(layer.w2_b);
        randomize_tensor_normal(layer.w3_a, rnd);
        ggml_set_zero(layer.w3_b);
    }

    free_random_normal_distribution(rnd);
}

static struct ggml_tensor * llama_build_lora_finetune_graphs(
        struct my_llama_model * model,
        struct my_llama_lora  * lora,
        struct ggml_allocr    * alloc,
        struct ggml_context   * ctx,
        struct ggml_cgraph    * gf,
        struct ggml_cgraph    * gb,
        struct ggml_cgraph    * gb_tmp,
        struct ggml_tensor  * * logits,
        struct ggml_tensor    * tokens_input,
        struct ggml_tensor    * targets,
        const  int              n_tokens,
        const  int              n_batch,
        const  bool             enable_flash_attn,
        const  bool             enable_checkpointing) {

    ggml_set_scratch(ctx, { 0, 0, nullptr, });
    const int n_past = 0;
    const int N = n_tokens;
    const auto & hparams  = model->hparams;
    const int n_ctx       = hparams.n_ctx;
    const int n_vocab     = hparams.n_vocab;
    const int n_embd      = hparams.n_embd;
    const int n_layer     = hparams.n_layer;
    const int n_head      = hparams.n_head;
    const int n_head_kv   = hparams.n_head_kv;
    const int n_ff        = hparams.n_ff;
    const int n_rot       = hparams.n_embd_head();
    const int n_embd_head = hparams.n_embd_head();
    const int n_embd_gqa  = hparams.n_embd_gqa();

    const float rms_norm_eps    = hparams.f_norm_rms_eps;
    const float rope_freq_base  = hparams.rope_freq_base;
    const float rope_freq_scale = hparams.rope_freq_scale;

    GGML_ASSERT((size_t) n_layer == lora->layers.size());

    auto set_name = [](struct ggml_tensor * t, const char * n) {
        ggml_set_name(t, n);
        if (t->grad) {
            ggml_format_name(t->grad, "%s->grad", n);
        }
    };

    // KQ_pos - contains the positions
    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_allocr_alloc(alloc, KQ_pos);
    if (!ggml_allocr_is_measure(alloc)) {
        int * data = (int *) KQ_pos->data;
        for (int i = 0; i < N; ++i) {
            data[i] = n_past + i;
        }
    }

    // rope has so much parameters that we make a custom function for it
    auto rope = [ctx, KQ_pos, n_rot, n_ctx, rope_freq_base, rope_freq_scale]
                (struct ggml_tensor * t) -> struct ggml_tensor * {
        // not capturing these, to silcence warnings
        const int rope_mode = 0;

        return ggml_rope_custom(ctx,
            t, KQ_pos, n_rot, rope_mode, n_ctx, 0,
            rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f
        );
    };

    set_name(tokens_input, "tokens_input");
    set_name(targets,      "targets");

    GGML_ASSERT(tokens_input->type == GGML_TYPE_I32);

    auto add_to_f32 = [] (struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
        if (ggml_is_quantized(a->type) || a->type == GGML_TYPE_F16) {
            return ggml_add_cast(ctx, a, b, GGML_TYPE_F32);
        } else if (a->type == GGML_TYPE_F32) {
            return ggml_add(ctx, a, b);
        } else {
            die_fmt("%s: Finetuning on tensors with type '%s' is not yet supported.\n",
                __func__, ggml_type_name(a->type));
        }
    };

    struct ggml_tensor * tok_embeddings = add_to_f32(ctx, model->tok_embeddings, ggml_mul_mat(ctx, lora->tok_embeddings_a, lora->tok_embeddings_b));
    struct ggml_tensor * norm           = add_to_f32(ctx, model->norm, ggml_mul_mat(ctx, lora->norm_a, lora->norm_b));
    struct ggml_tensor * output         = add_to_f32(ctx, model->output, ggml_mul_mat(ctx, lora->output_a, lora->output_b));

    struct ggml_tensor * t00 = ggml_reshape_1d(ctx, tokens_input, N*n_batch);  set_name(t00, "t00"); assert_shape_1d(t00, N*n_batch);
    struct ggml_tensor * t01 = ggml_get_rows(ctx, tok_embeddings, t00);        set_name(t01, "t01"); assert_shape_2d(t01, n_embd, N*n_batch);

    struct ggml_tensor * cur = t01;

    std::vector<struct ggml_tensor *> checkpoints;
    if (enable_checkpointing) {
        checkpoints.push_back(tokens_input);
        checkpoints.push_back(targets);
        checkpoints.push_back(t00);
        checkpoints.push_back(t01);
    }

    const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);

    for (int il = 0; il < n_layer; ++il) {
        struct my_llama_layer & layer = model->layers[il];
        struct my_llama_lora_layer & llayer = lora->layers[il];

        struct ggml_tensor * attention_norm = add_to_f32(ctx, layer.attention_norm, ggml_mul_mat(ctx, llayer.attention_norm_a, llayer.attention_norm_b));
        struct ggml_tensor * ffn_norm = add_to_f32(ctx, layer.ffn_norm, ggml_mul_mat(ctx, llayer.ffn_norm_a, llayer.ffn_norm_b));
        struct ggml_tensor * wq = add_to_f32(ctx, layer.wq, ggml_mul_mat(ctx, llayer.wq_a, llayer.wq_b));
        struct ggml_tensor * wk = add_to_f32(ctx, layer.wk, ggml_mul_mat(ctx, llayer.wk_a, llayer.wk_b));
        struct ggml_tensor * wv = add_to_f32(ctx, layer.wv, ggml_mul_mat(ctx, llayer.wv_a, llayer.wv_b));
        struct ggml_tensor * wo = add_to_f32(ctx, layer.wo, ggml_mul_mat(ctx, llayer.wo_a, llayer.wo_b));
        struct ggml_tensor * w1 = add_to_f32(ctx, layer.w1, ggml_mul_mat(ctx, llayer.w1_a, llayer.w1_b));
        struct ggml_tensor * w2 = add_to_f32(ctx, layer.w2, ggml_mul_mat(ctx, llayer.w2_a, llayer.w2_b));
        struct ggml_tensor * w3 = add_to_f32(ctx, layer.w3, ggml_mul_mat(ctx, llayer.w3_a, llayer.w3_b));

        struct ggml_tensor * t02 = ggml_rms_norm     (ctx, cur, rms_norm_eps);                       set_name(t02, "t02");     assert_shape_2d(t02, n_embd, N*n_batch);
        struct ggml_tensor * t03 = ggml_repeat       (ctx, attention_norm, t02);                     set_name(t03, "t03");     assert_shape_2d(t03, n_embd, N*n_batch);
        struct ggml_tensor * t04 = ggml_mul          (ctx, t03, t02);                                set_name(t04, "t04");     assert_shape_2d(t04, n_embd, N*n_batch);
        struct ggml_tensor * t05 = ggml_mul_mat      (ctx, wq, t04);                                 set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
        struct ggml_tensor * t06 = ggml_reshape_4d   (ctx, t05, n_embd_head, n_head, N, n_batch);    set_name(t06, "t06");     assert_shape_4d(t06, n_embd_head, n_head, N, n_batch);
        struct ggml_tensor * t07 = rope              (t06);                                          set_name(t07, "t07");     assert_shape_4d(t07, n_embd_head, n_head, N, n_batch);
        struct ggml_tensor * t08 = ggml_mul_mat      (ctx, wk, t04);                                 set_name(t08, "t08");     assert_shape_2d(t08, n_embd_gqa, N*n_batch);
        struct ggml_tensor * t09 = ggml_reshape_4d   (ctx, t08, n_embd_head, n_head_kv, N, n_batch); set_name(t09, "t09");     assert_shape_4d(t09, n_embd_head, n_head_kv, N, n_batch);
        struct ggml_tensor * t10 = rope              (t09);                                          set_name(t10, "t10");     assert_shape_4d(t10, n_embd_head, n_head_kv, N, n_batch);

        struct ggml_tensor * t11;
        if (ggml_is_quantized(wv->type)) {
            struct ggml_tensor * t11_1 = ggml_mul_mat  (ctx, wv, t04);                               set_name(t11_1, "t11_1"); assert_shape_2d(t11_1, n_embd_gqa, N*n_batch);
            struct ggml_tensor * t11_2 = ggml_transpose(ctx, t11_1);                                 set_name(t11_2, "t11_2"); assert_shape_2d(t11_2, N*n_batch, n_embd_gqa);
                                 t11   = ggml_cont     (ctx, t11_2);                                 set_name(t11, "t11");     assert_shape_2d(t11, N*n_batch, n_embd_gqa);
        } else {
                                 t11   = ggml_mul_mat  (ctx, t04, wv);                               set_name(t11, "t11");     assert_shape_2d(t11, N*n_batch, n_embd_gqa);
        }

        struct ggml_tensor * t12 = ggml_reshape_4d   (ctx, t11, N, n_batch, n_embd_head, n_head_kv); set_name(t12, "t12");     assert_shape_4d(t12, N, n_batch, n_embd_head, n_head_kv);
        struct ggml_tensor * t13 = ggml_permute      (ctx, t07, 0, 2, 1, 3);                         set_name(t13, "t13");     assert_shape_4d(t13, n_embd_head, N, n_head, n_batch);
        struct ggml_tensor * t14 = ggml_permute      (ctx, t10, 0, 2, 1, 3);                         set_name(t14, "t14");     assert_shape_4d(t14, n_embd_head, N, n_head_kv, n_batch);
        struct ggml_tensor * t15 = ggml_permute      (ctx, t12, 0, 3, 1, 2);                         set_name(t15, "t15");     assert_shape_4d(t15, N, n_embd_head, n_head_kv, n_batch);
        struct ggml_tensor * t16;
        if (enable_flash_attn) {
            t16 = ggml_flash_attn(ctx, t13, t14, t15, true);                                         set_name(t16, "t16");     assert_shape_4d(t16, n_embd_head, N, n_head, n_batch);
        } else {
            struct ggml_tensor * t16_0 = ggml_mul_mat              (ctx, t14, t13);                  set_name(t16_0, "t16_0"); assert_shape_4d(t16_0, N, N, n_head, n_batch);
            struct ggml_tensor * t16_1 = ggml_scale_inplace        (ctx, t16_0, kv_scale);           set_name(t16_1, "t16_1"); assert_shape_4d(t16_1, N, N, n_head, n_batch);
            struct ggml_tensor * t16_2 = ggml_diag_mask_inf_inplace(ctx, t16_1, n_past);             set_name(t16_2, "t16_2"); assert_shape_4d(t16_2, N, N, n_head, n_batch);
            struct ggml_tensor * t16_3 = ggml_soft_max_inplace     (ctx, t16_2);                     set_name(t16_3, "t16_3"); assert_shape_4d(t16_3, N, N, n_head, n_batch);
            t16 = ggml_mul_mat(ctx, t15, t16_3);                                                     set_name(t16, "t16");     assert_shape_4d(t16, n_embd_head, N, n_head, n_batch);
        }
        struct ggml_tensor * t17 = ggml_permute      (ctx, t16, 0, 2, 1, 3);                         set_name(t17, "t17");     assert_shape_4d(t17, n_embd_head, n_head, N, n_batch);
        struct ggml_tensor * t18 = ggml_cont         (ctx, t17);                                     set_name(t18, "t18");     assert_shape_4d(t18, n_embd_head, n_head, N, n_batch);
        struct ggml_tensor * t19 = ggml_reshape_2d   (ctx, t18, n_embd, N*n_batch);                  set_name(t19, "t19");     assert_shape_2d(t19, n_embd, N*n_batch);
        struct ggml_tensor * t20 = ggml_mul_mat      (ctx, wo, t19);                                 set_name(t20, "t20");     assert_shape_2d(t20, n_embd, N*n_batch);
        struct ggml_tensor * t21 = ggml_add          (ctx, t20, cur);                                set_name(t21, "t21");     assert_shape_2d(t21, n_embd, N*n_batch);
        struct ggml_tensor * t22 = ggml_rms_norm     (ctx, t21, rms_norm_eps);                       set_name(t22, "t22");     assert_shape_2d(t22, n_embd, N*n_batch);
        struct ggml_tensor * t23 = ggml_repeat       (ctx, ffn_norm, t22);                           set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
        struct ggml_tensor * t24 = ggml_mul          (ctx, t23, t22);                                set_name(t24, "t24");     assert_shape_2d(t24, n_embd, N*n_batch);
        struct ggml_tensor * t25 = ggml_mul_mat      (ctx, w3, t24);                                 set_name(t25, "t25");     assert_shape_2d(t25, n_ff, N*n_batch);
        struct ggml_tensor * t26 = ggml_mul_mat      (ctx, w1, t24);                                 set_name(t26, "t26");     assert_shape_2d(t26, n_ff, N*n_batch);
        struct ggml_tensor * t27 = ggml_silu         (ctx, t26);                                     set_name(t27, "t27");     assert_shape_2d(t27, n_ff, N*n_batch);
        struct ggml_tensor * t28 = ggml_mul          (ctx, t27, t25);                                set_name(t28, "t28");     assert_shape_2d(t28, n_ff, N*n_batch);
        struct ggml_tensor * t29 = ggml_mul_mat      (ctx, w2, t28);                                 set_name(t29, "t29");     assert_shape_2d(t29, n_embd, N*n_batch);
        struct ggml_tensor * t30 = ggml_add          (ctx, t29, t21);                                set_name(t30, "t30");     assert_shape_2d(t30, n_embd, N*n_batch);
        cur = t30;
        if (enable_checkpointing) {
            checkpoints.push_back(cur);
        }
    }
    struct ggml_tensor * t31   = ggml_rms_norm          (ctx, cur, rms_norm_eps);                    set_name(t31, "t31");     assert_shape_2d(t31, n_embd, N*n_batch);
    struct ggml_tensor * t32   = ggml_repeat            (ctx, norm, t31);                            set_name(t32, "t32");     assert_shape_2d(t32, n_embd, N*n_batch);
    struct ggml_tensor * t33   = ggml_mul               (ctx, t32, t31);                             set_name(t33, "t33");     assert_shape_2d(t33, n_embd, N*n_batch);
    struct ggml_tensor * t34   = ggml_mul_mat           (ctx, output, t33);                          set_name(t34, "t34");     assert_shape_2d(t34, n_vocab, N*n_batch);
    struct ggml_tensor * t35   = ggml_reshape_3d        (ctx, t34, n_vocab, N, n_batch);             set_name(t35, "t35");     assert_shape_3d(t35, n_vocab, N, n_batch);
    struct ggml_tensor * t36   = ggml_cross_entropy_loss(ctx, t35, targets);                         set_name(t36, "t36");     assert_shape_1d(t36, 1);

    if (enable_checkpointing) {
        checkpoints.push_back(t31);
        checkpoints.push_back(t32);
        checkpoints.push_back(t33);
        checkpoints.push_back(t34);
        checkpoints.push_back(t35);
        checkpoints.push_back(t36);
    }

    ggml_build_forward_expand(gf, t36);

    if (enable_checkpointing) {
        ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints.data(), (int) checkpoints.size());
    } else {
        ggml_graph_cpy(gf, gb);
        ggml_build_backward_expand(ctx, gf, gb, true);
    }

    GGML_ASSERT(alloc != NULL);

    // make sure some tensors are not reallocated by inserting new temporary nodes depending on them
    int n_leafs_before = gb->n_leafs;
    int n_nodes_before = gb->n_nodes;

    // output tensors
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t35, 1.0f));
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36, 1.0f));
    // input gradient
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36->grad, 1.0f));
    GGML_ASSERT(t36->grad->data == NULL && t36->grad->view_src == NULL);
    ggml_allocr_alloc(alloc, t36->grad);
    // KQ_pos
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, KQ_pos, 1.0f));

    // make sure base model tensors data cannot be used in viewable operations
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, model->tok_embeddings, 1.0f));
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, model->norm, 1.0f));
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, model->output, 1.0f));
    for (int il = 0; il < n_layer; ++il) {
        struct my_llama_layer & layer = model->layers[il];
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.attention_norm, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.ffn_norm, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wq, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wk, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wv, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wo, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.w1, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.w2, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.w3, 1.0f));
    }

    // allocating checkpoints in one block to reduce memory fragmentation
    // note: they will be freed in reverse order
    for (unsigned int i = 0; i < checkpoints.size(); ++i) {
        if (checkpoints[i]->data == NULL && checkpoints[i]->view_src == NULL) {
            ggml_allocr_alloc(alloc, checkpoints[i]);
        }
    }

    ggml_allocr_alloc_graph(alloc, gb);

    // remove the additional nodes and leafs
    for (int i = n_leafs_before; i < gb->n_leafs; ++i) {
        gb->leafs[i] = NULL;
    }
    for (int i = n_nodes_before; i < gb->n_nodes; ++i) {
        gb->nodes[i] = NULL;
    }
    gb->n_leafs = n_leafs_before;
    gb->n_nodes = n_nodes_before;

    *logits = t35;
    return t36;
}

static void load_llama_lora_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct my_llama_model * model, struct my_llama_lora * lora) {
    // NOTE: gguf_context must be initialized with f_ggml_ctx and no_alloc=false, otherwise tensor data can not be read

    std::string arch;

    std::vector<char> keybuf;
    keybuf.resize(512);

    GGUF_GET_KEY(fctx, arch, gguf_get_val_str, GGUF_TYPE_STRING, true, LLM_KV_GENERAL_ARCHITECTURE);
    GGML_ASSERT(arch == "llama");

    uint32_t ftype_u;
    GGUF_GET_KEY(fctx, ftype_u, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_GENERAL_FILE_TYPE);
    GGML_ASSERT((enum llama_ftype) ftype_u == LLAMA_FTYPE_ALL_F32);

    struct my_llama_hparams hparams;
    load_model_hparams_gguf(fctx, &hparams, arch.c_str());

    // parameters that define tensor shapes must match
    GGML_ASSERT(hparams.n_embd    == model->hparams.n_embd);
    GGML_ASSERT(hparams.n_ff      == model->hparams.n_ff);
    GGML_ASSERT(hparams.n_head    == model->hparams.n_head);
    GGML_ASSERT(hparams.n_head_kv == model->hparams.n_head_kv);
    GGML_ASSERT(hparams.n_layer   == model->hparams.n_layer);

    GGUF_GET_KEY(fctx, lora->hparams.n_rank_tok_embeddings, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_TOKEN_EMBD);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_norm,           gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_OUTPUT_NORM);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_output,         gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_OUTPUT);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_attention_norm, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_ATTN_NORM);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_wq,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_ATTN_Q);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_wk,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_ATTN_K);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_wv,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_ATTN_V);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_wo,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_ATTN_OUT);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_ffn_norm,       gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_FFN_NORM);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_w1,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_FFN_GATE);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_w2,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_FFN_DOWN);
    GGUF_GET_KEY(fctx, lora->hparams.n_rank_w3,             gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_LORA_RANK_FFN_UP);

    init_lora(model, lora);

    copy_tensor_by_name(lora->tok_embeddings_a, f_ggml_ctx, ggml_get_name(lora->tok_embeddings_a));
    copy_tensor_by_name(lora->tok_embeddings_b, f_ggml_ctx, ggml_get_name(lora->tok_embeddings_b));
    copy_tensor_by_name(lora->norm_a,           f_ggml_ctx, ggml_get_name(lora->norm_a));
    copy_tensor_by_name(lora->norm_b,           f_ggml_ctx, ggml_get_name(lora->norm_b));
    copy_tensor_by_name(lora->output_a,         f_ggml_ctx, ggml_get_name(lora->output_a));
    copy_tensor_by_name(lora->output_b,         f_ggml_ctx, ggml_get_name(lora->output_b));

    for (uint32_t i = 0; i < lora->layers.size(); ++i) {
        auto & layer = lora->layers[i];
        copy_tensor_by_name(layer.attention_norm_a, f_ggml_ctx, ggml_get_name(layer.attention_norm_a));
        copy_tensor_by_name(layer.attention_norm_b, f_ggml_ctx, ggml_get_name(layer.attention_norm_b));
        copy_tensor_by_name(layer.wq_a,             f_ggml_ctx, ggml_get_name(layer.wq_a));
        copy_tensor_by_name(layer.wq_b,             f_ggml_ctx, ggml_get_name(layer.wq_b));
        copy_tensor_by_name(layer.wk_a,             f_ggml_ctx, ggml_get_name(layer.wk_a));
        copy_tensor_by_name(layer.wk_b,             f_ggml_ctx, ggml_get_name(layer.wk_b));
        copy_tensor_by_name(layer.wv_a,             f_ggml_ctx, ggml_get_name(layer.wv_a));
        copy_tensor_by_name(layer.wv_b,             f_ggml_ctx, ggml_get_name(layer.wv_b));
        copy_tensor_by_name(layer.wo_a,             f_ggml_ctx, ggml_get_name(layer.wo_a));
        copy_tensor_by_name(layer.wo_b,             f_ggml_ctx, ggml_get_name(layer.wo_b));
        copy_tensor_by_name(layer.ffn_norm_a,       f_ggml_ctx, ggml_get_name(layer.ffn_norm_a));
        copy_tensor_by_name(layer.ffn_norm_b,       f_ggml_ctx, ggml_get_name(layer.ffn_norm_b));
        copy_tensor_by_name(layer.w1_a,             f_ggml_ctx, ggml_get_name(layer.w1_a));
        copy_tensor_by_name(layer.w1_b,             f_ggml_ctx, ggml_get_name(layer.w1_b));
        copy_tensor_by_name(layer.w2_a,             f_ggml_ctx, ggml_get_name(layer.w2_a));
        copy_tensor_by_name(layer.w2_b,             f_ggml_ctx, ggml_get_name(layer.w2_b));
        copy_tensor_by_name(layer.w3_a,             f_ggml_ctx, ggml_get_name(layer.w3_a));
        copy_tensor_by_name(layer.w3_b,             f_ggml_ctx, ggml_get_name(layer.w3_b));
    }
}

static void save_llama_lora_gguf(struct gguf_context * fctx, struct my_llama_model * model, struct my_llama_lora * lora) {
    const char * arch = "llama";
    enum llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::vector<char> keybuf;
    keybuf.resize(512);
    auto kv = [arch, &keybuf](const char * key) -> const char * {
        snprintf(keybuf.data(), keybuf.size(), key, arch);
        return keybuf.data();
    };

    gguf_set_val_str(fctx, LLM_KV_GENERAL_ARCHITECTURE, arch);
    gguf_set_val_u32(fctx, LLM_KV_GENERAL_FILE_TYPE, ftype);

    gguf_set_val_u32(fctx, kv(LLM_KV_CONTEXT_LENGTH),              model->hparams.n_ctx);
    gguf_set_val_u32(fctx, kv(LLM_KV_EMBEDDING_LENGTH),            model->hparams.n_embd);
    gguf_set_val_u32(fctx, kv(LLM_KV_FEED_FORWARD_LENGTH),         model->hparams.n_ff);
    gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT),        model->hparams.n_head);
    gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT_KV),     model->hparams.n_head_kv);
    gguf_set_val_u32(fctx, kv(LLM_KV_BLOCK_COUNT),                 model->hparams.n_layer);
    gguf_set_val_u32(fctx, kv(LLM_KV_ROPE_DIMENSION_COUNT),        model->hparams.n_embd_head());
    gguf_set_val_f32(fctx, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS), model->hparams.f_norm_rms_eps);
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_FREQ_BASE),              model->hparams.rope_freq_base);
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_SCALE_LINEAR),           model->hparams.rope_freq_scale);

    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_TOKEN_EMBD,   lora->hparams.n_rank_tok_embeddings);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_OUTPUT_NORM,  lora->hparams.n_rank_norm);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_OUTPUT,       lora->hparams.n_rank_output);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_ATTN_NORM,    lora->hparams.n_rank_attention_norm);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_ATTN_Q,       lora->hparams.n_rank_wq);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_ATTN_K,       lora->hparams.n_rank_wk);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_ATTN_V,       lora->hparams.n_rank_wv);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_ATTN_OUT,     lora->hparams.n_rank_wo);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_FFN_NORM,     lora->hparams.n_rank_ffn_norm);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_FFN_GATE,     lora->hparams.n_rank_w1);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_FFN_DOWN,     lora->hparams.n_rank_w2);
    gguf_set_val_u32(fctx, LLM_KV_TRAINING_LORA_RANK_FFN_UP,       lora->hparams.n_rank_w3);

    gguf_add_tensor(fctx, lora->tok_embeddings_a);
    gguf_add_tensor(fctx, lora->tok_embeddings_b);
    gguf_add_tensor(fctx, lora->norm_a);
    gguf_add_tensor(fctx, lora->norm_b);
    gguf_add_tensor(fctx, lora->output_a);
    gguf_add_tensor(fctx, lora->output_b);

    for (uint32_t i = 0; i < lora->layers.size(); ++i) {
        auto & layer = lora->layers[i];

        gguf_add_tensor(fctx, layer.attention_norm_a);
        gguf_add_tensor(fctx, layer.attention_norm_b);
        gguf_add_tensor(fctx, layer.wq_a);
        gguf_add_tensor(fctx, layer.wq_b);
        gguf_add_tensor(fctx, layer.wk_a);
        gguf_add_tensor(fctx, layer.wk_b);
        gguf_add_tensor(fctx, layer.wv_a);
        gguf_add_tensor(fctx, layer.wv_b);
        gguf_add_tensor(fctx, layer.wo_a);
        gguf_add_tensor(fctx, layer.wo_b);
        gguf_add_tensor(fctx, layer.ffn_norm_a);
        gguf_add_tensor(fctx, layer.ffn_norm_b);
        gguf_add_tensor(fctx, layer.w1_a);
        gguf_add_tensor(fctx, layer.w1_b);
        gguf_add_tensor(fctx, layer.w2_a);
        gguf_add_tensor(fctx, layer.w2_b);
        gguf_add_tensor(fctx, layer.w3_a);
        gguf_add_tensor(fctx, layer.w3_b);
    }
}

static void load_checkpoint_lora_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct my_llama_model * model, struct my_llama_lora * lora, struct train_state * train) {
    std::string train_type = LLM_KV_TRAINING_TYPE_FINETUNE_LORA;
    GGUF_GET_KEY(fctx, train_type, gguf_get_val_str, GGUF_TYPE_STRING, false, LLM_KV_TRAINING_TYPE);
    GGML_ASSERT(train_type == LLM_KV_TRAINING_TYPE_FINETUNE_LORA);

    load_train_state_gguf(fctx, f_ggml_ctx, train);
    load_llama_lora_gguf(fctx, f_ggml_ctx, model, lora);
}

static void save_checkpoint_lora_gguf(struct gguf_context * fctx, struct my_llama_model * model, struct my_llama_lora * lora, struct train_state * train) {
    gguf_set_val_str(fctx, LLM_KV_TRAINING_TYPE, LLM_KV_TRAINING_TYPE_FINETUNE_LORA);
    save_llama_lora_gguf(fctx, model, lora);
    save_train_state_gguf(fctx, train);
}

static bool load_checkpoint_lora_file(const char * filename, struct my_llama_model * model, struct my_llama_lora * lora, struct train_state * train) {
    struct ggml_context * f_ggml_ctx;
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &f_ggml_ctx;
    struct gguf_context * fctx = gguf_init_from_file(filename, params);
    if (fctx == NULL) {
        return false;
    }

    load_checkpoint_lora_gguf(fctx, f_ggml_ctx, model, lora, train);

    gguf_free(fctx);
    return true;
}

static void save_checkpoint_lora_file(const char * filename, struct my_llama_model * model, struct my_llama_lora * lora, struct train_state * train) {
    printf("%s: saving to %s\n", __func__, filename);
    struct gguf_context * fctx = gguf_init_empty();

    save_checkpoint_lora_gguf(fctx, model, lora, train);

    // write file
    const bool only_meta = false;
    gguf_write_to_file(fctx, filename, only_meta);
    gguf_free(fctx);
}

struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            size = 0;
        } else {
            seek(0, SEEK_END);
            size = tell();
            seek(0, SEEK_SET);
        }
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    void read_raw(void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, size, 1, fp);
        if (ferror(fp)) {
            die_fmt("read error: %s", strerror(errno));
        }
        if (ret != 1) {
            die("unexpectedly reached end of file");
        }
    }

    std::uint32_t read_u32() {
        std::uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    std::string read_string(std::uint32_t len) {
        std::vector<char> chars(len);
        read_raw(chars.data(), len);
        return std::string(chars.data(), len);
    }

    void write_raw(const void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, size, 1, fp);
        if (ret != 1) {
            die_fmt("write error: %s", strerror(errno));
        }
    }

    void write_u32(std::uint32_t val) {
        write_raw(&val, sizeof(val));
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

static void write_tensor(struct llama_file * file, struct ggml_tensor * tensor, const char * name) {
    if (tensor == NULL) {
        file->write_u32(0);
        file->write_u32(0);
        file->write_u32(GGML_TYPE_F32);
        file->seek((0-file->tell()) & 31, SEEK_CUR);
        return;
    }
    if (name == NULL) {
        name = ggml_get_name(tensor);
    }
    uint32_t name_len = strlen(name);
    uint32_t nd = ggml_n_dims(tensor);
    uint32_t ne[4] = { (uint32_t)tensor->ne[0],
                       (uint32_t)tensor->ne[1],
                       (uint32_t)tensor->ne[2],
                       (uint32_t)tensor->ne[3] };
    file->write_u32(nd);
    file->write_u32(name_len);
    file->write_u32(tensor->type);
    file->write_raw(ne, sizeof(ne[0]) * nd);
    file->write_raw(name, name_len);
    file->seek((0-file->tell()) & 31, SEEK_CUR);
    file->write_raw(tensor->data, ggml_nbytes(tensor));
}

static void save_as_llama_lora(const char * filename, struct my_llama_lora * lora) {
    printf("%s: saving to %s\n", __func__, filename);
    struct llama_file file(filename, "wb");
    if (file.fp == NULL) {
        return;
    }

    std::vector<char> tn_buf;
    tn_buf.resize(GGML_MAX_NAME);

    auto tn = [&tn_buf](const char * key, const char * suffix) -> const char * {
        snprintf(tn_buf.data(), tn_buf.size(), "%s%s", key, suffix);
        return tn_buf.data();
    };

    auto tni = [&tn_buf](const char * key, int bid, const char * suffix) -> const char * {
        snprintf(tn_buf.data(), tn_buf.size(), key, bid);
        std::string s = tn_buf.data();
        snprintf(tn_buf.data(), tn_buf.size(), "%s%s", s.c_str(), suffix);
        return tn_buf.data();
    };

    // write_magic
    file.write_u32(LLAMA_FILE_MAGIC_GGLA);   // magic
    file.write_u32(1); // version
    // write_hparams
    file.write_u32(lora->hparams.lora_r);
    file.write_u32(lora->hparams.lora_alpha);
    // write tensors
    write_tensor(&file, lora->tok_embeddings_a, tn(LLM_TENSOR_TOKEN_EMBD,  ".weight.loraA"));
    write_tensor(&file, lora->tok_embeddings_b, tn(LLM_TENSOR_TOKEN_EMBD,  ".weight.loraB"));
    write_tensor(&file, lora->norm_a,           tn(LLM_TENSOR_OUTPUT_NORM, ".weight.loraA"));
    write_tensor(&file, lora->norm_b,           tn(LLM_TENSOR_OUTPUT_NORM, ".weight.loraB"));
    write_tensor(&file, lora->output_a,         tn(LLM_TENSOR_OUTPUT,      ".weight.loraA"));
    write_tensor(&file, lora->output_b,         tn(LLM_TENSOR_OUTPUT,      ".weight.loraB"));
    for (uint32_t i = 0; i < lora->layers.size(); ++i) {
        auto & layer = lora->layers[i];
        write_tensor(&file, layer.attention_norm_a, tni(LLM_TENSOR_ATTN_NORM, i, ".weight.loraA"));
        write_tensor(&file, layer.attention_norm_b, tni(LLM_TENSOR_ATTN_NORM, i, ".weight.loraB"));
        write_tensor(&file, layer.wq_a,             tni(LLM_TENSOR_ATTN_Q,    i, ".weight.loraA"));
        write_tensor(&file, layer.wq_b,             tni(LLM_TENSOR_ATTN_Q,    i, ".weight.loraB"));
        write_tensor(&file, layer.wk_a,             tni(LLM_TENSOR_ATTN_K,    i, ".weight.loraA"));
        write_tensor(&file, layer.wk_b,             tni(LLM_TENSOR_ATTN_K,    i, ".weight.loraB"));
        write_tensor(&file, layer.wv_a,             tni(LLM_TENSOR_ATTN_V,    i, ".weight.loraA"));
        write_tensor(&file, layer.wv_b,             tni(LLM_TENSOR_ATTN_V,    i, ".weight.loraB"));
        write_tensor(&file, layer.wo_a,             tni(LLM_TENSOR_ATTN_OUT,  i, ".weight.loraA"));
        write_tensor(&file, layer.wo_b,             tni(LLM_TENSOR_ATTN_OUT,  i, ".weight.loraB"));
        write_tensor(&file, layer.ffn_norm_a,       tni(LLM_TENSOR_FFN_NORM,  i, ".weight.loraA"));
        write_tensor(&file, layer.ffn_norm_b,       tni(LLM_TENSOR_FFN_NORM,  i, ".weight.loraB"));
        write_tensor(&file, layer.w1_a,             tni(LLM_TENSOR_FFN_GATE,  i, ".weight.loraA"));
        write_tensor(&file, layer.w1_b,             tni(LLM_TENSOR_FFN_GATE,  i, ".weight.loraB"));
        write_tensor(&file, layer.w2_a,             tni(LLM_TENSOR_FFN_DOWN,  i, ".weight.loraA"));
        write_tensor(&file, layer.w2_b,             tni(LLM_TENSOR_FFN_DOWN,  i, ".weight.loraB"));
        write_tensor(&file, layer.w3_a,             tni(LLM_TENSOR_FFN_UP,    i, ".weight.loraA"));
        write_tensor(&file, layer.w3_b,             tni(LLM_TENSOR_FFN_UP,    i, ".weight.loraB"));
    }
}

struct train_params {
    struct train_params_common common;

    const char * fn_model_base;
    const char * fn_lora_out;

    bool only_write_lora;

    float f_norm_rms_eps;
    float rope_freq_base;
    float rope_freq_scale;

    bool custom_f_norm_rms_eps;
    bool custom_rope_freq_base;
    bool custom_rope_freq_scale;

    int32_t lora_r;
    int32_t lora_alpha;
    bool custom_lora_alpha;

    uint32_t n_rank_attention_norm;
    uint32_t n_rank_wq;
    uint32_t n_rank_wk;
    uint32_t n_rank_wv;
    uint32_t n_rank_wo;
    uint32_t n_rank_ffn_norm;
    uint32_t n_rank_w1;
    uint32_t n_rank_w2;
    uint32_t n_rank_w3;
    uint32_t n_rank_tok_embeddings;
    uint32_t n_rank_norm;
    uint32_t n_rank_output;

    bool custom_n_rank_attention_norm;
    bool custom_n_rank_wq;
    bool custom_n_rank_wk;
    bool custom_n_rank_wv;
    bool custom_n_rank_wo;
    bool custom_n_rank_ffn_norm;
    bool custom_n_rank_w1;
    bool custom_n_rank_w2;
    bool custom_n_rank_w3;
    bool custom_n_rank_tok_embeddings;
    bool custom_n_rank_norm;
    bool custom_n_rank_output;
};

static struct train_params get_default_train_params() {
    struct train_params params;
    params.common = get_default_train_params_common();
    params.fn_model_base     = "";
    params.fn_lora_out       = "ggml-lora-ITERATION-f32.gguf";

    params.only_write_lora = false;

    params.f_norm_rms_eps  = 1e-5f;
    params.rope_freq_base  = 10000.0f;
    params.rope_freq_scale = 1.0f;

    params.custom_f_norm_rms_eps  = false;
    params.custom_rope_freq_base  = false;
    params.custom_rope_freq_scale = false;

    params.lora_r      = 4;
    params.lora_alpha  = 4;
    params.custom_lora_alpha = false;

    params.n_rank_attention_norm = 1;
    params.n_rank_wq             = 4;
    params.n_rank_wk             = 4;
    params.n_rank_wv             = 4;
    params.n_rank_wo             = 4;
    params.n_rank_ffn_norm       = 1;
    params.n_rank_w1             = 4;
    params.n_rank_w2             = 4;
    params.n_rank_w3             = 4;
    params.n_rank_tok_embeddings = 4;
    params.n_rank_norm           = 1;
    params.n_rank_output         = 4;

    params.custom_n_rank_attention_norm = false;
    params.custom_n_rank_wq             = false;
    params.custom_n_rank_wk             = false;
    params.custom_n_rank_wv             = false;
    params.custom_n_rank_wo             = false;
    params.custom_n_rank_ffn_norm       = false;
    params.custom_n_rank_w1             = false;
    params.custom_n_rank_w2             = false;
    params.custom_n_rank_w3             = false;
    params.custom_n_rank_tok_embeddings = false;
    params.custom_n_rank_norm           = false;
    params.custom_n_rank_output         = false;

    return params;
}

static void train_print_usage(int argc, char ** argv, const struct train_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                 show this help message and exit\n");

    fprintf(stderr, "  --model-base FNAME         model path from which to load base model (default '%s')\n", params->fn_model_base);
    fprintf(stderr, "  --lora-out FNAME           path to save llama lora (default '%s')\n", params->fn_lora_out);
    fprintf(stderr, "  --only-write-lora          only save llama lora, don't do any training.  use this if you only want to convert a checkpoint to a lora adapter.\n");
    fprintf(stderr, "  --norm-rms-eps F           RMS-Norm epsilon value (default %f)\n", params->f_norm_rms_eps);
    fprintf(stderr, "  --rope-freq-base F         Frequency base for ROPE (default %f)\n", params->rope_freq_base);
    fprintf(stderr, "  --rope-freq-scale F        Frequency scale for ROPE (default %f)\n", params->rope_freq_scale);
    fprintf(stderr, "  --lora-alpha N             LORA alpha : resulting LORA scaling is alpha/r. (default %d)\n", params->lora_alpha);
    fprintf(stderr, "  --lora-r N                 LORA r: default rank. Also specifies resulting scaling together with lora-alpha. (default %d)\n", params->lora_r);
    fprintf(stderr, "  --rank-att-norm N          LORA rank for attention norm tensor, overrides default rank. Norm tensors should generally have rank 1.\n");
    fprintf(stderr, "  --rank-ffn-norm N          LORA rank for feed-forward norm tensor, overrides default rank. Norm tensors should generally have rank 1.\n");
    fprintf(stderr, "  --rank-out-norm N          LORA rank for output norm tensor, overrides default rank. Norm tensors should generally have rank 1.\n");
    fprintf(stderr, "  --rank-tok-embd N          LORA rank for token embeddings tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-out N               LORA rank for output tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-wq N                LORA rank for wq tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-wk N                LORA rank for wk tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-wv N                LORA rank for wv tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-wo N                LORA rank for wo tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-w1 N                LORA rank for w1 tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-w2 N                LORA rank for w2 tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-w3 N                LORA rank for w3 tensor, overrides default rank.\n");

    print_common_train_usage(argc, argv, &params->common);
}

static bool train_params_parse(int argc, char ** argv, struct train_params * params) {
    bool invalid_param = false;
    std::string arg;
    struct train_params default_params = get_default_train_params();
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (consume_common_train_arg(argc, argv, &i, &params->common, &invalid_param)) {
            if (invalid_param) {
                break;
            } else if (params->common.print_usage) {
                train_print_usage(argc, argv, &default_params);
                exit(0);
            }
        } else if (arg == "--model-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_model_base = argv[i];
        } else if (arg == "--lora-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_lora_out = argv[i];
        } else if (arg == "--only-write-lora") {
            params->only_write_lora = true;
        } else if (arg == "--norm-rms-eps") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->f_norm_rms_eps = std::stof(argv[i]);
            params->custom_f_norm_rms_eps = true;
        } else if (arg == "--rope-freq-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->rope_freq_base = std::stof(argv[i]);
            params->custom_rope_freq_base = true;
        } else if (arg == "--rope-freq-scale") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->rope_freq_scale = std::stof(argv[i]);
            params->custom_rope_freq_scale = true;
        } else if (arg == "--lora-alpha") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->lora_alpha = std::stoi(argv[i]);
            params->custom_lora_alpha = true;
        } else if (arg == "--lora-r") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->lora_r = std::stoi(argv[i]);
        } else if (arg == "--rank-att-norm") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_attention_norm = std::stoi(argv[i]);
            params->custom_n_rank_attention_norm = true;
        } else if (arg == "--rank-ffn-norm") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_ffn_norm = std::stoi(argv[i]);
            params->custom_n_rank_ffn_norm = true;
        } else if (arg == "--rank-out-norm") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_norm = std::stoi(argv[i]);
            params->custom_n_rank_norm = true;
        } else if (arg == "--rank-tok-embd") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_tok_embeddings = std::stoi(argv[i]);
            params->custom_n_rank_tok_embeddings = true;
        } else if (arg == "--rank-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_output = std::stoi(argv[i]);
            params->custom_n_rank_output = true;
        } else if (arg == "--rank-wq") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_wq = std::stoi(argv[i]);
            params->custom_n_rank_wq = true;
        } else if (arg == "--rank-wk") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_wk = std::stoi(argv[i]);
            params->custom_n_rank_wk = true;
        } else if (arg == "--rank-wv") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_wv = std::stoi(argv[i]);
            params->custom_n_rank_wv = true;
        } else if (arg == "--rank-wo") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_wo = std::stoi(argv[i]);
            params->custom_n_rank_wo = true;
        } else if (arg == "--rank-w1") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_w1 = std::stoi(argv[i]);
            params->custom_n_rank_w1 = true;
        } else if (arg == "--rank-w2") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_w2 = std::stoi(argv[i]);
            params->custom_n_rank_w2 = true;
        } else if (arg == "--rank-w3") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rank_w3 = std::stoi(argv[i]);
            params->custom_n_rank_w3 = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            train_print_usage(argc, argv, &default_params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        train_print_usage(argc, argv, &default_params);
        exit(1);
    }
    finish_processing_train_args(&params->common);
    return true;
}

struct save_train_files_data {
    const char            * fn_checkpoint_out;
    const char            * fn_lora_out;
    const char            * pattern_fn_it;
    const char            * fn_latest;
    struct my_llama_model * model;
    struct my_llama_lora  * lora;
};

static void save_train_files(void * vdata, struct train_state * train) {
    struct save_train_files_data * data   = (struct save_train_files_data *) vdata;

    int64_t iter = train->opt->iter;

    if (strlen(data->fn_checkpoint_out) > 0) {
        save_checkpoint_lora_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->model, data->lora, train);
        save_checkpoint_lora_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->model, data->lora, train);
    }
    if (strlen(data->fn_lora_out) > 0) {
        save_as_llama_lora(get_train_filename(data->fn_lora_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->lora);
        save_as_llama_lora(get_train_filename(data->fn_lora_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->lora);
    }
}

static int64_t get_parameter_count(struct my_llama_lora* lora) {
    int64_t nx = 0;
    nx += ggml_nelements(lora->tok_embeddings_a);
    nx += ggml_nelements(lora->tok_embeddings_b);
    nx += ggml_nelements(lora->norm_a);
    nx += ggml_nelements(lora->norm_b);
    nx += ggml_nelements(lora->output_a);
    nx += ggml_nelements(lora->output_b);

    for (uint32_t i = 0; i < lora->layers.size(); ++i) {
        auto & layer = lora->layers[i];
        nx += ggml_nelements(layer.attention_norm_a);
        nx += ggml_nelements(layer.attention_norm_b);
        nx += ggml_nelements(layer.wq_a);
        nx += ggml_nelements(layer.wq_b);
        nx += ggml_nelements(layer.wk_a);
        nx += ggml_nelements(layer.wk_b);
        nx += ggml_nelements(layer.wv_a);
        nx += ggml_nelements(layer.wv_b);
        nx += ggml_nelements(layer.wo_a);
        nx += ggml_nelements(layer.wo_b);
        nx += ggml_nelements(layer.ffn_norm_a);
        nx += ggml_nelements(layer.ffn_norm_b);
        nx += ggml_nelements(layer.w1_a);
        nx += ggml_nelements(layer.w1_b);
        nx += ggml_nelements(layer.w2_a);
        nx += ggml_nelements(layer.w2_b);
        nx += ggml_nelements(layer.w3_a);
        nx += ggml_nelements(layer.w3_b);
    }
    return nx;
}

int main(int argc, char ** argv) {
    struct train_params params = get_default_train_params();

    if (!train_params_parse(argc, argv, &params)) {
        return 1;
    }

    if (params.common.seed == LLAMA_DEFAULT_SEED) {
        params.common.seed = time(NULL);
    }
    printf("%s: seed: %u\n", __func__, params.common.seed);
    srand(params.common.seed);

    struct llama_model_params llama_mparams = llama_model_default_params();
    llama_mparams.n_gpu_layers = params.common.n_gpu_layers;
    llama_mparams.vocab_only = false;

    printf("%s: model base = '%s'\n", __func__, params.fn_model_base);
    struct llama_model * lmodel = llama_load_model_from_file(params.fn_model_base, llama_mparams);

    struct llama_context_params llama_cparams = llama_context_default_params();
    struct llama_context * lctx = llama_new_context_with_model(lmodel, llama_cparams);

    struct my_llama_model model;
    init_model(lmodel, &model, params.fn_model_base, params.common.n_ctx);

    struct my_llama_lora lora;

    struct train_state      * train = init_train_state();
    struct ggml_opt_context * opt   = train->opt;

    // set params from command line
    if (params.custom_f_norm_rms_eps) {
        model.hparams.f_norm_rms_eps  = params.f_norm_rms_eps;
    }
    if (params.custom_rope_freq_base) {
        model.hparams.rope_freq_base  = params.rope_freq_base;
    }
    if (params.custom_rope_freq_scale) {
        model.hparams.rope_freq_scale = params.rope_freq_scale;
    }
    lora.hparams.lora_r                = params.lora_r;
    lora.hparams.lora_alpha            = params.custom_lora_alpha            ? params.lora_alpha            : params.lora_r;
    uint32_t n_rank_attention_norm     = params.custom_n_rank_attention_norm ? params.n_rank_attention_norm : 1;
    uint32_t n_rank_wq                 = params.custom_n_rank_wq             ? params.n_rank_wq             : params.lora_r;
    uint32_t n_rank_wk                 = params.custom_n_rank_wk             ? params.n_rank_wk             : params.lora_r;
    uint32_t n_rank_wv                 = params.custom_n_rank_wv             ? params.n_rank_wv             : params.lora_r;
    uint32_t n_rank_wo                 = params.custom_n_rank_wo             ? params.n_rank_wo             : params.lora_r;
    uint32_t n_rank_ffn_norm           = params.custom_n_rank_ffn_norm       ? params.n_rank_ffn_norm       : 1;
    uint32_t n_rank_w1                 = params.custom_n_rank_w1             ? params.n_rank_w1             : params.lora_r;
    uint32_t n_rank_w2                 = params.custom_n_rank_w2             ? params.n_rank_w2             : params.lora_r;
    uint32_t n_rank_w3                 = params.custom_n_rank_w3             ? params.n_rank_w3             : params.lora_r;
    uint32_t n_rank_tok_embeddings     = params.custom_n_rank_tok_embeddings ? params.n_rank_tok_embeddings : params.lora_r;
    uint32_t n_rank_norm               = params.custom_n_rank_norm           ? params.n_rank_norm           : 1;
    uint32_t n_rank_output             = params.custom_n_rank_output         ? params.n_rank_output         : params.lora_r;
    lora.hparams.n_rank_attention_norm = n_rank_attention_norm;
    lora.hparams.n_rank_wq             = n_rank_wq;
    lora.hparams.n_rank_wk             = n_rank_wk;
    lora.hparams.n_rank_wv             = n_rank_wv;
    lora.hparams.n_rank_wo             = n_rank_wo;
    lora.hparams.n_rank_ffn_norm       = n_rank_ffn_norm;
    lora.hparams.n_rank_w1             = n_rank_w1;
    lora.hparams.n_rank_w2             = n_rank_w2;
    lora.hparams.n_rank_w3             = n_rank_w3;
    lora.hparams.n_rank_tok_embeddings = n_rank_tok_embeddings;
    lora.hparams.n_rank_norm           = n_rank_norm;
    lora.hparams.n_rank_output         = n_rank_output;

    // set opt params from command line
    opt->params = ggml_opt_default_params(GGML_OPT_ADAM);
    opt->params.print_forward_graph     = false;
    opt->params.print_backward_graph    = false;
    opt->params.graph_size              = LLAMA_TRAIN_MAX_NODES;
    opt->params.n_threads               = params.common.n_threads;
    opt->params.past                    = params.common.opt_past;
    opt->params.delta                   = params.common.opt_delta;
    opt->params.max_no_improvement      = params.common.opt_max_no_improvement;
    opt->params.n_gradient_accumulation = params.common.n_gradient_accumulation;
    opt->params.adam.n_iter             = params.common.adam_n_iter;
    opt->params.adam.sched              = 1.0f;
    opt->params.adam.alpha              = params.common.adam_alpha;
    opt->params.adam.decay              = params.common.adam_decay;
    opt->params.adam.decay_min_ndim     = params.common.adam_decay_min_ndim;
    opt->params.adam.beta1              = params.common.adam_beta1;
    opt->params.adam.beta2              = params.common.adam_beta2;
    opt->params.adam.gclip              = params.common.adam_gclip;
    opt->params.adam.eps_f              = params.common.adam_eps_f;

    printf("%s: init model\n", __func__);
    bool existed = load_checkpoint_lora_file(params.common.fn_checkpoint_in, &model, &lora, train);

    if (existed) {
        // overwrite last n_ctx with user provided n_ctx
        if (params.common.custom_n_ctx) {
            model.hparams.n_ctx = params.common.n_ctx;
        }

        const bool opt_param_count_changed = (
           (lora.hparams.n_rank_attention_norm != n_rank_attention_norm)
        || (lora.hparams.n_rank_wq             != n_rank_wq)
        || (lora.hparams.n_rank_wk             != n_rank_wk)
        || (lora.hparams.n_rank_wv             != n_rank_wv)
        || (lora.hparams.n_rank_wo             != n_rank_wo)
        || (lora.hparams.n_rank_ffn_norm       != n_rank_ffn_norm)
        || (lora.hparams.n_rank_w1             != n_rank_w1)
        || (lora.hparams.n_rank_w2             != n_rank_w2)
        || (lora.hparams.n_rank_w3             != n_rank_w3)
        || (lora.hparams.n_rank_tok_embeddings != n_rank_tok_embeddings)
        || (lora.hparams.n_rank_norm           != n_rank_norm)
        || (lora.hparams.n_rank_output         != n_rank_output)
        );

        const bool opt_past_changed = opt->params.past != params.common.opt_past;

        if (opt_param_count_changed) {
            print_lora_params(&lora.hparams);
            die("Provided rank differs from checkpoint file. To use different rank start finetune from scratch with empty input checkpoint, e.g --checkpoint-in ''. Aborting.");
            // need to discard previous optimizer gradient statistics and opt_init with new shapes
            // TODO
        }
        if (opt_past_changed) {
            die("Optimizer parameter '--opt-past N' differs from checkpoint file. To use different value finetune from scratch with empty input checkpoint, e.g --checkpoint-in ''. Aborting");
            // need to discard previous optimizer past function value statistics and opt_init with new shapes
            // TODO
        }
    } else { // existed == false
        init_lora(&model, &lora);
        randomize_lora(&lora, params.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
        if (!params.only_write_lora) {
            ggml_opt_init(opt->ctx, opt, opt->params, get_parameter_count(&lora));
        }
    }
    opt->iter = train->train_its;

    print_params(&model.hparams);
    print_lora_params(&lora.hparams);
    printf("%s: total train_iterations %llu\n", __func__, (long long unsigned) train->train_its);
    printf("%s: seen train_samples     %llu\n", __func__, (long long unsigned) train->train_samples);
    printf("%s: seen train_tokens      %llu\n", __func__, (long long unsigned) train->train_tokens);
    printf("%s: completed train_epochs %llu\n", __func__, (long long unsigned) train->train_epochs);
    printf("%s: lora_size = %zu bytes (%.1f MB)\n", __func__, (ggml_used_mem(lora.ctx) + lora.data.size()), (float) (ggml_used_mem(lora.ctx) + lora.data.size()) / (1024.0f*1024.0f));

    if (params.only_write_lora) {
        save_train_files_data save_data;
        save_data.fn_checkpoint_out = "";
        save_data.fn_lora_out       = params.fn_lora_out;
        save_data.pattern_fn_it     = params.common.pattern_fn_it;
        save_data.fn_latest         = params.common.fn_latest;
        save_data.model             = &model;
        save_data.lora              = &lora;

        save_train_files(&save_data, train);

        free_train_state(train);
        ggml_free(lora.ctx);
        llama_free(lctx);
        llama_free_model(lmodel);
        return 0;
    }

    printf("%s: opt_size  = %zu bytes (%.1f MB)\n", __func__, ggml_get_mem_size(opt->ctx), (float) ggml_get_mem_size(opt->ctx) / (1024.0f*1024.0f));
    printf("%s: opt iter %d\n", __func__, opt->iter);

    int n_tokens = model.hparams.n_ctx;
    int n_vocab  = model.hparams.n_vocab;
    int n_batch  = params.common.n_batch;


    std::vector<uint8_t> mem_input_data;
    std::vector<uint8_t> mem_compute_data;

    // context for input tensors without their data
    struct ggml_init_params ctx_input_params = {
        ggml_tensor_overhead() * 2, // mem_size
        NULL,                       // mem_buffer
        true,                       // no_alloc
    };
    struct ggml_context * ctx_input = ggml_init(ctx_input_params);

    // the input tensors
    struct ggml_tensor * tokens_input  = ggml_new_tensor_2d(ctx_input, GGML_TYPE_I32, n_tokens, n_batch);
    struct ggml_tensor * target_probs  = ggml_new_tensor_3d(ctx_input, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);

    // measure required memory for input tensors
    size_t max_input_size = GGML_PAD(ggml_nbytes(tokens_input), tensor_alignment) +
                            GGML_PAD(ggml_nbytes(target_probs), tensor_alignment) +
                            tensor_alignment;
    printf("%s: input_size = %zu bytes (%.1f MB)\n", __func__, max_input_size, (float) max_input_size / (1024.0f*1024.0f));

    // allocate input tensors
    mem_input_data.resize(max_input_size);
    ggml_allocr_t alloc_inps = ggml_allocr_new(mem_input_data.data(), mem_input_data.size(), tensor_alignment);
    ggml_allocr_alloc(alloc_inps, tokens_input);
    ggml_allocr_alloc(alloc_inps, target_probs);

    // context for compute tensors without their data
    const size_t estimated_compute_size_wo_data = (
            2*LLAMA_TRAIN_MAX_NODES*ggml_tensor_overhead() +
            (params.common.use_checkpointing ? 3 : 2)*(GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true))
    );
    struct ggml_init_params ctx_compute_params = {
        estimated_compute_size_wo_data, // mem_size
        NULL,                           // mem_buffer
        true,                           // no_alloc
    };
    struct ggml_context * ctx_compute = NULL;

    struct ggml_tensor * loss   = NULL;
    struct ggml_tensor * logits = NULL;

    struct ggml_cgraph * gf     = NULL;
    struct ggml_cgraph * gb     = NULL;
    struct ggml_cgraph * gb_tmp = NULL;

    // measure required memory for compute tensors
    size_t best_compute_size = SIZE_MAX;
    enum ggml_cgraph_eval_order best_order = GGML_CGRAPH_EVAL_ORDER_COUNT;
    // find best evaluation order
    for (unsigned order = 0; order < (unsigned) GGML_CGRAPH_EVAL_ORDER_COUNT; ++order) {
        ctx_compute = ggml_init(ctx_compute_params);
        ggml_allocr_t alloc = ggml_allocr_new_measure(tensor_alignment);
        gf = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
        gf->order = (enum ggml_cgraph_eval_order) order;
        gb = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
        gb_tmp = params.common.use_checkpointing
            ? ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true)
            : NULL;
        loss = llama_build_lora_finetune_graphs(
            &model, &lora, alloc, ctx_compute,
            gf, gb, gb_tmp,
            &logits, tokens_input, target_probs,
            n_tokens, n_batch,
            params.common.use_flash,
            params.common.use_checkpointing
        );
        size_t max_compute_size = ggml_allocr_max_size(alloc) + tensor_alignment;
        if (max_compute_size < best_compute_size) {
            best_compute_size = max_compute_size;
            best_order = gf->order;
        }
        ggml_allocr_free(alloc);
        ggml_free(ctx_compute);
    }
    size_t max_compute_size = best_compute_size;
    printf("%s: compute_size = %zu bytes (%.1f MB)\n", __func__, max_compute_size, (float) max_compute_size / (1024.0f*1024.0f));
    printf("%s: evaluation order = %s\n", __func__,
        (best_order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? "LEFT_TO_RIGHT" :
        (best_order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? "RIGHT_TO_LEFT" :
        "invalid");

    // allocate compute tensors
    mem_compute_data.resize(max_compute_size);
    ctx_compute = ggml_init(ctx_compute_params);
    ggml_allocr_t alloc = ggml_allocr_new(mem_compute_data.data(), mem_compute_data.size(), tensor_alignment);
    gf = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
    gf->order = best_order;
    gb = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
    gb_tmp = params.common.use_checkpointing
        ? ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true)
        : NULL;
    loss = llama_build_lora_finetune_graphs(
        &model, &lora, alloc, ctx_compute,
        gf, gb, gb_tmp,
        &logits, tokens_input, target_probs,
        n_tokens, n_batch,
        params.common.use_flash,
        params.common.use_checkpointing
    );
    ggml_allocr_free(alloc);
    ggml_allocr_free(alloc_inps);


    // tokenize data
    std::vector<llama_token> train_tokens;
    std::vector<size_t> train_samples_begin;
    std::vector<size_t> train_samples_size;
    printf("%s: tokenize training data from %s\n", __func__, params.common.fn_train_data);
    tokenize_file(lctx,
            params.common.fn_train_data,
            params.common.sample_start,
            params.common.include_sample_start,
            params.common.overlapping_samples,
            n_tokens,
            train_tokens,
            train_samples_begin,
            train_samples_size);
    GGML_ASSERT(train_samples_begin.size() == train_samples_size.size());

    printf("%s: number of training tokens: %zu\n", __func__, train_tokens.size());

    std::vector<size_t> token_noccurs;
    token_noccurs.resize(model.hparams.n_vocab, 0);
    for (unsigned int i = 0; i < train_tokens.size(); ++i) {
        ++token_noccurs[train_tokens[i]];
    }
    int n_unique_tokens = 0;
    for (unsigned int i = 0; i < token_noccurs.size(); ++i) {
        if (token_noccurs[i] == 0) continue;
        ++n_unique_tokens;
    }
    printf("%s: number of unique tokens: %d\n", __func__, n_unique_tokens);

    size_t shuffle_samples_hash = compute_samples_hash(params.common.fn_train_data, train_samples_begin.data(), train_samples_size.data(), train_samples_size.size());
    const bool changed_train_data = (shuffle_samples_hash != train->shuffle_samples_hash) || (train->shuffle_sample_count != train_samples_size.size());
    if (changed_train_data) {
        printf("%s: train data seems to have changed. restarting shuffled epoch.\n", __func__);
    }
    if (params.common.force_reshuffle) {
        printf("%s: forced reshuffling of data. restarting with newly shuffled epoch.\n", __func__);
    }
    if ((train->shuffle_rng_state_current == "") || changed_train_data || params.common.force_reshuffle) {
        train->shuffle_rng_state_current = mt19937_seed_to_state(params.common.seed);
        train->shuffle_sample_count = train_samples_size.size();
        train->shuffle_next_sample = 0;
        train->shuffle_samples_hash = shuffle_samples_hash;
    }
    std::vector<size_t> train_shuffled_samples_offs;
    std::vector<size_t> train_shuffled_samples_begin;
    std::vector<size_t> train_shuffled_samples_size;
    train_shuffled_samples_offs.resize(train_samples_begin.size());
    train_shuffled_samples_begin.resize(train_samples_begin.size());
    train_shuffled_samples_size.resize(train_samples_size.size());
    train->shuffle_rng_state_next = shuffle_samples(
        train->shuffle_rng_state_current,
        train_shuffled_samples_offs.data(),
        train_shuffled_samples_begin.data(),
        train_shuffled_samples_size.data(),
        train_samples_begin.data(),
        train_samples_size.data(),
        train_samples_size.size());

    printf("%s: begin training\n", __func__);

    save_train_files_data save_data;
    save_data.fn_checkpoint_out = params.common.fn_checkpoint_out;
    save_data.fn_lora_out       = params.fn_lora_out;
    save_data.pattern_fn_it     = params.common.pattern_fn_it;
    save_data.fn_latest         = params.common.fn_latest;
    save_data.model             = &model;
    save_data.lora              = &lora;

    struct train_opt_callback_data opt_cb_data;
    opt_cb_data.params                 = &params.common;
    opt_cb_data.train                  = train;
    opt_cb_data.save_cb                = &save_train_files;
    opt_cb_data.save_data              = &save_data;
    opt_cb_data.lctx                   = lctx;
    opt_cb_data.last_save_iter         = opt->iter;
    opt_cb_data.tokens_data            = train_tokens.data();
    opt_cb_data.tokens_size            = train_tokens.size();
    opt_cb_data.samples_begin          = train_samples_begin.data();
    opt_cb_data.samples_size           = train_samples_size.data();
    opt_cb_data.shuffled_samples_offs  = train_shuffled_samples_offs.data();
    opt_cb_data.shuffled_samples_begin = train_shuffled_samples_begin.data();
    opt_cb_data.shuffled_samples_size  = train_shuffled_samples_size.data();
    opt_cb_data.samples_count          = train_samples_size.size();
    opt_cb_data.tokens_input           = tokens_input;
    opt_cb_data.target_probs           = target_probs;
    opt_cb_data.first_iter             = opt->iter;
    opt_cb_data.first_epoch            = train->train_epochs;
    opt_cb_data.iter_at_last_epoch     = -1;
    opt_cb_data.last_time              = ggml_time_ms();
    opt_cb_data.millis_per_iter        = 0.0;

    // measure required memory for work buffer
    size_t max_work_size = ggml_graph_plan(gb, params.common.n_threads).work_size + GGML_OBJECT_SIZE;
    printf("%s: work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float) max_work_size / (1024.0f*1024.0f));

    // context for work buffer
    struct ggml_init_params ctx_work_params = {
        max_work_size, // mem_size
        NULL,          // mem_buffer
        false,         // no_alloc
    };
    struct ggml_context * ctx_work = ggml_init(ctx_work_params);

    int64_t t0 = ggml_time_ms();

    ggml_opt_resume_g(ctx_work, opt, loss, gf, gb, &train_opt_callback, (void *) &opt_cb_data);

    ggml_free(ctx_work);
    ggml_free(ctx_compute);
    ggml_free(ctx_input);

    int64_t t1 = ggml_time_ms();
    printf("%s: total training time: ", __func__);
    print_duration((double) (t1 - t0));
    printf("\n");

    int new_iters = opt->iter - opt_cb_data.last_save_iter;
    if (new_iters > 0) {
        train->train_its     += new_iters;
        train->train_tokens  += new_iters * opt->params.n_gradient_accumulation * n_batch * n_tokens;

        save_train_files(&save_data, train);
        opt_cb_data.last_save_iter = opt->iter;
    }

    ggml_free(opt->ctx);
    free_train_state(train);
    ggml_free(lora.ctx);
    llama_free(lctx);
    llama_free_model(lmodel);
    return 0;
}
