#include "ggml.h"
#include "ggml-alloc.h"
#include "common.h"
#include "train.h"
#include "llama.h"
#include <unordered_map>
#include <vector>
#include <cassert>
#include <climits>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static const size_t tensor_alignment = 32;

struct my_llama_hparams {
    uint32_t n_vocab = 32000;
    uint32_t n_ctx   = 512;
    uint32_t n_embd  = 4096;
    uint32_t n_head  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;
    uint32_t n_ff    = 11008;

    // float f_norm_eps     = 1e-5f; // falcon
    float f_norm_rms_eps = 1e-5f; // llama

    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;
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
    struct ggml_context * ctx = NULL;
    std::vector<uint8_t> data;

    my_llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<my_llama_layer> layers;
};

// gguf constants (sync with gguf.py)
static const char * LLM_KV_TRAINING_TYPE_TRAIN_MODEL     = "train_model";
static const char * LLM_KV_TRAINING_TYPE                 = "training.type";

static const char * LLM_KV_GENERAL_ARCHITECTURE        = "general.architecture";
static const char * LLM_KV_GENERAL_FILE_TYPE           = "general.file_type";

static const char * LLM_KV_CONTEXT_LENGTH              = "%s.context_length";
static const char * LLM_KV_EMBEDDING_LENGTH            = "%s.embedding_length";
static const char * LLM_KV_BLOCK_COUNT                 = "%s.block_count";
static const char * LLM_KV_FEED_FORWARD_LENGTH         = "%s.feed_forward_length";
static const char * LLM_KV_ATTENTION_HEAD_COUNT        = "%s.attention.head_count";
static const char * LLM_KV_ATTENTION_LAYERNORM_RMS_EPS = "%s.attention.layer_norm_rms_epsilon";
static const char * LLM_KV_ROPE_DIMENSION_COUNT        = "%s.rope.dimension_count";
static const char * LLM_KV_ROPE_FREQ_BASE              = "%s.rope.freq_base"; // TODO load in llama.cpp
static const char * LLM_KV_ROPE_SCALE_LINEAR           = "%s.rope.scale_linear";

static const char * LLM_KV_TOKENIZER_MODEL             = "tokenizer.ggml.model";
static const char * LLM_KV_TOKENIZER_LIST              = "tokenizer.ggml.tokens";
static const char * LLM_KV_TOKENIZER_TOKEN_TYPE        = "tokenizer.ggml.token_type";
static const char * LLM_KV_TOKENIZER_SCORES            = "tokenizer.ggml.scores";
static const char * LLM_KV_TOKENIZER_MERGES            = "tokenizer.ggml.merges";
static const char * LLM_KV_TOKENIZER_BOS_ID            = "tokenizer.ggml.bos_token_id";
static const char * LLM_KV_TOKENIZER_EOS_ID            = "tokenizer.ggml.eos_token_id";
static const char * LLM_KV_TOKENIZER_UNK_ID            = "tokenizer.ggml.unknown_token_id";
static const char * LLM_KV_TOKENIZER_SEP_ID            = "tokenizer.ggml.seperator_token_id";
static const char * LLM_KV_TOKENIZER_PAD_ID            = "tokenizer.ggml.padding_token_id";

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
    printf("%s: n_vocab: %d\n", __func__, params->n_vocab);
    printf("%s: n_ctx:   %d\n", __func__, params->n_ctx);
    printf("%s: n_embd:  %d\n", __func__, params->n_embd);
    printf("%s: n_head:  %d\n", __func__, params->n_head);
    printf("%s: n_ff:    %d\n", __func__, params->n_ff);
    printf("%s: n_layer: %d\n", __func__, params->n_layer);
    printf("%s: n_rot:   %d\n", __func__, params->n_rot);
}

static void set_param_model(struct my_llama_model * model) {
    const auto& hparams = model->hparams;

    const uint32_t n_layer = hparams.n_layer;

    struct ggml_context* ctx = model->ctx;

    ggml_set_param(ctx, model->tok_embeddings);
    ggml_set_param(ctx, model->norm);
    ggml_set_param(ctx, model->output);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        ggml_set_param(ctx, layer.attention_norm);
        ggml_set_param(ctx, layer.wq);
        ggml_set_param(ctx, layer.wk);
        ggml_set_param(ctx, layer.wv);
        ggml_set_param(ctx, layer.wo);
        ggml_set_param(ctx, layer.ffn_norm);
        ggml_set_param(ctx, layer.w1);
        ggml_set_param(ctx, layer.w2);
        ggml_set_param(ctx, layer.w3);
    }
}

static void alloc_model(struct ggml_allocr * alloc, struct my_llama_model * model) {
    ggml_allocr_alloc(alloc, model->tok_embeddings);
    ggml_allocr_alloc(alloc, model->norm);
    ggml_allocr_alloc(alloc, model->output);
    for (uint32_t i = 0; i < model->layers.size(); ++i) {
        auto & layer = model->layers[i];
        ggml_allocr_alloc(alloc, layer.attention_norm);
        ggml_allocr_alloc(alloc, layer.wq);
        ggml_allocr_alloc(alloc, layer.wk);
        ggml_allocr_alloc(alloc, layer.wv);
        ggml_allocr_alloc(alloc, layer.wo);
        ggml_allocr_alloc(alloc, layer.ffn_norm);
        ggml_allocr_alloc(alloc, layer.w1);
        ggml_allocr_alloc(alloc, layer.w2);
        ggml_allocr_alloc(alloc, layer.w3);
    }
    ggml_allocr_alloc(alloc, model->tok_embeddings->grad);
    ggml_allocr_alloc(alloc, model->norm->grad);
    ggml_allocr_alloc(alloc, model->output->grad);
    for (uint32_t i = 0; i < model->layers.size(); ++i) {
        auto & layer = model->layers[i];
        ggml_allocr_alloc(alloc, layer.attention_norm->grad);
        ggml_allocr_alloc(alloc, layer.wq->grad);
        ggml_allocr_alloc(alloc, layer.wk->grad);
        ggml_allocr_alloc(alloc, layer.wv->grad);
        ggml_allocr_alloc(alloc, layer.wo->grad);
        ggml_allocr_alloc(alloc, layer.ffn_norm->grad);
        ggml_allocr_alloc(alloc, layer.w1->grad);
        ggml_allocr_alloc(alloc, layer.w2->grad);
        ggml_allocr_alloc(alloc, layer.w3->grad);
    }
}

static void init_model(struct my_llama_model * model) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;
    const uint32_t n_ff    = hparams.n_ff;


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

    // context for model tensors without their data
    struct ggml_init_params ctx_model_params;
    ctx_model_params.mem_size   = ggml_tensor_overhead()*2*(6 + n_layer*18);
    ctx_model_params.mem_buffer = NULL;
    ctx_model_params.no_alloc   = true;

    struct ggml_context * ctx = ggml_init(ctx_model_params);
    model->ctx = ctx;

    model->tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    model->norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    model->output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);

    ggml_set_name(model->tok_embeddings, tn(LLM_TENSOR_TOKEN_EMBD));
    ggml_set_name(model->norm,           tn(LLM_TENSOR_OUTPUT_NORM));
    ggml_set_name(model->output,         tn(LLM_TENSOR_OUTPUT));

    model->layers.resize(n_layer);
    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        layer.wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);

        layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        layer.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);
        layer.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff, n_embd);
        layer.w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);

        ggml_set_name(layer.attention_norm, tni(LLM_TENSOR_ATTN_NORM, i));

        ggml_set_name(layer.wq,             tni(LLM_TENSOR_ATTN_Q, i));
        ggml_set_name(layer.wk,             tni(LLM_TENSOR_ATTN_K, i));
        ggml_set_name(layer.wv,             tni(LLM_TENSOR_ATTN_V, i));
        ggml_set_name(layer.wo,             tni(LLM_TENSOR_ATTN_OUT, i));

        ggml_set_name(layer.ffn_norm,       tni(LLM_TENSOR_FFN_NORM, i));

        ggml_set_name(layer.w1,             tni(LLM_TENSOR_FFN_GATE, i));
        ggml_set_name(layer.w2,             tni(LLM_TENSOR_FFN_DOWN, i));
        ggml_set_name(layer.w3,             tni(LLM_TENSOR_FFN_UP, i));
    }

    set_param_model(model);

    // measure data size
    size_t size = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size += GGML_PAD(ggml_nbytes(t), tensor_alignment);
    }

    // allocate data
    struct ggml_allocr * alloc = NULL;
    model->data.resize(size + tensor_alignment);
    alloc = ggml_allocr_new(model->data.data(), model->data.size(), tensor_alignment);
    alloc_model(alloc, model);
}

static void randomize_model(struct my_llama_model * model, int seed, float mean, float std, float min, float max) {
    const auto & hparams = model->hparams;

    const uint32_t n_layer = hparams.n_layer;

    struct random_normal_distribution * rnd = init_random_normal_distribution(seed, mean, std, min, max);

    randomize_tensor_normal(model->tok_embeddings, rnd);
    randomize_tensor_normal(model->norm,           rnd);
    randomize_tensor_normal(model->output,         rnd);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];
        randomize_tensor_normal(layer.attention_norm, rnd);

        randomize_tensor_normal(layer.wq, rnd);
        randomize_tensor_normal(layer.wk, rnd);
        randomize_tensor_normal(layer.wv, rnd);
        randomize_tensor_normal(layer.wo, rnd);

        randomize_tensor_normal(layer.ffn_norm, rnd);

        randomize_tensor_normal(layer.w1, rnd);
        randomize_tensor_normal(layer.w2, rnd);
        randomize_tensor_normal(layer.w3, rnd);
    }

    free_random_normal_distribution(rnd);
}

static struct ggml_tensor * llama_build_train_graphs(
        struct my_llama_model * model,
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
    const auto & hparams = model->hparams;
    const int n_ctx      = hparams.n_ctx;
    const int n_vocab    = hparams.n_vocab;
    const int n_embd     = hparams.n_embd;
    const int n_layer    = hparams.n_layer;
    const int n_head     = hparams.n_head;
    const int n_rot      = hparams.n_rot;
    const int n_ff       = hparams.n_ff;
    const float f_norm_rms_eps  = hparams.f_norm_rms_eps;
    const float rope_freq_base  = hparams.rope_freq_base;
    const float rope_freq_scale = hparams.rope_freq_scale;

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

        return ggml_rope_custom(
            ctx, t, KQ_pos, n_rot, rope_mode, n_ctx, 0, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f
        );
    };

    set_name(tokens_input, "tokens_input");
    set_name(targets,      "targets");

    GGML_ASSERT(tokens_input->type == GGML_TYPE_I32);
    struct ggml_tensor * t00 = ggml_reshape_1d(ctx, tokens_input, N*n_batch);  set_name(t00, "t00"); assert_shape_1d(t00, N*n_batch);
    struct ggml_tensor * t01 = ggml_get_rows(ctx, model->tok_embeddings, t00); set_name(t01, "t01"); assert_shape_2d(t01, n_embd, N*n_batch);

    struct ggml_tensor * cur = t01;

    std::vector<struct ggml_tensor *> checkpoints;
    checkpoints.push_back(tokens_input);
    checkpoints.push_back(targets);
    checkpoints.push_back(t00);
    checkpoints.push_back(t01);

    const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);

    for (int il = 0; il < n_layer; ++il) {
        struct my_llama_layer & layer = model->layers[il];
        struct ggml_tensor * t02 = ggml_rms_norm     (ctx, cur, f_norm_rms_eps);                    set_name(t02, "t02");     assert_shape_2d(t02, n_embd, N*n_batch);
        struct ggml_tensor * t03 = ggml_repeat       (ctx, layer.attention_norm, t02);              set_name(t03, "t03");     assert_shape_2d(t03, n_embd, N*n_batch);
        struct ggml_tensor * t04 = ggml_mul          (ctx, t03, t02);                               set_name(t04, "t04");     assert_shape_2d(t04, n_embd, N*n_batch);
        struct ggml_tensor * t05 = ggml_mul_mat      (ctx, layer.wq, t04);                          set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
        struct ggml_tensor * t06 = ggml_reshape_4d   (ctx, t05, n_embd/n_head, n_head, N, n_batch); set_name(t06, "t06");     assert_shape_4d(t06, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t07 = rope              (t06);                                         set_name(t07, "t07");     assert_shape_4d(t07, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t08 = ggml_mul_mat      (ctx, layer.wk, t04);                          set_name(t08, "t08");     assert_shape_2d(t08, n_embd, N*n_batch);
        struct ggml_tensor * t09 = ggml_reshape_4d   (ctx, t08, n_embd/n_head, n_head, N, n_batch); set_name(t09, "t09");     assert_shape_4d(t09, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t10 = rope              (t09);                                         set_name(t10, "t10");     assert_shape_4d(t10, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t11 = ggml_mul_mat      (ctx, t04, layer.wv);                          set_name(t11, "t11");     assert_shape_2d(t11, N*n_batch, n_embd);
        struct ggml_tensor * t12 = ggml_reshape_4d   (ctx, t11, N, n_batch, n_embd/n_head, n_head); set_name(t12, "t12");     assert_shape_4d(t12, N, n_batch, n_embd/n_head, n_head);
        struct ggml_tensor * t13 = ggml_permute      (ctx, t07, 0, 2, 1, 3);                        set_name(t13, "t13");     assert_shape_4d(t13, n_embd/n_head, N, n_head, n_batch);
        struct ggml_tensor * t14 = ggml_permute      (ctx, t10, 0, 2, 1, 3);                        set_name(t14, "t14");     assert_shape_4d(t14, n_embd/n_head, N, n_head, n_batch);
        struct ggml_tensor * t15 = ggml_permute      (ctx, t12, 0, 3, 1, 2);                        set_name(t15, "t15");     assert_shape_4d(t15, N, n_embd/n_head, n_head, n_batch);
        struct ggml_tensor * t16;
        if (enable_flash_attn) {
            t16 = ggml_flash_attn(ctx, t13, t14, t15, true);                                        set_name(t16, "t16");     assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);
        } else {
            struct ggml_tensor * t16_0 = ggml_mul_mat              (ctx, t14, t13);                 set_name(t16_0, "t16_0"); assert_shape_4d(t16_0, N, N, n_head, n_batch);
            struct ggml_tensor * t16_1 = ggml_scale_inplace        (ctx, t16_0, kv_scale);          set_name(t16_1, "t16_1"); assert_shape_4d(t16_1, N, N, n_head, n_batch);
            struct ggml_tensor * t16_2 = ggml_diag_mask_inf_inplace(ctx, t16_1, n_past);            set_name(t16_2, "t16_2"); assert_shape_4d(t16_2, N, N, n_head, n_batch);
            struct ggml_tensor * t16_3 = ggml_soft_max_inplace     (ctx, t16_2);                    set_name(t16_3, "t16_3"); assert_shape_4d(t16_3, N, N, n_head, n_batch);
            t16 = ggml_mul_mat(ctx, t15, t16_3);                                                    set_name(t16, "t16");     assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);
        }
        struct ggml_tensor * t17 = ggml_permute      (ctx, t16, 0, 2, 1, 3);                        set_name(t17, "t17");     assert_shape_4d(t17, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t18 = ggml_cont         (ctx, t17);                                    set_name(t18, "t18");     assert_shape_4d(t18, n_embd/n_head, n_head, N, n_batch);
        struct ggml_tensor * t19 = ggml_reshape_2d   (ctx, t18, n_embd, N*n_batch);                 set_name(t19, "t19");     assert_shape_2d(t19, n_embd, N*n_batch);
        struct ggml_tensor * t20 = ggml_mul_mat      (ctx, layer.wo, t19);                          set_name(t20, "t20");     assert_shape_2d(t20, n_embd, N*n_batch);
        struct ggml_tensor * t21 = ggml_add          (ctx, t20, cur);                               set_name(t21, "t21");     assert_shape_2d(t21, n_embd, N*n_batch);
        struct ggml_tensor * t22 = ggml_rms_norm     (ctx, t21, f_norm_rms_eps);                    set_name(t22, "t22");     assert_shape_2d(t22, n_embd, N*n_batch);
        struct ggml_tensor * t23 = ggml_repeat       (ctx, layer.ffn_norm, t22);                    set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
        struct ggml_tensor * t24 = ggml_mul          (ctx, t23, t22);                               set_name(t24, "t24");     assert_shape_2d(t24, n_embd, N*n_batch);
        struct ggml_tensor * t25 = ggml_mul_mat      (ctx, layer.w3, t24);                          set_name(t25, "t25");     assert_shape_2d(t25, n_ff, N*n_batch);
        struct ggml_tensor * t26 = ggml_mul_mat      (ctx, layer.w1, t24);                          set_name(t26, "t26");     assert_shape_2d(t26, n_ff, N*n_batch);
        struct ggml_tensor * t27 = ggml_silu         (ctx, t26);                                    set_name(t27, "t27");     assert_shape_2d(t27, n_ff, N*n_batch);
        struct ggml_tensor * t28 = ggml_mul          (ctx, t27, t25);                               set_name(t28, "t28");     assert_shape_2d(t28, n_ff, N*n_batch);
        struct ggml_tensor * t29 = ggml_mul_mat      (ctx, layer.w2, t28);                          set_name(t29, "t29");     assert_shape_2d(t29, n_embd, N*n_batch);
        struct ggml_tensor * t30 = ggml_add          (ctx, t29, t21);                               set_name(t30, "t30");     assert_shape_2d(t30, n_embd, N*n_batch);
        cur = t30;
        checkpoints.push_back(cur);
    }
    struct ggml_tensor * t31   = ggml_rms_norm          (ctx, cur, f_norm_rms_eps);                 set_name(t31, "t31");     assert_shape_2d(t31, n_embd, N*n_batch);
    struct ggml_tensor * t32   = ggml_repeat            (ctx, model->norm, t31);                    set_name(t32, "t32");     assert_shape_2d(t32, n_embd, N*n_batch);
    struct ggml_tensor * t33   = ggml_mul               (ctx, t32, t31);                            set_name(t33, "t33");     assert_shape_2d(t33, n_embd, N*n_batch);
    struct ggml_tensor * t34   = ggml_mul_mat           (ctx, model->output, t33);                  set_name(t34, "t34");     assert_shape_2d(t34, n_vocab, N*n_batch);
    struct ggml_tensor * t35   = ggml_reshape_3d        (ctx, t34, n_vocab, N, n_batch);            set_name(t35, "t35");     assert_shape_3d(t35, n_vocab, N, n_batch);
    struct ggml_tensor * t36   = ggml_cross_entropy_loss(ctx, t35, targets);                        set_name(t36, "t36");     assert_shape_1d(t36, 1);

    checkpoints.push_back(t31);
    checkpoints.push_back(t32);
    checkpoints.push_back(t33);
    checkpoints.push_back(t34);
    checkpoints.push_back(t35);
    checkpoints.push_back(t36);

    ggml_build_forward_expand(gf, t36);

    if (enable_checkpointing) {
        ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints.data(), (int) checkpoints.size());
    } else {
        ggml_graph_cpy(gf, gb);
        ggml_build_backward_expand(ctx, gf, gb, true);
    }

    if (alloc) {
        // make sure some tensors are not reallocated by inserting new temporary nodes depending on them
        int n_leafs_before = gb->n_leafs;
        int n_nodes_before = gb->n_nodes;
        // output tensors
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t35, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36, 1.0f));
        // input gradient
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36->grad, 1.0f));
        // KQ_pos
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, KQ_pos, 1.0f));
        GGML_ASSERT(t36->grad->data == NULL && t36->grad->view_src == NULL);

        ggml_allocr_alloc(alloc, t36->grad);

        // allocating checkpoints in one block to reduce memory fragmentation
        // note: they will be freed in reverse order
        for (int i = 0; i < (int) checkpoints.size(); ++i) {
            if (checkpoints[i]->data == NULL && checkpoints[i]->view_src == NULL) {
                ggml_allocr_alloc(alloc, checkpoints[i]);
            }
        }

        //int n_leafs_after = gb->n_leafs;
        //int n_nodes_after = gb->n_nodes;

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
    }

    *logits = t35;
    return t36;
}

#define GGUF_GET_KEY(ctx, dst, func, type, req, key) \
do { \
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
} while (0)

static void load_llama_model_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct my_llama_model * model) {
    // NOTE: gguf_context must be initialized with f_ggml_ctx and no_alloc=false, otherwise tensor data can not be read
    std::string arch;

    std::vector<char> keybuf;
    keybuf.resize(512);
    auto kv = [&arch, &keybuf](const char * key) -> const char * {
        snprintf(keybuf.data(), keybuf.size(), key, arch.c_str());
        return keybuf.data();
    };

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

    GGUF_GET_KEY(fctx, arch, gguf_get_val_str, GGUF_TYPE_STRING, true, LLM_KV_GENERAL_ARCHITECTURE);
    GGML_ASSERT(arch == "llama");

    uint32_t ftype_u;
    GGUF_GET_KEY(fctx, ftype_u, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_GENERAL_FILE_TYPE);
    GGML_ASSERT((enum llama_ftype) ftype_u == LLAMA_FTYPE_ALL_F32);

    // n_ctx was not saved in earlier checkpoint file versions, so we make it optional here
    GGUF_GET_KEY(fctx, model->hparams.n_ctx,   gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_CONTEXT_LENGTH));

    GGUF_GET_KEY(fctx, model->hparams.n_embd,  gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_EMBEDDING_LENGTH));
    GGUF_GET_KEY(fctx, model->hparams.n_ff,    gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_FEED_FORWARD_LENGTH));
    GGUF_GET_KEY(fctx, model->hparams.n_head,  gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_ATTENTION_HEAD_COUNT));
    GGUF_GET_KEY(fctx, model->hparams.n_layer, gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_BLOCK_COUNT));

    model->hparams.n_rot = model->hparams.n_embd / model->hparams.n_head;
    GGUF_GET_KEY(fctx, model->hparams.n_rot,   gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_ROPE_DIMENSION_COUNT));

    float rope_freq_scale = 1.0f;
    GGUF_GET_KEY(fctx, model->hparams.f_norm_rms_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS));
    GGUF_GET_KEY(fctx, model->hparams.rope_freq_base, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_FREQ_BASE));
    GGUF_GET_KEY(fctx, rope_freq_scale, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_SCALE_LINEAR));
    if (rope_freq_scale != 1.0f) {
        model->hparams.rope_freq_scale = 1.0f / rope_freq_scale;
    }

    init_model(model);

    copy_tensor_by_name(model->tok_embeddings, f_ggml_ctx, tn(LLM_TENSOR_TOKEN_EMBD));
    copy_tensor_by_name(model->norm,           f_ggml_ctx, tn(LLM_TENSOR_OUTPUT_NORM));
    copy_tensor_by_name(model->output,         f_ggml_ctx, tn(LLM_TENSOR_OUTPUT));

    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        auto & layer = model->layers[i];

        copy_tensor_by_name(layer.attention_norm, f_ggml_ctx, tni(LLM_TENSOR_ATTN_NORM, i));
        copy_tensor_by_name(layer.wq,             f_ggml_ctx, tni(LLM_TENSOR_ATTN_Q, i));
        copy_tensor_by_name(layer.wk,             f_ggml_ctx, tni(LLM_TENSOR_ATTN_K, i));
        copy_tensor_by_name(layer.wv,             f_ggml_ctx, tni(LLM_TENSOR_ATTN_V, i));
        copy_tensor_by_name(layer.wo,             f_ggml_ctx, tni(LLM_TENSOR_ATTN_OUT, i));
        copy_tensor_by_name(layer.ffn_norm,       f_ggml_ctx, tni(LLM_TENSOR_FFN_NORM, i));
        copy_tensor_by_name(layer.w1,             f_ggml_ctx, tni(LLM_TENSOR_FFN_GATE, i));
        copy_tensor_by_name(layer.w2,             f_ggml_ctx, tni(LLM_TENSOR_FFN_DOWN, i));
        copy_tensor_by_name(layer.w3,             f_ggml_ctx, tni(LLM_TENSOR_FFN_UP, i));
    }
}

static void save_llama_model_gguf(struct gguf_context * fctx, const char * fn_vocab_model, struct my_llama_model * model) {
    const char * arch = "llama";
    enum llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::vector<char> keybuf;
    keybuf.resize(512);
    auto kv = [arch, &keybuf](const char * key) -> const char * {
        snprintf(keybuf.data(), keybuf.size(), key, arch);
        return keybuf.data();
    };

    // set arch
    gguf_set_val_str(fctx, LLM_KV_GENERAL_ARCHITECTURE, arch);
    gguf_set_val_u32(fctx, LLM_KV_GENERAL_FILE_TYPE, ftype);

    // set hparams
    gguf_set_val_u32(fctx, kv(LLM_KV_CONTEXT_LENGTH),              model->hparams.n_ctx                  );
    gguf_set_val_u32(fctx, kv(LLM_KV_EMBEDDING_LENGTH),            model->hparams.n_embd                 );
    gguf_set_val_u32(fctx, kv(LLM_KV_FEED_FORWARD_LENGTH),         model->hparams.n_ff                   );
    gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT),        model->hparams.n_head                 );
    gguf_set_val_u32(fctx, kv(LLM_KV_BLOCK_COUNT),                 model->hparams.n_layer                );
    gguf_set_val_u32(fctx, kv(LLM_KV_ROPE_DIMENSION_COUNT),        model->hparams.n_rot                  );

    gguf_set_val_f32(fctx, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS), model->hparams.f_norm_rms_eps         );
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_FREQ_BASE),              model->hparams.rope_freq_base         ); // TODO load in llama.cpp
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_SCALE_LINEAR),           1.0f / model->hparams.rope_freq_scale );

    // set vocab by copying from vocab_model gguf file
    {
        struct gguf_init_params params = {
            /*.no_alloc = */ false,
            /*.ctx      = */ NULL,
        };
        struct gguf_context * vctx = gguf_init_from_file(fn_vocab_model, params);

        const int token_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_LIST));
        if (token_idx == -1) {
            die("cannot find tokenizer vocab in model file");
        }
        const uint32_t n_vocab = gguf_get_arr_n(vctx, token_idx);

        const int score_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_SCORES));
        if (score_idx == -1) {
            die("cannot find tokenizer scores in model file");
        }

        const float * scores = (const float * ) gguf_get_arr_data(vctx, score_idx);

        const int toktype_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE));
        if (toktype_idx == -1) {
            die("cannot find token type list in GGUF file");
        }

        const int * toktypes = (const int * ) gguf_get_arr_data(vctx, toktype_idx);

        std::string tokenizer_name;
        GGUF_GET_KEY(vctx, tokenizer_name, gguf_get_val_str, GGUF_TYPE_STRING, true, kv(LLM_KV_TOKENIZER_MODEL));

        gguf_set_val_str(fctx, kv(LLM_KV_TOKENIZER_MODEL), tokenizer_name.c_str());
        gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_SCORES), GGUF_TYPE_FLOAT32, scores, n_vocab);
        gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE), GGUF_TYPE_INT32, toktypes, n_vocab);

        int32_t special_bos_id = 1;
        int32_t special_eos_id = 2;
        int32_t special_unk_id = 0;
        int32_t special_sep_id = -1;
        int32_t special_pad_id = -1;
        if (tokenizer_name == "llama") {
            // default special tokens
            special_bos_id = 1;
            special_eos_id = 2;
            special_unk_id = 0;
            special_sep_id = -1;
            special_pad_id = -1;
        } else if (tokenizer_name == "gpt2") {
            // read and copy bpe merges
            const int merges_keyidx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_MERGES));
            if (merges_keyidx == -1) {
                die("cannot find tokenizer merges in model file");
            }

            const int n_merges = gguf_get_arr_n(vctx, merges_keyidx);

            std::vector<const char*> merges;
            merges.resize(n_merges);
            for (int i = 0; i < n_merges; i++) {
                merges[i] = gguf_get_arr_str(vctx, merges_keyidx, i);
            }
            gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_MERGES), merges.data(), n_merges);

            // default special tokens
            special_bos_id = 11;
            special_eos_id = 11;
            special_unk_id = -1;
            special_sep_id = -1;
            special_pad_id = -1;
        } else {
            fprintf(stderr, "%s: unknown tokenizer: '%s'", __func__, tokenizer_name.c_str());
            fprintf(stderr, "%s: using default tokenizer: 'llama'", __func__);
        }

        std::vector<const char*> tokens;
        tokens.resize(n_vocab);
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokens[i] = gguf_get_arr_str(vctx, token_idx, i);
        }
        gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_LIST), tokens.data(), n_vocab);

        GGUF_GET_KEY(vctx, special_bos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_BOS_ID));
        GGUF_GET_KEY(vctx, special_eos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_EOS_ID));
        GGUF_GET_KEY(vctx, special_unk_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_UNK_ID));
        GGUF_GET_KEY(vctx, special_sep_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_SEP_ID));
        GGUF_GET_KEY(vctx, special_pad_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_PAD_ID));

        gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_BOS_ID), special_bos_id);
        gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_EOS_ID), special_eos_id);
        gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_UNK_ID), special_unk_id);
        gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_SEP_ID), special_sep_id);
        gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_PAD_ID), special_pad_id);

        gguf_free(vctx);
    }

    // add tensors
    gguf_add_tensor(fctx, model->tok_embeddings);
    gguf_add_tensor(fctx, model->norm);
    gguf_add_tensor(fctx, model->output);
    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        auto & layer = model->layers[i];


        gguf_add_tensor(fctx, layer.attention_norm);
        gguf_add_tensor(fctx, layer.wq);
        gguf_add_tensor(fctx, layer.wk);
        gguf_add_tensor(fctx, layer.wv);
        gguf_add_tensor(fctx, layer.wo);
        gguf_add_tensor(fctx, layer.ffn_norm);
        gguf_add_tensor(fctx, layer.w1);
        gguf_add_tensor(fctx, layer.w2);
        gguf_add_tensor(fctx, layer.w3);
    }
}

static void save_llama_model_file(const char * filename, const char * fn_vocab_model, struct my_llama_model * model) {
    printf("%s: saving to %s\n", __func__, filename);
    struct gguf_context * fctx = gguf_init_empty();

    save_llama_model_gguf(fctx, fn_vocab_model, model);

    // write file
    const bool only_meta = false;
    gguf_write_to_file(fctx, filename, only_meta);
    gguf_free(fctx);
}

static void load_checkpoint_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct my_llama_model * model, struct train_state * train) {
    load_llama_model_gguf(fctx, f_ggml_ctx, model);
    if (load_train_state_gguf(fctx, f_ggml_ctx, train)) {
        std::string train_type = LLM_KV_TRAINING_TYPE_TRAIN_MODEL;
        GGUF_GET_KEY(fctx, train_type, gguf_get_val_str, GGUF_TYPE_STRING, false, LLM_KV_TRAINING_TYPE);
        GGML_ASSERT(train_type == LLM_KV_TRAINING_TYPE_TRAIN_MODEL);
    } else {
        printf("%s: loaded llama model as checkpoint\n", __func__);
    }
}

static void save_checkpoint_gguf(struct gguf_context * fctx, const char * fn_vocab_model, struct my_llama_model * model, struct train_state * train) {
    gguf_set_val_str(fctx, LLM_KV_TRAINING_TYPE, LLM_KV_TRAINING_TYPE_TRAIN_MODEL);
    save_llama_model_gguf(fctx, fn_vocab_model, model);
    save_train_state_gguf(fctx, train);
}

static bool load_checkpoint_file(const char * filename, struct my_llama_model * model, struct train_state * train) {
    struct ggml_context * f_ggml_ctx;
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &f_ggml_ctx;
    struct gguf_context * fctx = gguf_init_from_file(filename, params);
    if (fctx == NULL) {
        return false;
    }

    load_checkpoint_gguf(fctx, f_ggml_ctx, model, train);

    return true;
}

static void save_checkpoint_file(const char * filename, const char * fn_vocab_model, struct my_llama_model * model, struct train_state * train) {
    printf("%s: saving to %s\n", __func__, filename);
    struct gguf_context * fctx = gguf_init_empty();

    save_checkpoint_gguf(fctx, fn_vocab_model, model, train);

    // write file
    const bool only_meta = false;
    gguf_write_to_file(fctx, filename, only_meta);
    gguf_free(fctx);
}

struct train_params {
    struct train_params_common common;

    const char * fn_vocab_model;
    const char * fn_model_out;

    bool only_write_model;

    int n_ctx;
    int n_embd;
    int n_head;
    int n_layer;
    int n_ff;

    float f_norm_rms_eps;
    float rope_freq_base;
    float rope_freq_scale;
};

static struct train_params get_default_train_params() {
    struct train_params params;
    params.common = get_default_train_params_common();
    params.fn_vocab_model    = "ggml-vic7b-uncensored-q4_0.bin";
    params.fn_model_out      = "ggml-checkpoint-f32.bin";

    params.only_write_model = false;

    params.n_ctx      =  128;
    params.n_embd     =  256;
    params.n_head     =    8;
    params.n_layer    =   16;
    params.n_ff       =  768;

    params.f_norm_rms_eps  = 1e-5f;
    params.rope_freq_base  = 10000.0f;
    params.rope_freq_scale = 1.0f;

    return params;
}

static void train_print_usage(int argc, char ** argv, const struct train_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                 show this help message and exit\n");

    fprintf(stderr, "  --vocab-model FNAME        model path from which to load vocab (default '%s')\n", params->fn_vocab_model);
    fprintf(stderr, "  --model-out FNAME          path to save ggml model (default '%s')\n", params->fn_model_out);
    fprintf(stderr, "  --only-write-model         only save llama model, don't do any training. use this if you only want to convert a checkpoint to a model.\n");
    fprintf(stderr, "  --embd N                   Embedding size used for new models (default %d)\n", params->n_embd);
    fprintf(stderr, "  --ff N                     Feedforward size used for new models. (default %d)\n", params->n_ff);
    fprintf(stderr, "  --head N                   Number of heads for new models (default %d)\n", params->n_head);
    fprintf(stderr, "  --layer N                  Number of layers for new models (default %d)\n", params->n_layer);
    fprintf(stderr, "  --norm-rms-eps F           RMS-Norm epsilon value (default %f)\n", params->f_norm_rms_eps);
    fprintf(stderr, "  --rope-freq-base F         Frequency base for ROPE (default %f)\n", params->rope_freq_base);
    fprintf(stderr, "  --rope-freq-scale F        Frequency scale for ROPE (default %f)\n", params->rope_freq_scale);

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
        } else if (arg == "--vocab-model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_vocab_model = argv[i];
        } else if (arg == "--model-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_model_out = argv[i];
        } else if (arg == "--only-write-model") {
            params->only_write_model = true;
        } else if (arg == "--embd") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_embd = std::stoi(argv[i]);
        } else if (arg == "--ff") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_ff = std::stoi(argv[i]);
        } else if (arg == "--head") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_head = std::stoi(argv[i]);
        } else if (arg == "--layer") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_layer = std::stoi(argv[i]);
        } else if (arg == "--norm-rms-eps") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->f_norm_rms_eps = std::stof(argv[i]);
        } else if (arg == "--rope-freq-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->rope_freq_base = std::stof(argv[i]);
        } else if (arg == "--rope-freq-scale") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->rope_freq_scale = std::stof(argv[i]);
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
    const char            * fn_model_out;
    const char            * fn_vocab_model;
    const char            * pattern_fn_it;
    const char            * fn_latest;
    struct my_llama_model * model;
};

static void save_train_files(void * vdata, struct train_state * train) {
    struct save_train_files_data * data   = (struct save_train_files_data *) vdata;
    int64_t iter = train->opt->iter;

    if (strlen(data->fn_checkpoint_out) > 0) {
        save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_vocab_model, data->model, train);
        save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->fn_vocab_model, data->model, train);

    }
    if (strlen(data->fn_model_out) > 0) {
        save_llama_model_file(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_vocab_model, data->model);
        save_llama_model_file(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->fn_vocab_model, data->model);
    }
}

static int64_t get_parameter_count(struct my_llama_model* model) {
    int64_t nx = 0;
    nx += ggml_nelements(model->tok_embeddings);
    nx += ggml_nelements(model->norm);
    nx += ggml_nelements(model->output);

    for (uint32_t i = 0; i < model->layers.size(); ++i) {
        auto & layer = model->layers[i];
        nx += ggml_nelements(layer.attention_norm);
        nx += ggml_nelements(layer.wq);
        nx += ggml_nelements(layer.wk);
        nx += ggml_nelements(layer.wv);
        nx += ggml_nelements(layer.wo);
        nx += ggml_nelements(layer.ffn_norm);
        nx += ggml_nelements(layer.w1);
        nx += ggml_nelements(layer.w2);
        nx += ggml_nelements(layer.w3);
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

    struct llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;

    struct llama_context_params cparams = llama_context_default_params();

    struct llama_model * lmodel = llama_load_model_from_file(params.fn_vocab_model, mparams);
    struct llama_context * lctx = llama_new_context_with_model(lmodel, cparams);

    struct my_llama_model model;
    model.hparams.n_vocab = llama_n_vocab(lmodel);
    model.hparams.n_ctx   = params.common.n_ctx;
    model.hparams.n_embd  = params.n_embd;
    model.hparams.n_head  = params.n_head;
    model.hparams.n_layer = params.n_layer;
    model.hparams.n_ff    = params.n_ff;
    // llama.cpp requires n_rot to be exactly n_embd / n_head
    model.hparams.n_rot   = model.hparams.n_embd / model.hparams.n_head;
    model.hparams.f_norm_rms_eps  = params.f_norm_rms_eps;
    model.hparams.rope_freq_base  = params.rope_freq_base;
    model.hparams.rope_freq_scale = params.rope_freq_scale;

    struct train_state      * train = init_train_state();
    struct ggml_opt_context * opt   = train->opt;

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
    bool existed = load_checkpoint_file(params.common.fn_checkpoint_in, &model, train);
    if (existed) {
        // overwrite last n_ctx with user provided n_ctx
        if (params.common.custom_n_ctx) {
            model.hparams.n_ctx = params.common.n_ctx;
        }

        const bool opt_past_changed = opt->params.past != params.common.opt_past;

        if (opt_past_changed) {
            die("Optimizer parameter '--opt-past N' differs from checkpoint file. To use different value train from scratch with empty input checkpoint, e.g --checkpoint-in ''. Aborting");
            // need to discard previous optimizer past function value statistics and opt_init with new shapes
            // TODO
        }
    } else {
        init_model(&model);
        randomize_model(&model, params.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
        if (!params.only_write_model) {
            ggml_opt_init(opt->ctx, opt, opt->params, get_parameter_count(&model));
        }
    }
    opt->iter = train->train_its;

    print_params(&model.hparams);
    printf("%s: total train_iterations %llu\n", __func__, (long long unsigned) train->train_its);
    printf("%s: seen train_samples     %llu\n", __func__, (long long unsigned) train->train_samples);
    printf("%s: seen train_tokens      %llu\n", __func__, (long long unsigned) train->train_tokens);
    printf("%s: completed train_epochs %llu\n", __func__, (long long unsigned) train->train_epochs);
    printf("%s: model_size = %zu bytes (%.1f MB)\n", __func__, (ggml_used_mem(model.ctx) + model.data.size()), (float) (ggml_used_mem(model.ctx) + model.data.size()) / (1024.0f*1024.0f));

    if (params.only_write_model) {
        save_train_files_data save_data;
        save_data.fn_checkpoint_out = "";
        save_data.fn_model_out      = params.fn_model_out;
        save_data.fn_vocab_model    = params.fn_vocab_model;
        save_data.pattern_fn_it     = params.common.pattern_fn_it;
        save_data.fn_latest         = params.common.fn_latest;
        save_data.model             = &model;

        save_train_files(&save_data, train);

        free_train_state(train);
        ggml_free(model.ctx);
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

    ggml_allocr * alloc = NULL;

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
    alloc = ggml_allocr_new(mem_input_data.data(), mem_input_data.size(), tensor_alignment);
    ggml_allocr_alloc(alloc, tokens_input);
    ggml_allocr_alloc(alloc, target_probs);

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
        alloc = ggml_allocr_new_measure(tensor_alignment);
        gf = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
        gf->order = (enum ggml_cgraph_eval_order) order;
        gb = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
        gb_tmp = params.common.use_checkpointing
            ? ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true)
            : NULL;
        loss = llama_build_train_graphs(
            &model, alloc, ctx_compute,
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
    alloc = ggml_allocr_new(mem_compute_data.data(), mem_compute_data.size(), tensor_alignment);
    gf = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
    gf->order = best_order;
    gb = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
    gb_tmp = params.common.use_checkpointing
        ? ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true)
        : NULL;
    loss = llama_build_train_graphs(
        &model, alloc, ctx_compute,
        gf, gb, gb_tmp,
        &logits, tokens_input, target_probs,
        n_tokens, n_batch,
        params.common.use_flash,
        params.common.use_checkpointing
    );

    std::vector<llama_token> train_tokens;
    std::vector<size_t> train_samples_begin;
    std::vector<size_t> train_samples_size;
    printf("%s: tokenize training data\n", __func__);
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
    save_data.fn_model_out      = params.fn_model_out;
    save_data.fn_vocab_model    = params.fn_vocab_model;
    save_data.pattern_fn_it     = params.common.pattern_fn_it;
    save_data.fn_latest         = params.common.fn_latest;
    save_data.model             = &model;

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
    ggml_free(model.ctx);
    llama_free(lctx);
    llama_free_model(lmodel);
    return 0;
}
