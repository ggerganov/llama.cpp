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

    my_llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<my_llama_layer> layers;
};

// gguf constants (sync with gguf.py)

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

static void init_model(struct my_llama_model * model) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;
    const uint32_t n_ff    = hparams.n_ff;

    struct ggml_context * ctx = model->ctx;

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

    // rope has so much parameters that we make a custom function for it
    auto rope = [ctx, n_rot, n_ctx, rope_freq_base, rope_freq_scale]
                (struct ggml_tensor * t) -> struct ggml_tensor * {
        // not capturing these, to silcence warnings
        const int n_past    = 0;
        const int rope_mode = 0;

        return ggml_rope_custom(ctx,
            t, n_past, n_rot, rope_mode, n_ctx,
            rope_freq_base, rope_freq_scale);
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

    struct ggml_tensor * kv_scale = NULL;
    if (!enable_flash_attn) {
        kv_scale = ggml_new_f32(ctx, 1.0f/sqrtf(float(n_embd)/n_head));
    }

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
        *gb = *gf;
        ggml_build_backward_expand(ctx, gf, gb, true);
    }

    if (alloc) {
        // make sure some tensors are not reallocated by inserting new temporary nodes depending on them
        int n_leafs_before = gb->n_leafs;
        int n_nodes_before = gb->n_nodes;
        struct ggml_tensor * one = ggml_new_f32(ctx, 1.0f);
        // output tensors
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t35, one));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36, one));
        // input gradient
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36->grad, one));
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

static void save_llama_model_file(const char * filename, const char * fn_vocab_model, struct my_llama_model * model, const char * pattern_it, int iteration, const char * latest) {
    std::string sit = (iteration >= 0) ? std::to_string(iteration) : std::string(latest);
    std::string fn = replace_str(filename, pattern_it, sit.c_str());
    printf("%s: saving to %s\n", __func__, fn.c_str());
    struct gguf_context * fctx = gguf_init_empty();

    save_llama_model_gguf(fctx, fn_vocab_model, model);

    // write file
    const bool only_meta = false;
    gguf_write_to_file(fctx, fn.c_str(), only_meta);
    gguf_free(fctx);
}

static void load_checkpoint_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct my_llama_model * model, struct train_state * train) {
    load_llama_model_gguf(fctx, f_ggml_ctx, model);
    if (!load_train_state_gguf(fctx, f_ggml_ctx, train)) {
        printf("%s: loaded llama model as checkpoint\n", __func__);
    }
}

static void save_checkpoint_gguf(struct gguf_context * fctx, const char * fn_vocab_model, struct my_llama_model * model, struct train_state * train) {
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

static void save_checkpoint_file(const char * filename, const char * fn_vocab_model, struct my_llama_model * model, struct train_state * train, const char * pattern_it, int iteration, const char * latest) {
    std::string sit = (iteration >= 0) ? std::to_string(iteration) : std::string(latest);
    std::string fn = replace_str(filename, pattern_it, sit.c_str());
    printf("%s: saving to %s\n", __func__, fn.c_str());
    struct gguf_context * fctx = gguf_init_empty();

    save_checkpoint_gguf(fctx, fn_vocab_model, model, train);

    // write file
    const bool only_meta = false;
    gguf_write_to_file(fctx, fn.c_str(), only_meta);
    gguf_free(fctx);
}

struct train_params {
    const char * fn_vocab_model;
    const char * fn_train_data;
    const char * fn_checkpoint_in;
    const char * fn_checkpoint_out;
    const char * fn_model_out;
    const char * pattern_fn_it;
    const char * fn_latest;

    int save_every;

    uint32_t seed;

    int n_ctx;
    int n_embd;
    int n_head;
    int n_layer;
    int n_ff;

    int n_threads;
    int n_examples;
    int n_batch;
    int n_gradient_accumulation;

    float f_norm_rms_eps;
    float rope_freq_base;
    float rope_freq_scale;

    int print_info_interval;

    bool use_flash;
    bool use_checkpointing;
    bool use_alloc;

    std::string sample_start;
    bool include_sample_start;
    bool escape;
    bool overlapping_samples;
    bool fill_with_next_samples;
    bool separate_with_eos;
    bool separate_with_bos;

    bool force_reshuffle;

    int   warmup;
    int   cos_decay_steps;
    float cos_decay_restart;
    float cos_decay_min;
    bool  enable_restart;

    int   opt_past;
    float opt_delta;
    int   opt_max_no_improvement;

    int   adam_n_iter;
    float adam_alpha;
    float adam_min_alpha;
    float adam_decay;
    int   adam_decay_min_ndim;
    float adam_beta1;
    float adam_beta2;
    float adam_gclip;
    float adam_eps_f;

    int mem_model_gb;
    int mem_compute_gb;
    int mem_compute0_gb;
};

struct train_params get_default_train_params() {
    struct train_params params;
    params.fn_vocab_model    = "ggml-vic7b-uncensored-q4_0.bin";
    params.fn_train_data     = "shakespeare.txt";
    params.fn_checkpoint_in  = "checkpoint.bin";
    params.fn_checkpoint_out = "checkpoint.bin";
    params.fn_model_out      = "ggml-checkpoint-f32.bin";
    params.pattern_fn_it     = "ITERATION";
    params.fn_latest         = "LATEST";

    params.save_every = 10;

    params.seed       =   -1;

    params.n_ctx      =  128;
    params.n_embd     =  256;
    params.n_head     =    8;
    params.n_layer    =   16;
    params.n_ff       =  768;

    params.n_threads  =    6;
    params.n_examples =    1;
    params.n_batch    =    8;
    params.n_gradient_accumulation = 1;

    params.f_norm_rms_eps  = 1e-5f;
    params.rope_freq_base  = 10000.0f;
    params.rope_freq_scale = 1.0f;

    params.print_info_interval    = 1;

    params.use_flash              = true;
    params.use_checkpointing      = true;
    params.use_alloc              = true;

    params.sample_start           = "";
    params.include_sample_start   = false;
    params.escape                 = false;
    params.overlapping_samples    = false;
    params.fill_with_next_samples = false;
    params.separate_with_eos      = false;
    params.separate_with_bos      = true;
    params.force_reshuffle        = false;

    params.opt_past               = 0;
    params.opt_delta              = 1e-5f;
    params.opt_max_no_improvement = 0;

    params.warmup            =  100;
    params.cos_decay_steps   = 1000;
    params.cos_decay_restart = 1.1f;
    params.cos_decay_min     = 0.1f;
    params.enable_restart    = false;

    params.adam_n_iter         = 256;
    params.adam_alpha          = 1e-3f;
    params.adam_min_alpha      = 0;
    params.adam_decay          = 1e-1f;
    params.adam_decay_min_ndim = 2;
    params.adam_beta1          = 0.9f;
    params.adam_beta2          = 0.999f;
    params.adam_gclip          = 1.0f;
    params.adam_eps_f          = 0.0f;

    params.mem_model_gb   =  2;
    params.mem_compute_gb = 24;
    params.mem_compute0_gb = 8;
    return params;
}

static void train_print_usage(int /*argc*/, char ** argv, const struct train_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                 show this help message and exit\n");
    fprintf(stderr, "  --vocab-model FNAME        model path from which to load vocab (default '%s')\n", params->fn_vocab_model);
    fprintf(stderr, "  --train-data FNAME         path from which to load training data (default '%s')\n", params->fn_train_data);
    fprintf(stderr, "  --checkpoint-in FNAME      path from which to load training checkpoint (default '%s')\n", params->fn_checkpoint_in);
    fprintf(stderr, "  --checkpoint-out FNAME     path to save training checkpoint (default '%s')\n", params->fn_checkpoint_out);
    fprintf(stderr, "  --model-out FNAME          path to save ggml model (default '%s')\n", params->fn_model_out);
    fprintf(stderr, "  --pattern-fn-it STR        pattern in output filenames to be replaced by iteration number (default '%s')\n", params->pattern_fn_it);
    fprintf(stderr, "  --fn-latest STR            string to use instead of iteration number for saving latest output (default '%s')\n", params->fn_latest);
    fprintf(stderr, "  --save-every N             save checkpoint and lora every N iterations. Disabled when N <= 0. (default '%d')\n", params->save_every);
    fprintf(stderr, "  -s SEED, --seed SEED       RNG seed (default: -1, use random seed for -1)\n");
    fprintf(stderr, "  -c N, --ctx N              Context size used during training (default %d)\n", params->n_ctx);
    fprintf(stderr, "  --embd N                   Embedding size used for new models (default %d)\n", params->n_embd);
    fprintf(stderr, "  --ff N                     Feedforward size used for new models. (default %d)\n", params->n_ff);
    fprintf(stderr, "  --head N                   Number of heads for new models (default %d)\n", params->n_head);
    fprintf(stderr, "  --layer N                  Number of layers for new models (default %d)\n", params->n_layer);
    fprintf(stderr, "  --norm-rms-eps F           RMS-Norm epsilon value (default %f)\n", params->f_norm_rms_eps);
    fprintf(stderr, "  --rope-freq-base F         Frequency base for ROPE (default %f)\n", params->rope_freq_base);
    fprintf(stderr, "  --rope-freq-scale F        Frequency scale for ROPE (default %f)\n", params->rope_freq_scale);
    fprintf(stderr, "  -t N, --threads N          Number of threads (default %d)\n", params->n_threads);
    fprintf(stderr, "  -n N, --examples N         Number of examples to train (default %d)\n", params->n_examples);
    fprintf(stderr, "  -b N, --batch N            Parallel batch size (default %d)\n", params->n_batch);
    fprintf(stderr, "  --grad-acc N               Number of gradient accumulation steps (simulates larger batch size of batch*gradacc) (default %d)\n", params->n_gradient_accumulation);
    fprintf(stderr, "  --print-info-interval N    Print infos during training each N examples (default %d)\n", params->print_info_interval);
    fprintf(stderr, "  --sample-start STR         Sets the starting point for samples after the specified pattern. If empty use every token position as sample start. (default '%s')\n", params->sample_start.c_str());
    fprintf(stderr, "  --include-sample-start     Include the sample start in the samples. (default off)\n");
    fprintf(stderr, "  --escape                   process sample start escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\)\n");
    fprintf(stderr, "  --overlapping-samples      Samples my overlap, will include sample-start of second and following samples. When off, samples will end at begin of next sample. (default off)\n");
    fprintf(stderr, "  --fill-with-next-samples   Samples shorter than context length will be followed by the next (shuffled) samples. (default off)\n");
    fprintf(stderr, "  --separate-with-eos        When fill-with-next-samples, insert end-of-sequence token between samples.%s\n", params->separate_with_eos ? " (default)" : "");
    fprintf(stderr, "  --separate-with-bos        When fill-with-next-samples, insert begin-of-sequence token between samples.%s\n", params->separate_with_bos ? " (default)" : "");
    fprintf(stderr, "  --no-separate-with-eos     When fill-with-next-samples, don't insert end-of-sequence token between samples.%s\n", !params->separate_with_eos ? " (default)" : "");
    fprintf(stderr, "  --no-separate-with-bos     When fill-with-next-samples, don't insert begin-of-sequence token between samples.%s\n", !params->separate_with_bos ? " (default)" : "");
    fprintf(stderr, "  --force-reshuffle          Force a reshuffling of data at program start, otherwise the shuffling of loaded checkpoint is resumed.\n");
    fprintf(stderr, "  --no-flash                 Don't use flash attention \n");
    fprintf(stderr, "  --use-flash                Use flash attention (default)\n");
    fprintf(stderr, "  --no-checkpointing         Don't use gradient checkpointing\n");
    fprintf(stderr, "  --use-checkpointing        Use gradient checkpointing (default)\n");
    fprintf(stderr, "  --no-alloc                 Don't use allocator\n");
    fprintf(stderr, "  --use-alloc                Use allocator (default)\n");
    fprintf(stderr, "  --warmup N                 Only for Adam optimizer. Number of warmup steps (default %d)\n", params->warmup);
    fprintf(stderr, "  --cos-decay-steps N        Only for Adam optimizer. Number of cosine decay steps (default %d)\n", params->cos_decay_steps);
    fprintf(stderr, "  --cos-decay-restart N      Only for Adam optimizer. Increase of cosine decay steps after restart (default %f)\n", params->cos_decay_restart);
    fprintf(stderr, "  --cos-decay-min N          Only for Adam optimizer. Cosine decay minimum (default %f)\n", params->cos_decay_min);
    fprintf(stderr, "  --enable-restart N         Only for Adam optimizer. Enable restarts of cos-decay %s\n", params->enable_restart ? "(default)" : "");
    fprintf(stderr, "  --disable-restart N        Only for Adam optimizer. Disable restarts of cos-decay %s\n", !params->enable_restart ? "(default)" : "");
    fprintf(stderr, "  --opt-past N               Number of optimization iterations to track for delta convergence test. Disabled when zero. (default %d)\n", params->opt_past);
    fprintf(stderr, "  --opt-delta N              Maximum delta for delta convergence test. Disabled when <= zero. (default %f)\n", params->opt_delta);
    fprintf(stderr, "  --opt-max-no-improvement N Maximum number of optimization iterations with no improvement. Disabled when <= zero. (default %d)\n", params->opt_max_no_improvement);
    fprintf(stderr, "  --adam-epsf N              AdamW epsilon for convergence test. Disabled when <= zero. (default %f)\n", params->adam_eps_f);
    fprintf(stderr, "  --adam-iter N              Maximum number of Adam optimization iterations for each batch (default %d)\n", params->adam_n_iter);
    fprintf(stderr, "  --adam-alpha N             Adam learning rate alpha (default %f)\n", params->adam_alpha);
    fprintf(stderr, "  --adam-min-alpha N         Adam minimum learning rate alpha - including warmup phase (default %f)\n", params->adam_min_alpha);
    fprintf(stderr, "  --adam-decay N             AdamW weight decay. Values greater zero enable AdamW instead of regular Adam. (default %f)\n", params->adam_decay);
    fprintf(stderr, "  --adam-decay-min-ndim N    Minimum number of tensor dimensions to apply AdamW weight decay. Weight decay is not applied to tensors with less n_dims. (default %d)\n", params->adam_decay_min_ndim);
    fprintf(stderr, "  --adam-beta1 N             AdamW beta1 in interval [0,1). How much to smooth the first moment of gradients. (default %f)\n", params->adam_beta1);
    fprintf(stderr, "  --adam-beta2 N             AdamW beta2 in interval [0,1). How much to smooth the second moment of gradients. (default %f)\n", params->adam_beta2);
    fprintf(stderr, "  --adam-gclip N             AdamW gradient clipping. Disabled when zero. (default %f)\n", params->adam_gclip);
    fprintf(stderr, "  --mem-model N              Memory to allocate for model and cache in gigabytes. (default %d)\n", params->mem_model_gb);
    fprintf(stderr, "  --mem-compute N            Memory to allocate for compute in gigabytes. (default %d)\n", params->mem_compute_gb);
    fprintf(stderr, "  --mem-compute0 N           Memory to allocate for automatic memory allocator in gigabytes. (default %d)\n", params->mem_compute0_gb);
    fprintf(stderr, "\n");
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

        if (arg == "--vocab-model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_vocab_model = argv[i];
        } else if (arg == "--train-data") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_train_data = argv[i];
        } else if (arg == "--checkpoint-in") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_checkpoint_in = argv[i];
        } else if (arg == "--checkpoint-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_checkpoint_out = argv[i];
        } else if (arg == "--model-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_model_out = argv[i];
        } else if (arg == "--pattern-fn-it") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->pattern_fn_it = argv[i];
        } else if (arg == "--fn-latest") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_latest = argv[i];
        } else if (arg == "--save-every") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->save_every = std::stoi(argv[i]);
        } else if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->seed = std::stoi(argv[i]);
        } else if (arg == "-c" || arg == "--ctx") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_ctx = std::stoi(argv[i]);
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
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_threads = std::stoi(argv[i]);
        } else if (arg == "-b" || arg == "--batch") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_batch = std::stoi(argv[i]);
        } else if (arg == "--grad-acc") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_gradient_accumulation = std::max(1, std::stoi(argv[i]));
        } else if (arg == "-n" || arg == "--examples") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_examples = std::stoi(argv[i]);
        } else if (arg == "--print-info-interval") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->print_info_interval = std::stoi(argv[i]);
         } else if (arg == "--sample-start") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->sample_start = std::string(argv[i]);
        } else if (arg == "--escape") {
            params->escape = true;
        } else if (arg == "--include-sample-start") {
            params->include_sample_start = true;
        } else if (arg == "--overlapping-samples") {
            params->overlapping_samples = true;
        } else if (arg == "--fill-with-next-samples") {
            params->fill_with_next_samples = true;
        } else if (arg == "--separate-with-eos") {
            params->separate_with_eos = true;
        } else if (arg == "--separate-with-bos") {
            params->separate_with_bos = true;
        } else if (arg == "--no-separate-with-eos") {
            params->separate_with_eos = false;
        } else if (arg == "--no-separate-with-bos") {
            params->separate_with_bos = false;
        } else if (arg == "--force-reshuffle") {
            params->force_reshuffle = true;
        } else if (arg == "--no-flash") {
            params->use_flash = false;
        } else if (arg == "--use-flash") {
            params->use_flash = true;
        } else if (arg == "--no-checkpointing") {
            params->use_checkpointing = false;
        } else if (arg == "--use-checkpointing") {
            params->use_checkpointing = true;
        } else if (arg == "--no-alloc") {
            params->use_alloc = false;
        } else if (arg == "--use-alloc") {
            params->use_alloc = true;
        } else if (arg == "--warmup") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->warmup = std::stoi(argv[i]);
        } else if (arg == "--cos-decay-steps") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->cos_decay_steps = std::stof(argv[i]);
        } else if (arg == "--cos-decay-restart") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->cos_decay_restart = std::stof(argv[i]);
        } else if (arg == "--cos-decay-min") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->cos_decay_min = std::stof(argv[i]);
        } else if (arg == "--enable-restart") {
            params->enable_restart = true;
        } else if (arg == "--disable-restart") {
            params->enable_restart = false;
        } else if (arg == "--opt-past") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->opt_past = std::stoi(argv[i]);
        } else if (arg == "--opt-delta") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->opt_delta = std::stof(argv[i]);
        } else if (arg == "--opt-max-no-improvement") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->opt_max_no_improvement = std::stoi(argv[i]);
        } else if (arg == "--adam-epsf") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_eps_f = std::stof(argv[i]);
        } else if (arg == "--adam-iter") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_n_iter = std::stoi(argv[i]);
        } else if (arg == "--adam-alpha") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_alpha = std::stof(argv[i]);
        } else if (arg == "--adam-min-alpha") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_min_alpha = std::stof(argv[i]);
        } else if (arg == "--adam-decay") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_decay = std::stof(argv[i]);
        } else if (arg == "--adam-decay-min-ndim") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_decay_min_ndim = std::stoi(argv[i]);
        } else if (arg == "--adam-beta1") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_beta1 = std::stof(argv[i]);
        } else if (arg == "--adam-beta2") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_beta2 = std::stof(argv[i]);
        } else if (arg == "--adam-gclip") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_gclip = std::stof(argv[i]);
        } else if (arg == "--mem-model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->mem_model_gb = std::stoi(argv[i]);
        } else if (arg == "--mem-compute") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->mem_compute_gb = std::stoi(argv[i]);
        } else if (arg == "--mem-compute0") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->mem_compute0_gb = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            train_print_usage(argc, argv, &default_params);
            exit(0);
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
    if (params->escape) {
        process_escapes(params->sample_start);
    }

    return true;
}

struct opt_callback_data {
    struct train_params   * params;
    struct train_state    * train;
    struct my_llama_model * model;
    struct llama_context  * lctx;
    int                     last_save_iter;
    llama_token           * tokens_data;
    size_t                  tokens_size;
    size_t                * samples_begin;
    size_t                * samples_size;
    size_t                * shuffled_samples_begin;
    size_t                * shuffled_samples_size;
    size_t                  samples_count;
    struct ggml_tensor    * tokens_input;
    struct ggml_tensor    * target_logits;
    struct ggml_tensor    * target_probs;
    int                     first_iter;
    int64_t                 last_time;
    double                  millis_per_iter;
};

static void opt_callback(void * vdata, int accum_step, float * sched) {
    struct opt_callback_data * data   = (struct opt_callback_data *) vdata;
    struct train_params      * params = data->params;
    struct train_state       * train  = data->train;
    struct ggml_opt_context  * opt    = train->opt;
    int n_batch = params->n_batch;
    int n_ctx = params->n_ctx;

    if (accum_step == 0) {
        // time measurement
        int64_t now = ggml_time_ms();
        if (now > data->last_time && opt->iter > data->first_iter) {
            double dt = now - data->last_time;
            if (data->millis_per_iter == 0.0) {
                data->millis_per_iter = dt;
            } else {
                const double gain = 0.7;
                data->millis_per_iter = data->millis_per_iter*(1.0-gain) + dt*gain;
            }
        }

        double remaining_millis = 0.0;
        if (data->millis_per_iter > 0.0) {
            const int n_iter = params->adam_n_iter;
            const int done_iter = opt->iter - data->first_iter;
            const int remaining_iter = n_iter - done_iter;
            remaining_millis = remaining_iter * data->millis_per_iter;
        }

        // file saving
        const bool save_now = (params->save_every > 0) && (opt->iter - data->last_save_iter >= params->save_every);
        if (save_now) {
            int new_iters = opt->iter - data->last_save_iter;
            train->train_its += new_iters;
            train->train_samples += new_iters * opt->params.n_gradient_accumulation * n_batch;
            train->train_tokens  += new_iters * opt->params.n_gradient_accumulation * n_batch * n_ctx;

            if (strlen(params->fn_checkpoint_out) > 0) {
                save_checkpoint_file(params->fn_checkpoint_out, params->fn_vocab_model, data->model, train, params->pattern_fn_it, opt->iter, params->fn_latest);
                save_checkpoint_file(params->fn_checkpoint_out, params->fn_vocab_model, data->model, train, params->pattern_fn_it, -1, params->fn_latest);

            }
            if (strlen(params->fn_model_out) > 0) {
                save_llama_model_file(params->fn_model_out, params->fn_vocab_model, data->model, params->pattern_fn_it, opt->iter, params->fn_latest);
                save_llama_model_file(params->fn_model_out, params->fn_vocab_model, data->model, params->pattern_fn_it, -1, params->fn_latest);
            }
            data->last_save_iter = opt->iter;
        }

        // exclude file saving from time measurement, by measuring last_time after saving
        data->last_time = ggml_time_ms();

        *sched = learning_schedule(
            opt->iter,
            params->warmup,
            params->cos_decay_steps,
            params->adam_alpha,
            params->adam_min_alpha,
            params->cos_decay_min,
            params->cos_decay_restart,
            params->enable_restart);

        int impr_plot = -(int)(1 + (opt->loss_before - opt->loss_after) * 10.0f + 0.5f);
        if (impr_plot > 0) impr_plot = 0;
        if (std::isnan(opt->loss_before) || std::isnan(opt->loss_before)) impr_plot = 0;
        printf("%s: iter=%6d sample=%zu/%zu sched=%f loss=%f",
            __func__, opt->iter, std::min(1+train->shuffle_next_sample, train->shuffle_sample_count), train->shuffle_sample_count,
            *sched, opt->loss_after);


        if (data->millis_per_iter > 0) {
            printf(" dt=");
            print_duration(data->millis_per_iter);
            printf(" eta=");
            print_duration(remaining_millis);
        }

        float improvement = opt->loss_before - opt->loss_after;
        const float plot_scale = 10.0f;
        int bar_len = (int)(1 + improvement*plot_scale + 0.5);
        printf(" |");
        for (int i=0; i<bar_len; ++i) {
            printf("-");
        }
        printf(">");
        printf("\n");
    }

    int64_t used_samples = get_example_targets_batch(
        data->lctx,
        data->tokens_input,
        data->target_probs,
        train->shuffle_next_sample,
        data->shuffled_samples_begin,
        data->shuffled_samples_size,
        data->samples_count,
        data->tokens_data,
        data->tokens_size,
        params->separate_with_eos,
        params->separate_with_bos,
        params->fill_with_next_samples);

    train->shuffle_next_sample += used_samples;

    if (train->shuffle_next_sample >= train->shuffle_sample_count) {
        ++train->train_epochs;
        printf("%s: reshuffle samples. completed epochs: %llu\n", __func__, (long long unsigned) train->train_epochs);
        // note: we may have used some samples from the current shuffling more than once
        train->shuffle_rng_state_current = train->shuffle_rng_state_next;
        train->shuffle_rng_state_next = shuffle_samples(
            train->shuffle_rng_state_current,
            data->shuffled_samples_begin,
            data->shuffled_samples_size,
            data->samples_begin,
            data->samples_size,
            data->samples_count);
        train->shuffle_next_sample = 0;
    }

}

int main(int argc, char ** argv) {
    struct train_params params = get_default_train_params();

    if (!train_params_parse(argc, argv, &params)) {
        return 1;
    }

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }
    printf("%s: seed: %u\n", __func__, params.seed);
    srand(params.seed);

    struct llama_context_params llama_params = llama_context_default_params();
    llama_params.vocab_only = true;

    struct llama_model * lmodel = llama_load_model_from_file(params.fn_vocab_model, llama_params);
    struct llama_context * lctx = llama_new_context_with_model(lmodel, llama_params);

    struct my_llama_model model;
    model.hparams.n_vocab = llama_n_vocab(lctx);
    model.hparams.n_ctx   = params.n_ctx;
    model.hparams.n_embd  = params.n_embd;
    model.hparams.n_head  = params.n_head;
    model.hparams.n_layer = params.n_layer;
    model.hparams.n_ff    = params.n_ff;
    // llama.cpp requires n_rot to be exactly n_embd / n_head
    model.hparams.n_rot   = model.hparams.n_embd / model.hparams.n_head;
    model.hparams.f_norm_rms_eps  = params.f_norm_rms_eps;
    model.hparams.rope_freq_base  = params.rope_freq_base;
    model.hparams.rope_freq_scale = params.rope_freq_scale;

    print_params(&model.hparams);

    struct ggml_init_params lcparams;
    lcparams.mem_size   = 1024ll*1024ll*1024ll*((size_t) params.mem_model_gb);
    lcparams.mem_buffer = NULL;
    lcparams.no_alloc   = false;

    model.ctx = ggml_init(lcparams);

    int n_tokens = model.hparams.n_ctx;
    int n_vocab  = model.hparams.n_vocab;
    int n_batch  = params.n_batch;

    struct train_state      * train = init_train_state(params.seed);
    struct ggml_opt_context * opt   = train->opt;

    struct ggml_opt_params opt_params_adam = ggml_opt_default_params(GGML_OPT_ADAM);
    opt_params_adam.print_forward_graph     = false;
    opt_params_adam.print_backward_graph    = false;
    opt_params_adam.n_threads               = params.n_threads;
    opt_params_adam.past                    = params.opt_past;
    opt_params_adam.delta                   = params.opt_delta;
    opt_params_adam.max_no_improvement      = params.opt_max_no_improvement;
    opt_params_adam.n_gradient_accumulation = params.n_gradient_accumulation;
    opt_params_adam.adam.n_iter             = params.adam_n_iter;
    opt_params_adam.adam.sched              = 1.0f;
    opt_params_adam.adam.alpha              = params.adam_alpha;
    opt_params_adam.adam.decay              = params.adam_decay;
    opt_params_adam.adam.decay_min_ndim     = params.adam_decay_min_ndim;
    opt_params_adam.adam.beta1              = params.adam_beta1;
    opt_params_adam.adam.beta2              = params.adam_beta2;
    opt_params_adam.adam.gclip              = params.adam_gclip;
    opt_params_adam.adam.eps_f              = params.adam_eps_f;

    opt->ctx = model.ctx;
    opt->params = opt_params_adam;

    printf("%s: init model\n", __func__);
    bool existed = load_checkpoint_file(params.fn_checkpoint_in, &model, train);
    if (!existed) {
        init_model(&model);
    }
    set_param_model(&model);

    opt->params = opt_params_adam;

    opt->iter = train->train_its;
    printf("%s: opt iter %d\n", __func__, opt->iter);

    bool from_scratch = !existed;
    if (from_scratch) {
        randomize_model(&model, params.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    }

    printf("used_mem model: %zu bytes\n", ggml_used_mem(model.ctx));
    // ggml_print_tensor_objects(model.ctx);

    // TODO: use std::vector<uint8_t> intead of "new"
    size_t    compute_size = 1024ll*1024ll*1024ll*((size_t) params.mem_compute_gb);
    uint8_t * compute_addr = new uint8_t[compute_size];

    size_t size_buf_0 = 1024ll*1024ll*1024ll*((size_t) params.mem_compute0_gb);
    uint8_t * compute_buf_0 = new uint8_t[size_buf_0];

    ggml_allocr * alloc = NULL;
    if (params.use_alloc) {
        static const size_t tensor_alignment = 32;
        alloc = ggml_allocr_new(compute_buf_0, size_buf_0, tensor_alignment);
    }

    std::vector<llama_token> train_tokens;
    std::vector<size_t> train_samples_begin;
    std::vector<size_t> train_samples_size;
    printf("%s: tokenize training data\n", __func__);
    tokenize_file(lctx,
            params.fn_train_data,
            params.sample_start,
            params.include_sample_start,
            params.overlapping_samples,
            n_tokens,
            train_tokens,
            train_samples_begin,
            train_samples_size);
    GGML_ASSERT(train_samples_begin.size() == train_samples_size.size());

    printf("%s: number of training tokens: %zu\n", __func__, train_tokens.size());

    size_t shuffle_samples_hash = compute_samples_hash(params.fn_train_data, train_samples_begin.data(), train_samples_size.data(), train_samples_size.size());
    const bool changed_train_data = (shuffle_samples_hash != train->shuffle_samples_hash) || (train->shuffle_sample_count != train_samples_size.size());
    if (changed_train_data) {
        printf("%s: train data seems to have changed. restarting shuffled epoch.\n", __func__);
    }
    if (params.force_reshuffle) {
        printf("%s: forced reshuffling of data. restarting with newly shuffled epoch.\n", __func__);
    }
    if ((train->shuffle_rng_state_current == "") || changed_train_data || params.force_reshuffle) {
        train->shuffle_rng_state_current = mt19937_seed_to_state(params.seed);
        train->shuffle_sample_count = train_samples_size.size();
        train->shuffle_next_sample = 0;
        train->shuffle_samples_hash = shuffle_samples_hash;
    }
    std::vector<size_t> train_shuffled_samples_begin;
    std::vector<size_t> train_shuffled_samples_size;
    train_shuffled_samples_begin.resize(train_samples_begin.size());
    train_shuffled_samples_size.resize(train_samples_size.size());
    train->shuffle_rng_state_next = shuffle_samples(
        train->shuffle_rng_state_current,
        train_shuffled_samples_begin.data(),
        train_shuffled_samples_size.data(),
        train_samples_begin.data(),
        train_samples_size.data(),
        train_samples_size.size());
    printf("%s: begin training\n", __func__);

    struct opt_callback_data opt_cb_data;
    opt_cb_data.params = &params;
    opt_cb_data.train = train;
    opt_cb_data.model = &model;
    opt_cb_data.lctx = lctx;
    opt_cb_data.last_save_iter = opt->iter;
    opt_cb_data.tokens_data = train_tokens.data();
    opt_cb_data.tokens_size = train_tokens.size();
    opt_cb_data.samples_begin          = train_samples_begin.data();
    opt_cb_data.samples_size           = train_samples_size.data();
    opt_cb_data.shuffled_samples_begin = train_shuffled_samples_begin.data();
    opt_cb_data.shuffled_samples_size  = train_shuffled_samples_size.data();
    opt_cb_data.samples_count          = train_samples_size.size();
    opt_cb_data.tokens_input           = NULL;
    opt_cb_data.target_logits          = NULL;
    opt_cb_data.target_probs           = NULL;
    opt_cb_data.first_iter             = opt->iter;
    opt_cb_data.last_time              = ggml_time_ms();
    opt_cb_data.millis_per_iter        = 0.0;

    int64_t t0 = ggml_time_ms();

    for (int ex = 0; ex < params.n_examples; ++ex) {

        struct ggml_init_params cparams = {
            compute_size, // mem_size
            compute_addr, // mem_buffer
            false,        // no_alloc
        };
        struct ggml_context * ctx0 = ggml_init(cparams);

        ggml_set_no_alloc(ctx0, false);

        // don't use alloc for input tensors, so we can safely fill them with data
        //struct ggml_tensor * after_opt_best_samples = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        //struct ggml_tensor * after_opt_probs        = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
        struct ggml_tensor * tokens_input           = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        struct ggml_tensor * target_logits          = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
        struct ggml_tensor * target_probs           = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);

        ggml_set_no_alloc(ctx0, (alloc != NULL));

        if (alloc) {
            ggml_allocr_reset(alloc);
        }

        opt_cb_data.tokens_input  = tokens_input;
        opt_cb_data.target_logits = target_logits;
        opt_cb_data.target_probs  = target_probs;

        int n_past = 0;

        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        struct ggml_cgraph * gb = ggml_new_graph(ctx0);
        struct ggml_cgraph * gb_tmp = params.use_checkpointing
            ? ggml_new_graph(ctx0)
            : NULL;

        GGML_ASSERT(n_past == 0);

        struct ggml_tensor * loss   = NULL;
        struct ggml_tensor * logits = NULL;

        loss = llama_build_train_graphs(
            &model, alloc, ctx0,
            gf, gb, gb_tmp,
            &logits, tokens_input, target_probs,
            n_tokens, n_batch,
            params.use_flash,
            params.use_checkpointing
        );

        size_t used_mem_before_opt = ggml_used_mem(ctx0);

        opt->params.adam.sched = learning_schedule(
            opt->iter,
            params.warmup,
            params.cos_decay_steps,
            params.adam_alpha,
            params.adam_min_alpha,
            params.cos_decay_min,
            params.cos_decay_restart,
            params.enable_restart);

        printf("%s: opt->params.adam.sched %.5f\n", __func__, opt->params.adam.sched);

        ggml_opt_resume_g(ctx0, opt, loss, gf, gb, &opt_callback, (void *) &opt_cb_data);

        size_t used_mem_after_opt = ggml_used_mem(ctx0);

        int n_iter = params.adam_n_iter;
        train->train_its = opt->iter;
        train->train_samples += n_batch * n_iter;
        train->train_tokens  += n_batch * n_tokens * n_iter;

        if (params.print_info_interval > 0 && ex % params.print_info_interval == 0) {
            printf("Example %d, opt iter %d\n", ex, opt->iter);
            printf("error_before_opt: %.6f\n", opt->loss_before);
            printf("error_after_opt:  %.6f\n", opt->loss_after);
            printf("used_mem_before_opt: %zu bytes\n", used_mem_before_opt);
            printf("used_mem_after_opt:  %zu bytes\n", used_mem_after_opt);
        }

        ggml_free(ctx0);
    }

    int64_t t1 = ggml_time_ms();
    int64_t d  = t1-t0;
    double  dd = (double) d * 1e-3;
    printf("%s: total training time=%f seconds\n", __func__, dd);

    int new_iters = opt->iter - opt_cb_data.last_save_iter;
    train->train_its += new_iters;
    train->train_samples += new_iters * opt->params.n_gradient_accumulation * n_batch;
    train->train_tokens  += new_iters * opt->params.n_gradient_accumulation * n_batch * n_tokens;

    if (params.n_examples > 0) {
        save_checkpoint_file(params.fn_checkpoint_out, params.fn_vocab_model, &model, train, params.pattern_fn_it, opt->iter, params.fn_latest);
        save_checkpoint_file(params.fn_checkpoint_out, params.fn_vocab_model, &model, train, params.pattern_fn_it, -1, params.fn_latest);
    }

    if (strlen(params.fn_model_out) > 0) {
        save_llama_model_file(params.fn_model_out, params.fn_vocab_model, &model, params.pattern_fn_it, opt->iter, params.fn_latest);
        save_llama_model_file(params.fn_model_out, params.fn_vocab_model, &model, params.pattern_fn_it, -1, params.fn_latest);
    }

    opt_cb_data.last_save_iter = opt->iter;

    if (alloc) {
        ggml_allocr_free(alloc);
    }

    delete[] compute_addr;
    delete[] compute_buf_0;
    free_train_state(train);
    ggml_free(model.ctx);
    llama_free(lctx);
    llama_free_model(lmodel);
    return 0;
}
