#include "ggml.h"
#include "ggml-alloc.h"
#include "llama.h"
#include "common.h"
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
#include <sstream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static const size_t tensor_alignment = 32;

struct random_normal_distribution {
    std::mt19937 gen;
    std::normal_distribution<float> rd;
    float min;
    float max;
};

struct random_uniform_distribution {
    std::mt19937 gen;
    std::uniform_real_distribution<float> rd;
};

void init_random_normal_distribution(struct random_normal_distribution * rnd, int seed, float mean, float std, float min, float max) {
    rnd->gen = std::mt19937(seed);
    rnd->rd = std::normal_distribution<float>{mean, std};
    rnd->min = min;
    rnd->max = max;
}

void init_random_uniform_distribution(struct random_uniform_distribution * rnd, int seed, float min, float max) {
    rnd->gen = std::mt19937(seed);
    rnd->rd = std::uniform_real_distribution<float>{min, max};
}

int clamp(const int v, const int min, const int max) {
    return ((v < min) ? (min) : (v > max) ? (max) : v);
}

float fclamp(const float v, const float min, const float max) {
    return ((v < min) ? (min) : (v > max) ? (max) : v);
}

float frand() {
    return (float)rand()/(float)RAND_MAX;
}

float frand_normal(struct random_normal_distribution * rnd) {
    return fclamp(rnd->rd(rnd->gen), rnd->min, rnd->max);
}

float frand_uniform(struct random_uniform_distribution * rnd) {
    return rnd->rd(rnd->gen);
}

struct ggml_tensor * randomize_tensor_normal(struct ggml_tensor * tensor, struct random_normal_distribution * rnd) {
    float scale = 1.0f; // xavier
    switch (tensor->n_dims) {
        case 1:
            scale /= sqrtf(tensor->ne[0]);
            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0]);
                *dst = scale * frand_normal(rnd);
            }
            break;
        case 2:
            scale /= sqrtf(tensor->ne[0]+tensor->ne[1]);
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
                    *dst = scale * frand_normal(rnd);
                }
            }
            break;
        case 3:
            scale /= sqrtf(tensor->ne[0]+tensor->ne[1]);
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
                        *dst = scale * frand_normal(rnd);
                    }
                }
            }
            break;
        case 4:
            scale /= sqrtf(tensor->ne[0]+tensor->ne[1]);
            for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
                for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                            float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3]);
                            *dst = scale * frand_normal(rnd);
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };
    return tensor;
}

struct ggml_tensor * randomize_tensor_uniform(struct ggml_tensor * tensor, struct random_uniform_distribution * rnd) {
    switch (tensor->n_dims) {
        case 1:
            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0]);
                *dst = frand_uniform(rnd);
            }
            break;
        case 2:
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
                    *dst = frand_uniform(rnd);
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
                        *dst = frand_uniform(rnd);
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
                for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                            float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3]);
                            *dst = frand_uniform(rnd);
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };
    return tensor;
}

struct my_llama_hparams {
    uint32_t n_vocab    = 32000;
    uint32_t n_ctx      = 512;
    uint32_t n_embd     = 4096;
    uint32_t n_ff       = 11008;
    uint32_t n_head     = 32;
    uint32_t n_head_kv  = 32;
    uint32_t n_layer    = 32;
    uint32_t n_rot      = 64;

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
    my_llama_hparams hparams;

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

    // float f_norm_eps     = 1e-5f; // falcon
    float f_norm_rms_eps = 1e-5f; // llama

    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;

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

    uint64_t train_its = 0;
    uint64_t train_samples = 0;
    uint64_t train_tokens = 0;
    uint64_t train_epochs = 0;

    size_t      shuffle_samples_hash; // fn, sample_count, *zip(sample_begins, sample_sizes)
    std::string shuffle_rng_state_current;
    std::string shuffle_rng_state_next;
    size_t      shuffle_sample_count;
    size_t      shuffle_next_sample;
};

// gguf constants
const char * LLM_KV_OPTIMIZER_TYPE = "optimizer.type";
const char * LLM_KV_OPTIMIZER_TYPE_ADAM  = "adam";
const char * LLM_KV_OPTIMIZER_TYPE_LBFGS = "lbfgs";
const char * LLM_KV_OPTIMIZER_FILE_VERSION               = "optimizer.file_version";
const char * LLM_KV_OPTIMIZER_CONVERGENCE_PAST_COUNT     = "optimizer.convergence_past_count";
const char * LLM_KV_OPTIMIZER_PARAMETER_COUNT            = "optimizer.parameter_count";
const char * LLM_KV_OPTIMIZER_ITERATION_COUNT            = "optimizer.iteration_count";
const char * LLM_KV_OPTIMIZER_JUST_INITIALIZED           = "optimizer.just_initialized";
const char * LLM_KV_OPTIMIZER_ADAM_BEST_LOSS             = "optimizer.adam.best_loss";
const char * LLM_KV_OPTIMIZER_ADAM_PREVIOUS_LOSS         = "optimizer.adam.previous_loss";
const char * LLM_KV_OPTIMIZER_ADAM_NO_IMPROVEMENT_COUNT  = "optimizer.adam.no_improvement_count";
const char * LLM_KV_OPTIMIZER_LBFGS_APPROX_HESSIAN_COUNT = "optimizer.lbfgs.approx_hessian_count";
const char * LLM_KV_OPTIMIZER_LBFGS_BEST_LOSS            = "optimizer.lbfgs.best_loss";
const char * LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_STEP     = "optimizer.lbfgs.line_search_step";
const char * LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_J        = "optimizer.lbfgs.line_search_j";
const char * LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_K        = "optimizer.lbfgs.line_search_k";
const char * LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_END      = "optimizer.lbfgs.line_search_end";
const char * LLM_KV_OPTIMIZER_LBFGS_NO_IMPROVEMENT_COUNT = "optimizer.lbfgs.no_improvement_count";

const char * LLM_TENSOR_OPTIMIZER_ADAM_FIRST_MOMENTS    = "optimizer.adam.first_moments";
const char * LLM_TENSOR_OPTIMIZER_ADAM_SECOND_MOMENTS   = "optimizer.adam.second_moments";
const char * LLM_TENSOR_OPTIMIZER_ADAM_PAST_LOSS_VALUES = "optimizer.adam.past_loss_values";

const char * LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_PARAMETERS  = "optimizer.lbfgs.current_parameters";
const char * LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_PARAMETERS = "optimizer.lbfgs.previous_parameters";
const char * LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_GRADIENTS   = "optimizer.lbfgs.current_gradients";
const char * LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_GRADIENTS  = "optimizer.lbfgs.previous_gradients";
const char * LLM_TENSOR_OPTIMIZER_LBFGS_SEARCH_DIRECTION    = "optimizer.lbfgs.search_direction";
const char * LLM_TENSOR_OPTIMIZER_LBFGS_PAST_LOSS_VALUES    = "optimizer.lbfgs.past_loss_values";
const char * LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_ALPHA        = "optimizer.lbfgs.memory_alpha";
const char * LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_YS           = "optimizer.lbfgs.memory_ys";
const char * LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_S            = "optimizer.lbfgs.memory_s";
const char * LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_Y            = "optimizer.lbfgs.memory_y";

const char * LLM_KV_TRAINING_TYPE_TRAIN_MODEL   = "train_model";
const char * LLM_KV_TRAINING_TYPE_FINETUNE_LORA = "finetune_lora";
const char * LLM_KV_TRAINING_TYPE               = "training.type";
const char * LLM_KV_TRAINING_FILE_VERSION       = "training.file_version";
const char * LLM_KV_TRAINING_ITERATION_COUNT    = "training.iteration_count";
const char * LLM_KV_TRAINING_SAMPLE_COUNT       = "training.sample_count";
const char * LLM_KV_TRAINING_TOKEN_COUNT        = "training.token_count";
const char * LLM_KV_TRAINING_EPOCH_COUNT        = "training.epoch_count";
const char * LLM_KV_TRAINING_SAMPLES_HASH       = "training.samples_hash";
const char * LLM_KV_TRAINING_SHUFFLE_SAMPLES_HASH = "training.shuffle.samples_hash";
const char * LLM_KV_TRAINING_SHUFFLE_RNG_STATE    = "training.shuffle.rng_state";
const char * LLM_KV_TRAINING_SHUFFLE_SAMPLE_COUNT = "training.shuffle.sample_count";
const char * LLM_KV_TRAINING_SHUFFLE_NEXT_SAMPLE  = "training.shuffle.next_sample";

const char * LLM_KV_TRAINING_LORA_RANK_TOKEN_EMBD  = "training.lora.rank.token_embd";
const char * LLM_KV_TRAINING_LORA_RANK_OUTPUT_NORM = "training.lora.rank.output_norm";
const char * LLM_KV_TRAINING_LORA_RANK_OUTPUT      = "training.lora.rank.output";
const char * LLM_KV_TRAINING_LORA_RANK_ATTN_NORM   = "training.lora.rank.attn_norm";
const char * LLM_KV_TRAINING_LORA_RANK_ATTN_Q      = "training.lora.rank.attn_q";
const char * LLM_KV_TRAINING_LORA_RANK_ATTN_K      = "training.lora.rank.attn_k";
const char * LLM_KV_TRAINING_LORA_RANK_ATTN_V      = "training.lora.rank.attn_v";
const char * LLM_KV_TRAINING_LORA_RANK_ATTN_OUT    = "training.lora.rank.attn_output";
const char * LLM_KV_TRAINING_LORA_RANK_FFN_NORM    = "training.lora.rank.ffn_norm";
const char * LLM_KV_TRAINING_LORA_RANK_FFN_GATE    = "training.lora.rank.ffn_gate";
const char * LLM_KV_TRAINING_LORA_RANK_FFN_DOWN    = "training.lora.rank.ffn_down";
const char * LLM_KV_TRAINING_LORA_RANK_FFN_UP      = "training.lora.rank.ffn_up";

// gguf constants (sync with gguf.py)

const char * LLM_KV_GENERAL_ARCHITECTURE        = "general.architecture";
const char * LLM_KV_GENERAL_FILE_TYPE           = "general.file_type";

const char * LLM_KV_CONTEXT_LENGTH              = "%s.context_length";
const char * LLM_KV_EMBEDDING_LENGTH            = "%s.embedding_length";
const char * LLM_KV_BLOCK_COUNT                 = "%s.block_count";
const char * LLM_KV_FEED_FORWARD_LENGTH         = "%s.feed_forward_length";
const char * LLM_KV_ATTENTION_HEAD_COUNT        = "%s.attention.head_count";
const char * LLM_KV_ATTENTION_LAYERNORM_RMS_EPS = "%s.attention.layer_norm_rms_epsilon";
const char * LLM_KV_ROPE_DIMENSION_COUNT        = "%s.rope.dimension_count";
const char * LLM_KV_ROPE_FREQ_BASE              = "%s.rope.freq_base"; // TODO load in llama.cpp
const char * LLM_KV_ROPE_SCALE_LINEAR           = "%s.rope.scale_linear";

const char * LLM_TENSOR_TOKEN_EMBD    = "token_embd";
const char * LLM_TENSOR_OUTPUT_NORM   = "output_norm";
const char * LLM_TENSOR_OUTPUT        = "output";
const char * LLM_TENSOR_ATTN_NORM     = "blk.%d.attn_norm";
const char * LLM_TENSOR_ATTN_Q        = "blk.%d.attn_q";
const char * LLM_TENSOR_ATTN_K        = "blk.%d.attn_k";
const char * LLM_TENSOR_ATTN_V        = "blk.%d.attn_v";
const char * LLM_TENSOR_ATTN_OUT      = "blk.%d.attn_output";
const char * LLM_TENSOR_FFN_NORM      = "blk.%d.ffn_norm";
const char * LLM_TENSOR_FFN_GATE      = "blk.%d.ffn_gate";
const char * LLM_TENSOR_FFN_DOWN      = "blk.%d.ffn_down";
const char * LLM_TENSOR_FFN_UP        = "blk.%d.ffn_up";

void print_params(struct my_llama_hparams * params) {
    printf("%s: n_vocab: %u\n", __func__, params->n_vocab);
    printf("%s: n_ctx:   %u\n", __func__, params->n_ctx);
    printf("%s: n_embd:  %u\n", __func__, params->n_embd);
    printf("%s: n_ff:    %u\n", __func__, params->n_ff);
    printf("%s: n_head:  %u\n", __func__, params->n_head);
    printf("%s: n_layer: %u\n", __func__, params->n_layer);
    printf("%s: n_rot:   %u\n", __func__, params->n_rot);
}

void print_lora_params(struct my_llama_lora_hparams * params) {
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
    printf("%s: norm_rms_eps          : %f\n", __func__, params->f_norm_rms_eps);
    printf("%s: rope_freq_base        : %f\n", __func__, params->rope_freq_base);
    printf("%s: rope_freq_scale       : %f\n", __func__, params->rope_freq_scale);
}

void init_model(struct llama_model * input, struct my_llama_model * model, uint32_t n_ctx) {
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

    hparams.n_vocab    = llama_model_n_vocab(input);
    hparams.n_ctx      = n_ctx;
    hparams.n_embd     = llama_model_n_embd(input);
    hparams.n_ff       = llama_model_n_ff(input);
    hparams.n_head     = llama_model_n_head(input);
    hparams.n_head_kv  = llama_model_n_head_kv(input);
    hparams.n_layer    = llama_model_n_layer(input);
    hparams.n_rot      = llama_model_n_rot(input);

    model->tok_embeddings = llama_get_model_tensor(input, tn(LLM_TENSOR_TOKEN_EMBD));
    model->norm           = llama_get_model_tensor(input, tn(LLM_TENSOR_OUTPUT_NORM));
    model->output         = llama_get_model_tensor(input, tn(LLM_TENSOR_OUTPUT));

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
    }
}

void set_param_lora(struct my_llama_lora * lora) {
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

void init_lora(const struct my_llama_model * model, struct my_llama_lora * lora) {
    const auto & lparams = lora->hparams;

    const uint32_t n_embd     = model->hparams.n_embd;
    const uint32_t n_embd_gqa = model->hparams.n_embd_gqa();
    const uint32_t n_layer    = model->hparams.n_layer;
    const uint32_t n_vocab    = model->hparams.n_vocab;
    const uint32_t n_ff       = model->hparams.n_ff;

    lora->train_its = 0;
    lora->train_samples = 0;
    lora->train_tokens = 0;

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
    ggml_allocr * alloc = NULL;
    alloc = ggml_allocr_new_measure(tensor_alignment);
    ggml_allocr_alloc(alloc, lora->tok_embeddings_a);
    ggml_allocr_alloc(alloc, lora->tok_embeddings_b);
    ggml_allocr_alloc(alloc, lora->norm_a);
    ggml_allocr_alloc(alloc, lora->norm_b);
    ggml_allocr_alloc(alloc, lora->output_a);
    ggml_allocr_alloc(alloc, lora->output_b);
    for (uint32_t i = 0; i < n_layer; ++i) {
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
    for (uint32_t i = 0; i < n_layer; ++i) {
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

    // allocate data
    lora->data.resize(ggml_allocr_max_size(alloc) + tensor_alignment);
    ggml_allocr_free(alloc);
    alloc = ggml_allocr_new(lora->data.data(), lora->data.size(), tensor_alignment);
    ggml_allocr_alloc(alloc, lora->tok_embeddings_a);
    ggml_allocr_alloc(alloc, lora->tok_embeddings_b);
    ggml_allocr_alloc(alloc, lora->norm_a);
    ggml_allocr_alloc(alloc, lora->norm_b);
    ggml_allocr_alloc(alloc, lora->output_a);
    ggml_allocr_alloc(alloc, lora->output_b);
    for (uint32_t i = 0; i < n_layer; ++i) {
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
    for (uint32_t i = 0; i < n_layer; ++i) {
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
    ggml_allocr_free(alloc);
}



void randomize_lora(struct my_llama_lora * lora, int seed, float mean, float std, float min, float max) {
    const uint32_t n_layer = lora->layers.size();

    struct random_normal_distribution rnd;
    init_random_normal_distribution(&rnd, seed, mean, std, min, max);

    randomize_tensor_normal(lora->tok_embeddings_a, &rnd);
    randomize_tensor_normal(lora->tok_embeddings_b, &rnd);
    randomize_tensor_normal(lora->norm_a,           &rnd);
    randomize_tensor_normal(lora->norm_b,           &rnd);
    randomize_tensor_normal(lora->output_a,         &rnd);
    randomize_tensor_normal(lora->output_b,         &rnd);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = lora->layers[i];
        randomize_tensor_normal(layer.attention_norm_a, &rnd);
        randomize_tensor_normal(layer.attention_norm_b, &rnd);

        randomize_tensor_normal(layer.wq_a, &rnd);
        randomize_tensor_normal(layer.wq_b, &rnd);
        randomize_tensor_normal(layer.wk_a, &rnd);
        randomize_tensor_normal(layer.wk_b, &rnd);
        randomize_tensor_normal(layer.wv_a, &rnd);
        randomize_tensor_normal(layer.wv_b, &rnd);
        randomize_tensor_normal(layer.wo_a, &rnd);
        randomize_tensor_normal(layer.wo_b, &rnd);

        randomize_tensor_normal(layer.ffn_norm_a, &rnd);
        randomize_tensor_normal(layer.ffn_norm_b, &rnd);

        randomize_tensor_normal(layer.w1_a, &rnd);
        randomize_tensor_normal(layer.w1_b, &rnd);
        randomize_tensor_normal(layer.w2_a, &rnd);
        randomize_tensor_normal(layer.w2_b, &rnd);
        randomize_tensor_normal(layer.w3_a, &rnd);
        randomize_tensor_normal(layer.w3_b, &rnd);
    }
}

void assert_shape_1d(struct ggml_tensor * tensor, int64_t ne0) {
    GGML_ASSERT(tensor->n_dims == 1);
    GGML_ASSERT(tensor->ne[0] == ne0);
}

void assert_shape_2d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    GGML_ASSERT(tensor->n_dims == 2);
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
}

void assert_shape_3d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2) {
    GGML_ASSERT(tensor->n_dims == 3);
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == ne2);
}

void assert_shape_4d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    GGML_ASSERT(tensor->n_dims == 4);
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == ne2);
    GGML_ASSERT(tensor->ne[3] == ne3);
}

struct ggml_tensor * llama_build_lora_finetune_graphs(
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
    const int n_rot       = hparams.n_rot;
    const int n_ff        = hparams.n_ff;
    const int n_embd_head = hparams.n_embd_head();
    const int n_embd_gqa  = hparams.n_embd_gqa();
    const float rms_norm_eps    = lora->hparams.f_norm_rms_eps;
    const float rope_freq_base  = lora->hparams.rope_freq_base;
    const float rope_freq_scale = lora->hparams.rope_freq_scale;

    GGML_ASSERT((size_t) n_layer == lora->layers.size());
    GGML_ASSERT(n_embd_head == n_rot);

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

    auto add_to_f32 = [] (struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
        if (ggml_is_quantized(a->type)) {
            return ggml_add_cast(ctx, a, b, GGML_TYPE_F32);
        } else {
            GGML_ASSERT(a->type == GGML_TYPE_F32);
            return ggml_add(ctx, a, b);
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

    struct ggml_tensor * kv_scale = NULL;
    if (!enable_flash_attn) {
        kv_scale = ggml_new_f32(ctx, 1.0f/sqrtf(float(n_embd)/n_head));
    }

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
        *gb = *gf;
        ggml_build_backward_expand(ctx, gf, gb, true);
    }

    GGML_ASSERT(alloc != NULL);

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

    // make sure base model tensors data cannot be used in viewable operations
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, model->tok_embeddings, one));
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, model->norm, one));
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, model->output, one));
    for (int il = 0; il < n_layer; ++il) {
        struct my_llama_layer & layer = model->layers[il];
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.attention_norm, one));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.ffn_norm, one));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wq, one));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wk, one));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wv, one));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.wo, one));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.w1, one));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.w2, one));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer.w3, one));
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

int get_example_targets_batch(
        struct llama_context * lctx,
        const size_t         * samples_begin,
        const size_t         * samples_size,
              size_t           samples_count,
        const llama_token    * train_data,
        size_t                 n_train_data,
        int                    example_id,
        struct ggml_tensor   * tokens_input,
        struct ggml_tensor   * target_probs,
        bool                   separate_with_eos,
        bool                   separate_with_bos,
        bool                   fill_with_next_samples) {

    GGML_ASSERT(tokens_input->n_dims  == 2);
    GGML_ASSERT(target_probs->n_dims  == 3);
    int n_vocab  = target_probs->ne[0];
    int n_tokens = tokens_input->ne[0];
    int n_batch  = tokens_input->ne[1];
    GGML_ASSERT(n_vocab  == target_probs->ne[0]);
    GGML_ASSERT(n_tokens == target_probs->ne[1]);
    GGML_ASSERT(n_batch  == target_probs->ne[2]);

    int used_samples = 0;

    ggml_set_f32(target_probs, 0.0f);
    int bos = llama_token_bos(lctx);
    int eos = llama_token_eos(lctx);
    // printf("%s: example_id=%d n_batch=%d n_train_samples=%zu\n", __func__, example_id, n_batch, n_train_samples);
    for (int k=0; k<n_batch; ++k) {
        // printf("%s: batch %d\n", __func__, k);
        size_t sample_offs  = 0;
        size_t sample_idx   = (example_id + used_samples) % samples_count;
        size_t sample_begin = samples_begin[sample_idx];
        size_t sample_size  = samples_size[sample_idx];
        ++used_samples;

        // printf("%s: sample_idx=%zu sample=%zu\n", __func__, sample_idx, sample);
        GGML_ASSERT(sample_begin+sample_size-1 < n_train_data);

        ggml_set_i32_nd(tokens_input, 0, k, 0, 0, bos);
        bool sample_separation_eos = !separate_with_eos;
        bool sample_separation_bos = !separate_with_bos;
        for (int i=0; i<n_tokens; ++i) {
            int token = eos;
            if (sample_offs >= sample_size && fill_with_next_samples) {
                if (!sample_separation_eos) {
                    // insert eos token to separate samples
                    sample_separation_eos = true;
                } else if (!sample_separation_bos) {
                    // insert bos token to separate samples
                    sample_separation_bos = true;
                    token = bos;
                } else {
                    // sample separation is done, continue with next sample
                    sample_separation_eos = !separate_with_eos;
                    sample_separation_bos = !separate_with_bos;
                    sample_offs  = 0;
                    sample_idx   = (example_id + used_samples) % samples_count;
                    sample_begin = samples_begin[sample_idx];
                    sample_size  = samples_size[sample_idx];
                    ++used_samples;
                }
            }
            // note: no else-if here
            if (sample_offs < sample_size) {
                token = clamp(train_data[sample_begin+sample_offs], 0, n_vocab-1);
                ++sample_offs;
            }
            ggml_set_f32_nd(target_probs,  token, i, k, 0, +1.0f);
            if (i+1<n_tokens) {
                ggml_set_i32_nd(tokens_input, i+1, k, 0, 0, token);
            }
        }
    }

    return used_samples;
}

#ifdef __GNUC__
#ifdef __MINGW32__
__attribute__((format(gnu_printf, 1, 2)))
#else
__attribute__((format(printf, 1, 2)))
#endif
#endif
static std::string format(const char * fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
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
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error(std::string("unexpectedly reached end of file"));
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
            throw std::runtime_error(format("write error: %s", strerror(errno)));
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

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

// mark each byte with its utf8 unit number.
// returns the number of utf8 characters.
// e.g. when bytes == '\x61\xD0\xB0\x62',
// then utf8_units will become [0,0,1,0]
// utf8_nunits will become [1,2,2,1] and 3 is returned.
// bytes where utf8_units is zero, are the begin of an utf8 character.
static size_t mark_utf8_units(const char* bytes, int * utf8_units, int * utf8_nunits, size_t count) {
    size_t offs = 0;
    size_t count_utf8 = 0;
    while(offs < count) {
        size_t len = utf8_len(bytes[offs]);
        for (size_t i=0; i<len; ++i) {
            utf8_units[offs+i] = i;
            utf8_nunits[offs+i] = len;
        }
        offs += len;
        ++count_utf8;
    }
    return count_utf8;
}

size_t tokenize_file(
        struct llama_context     * lctx,
        const char               * filename,
        const std::string        & sample_start,
        bool                       include_sample_start,
        bool                       overlapping_samples,
        unsigned                   context_length,
        std::vector<llama_token> & out_tokens,
        std::vector<size_t>      & out_samples_begin,
        std::vector<size_t>      & out_samples_size) {
    struct llama_file f(filename, "rb");

    if (f.size == 0) {
        out_tokens.clear();
        out_samples_begin.clear();
        out_samples_size.clear();
        printf("%s: warning: empty or not existing training data file '%s'\n",
            __func__, filename);
        return out_tokens.size();
    }

    std::vector<char> buf;
    buf.resize(f.size+1);

    f.read_raw(buf.data(), f.size);
    buf[f.size] = '\0';

    std::vector<int> utf8_units;
    std::vector<int> utf8_nunits;
    utf8_units.resize(buf.size());
    utf8_nunits.resize(buf.size());
    size_t n_utf8_chars = mark_utf8_units(buf.data(), utf8_units.data(), utf8_nunits.data(), buf.size());

    if (sample_start.size() == 0) {
        // tokenize all data at once
        out_tokens.resize(buf.size());

        int n_tokens = llama_tokenize(lctx, buf.data(), out_tokens.data(), buf.size(), false);
        if (n_tokens < 0) {
            out_tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(lctx, buf.data(), out_tokens.data(), buf.size(), false);
        }
        if (n_tokens >= 0) {
            out_tokens.resize(n_tokens);
        }

        // generate sample starts at all token positions
        out_samples_begin.clear();
        out_samples_begin.push_back(0);
        out_samples_size.push_back(std::min((size_t) context_length, out_tokens.size()));
        size_t end = (out_tokens.size() >= context_length) ? (out_tokens.size() - context_length) : 0;
        for (size_t sample_begin = 1; sample_begin < end; ++sample_begin) {
            out_samples_begin.push_back(sample_begin);
            out_samples_size.push_back(context_length);
        }
    } else {
        // split data into samples and tokenize each sample
        std::string data_str(buf.data(), buf.size()-1);
        out_samples_begin.clear();
        out_samples_size.clear();
        out_tokens.clear();

        // find all positions of pattern sample_start
        size_t sample_begin = data_str.find(sample_start, 0);
        while (sample_begin != std::string::npos) {
            out_samples_begin.push_back(sample_begin);
            const size_t search_start = sample_begin + sample_start.size();
            sample_begin = data_str.find(sample_start, search_start);
        }
        if (out_samples_begin.size() == 0) {
            printf("%s: warning: sample start pattern '%s' not found. inserting single sample at data begin\n",
                __func__, sample_start.c_str());
            out_samples_begin.push_back(0);
        }

        out_samples_size.resize(out_samples_begin.size(), 0);

        std::vector<char>        buf_sample;
        std::vector<llama_token> tok_sample;

        const size_t sample_begin_offset = (include_sample_start ? 0 : sample_start.size());
        size_t found_too_big_sample   = 0;
        size_t found_too_small_sample = 0;
        size_t found_empty_sample     = 0;
        size_t found_min_sample_size  = SIZE_MAX;
        size_t found_max_sample_size  = 0;

        size_t max_token_text_size = 0;
        int n_vocab = llama_n_vocab(lctx);
        for (llama_token token=0; token < n_vocab; ++token) {
            max_token_text_size = std::max(
                max_token_text_size,
                strlen(llama_token_get_text(lctx, token)));
        }

        // upper bound of context byte length.
        // strings with this byte length should always tokenize to at least context_length tokens.
        size_t context_byte_len = max_token_text_size*context_length;

        for (unsigned i=0; i<out_samples_begin.size(); ++i) {
            // determine sample begin and end from pattern positions
            size_t sample_begin = out_samples_begin[i] + sample_begin_offset;
            size_t sample_end   = overlapping_samples
                                    ? std::min(
                                        data_str.size(),
                                        sample_begin + context_byte_len)
                                    : (i+1 < out_samples_begin.size()
                                        ? out_samples_begin[i+1]
                                        : data_str.size());
            if (utf8_units[sample_end] > 0) {
                // sample end is in the middle of an utf8 character.
                // advance sample_end to the begin of the next utf8 character.
                sample_end += utf8_nunits[sample_end] - utf8_units[sample_end];
            }
            size_t sample_size = sample_end - sample_begin;
            if (sample_size == 0) {
                ++found_empty_sample;
            }

            if (sample_size > 0) {
                // llama_tokenize expects zero terminated string,
                // copy sample into buffer and zero terminate it.
                buf_sample.resize(sample_size+1);
                memcpy(buf_sample.data(), data_str.data() + sample_begin, sample_size);
                buf_sample[sample_size] = '\0';

                // printf("sample: '%s'\n", buf_sample.data());

                // tokenize the sample
                tok_sample.resize(sample_size);
                int n_tokens = llama_tokenize(lctx,
                    buf_sample.data(),
                    tok_sample.data(),
                    sample_size, false);
                if (n_tokens < 0) {
                    tok_sample.resize(-n_tokens);
                    n_tokens = llama_tokenize(lctx,
                        buf_sample.data(),
                        tok_sample.data(),
                        sample_size, false);
                    GGML_ASSERT(n_tokens >= 0);
                }
                GGML_ASSERT(n_tokens <= tok_sample.size());

                if ((size_t) n_tokens > context_length) {
                    ++found_too_big_sample;
                } else if ((size_t) n_tokens < context_length) {
                    ++found_too_small_sample;
                }
                found_max_sample_size = std::max(found_max_sample_size, (size_t) n_tokens);
                found_min_sample_size = std::min(found_min_sample_size, (size_t) n_tokens);

                // write out tokens, start and size of sample
                // overwrite the string start position with the token start position
                out_samples_begin[i] = out_tokens.size();
                out_samples_size[i] = (size_t) n_tokens;
                out_tokens.insert(out_tokens.end(), tok_sample.begin(), tok_sample.begin() + n_tokens);
            } else {
                out_samples_begin[i] = out_tokens.size();
                out_samples_size[i] = 0;
            }

        }
        if (found_too_big_sample > 0) {
            printf("%s: warning: found %zu samples (max length %zu) that exceed context length of %u. samples will be cut off.\n",
                __func__, found_too_big_sample, found_max_sample_size, context_length);
        }

        if (found_too_small_sample > 0) {
            printf("%s: warning: found %zu samples (min length %zu) that are shorter than context length of %u.\n",
                __func__, found_too_small_sample, found_min_sample_size, context_length);
        }

        if (found_empty_sample) {
            printf("%s: warning: found %zu empty samples.\n",
                __func__, found_empty_sample);
        }
    }
    printf("%s: total number of samples: %zu\n",
        __func__, out_samples_begin.size());

    GGML_ASSERT(out_samples_begin.size() == out_samples_size.size());

    return out_tokens.size();
}

void mt19937_set_state(std::mt19937& rng, const std::string& rng_state) {
    std::stringstream s_rng_state;
    s_rng_state.imbue(std::locale::classic());
    s_rng_state.exceptions(std::stringstream::failbit);
    s_rng_state.str(rng_state);
    s_rng_state >> rng;
}

std::string mt19937_get_state(const std::mt19937& rng) {
    std::stringstream s_rng_state;
    s_rng_state.imbue(std::locale::classic());
    s_rng_state << rng;
    return s_rng_state.str();
}

std::string mt19937_seed_to_state(unsigned seed) {
    std::mt19937 rng(seed);
    return mt19937_get_state(rng);
}

std::string shuffle_samples(const std::string& rng_state, size_t * begins, size_t * sizes, size_t count) {
    if (count == 0) return rng_state;

    std::mt19937 rng;
    mt19937_set_state(rng, rng_state);

    // sort indices by random value for each index
    std::vector<size_t> idcs;
    {
        std::vector<unsigned> rnd;
        idcs.resize(count);
        rnd.resize(count);
        for (unsigned i=0; i<count; ++i) {
            idcs[i] = i;
            rnd[i]  = rng();
        }

        std::sort(idcs.begin(), idcs.end(), [&rnd](size_t a, size_t b){
            // stable sort for reproducibility
            return (rnd[a] == rnd[b]) ? (a < b) : (rnd[a] < rnd[b]);
        });
    }

    // reorder begins and sizes by sorted indices
    std::vector<size_t> reordered;
    reordered.resize(count);

    for (unsigned i=0; i<count; ++i) {
        reordered[i] = begins[idcs[i]];
    }
    memcpy(begins, reordered.data(), sizeof(*begins)*reordered.size());

    for (unsigned i=0; i<count; ++i) {
        reordered[i] = sizes[idcs[i]];
    }
    memcpy(sizes, reordered.data(), sizeof(*sizes)*reordered.size());

    return mt19937_get_state(rng);
}

std::string replace_str(const char * s, const char * needle, const char * replacement) {
    std::string str = s;
    size_t pos = str.find(needle);
    if (pos != std::string::npos) {
        str.replace(pos, strlen(needle), replacement);
    }
    return str;
}

#define GGUF_GET_KEY(ctx, dst, func, type, req, key) \
{ \
    const std::string skey(key); \
    const int kid = gguf_find_key(ctx, skey.c_str()); \
    if (kid >= 0) { \
        enum gguf_type ktype = gguf_get_kv_type(ctx, kid); \
        if (ktype != (type)) { \
            throw std::runtime_error(format("key %s has wrong type: %s", skey.c_str(), gguf_type_name(ktype))); \
        } \
        (dst) = func(ctx, kid); \
    } else if (req) { \
        throw std::runtime_error(format("key not found in model: %s", skey.c_str())); \
    } \
}

bool are_same_layout(struct ggml_tensor * a, struct ggml_tensor * b) {
    GGML_ASSERT(a != NULL);
    GGML_ASSERT(b != NULL);
    GGML_ASSERT(a->type == b->type);
    GGML_ASSERT(ggml_are_same_shape(a, b));
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(b));

    return true;
}

void read_tensor_by_name(struct ggml_tensor * dst, struct ggml_context * ctx, const char * name) {
    if (dst == NULL) {
        return;
    }
    struct ggml_tensor * t  = ggml_get_tensor(ctx, name);
    GGML_ASSERT(are_same_layout(dst, t));
    memcpy(dst->data, t->data, ggml_nbytes(t));

    if (strlen(ggml_get_name(dst)) == 0) {
        ggml_set_name(dst, name);
    }
}

void load_default_lora_params_from_base_model(const char * fn_base_model, struct my_llama_lora_hparams * lora_params) {
    if (strlen(fn_base_model) == 0) {
        return;
    }
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = NULL;
    struct gguf_context * fctx = gguf_init_from_file(fn_base_model, params);
    if (fctx == NULL) {
        return;
    }

    const char * arch = "llama";
    std::vector<char> keybuf;
    keybuf.resize(512);
    auto kv = [arch, &keybuf](const char * key) -> const char * {
        snprintf(keybuf.data(), keybuf.size(), key, arch);
        return keybuf.data();
    };

    float rope_freq_scale = 1.0f;
    GGUF_GET_KEY(fctx, lora_params->f_norm_rms_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS));
    GGUF_GET_KEY(fctx, lora_params->rope_freq_base, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_FREQ_BASE));
    GGUF_GET_KEY(fctx, rope_freq_scale, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_SCALE_LINEAR));
    if (rope_freq_scale != 1.0f) {
        lora_params->rope_freq_scale = 1.0f / rope_freq_scale;
    }

    gguf_free(fctx);
}

void load_opt_context_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct ggml_opt_context * opt) {
    // NOTE: gguf_context must be initialized with f_ggml_ctx and no_alloc=false, otherwise tensor data can not be read

    uint32_t file_version;
    GGUF_GET_KEY(fctx, file_version, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_OPTIMIZER_FILE_VERSION);
    GGML_ASSERT(file_version == 0);

    GGUF_GET_KEY(fctx, opt->params.past, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_OPTIMIZER_CONVERGENCE_PAST_COUNT);
    GGUF_GET_KEY(fctx, opt->iter, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_OPTIMIZER_ITERATION_COUNT);
    GGUF_GET_KEY(fctx, opt->just_initialized, gguf_get_val_bool, GGUF_TYPE_BOOL, true, LLM_KV_OPTIMIZER_JUST_INITIALIZED);

    uint64_t nx;
    GGUF_GET_KEY(fctx, nx, gguf_get_val_u64, GGUF_TYPE_UINT64, true, LLM_KV_OPTIMIZER_PARAMETER_COUNT);
    opt->nx = (size_t) nx;

    // don't call ggml_opt_init until optimizer type and optimizer specific parameters are know

    std::string opt_type;
    GGUF_GET_KEY(fctx, opt_type, gguf_get_val_str, GGUF_TYPE_STRING, true, LLM_KV_OPTIMIZER_TYPE);
    if (opt_type == LLM_KV_OPTIMIZER_TYPE_ADAM) {
        opt->params.type = GGML_OPT_ADAM;

        GGUF_GET_KEY(fctx, opt->adam.fx_best,          gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, LLM_KV_OPTIMIZER_ADAM_BEST_LOSS);
        GGUF_GET_KEY(fctx, opt->adam.fx_prev,          gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, LLM_KV_OPTIMIZER_ADAM_PREVIOUS_LOSS);
        GGUF_GET_KEY(fctx, opt->adam.n_no_improvement, gguf_get_val_u32, GGUF_TYPE_UINT32,  true, LLM_KV_OPTIMIZER_ADAM_NO_IMPROVEMENT_COUNT);

        ggml_opt_init(opt->ctx, opt, opt->params, opt->nx);

        read_tensor_by_name(opt->adam.m,  f_ggml_ctx, LLM_TENSOR_OPTIMIZER_ADAM_FIRST_MOMENTS);
        read_tensor_by_name(opt->adam.v,  f_ggml_ctx, LLM_TENSOR_OPTIMIZER_ADAM_SECOND_MOMENTS);
        read_tensor_by_name(opt->adam.pf, f_ggml_ctx, LLM_TENSOR_OPTIMIZER_ADAM_PAST_LOSS_VALUES);
    } else if (opt_type == LLM_KV_OPTIMIZER_TYPE_LBFGS) {
        opt->params.type = GGML_OPT_LBFGS;

        GGUF_GET_KEY(fctx, opt->params.lbfgs.m,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, LLM_KV_OPTIMIZER_LBFGS_APPROX_HESSIAN_COUNT);
        GGUF_GET_KEY(fctx, opt->lbfgs.fx_best,          gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, LLM_KV_OPTIMIZER_LBFGS_BEST_LOSS);
        GGUF_GET_KEY(fctx, opt->lbfgs.step,             gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_STEP);
        GGUF_GET_KEY(fctx, opt->lbfgs.j,                gguf_get_val_i32, GGUF_TYPE_INT32,   true, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_J);
        GGUF_GET_KEY(fctx, opt->lbfgs.k,                gguf_get_val_i32, GGUF_TYPE_INT32,   true, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_K);
        GGUF_GET_KEY(fctx, opt->lbfgs.end,              gguf_get_val_i32, GGUF_TYPE_INT32,   true, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_END);
        GGUF_GET_KEY(fctx, opt->lbfgs.n_no_improvement, gguf_get_val_u32, GGUF_TYPE_UINT32,  true, LLM_KV_OPTIMIZER_LBFGS_NO_IMPROVEMENT_COUNT);

        ggml_opt_init(opt->ctx, opt, opt->params, opt->nx);

        read_tensor_by_name(opt->lbfgs.x,    f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_PARAMETERS);
        read_tensor_by_name(opt->lbfgs.xp,   f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_PARAMETERS);
        read_tensor_by_name(opt->lbfgs.g,    f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_GRADIENTS);
        read_tensor_by_name(opt->lbfgs.gp,   f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_GRADIENTS);
        read_tensor_by_name(opt->lbfgs.d,    f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_SEARCH_DIRECTION);
        read_tensor_by_name(opt->lbfgs.pf,   f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_PAST_LOSS_VALUES);
        read_tensor_by_name(opt->lbfgs.lmal, f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_ALPHA);
        read_tensor_by_name(opt->lbfgs.lmys, f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_YS);
        read_tensor_by_name(opt->lbfgs.lms,  f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_S);
        read_tensor_by_name(opt->lbfgs.lmy,  f_ggml_ctx, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_Y);
    } else {
        throw std::runtime_error("unknown optimizer type\n");
    }
}

void save_opt_context_gguf(struct gguf_context * fctx, struct ggml_opt_context * opt) {
    gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_FILE_VERSION, 0);
    gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_CONVERGENCE_PAST_COUNT, opt->params.past);
    gguf_set_val_u64(fctx, LLM_KV_OPTIMIZER_PARAMETER_COUNT, (uint64_t) opt->nx);
    gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_ITERATION_COUNT, opt->iter);
    gguf_set_val_bool(fctx, LLM_KV_OPTIMIZER_JUST_INITIALIZED, opt->just_initialized);

    switch (opt->params.type) {
        case GGML_OPT_ADAM:
            {
                gguf_set_val_str(fctx, LLM_KV_OPTIMIZER_TYPE, LLM_KV_OPTIMIZER_TYPE_ADAM);
                gguf_set_val_f32(fctx, LLM_KV_OPTIMIZER_ADAM_BEST_LOSS,            opt->adam.fx_best);
                gguf_set_val_f32(fctx, LLM_KV_OPTIMIZER_ADAM_PREVIOUS_LOSS,        opt->adam.fx_prev);
                gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_ADAM_NO_IMPROVEMENT_COUNT, opt->adam.n_no_improvement);

                ggml_set_name(opt->adam.m, LLM_TENSOR_OPTIMIZER_ADAM_FIRST_MOMENTS);
                ggml_set_name(opt->adam.v, LLM_TENSOR_OPTIMIZER_ADAM_SECOND_MOMENTS);
                if (opt->adam.pf) {
                    ggml_set_name(opt->adam.pf, LLM_TENSOR_OPTIMIZER_ADAM_PAST_LOSS_VALUES);
                }

                gguf_add_tensor(fctx, opt->adam.m);
                gguf_add_tensor(fctx, opt->adam.v);
                if (opt->adam.pf) {
                    gguf_add_tensor(fctx, opt->adam.pf);
                }
            } break;
        case GGML_OPT_LBFGS:
            {
                gguf_set_val_str(fctx, LLM_KV_OPTIMIZER_TYPE, LLM_KV_OPTIMIZER_TYPE_LBFGS);
                gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_LBFGS_APPROX_HESSIAN_COUNT, opt->params.lbfgs.m);
                gguf_set_val_f32(fctx, LLM_KV_OPTIMIZER_LBFGS_BEST_LOSS,            opt->lbfgs.fx_best);
                gguf_set_val_f32(fctx, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_STEP,     opt->lbfgs.step);
                gguf_set_val_i32(fctx, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_J,        opt->lbfgs.j);
                gguf_set_val_i32(fctx, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_K,        opt->lbfgs.k);
                gguf_set_val_i32(fctx, LLM_KV_OPTIMIZER_LBFGS_LINE_SEARCH_END,      opt->lbfgs.end);
                gguf_set_val_u32(fctx, LLM_KV_OPTIMIZER_LBFGS_NO_IMPROVEMENT_COUNT, opt->lbfgs.n_no_improvement);

                ggml_set_name(opt->lbfgs.x,    LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_PARAMETERS);
                ggml_set_name(opt->lbfgs.xp,   LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_PARAMETERS);
                ggml_set_name(opt->lbfgs.g,    LLM_TENSOR_OPTIMIZER_LBFGS_CURRENT_GRADIENTS);
                ggml_set_name(opt->lbfgs.gp,   LLM_TENSOR_OPTIMIZER_LBFGS_PREVIOUS_GRADIENTS);
                ggml_set_name(opt->lbfgs.d,    LLM_TENSOR_OPTIMIZER_LBFGS_SEARCH_DIRECTION);
                if (opt->lbfgs.pf) {
                    ggml_set_name(opt->lbfgs.pf, LLM_TENSOR_OPTIMIZER_LBFGS_PAST_LOSS_VALUES);
                }
                ggml_set_name(opt->lbfgs.lmal, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_ALPHA);
                ggml_set_name(opt->lbfgs.lmys, LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_YS);
                ggml_set_name(opt->lbfgs.lms,  LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_S);
                ggml_set_name(opt->lbfgs.lmy,  LLM_TENSOR_OPTIMIZER_LBFGS_MEMORY_Y);

                gguf_add_tensor(fctx, opt->lbfgs.x);
                gguf_add_tensor(fctx, opt->lbfgs.xp);
                gguf_add_tensor(fctx, opt->lbfgs.g);
                gguf_add_tensor(fctx, opt->lbfgs.gp);
                gguf_add_tensor(fctx, opt->lbfgs.d);
                if (opt->lbfgs.pf) {
                    gguf_add_tensor(fctx, opt->lbfgs.pf);
                }
                gguf_add_tensor(fctx, opt->lbfgs.lmal);
                gguf_add_tensor(fctx, opt->lbfgs.lmys);
                gguf_add_tensor(fctx, opt->lbfgs.lms);
                gguf_add_tensor(fctx, opt->lbfgs.lmy);
            } break;
    }
}

void load_llama_lora_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct my_llama_model * model, struct my_llama_lora * lora) {
    // NOTE: gguf_context must be initialized with f_ggml_ctx and no_alloc=false, otherwise tensor data can not be read

    std::string arch;

    std::vector<char> keybuf;
    keybuf.resize(512);
    auto kv = [&arch, &keybuf](const char * key) -> const char * {
        snprintf(keybuf.data(), keybuf.size(), key, arch.c_str());
        return keybuf.data();
    };

    GGUF_GET_KEY(fctx, arch, gguf_get_val_str, GGUF_TYPE_STRING, true, LLM_KV_GENERAL_ARCHITECTURE);
    GGML_ASSERT(arch == "llama");

    uint32_t ftype_u;
    GGUF_GET_KEY(fctx, ftype_u, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_GENERAL_FILE_TYPE);
    GGML_ASSERT((enum llama_ftype) ftype_u == LLAMA_FTYPE_ALL_F32);

    // n_ctx was not saved in earlier checkpoint file version, so we make it optional here
    GGUF_GET_KEY(fctx, model->hparams.n_ctx,   gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_CONTEXT_LENGTH));

    GGUF_GET_KEY(fctx, model->hparams.n_embd,  gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_EMBEDDING_LENGTH));
    GGUF_GET_KEY(fctx, model->hparams.n_ff,    gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_FEED_FORWARD_LENGTH));
    GGUF_GET_KEY(fctx, model->hparams.n_head,  gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_ATTENTION_HEAD_COUNT));
    GGUF_GET_KEY(fctx, model->hparams.n_layer, gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_BLOCK_COUNT));

    model->hparams.n_rot = model->hparams.n_embd / model->hparams.n_head;
    GGUF_GET_KEY(fctx, model->hparams.n_rot,   gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_ROPE_DIMENSION_COUNT));

    float rope_freq_scale = 1.0f;
    GGUF_GET_KEY(fctx, lora->hparams.f_norm_rms_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS));
    GGUF_GET_KEY(fctx, lora->hparams.rope_freq_base, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_FREQ_BASE));
    GGUF_GET_KEY(fctx, rope_freq_scale, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_SCALE_LINEAR));
    if (rope_freq_scale != 1.0f) {
        lora->hparams.rope_freq_scale = 1.0f / rope_freq_scale;
    }

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

    read_tensor_by_name(lora->tok_embeddings_a, f_ggml_ctx, ggml_get_name(lora->tok_embeddings_a));
    read_tensor_by_name(lora->tok_embeddings_b, f_ggml_ctx, ggml_get_name(lora->tok_embeddings_b));
    read_tensor_by_name(lora->norm_a,           f_ggml_ctx, ggml_get_name(lora->norm_a));
    read_tensor_by_name(lora->norm_b,           f_ggml_ctx, ggml_get_name(lora->norm_b));
    read_tensor_by_name(lora->output_a,         f_ggml_ctx, ggml_get_name(lora->output_a));
    read_tensor_by_name(lora->output_b,         f_ggml_ctx, ggml_get_name(lora->output_b));

    for (uint32_t i = 0; i < lora->layers.size(); ++i) {
        auto & layer = lora->layers[i];
        read_tensor_by_name(layer.attention_norm_a, f_ggml_ctx, ggml_get_name(layer.attention_norm_a));
        read_tensor_by_name(layer.attention_norm_b, f_ggml_ctx, ggml_get_name(layer.attention_norm_b));
        read_tensor_by_name(layer.wq_a,             f_ggml_ctx, ggml_get_name(layer.wq_a));
        read_tensor_by_name(layer.wq_b,             f_ggml_ctx, ggml_get_name(layer.wq_b));
        read_tensor_by_name(layer.wk_a,             f_ggml_ctx, ggml_get_name(layer.wk_a));
        read_tensor_by_name(layer.wk_b,             f_ggml_ctx, ggml_get_name(layer.wk_b));
        read_tensor_by_name(layer.wv_a,             f_ggml_ctx, ggml_get_name(layer.wv_a));
        read_tensor_by_name(layer.wv_b,             f_ggml_ctx, ggml_get_name(layer.wv_b));
        read_tensor_by_name(layer.wo_a,             f_ggml_ctx, ggml_get_name(layer.wo_a));
        read_tensor_by_name(layer.wo_b,             f_ggml_ctx, ggml_get_name(layer.wo_b));
        read_tensor_by_name(layer.ffn_norm_a,       f_ggml_ctx, ggml_get_name(layer.ffn_norm_a));
        read_tensor_by_name(layer.ffn_norm_b,       f_ggml_ctx, ggml_get_name(layer.ffn_norm_b));
        read_tensor_by_name(layer.w1_a,             f_ggml_ctx, ggml_get_name(layer.w1_a));
        read_tensor_by_name(layer.w1_b,             f_ggml_ctx, ggml_get_name(layer.w1_b));
        read_tensor_by_name(layer.w2_a,             f_ggml_ctx, ggml_get_name(layer.w2_a));
        read_tensor_by_name(layer.w2_b,             f_ggml_ctx, ggml_get_name(layer.w2_b));
        read_tensor_by_name(layer.w3_a,             f_ggml_ctx, ggml_get_name(layer.w3_a));
        read_tensor_by_name(layer.w3_b,             f_ggml_ctx, ggml_get_name(layer.w3_b));
    }
}

void save_llama_lora_gguf(struct gguf_context * fctx, struct my_llama_model * model, struct my_llama_lora * lora) {
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
    gguf_set_val_u32(fctx, kv(LLM_KV_BLOCK_COUNT),                 model->hparams.n_layer);
    gguf_set_val_u32(fctx, kv(LLM_KV_ROPE_DIMENSION_COUNT),        model->hparams.n_rot);
    gguf_set_val_f32(fctx, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS), lora->hparams.f_norm_rms_eps);
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_FREQ_BASE),              lora->hparams.rope_freq_base);
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_SCALE_LINEAR),           lora->hparams.rope_freq_scale);
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

void load_checkpoint_lora_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct my_llama_model * model, struct my_llama_lora * lora, struct ggml_opt_context * opt) {
    load_llama_lora_gguf(fctx, f_ggml_ctx, model, lora);

    uint32_t file_version;
    GGUF_GET_KEY(fctx, file_version,         gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_FILE_VERSION);
    GGML_ASSERT(file_version <= 1);

    std::string train_type = LLM_KV_TRAINING_TYPE_FINETUNE_LORA;
    GGUF_GET_KEY(fctx, train_type,           gguf_get_val_str, GGUF_TYPE_STRING, false, LLM_KV_TRAINING_TYPE);
    GGML_ASSERT(train_type == LLM_KV_TRAINING_TYPE_FINETUNE_LORA);

    if (file_version == 0) {

        GGUF_GET_KEY(fctx, lora->train_its,     gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_ITERATION_COUNT);
        GGUF_GET_KEY(fctx, lora->train_samples, gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_SAMPLE_COUNT);
        GGUF_GET_KEY(fctx, lora->train_tokens,  gguf_get_val_u32, GGUF_TYPE_UINT32, true, LLM_KV_TRAINING_TOKEN_COUNT);

    } else if (file_version == 1) {

        GGUF_GET_KEY(fctx, lora->train_its,     gguf_get_val_u64, GGUF_TYPE_UINT64, true, LLM_KV_TRAINING_ITERATION_COUNT);
        GGUF_GET_KEY(fctx, lora->train_samples, gguf_get_val_u64, GGUF_TYPE_UINT64, true, LLM_KV_TRAINING_SAMPLE_COUNT);
        GGUF_GET_KEY(fctx, lora->train_tokens,  gguf_get_val_u64, GGUF_TYPE_UINT64, true, LLM_KV_TRAINING_TOKEN_COUNT);
        GGUF_GET_KEY(fctx, lora->train_epochs,  gguf_get_val_u64, GGUF_TYPE_UINT64, true, LLM_KV_TRAINING_EPOCH_COUNT);

        GGUF_GET_KEY(fctx, lora->shuffle_samples_hash,      gguf_get_val_u64, GGUF_TYPE_UINT64, false, LLM_KV_TRAINING_SHUFFLE_SAMPLES_HASH);
        GGUF_GET_KEY(fctx, lora->shuffle_rng_state_current, gguf_get_val_str, GGUF_TYPE_STRING, false, LLM_KV_TRAINING_SHUFFLE_RNG_STATE);
        GGUF_GET_KEY(fctx, lora->shuffle_sample_count,      gguf_get_val_u64, GGUF_TYPE_UINT64, false, LLM_KV_TRAINING_SHUFFLE_SAMPLE_COUNT);
        GGUF_GET_KEY(fctx, lora->shuffle_next_sample,       gguf_get_val_u64, GGUF_TYPE_UINT64, false, LLM_KV_TRAINING_SHUFFLE_NEXT_SAMPLE);
    }

    load_opt_context_gguf(fctx, f_ggml_ctx, opt);
}

void save_checkpoint_lora_gguf(struct gguf_context * fctx, struct my_llama_model * model, struct my_llama_lora * lora, struct ggml_opt_context * opt) {
    save_llama_lora_gguf(fctx, model, lora);

    gguf_set_val_u32(fctx, LLM_KV_TRAINING_FILE_VERSION,    1);
    gguf_set_val_str(fctx, LLM_KV_TRAINING_TYPE,            LLM_KV_TRAINING_TYPE_FINETUNE_LORA);
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_ITERATION_COUNT, lora->train_its);
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_SAMPLE_COUNT,    lora->train_samples);
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_TOKEN_COUNT,     lora->train_tokens);
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_EPOCH_COUNT,     lora->train_epochs);

    gguf_set_val_u64(fctx, LLM_KV_TRAINING_SHUFFLE_SAMPLES_HASH, (uint64_t) lora->shuffle_samples_hash);
    gguf_set_val_str(fctx, LLM_KV_TRAINING_SHUFFLE_RNG_STATE,    lora->shuffle_rng_state_current.c_str());
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_SHUFFLE_SAMPLE_COUNT, (uint64_t) lora->shuffle_sample_count);
    gguf_set_val_u64(fctx, LLM_KV_TRAINING_SHUFFLE_NEXT_SAMPLE,  (uint64_t) lora->shuffle_next_sample);

    save_opt_context_gguf(fctx, opt);
}

bool load_checkpoint_lora_file(const char * filename, struct my_llama_model * model, struct my_llama_lora * lora, struct ggml_opt_context * opt) {
    struct ggml_context * f_ggml_ctx;
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &f_ggml_ctx;
    struct gguf_context * fctx = gguf_init_from_file(filename, params);
    if (fctx == NULL) {
        return false;
    }

    load_checkpoint_lora_gguf(fctx, f_ggml_ctx, model, lora, opt);

    gguf_free(fctx);
    return true;
}

void save_checkpoint_lora_file(const char * filename, struct my_llama_model * model, struct my_llama_lora * lora, struct ggml_opt_context * opt, const char * pattern_it, int iteration, const char * latest) {
    std::string sit = (iteration >= 0) ? std::to_string(iteration) : std::string(latest);
    std::string fn = replace_str(filename, pattern_it, sit.c_str());
    printf("%s: saving to %s\n", __func__, fn.c_str());
    struct gguf_context * fctx = gguf_init_empty();

    save_checkpoint_lora_gguf(fctx, model, lora, opt);

    // write file
    const bool only_meta = false;
    gguf_write_to_file(fctx, fn.c_str(), only_meta);
    gguf_free(fctx);
}

void write_tensor(struct llama_file * file, struct ggml_tensor * tensor, const char * name) {
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
    uint32_t nd = tensor->n_dims;
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

void save_as_llama_lora(struct my_llama_lora * lora, const char * filename, const char * pattern_it, int iteration, const char * latest) {
    std::string sit = (iteration >= 0) ? std::to_string(iteration) : std::string(latest);
    std::string fn = replace_str(filename, pattern_it, sit.c_str());
    printf("%s: saving to %s\n", __func__, fn.c_str());
    struct llama_file file(fn.c_str(), "wb");
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

    uint32_t LLAMA_FILE_MAGIC_LORA = 0x67676C61; // 'ggla'
    // write_magic
    file.write_u32(LLAMA_FILE_MAGIC_LORA);   // magic
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

float cosine_decay(const int decay_steps, const float minimum, int step) {
    if (step > decay_steps) {
        step = decay_steps;
    }
    const float cosine_decay = 0.50f*(1.0f + cosf(3.14159265359f*step/decay_steps));
    const float decay = (1 - minimum)*cosine_decay + minimum;
    return decay;
}

float cosine_decay_restart(int decay_steps, const float minimum, int step, float restart_step_mult, bool enable_restart) {
    if (enable_restart) {
        while (step > decay_steps) {
            step -= decay_steps;
            decay_steps = (int) restart_step_mult * decay_steps;
        }
    }
    return cosine_decay(decay_steps, minimum, step);
}

struct train_params {
    const char * fn_model_base;
    const char * fn_train_data;
    const char * fn_checkpoint_in;
    const char * fn_checkpoint_out;
    const char * fn_lora_out;
    const char * pattern_fn_it;
    const char * fn_latest;

    int save_every;

    uint32_t seed;

    int n_ctx;
    int n_threads;
    int n_batch;
    int n_gradient_accumulation;

    bool custom_n_ctx;

    bool only_write_lora;

    float f_norm_rms_eps;
    float rope_freq_base;
    float rope_freq_scale;

    bool custom_f_norm_rms_eps;
    bool custom_rope_freq_base;
    bool custom_rope_freq_scale;

    int32_t lora_r;
    int32_t lora_alpha;

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

    bool use_adam;
    bool use_flash;
    bool use_checkpointing;

    std::string sample_start;
    bool include_sample_start;
    bool escape;
    bool overlapping_samples;
    bool fill_with_next_samples;
    bool separate_with_eos;
    bool separate_with_bos;

    bool force_reshuffle;

    // only adam
    int   warmup;
    int   cos_decay_steps;
    float cos_decay_restart;
    float cos_decay_min;
    bool  enable_restart;

    int   opt_past;
    float opt_delta;
    int   opt_max_no_improvement;

    int   lbfgs_n_iter;
    int   adam_n_iter;
    float adam_alpha;
    float adam_min_alpha;
    float adam_decay;
    int   adam_decay_min_ndim;
    float adam_beta1;
    float adam_beta2;
    float adam_gclip;
    float adam_eps_f;
};

struct train_params get_default_train_params() {
    struct train_params params;
    params.fn_model_base     = "";
    params.fn_train_data     = "shakespeare.txt";
    params.fn_checkpoint_in  = "checkpoint.gguf";
    params.fn_checkpoint_out = "checkpoint-ITERATION.gguf";
    params.fn_lora_out       = "ggml-lora-ITERATION-f32.gguf";
    params.pattern_fn_it     = "ITERATION";
    params.fn_latest         = "LATEST";

    params.save_every = 10;

    params.seed       =   -1;

    params.n_ctx      =  128;
    params.n_threads  =    6;
    params.n_batch    =    8;
    params.n_gradient_accumulation = 1;

    params.custom_n_ctx = false;

    params.only_write_lora = false;

    params.f_norm_rms_eps  = 1e-5f;
    params.rope_freq_base  = 10000.0f;
    params.rope_freq_scale = 1.0f;

    params.custom_f_norm_rms_eps  = false;
    params.custom_rope_freq_base  = false;
    params.custom_rope_freq_scale = false;

    params.lora_alpha  = 4;
    params.lora_r      = 4;

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

    params.use_adam               = true;
    params.use_flash              = true;
    params.use_checkpointing      = true;

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

    // only adam
    params.warmup            =  100;
    params.cos_decay_steps   = 1000;
    params.cos_decay_restart = 1.1f;
    params.cos_decay_min     = 0.1f;
    params.enable_restart    = false;

    params.lbfgs_n_iter        = 256;
    params.adam_n_iter         = 256;
    params.adam_alpha          = 1e-3f;
    params.adam_min_alpha      = 0;
    params.adam_decay          = 1e-1f;
    params.adam_decay_min_ndim = 2;
    params.adam_beta1          = 0.9f;
    params.adam_beta2          = 0.999f;
    params.adam_gclip          = 1.0f;
    params.adam_eps_f          = 0.0f;
    return params;
}

void train_print_usage(int /*argc*/, char ** argv, const struct train_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                 show this help message and exit\n");
    fprintf(stderr, "  --model-base FNAME         model path from which to load base model (default '%s')\n", params->fn_model_base);
    fprintf(stderr, "  --train-data FNAME         path from which to load training data (default '%s')\n", params->fn_train_data);
    fprintf(stderr, "  --checkpoint-in FNAME      path from which to load training checkpoint (default '%s')\n", params->fn_checkpoint_in);
    fprintf(stderr, "  --checkpoint-out FNAME     path to save training checkpoint (default '%s')\n", params->fn_checkpoint_out);
    fprintf(stderr, "  --lora-out FNAME           path to save llama lora (default '%s')\n", params->fn_lora_out);
    fprintf(stderr, "  --pattern-fn-it STR        pattern in output filenames to be replaced by iteration number (default '%s')\n", params->pattern_fn_it);
    fprintf(stderr, "  --fn-latest STR            string to use instead of iteration number for saving latest output (default '%s')\n", params->fn_latest);
    fprintf(stderr, "  --save-every N             save checkpoint and lora every N iterations. Disabled when N <= 0. (default '%d')\n", params->save_every);
    fprintf(stderr, "  --only-write-lora          only save llama lora, don't do any training\n");
    fprintf(stderr, "  -s SEED, --seed SEED       RNG seed (default: -1, use random seed for -1)\n");
    fprintf(stderr, "  -c N, --ctx N              Context size used during training (default %d)\n", params->n_ctx);
    fprintf(stderr, "  -t N, --threads N          Number of threads (default %d)\n", params->n_threads);
    fprintf(stderr, "  -b N, --batch N            Parallel batch size (default %d)\n", params->n_batch);
    fprintf(stderr, "  --grad-acc N               Number of gradient accumulation steps (simulates larger batch size of batch*gradacc) (default %d)\n", params->n_gradient_accumulation);
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
    fprintf(stderr, "  --use-adam                 Use Adam optimizer (default)\n");
    fprintf(stderr, "  --no-flash                 Don't use flash attention \n");
    fprintf(stderr, "  --use-flash                Use flash attention (default)\n");
    fprintf(stderr, "  --no-checkpointing         Don't use gradient checkpointing\n");
    fprintf(stderr, "  --use-checkpointing        Use gradient checkpointing (default)\n");
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
    fprintf(stderr, "  --lbfgs-iter N             Maximum number of LBFGS optimization iterations for each batch (default %d)\n", params->lbfgs_n_iter);
    fprintf(stderr, "\n");
}

bool train_params_parse(int argc, char ** argv, struct train_params * params) {
    bool invalid_param = false;
    std::string arg;
    struct train_params default_params = get_default_train_params();
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "--model-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_model_base = argv[i];
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
        } else if (arg == "--lora-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_lora_out = argv[i];
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
        } else if (arg == "--only-write-lora") {
            params->only_write_lora = true;
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
            params->custom_n_ctx = true;
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
        } else if (arg == "--use-lbfgs") {
            params->use_adam = false;
        } else if (arg == "--use-adam") {
            params->use_adam = true;
        } else if (arg == "--no-flash") {
            params->use_flash = false;
        } else if (arg == "--use-flash") {
            params->use_flash = true;
        } else if (arg == "--no-checkpointing") {
            params->use_checkpointing = false;
        } else if (arg == "--use-checkpointing") {
            params->use_checkpointing = true;
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
        } else if (arg == "--lbfgs-iter") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->lbfgs_n_iter = std::stoi(argv[i]);
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
    struct train_params *     params;
    struct ggml_opt_context * opt;
    struct my_llama_model *   model;
    struct my_llama_lora  *   lora;
    struct llama_context *    lctx;
    int                       last_save_iter;
    llama_token *             tokens_data;
    size_t                    tokens_size;
    size_t *                  samples_begin;
    size_t *                  samples_size;
    size_t                    samples_count;
    struct ggml_tensor *      tokens_input;
    struct ggml_tensor *      target_probs;
    int                       first_iter;
    int64_t                   last_time;
    double                    millis_per_iter;
};

void print_duration(double fmillis) {
    if (fmillis < 1000.0f) {
        printf("%.1fms", (float) fmillis);
        return;
    }
    const int64_t one_sec  = 1000;
    const int64_t one_min  = one_sec  * 60;
    const int64_t one_hour = one_min  * 60;
    const int64_t one_day  = one_hour * 24;

    int64_t millis  = (int64_t) fmillis;
    int64_t days    = millis/one_day;
    int64_t hours   = (millis - days*one_day)/one_hour;
    int64_t minutes = (millis - days*one_day - hours*one_hour)/one_min;
    int64_t seconds = (millis - days*one_day - hours*one_hour - minutes*one_min)/one_sec;

    // to print int64_t either cast to (long long int) or use macro PRId64 from <inttypes.h>
    if (days > 0) {
        printf("%lldd ", (long long int) days);
    }
    printf("%02lld:%02lld:%02lld", (long long int) hours, (long long int) minutes, (long long int) seconds);
}

void opt_callback(void * vdata, int accum_step, float * sched) {
    struct opt_callback_data * data = (struct opt_callback_data *) vdata;
    struct train_params * params    = data->params;
    struct ggml_opt_context * opt   = data->opt;
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
            const int n_iter = params->use_adam ? params->adam_n_iter : params->lbfgs_n_iter;
            const int done_iter = opt->iter - data->first_iter;
            const int remaining_iter = n_iter - done_iter;
            remaining_millis = remaining_iter * data->millis_per_iter;
        }

        // file saving
        const bool save_now = (params->save_every > 0) && (opt->iter - data->last_save_iter >= params->save_every);
        if (save_now) {
            int new_iters = opt->iter - data->last_save_iter;
            data->lora->train_its += new_iters;
            data->lora->train_samples += new_iters * opt->params.n_gradient_accumulation * n_batch;
            data->lora->train_tokens  += new_iters * opt->params.n_gradient_accumulation * n_batch * n_ctx;

            if (strlen(params->fn_checkpoint_out) > 0) {
                save_checkpoint_lora_file(params->fn_checkpoint_out, data->model, data->lora, opt, params->pattern_fn_it, opt->iter, params->fn_latest);
                save_checkpoint_lora_file(params->fn_checkpoint_out, data->model, data->lora, opt, params->pattern_fn_it, -1, params->fn_latest);
            }
            if (strlen(params->fn_lora_out) > 0) {
                save_as_llama_lora(data->lora, params->fn_lora_out, params->pattern_fn_it, opt->iter, params->fn_latest);
                save_as_llama_lora(data->lora, params->fn_lora_out, params->pattern_fn_it, -1, params->fn_latest);
            }
            data->last_save_iter = opt->iter;
        }

        // exclude file saving from time measurement, by measuring last_time after saving
        data->last_time = ggml_time_ms();

        *sched = (opt->iter < params->warmup)
                    ? (float) opt->iter / (float) params->warmup
                    : cosine_decay_restart(
                        params->cos_decay_steps,
                        params->cos_decay_min,
                        opt->iter - params->warmup,
                        params->cos_decay_restart,
                        params->enable_restart);
        float min_sched = params->adam_min_alpha / params->adam_alpha;
        *sched = min_sched + *sched * (1.0f - min_sched);

        int impr_plot = -(int)(1 + (opt->loss_before - opt->loss_after) * 10.0f + 0.5f);
        if (impr_plot > 0) impr_plot = 0;
        if (std::isnan(opt->loss_before) || std::isnan(opt->loss_before)) impr_plot = 0;
        printf("%s: iter=%6d sample=%zu/%zu sched=%f loss=%f",
            __func__, opt->iter, std::min(1+data->lora->shuffle_next_sample, data->lora->shuffle_sample_count), data->lora->shuffle_sample_count,
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
        // printf("improvement: %*d>", impr_plot, (int)0);
        printf("\n");
    }

    int used_samples = get_example_targets_batch(
        data->lctx,
        data->samples_begin,
        data->samples_size,
        data->samples_count,
        data->tokens_data,
        data->tokens_size,
        data->lora->shuffle_next_sample,
        data->tokens_input,
        data->target_probs,
        params->separate_with_eos,
        params->separate_with_bos,
        params->fill_with_next_samples);

    data->lora->shuffle_next_sample += used_samples;

    if (data->lora->shuffle_next_sample >= data->lora->shuffle_sample_count) {
        ++data->lora->train_epochs;
        printf("%s: reshuffle samples. completed epochs: %llu\n", __func__, (long long unsigned) data->lora->train_epochs);
        // note: we may have used some samples from the current shuffling more than once
        data->lora->shuffle_rng_state_current = data->lora->shuffle_rng_state_next;
        data->lora->shuffle_rng_state_next = shuffle_samples(
            data->lora->shuffle_rng_state_current,
            data->samples_begin,
            data->samples_size,
            data->samples_count);
        data->lora->shuffle_next_sample = 0;
    }
}

int64_t get_parameter_count(struct my_llama_lora* lora) {
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

size_t hash_combine(size_t h1, size_t h2) {
    return h1 ^ (h2 << 1);
}

size_t compute_samples_hash(const char* fn, const size_t* samples_begin, const size_t* samples_size, size_t sample_count) {
    std::hash<std::string> h_string;
    std::hash<unsigned long long> h_ull;
    size_t h = h_string(std::string(fn));
    h = hash_combine(h, h_ull((unsigned long long) sample_count));
    for (size_t i=0; i< sample_count; ++i) {
        h = hash_combine(h, h_ull((unsigned long long) samples_begin[i]));
        h = hash_combine(h, h_ull((unsigned long long) samples_size[i]));
    }
    return h;
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
    llama_params.vocab_only = false;

    printf("%s: model base = '%s'\n", __func__, params.fn_model_base);
    struct llama_model * lmodel = llama_load_model_from_file(params.fn_model_base, llama_params);
    struct llama_context * lctx = llama_new_context_with_model(lmodel, llama_params);

    struct my_llama_model model;
    init_model(lmodel, &model, params.n_ctx);

    struct my_llama_lora lora;
    struct ggml_opt_context* opt = (struct ggml_opt_context*)alloca(sizeof(struct ggml_opt_context));
    memset(opt, 0, sizeof(struct ggml_opt_context));

    opt->ctx = NULL;

    load_default_lora_params_from_base_model(params.fn_model_base, &lora.hparams);

    // set lora params from command line
    if (params.custom_f_norm_rms_eps) {
        lora.hparams.f_norm_rms_eps  = params.f_norm_rms_eps;
    }
    if (params.custom_rope_freq_base) {
        lora.hparams.rope_freq_base  = params.rope_freq_base;
    }
    if (params.custom_rope_freq_scale) {
        lora.hparams.rope_freq_scale = params.rope_freq_scale;
    }
    lora.hparams.lora_r                = params.lora_r;
    lora.hparams.lora_alpha            = params.lora_alpha;
    int n_rank_attention_norm          = params.custom_n_rank_attention_norm ? params.n_rank_attention_norm : 1;
    int n_rank_wq                      = params.custom_n_rank_wq             ? params.n_rank_wq             : params.lora_r;
    int n_rank_wk                      = params.custom_n_rank_wk             ? params.n_rank_wk             : params.lora_r;
    int n_rank_wv                      = params.custom_n_rank_wv             ? params.n_rank_wv             : params.lora_r;
    int n_rank_wo                      = params.custom_n_rank_wo             ? params.n_rank_wo             : params.lora_r;
    int n_rank_ffn_norm                = params.custom_n_rank_ffn_norm       ? params.n_rank_ffn_norm       : 1;
    int n_rank_w1                      = params.custom_n_rank_w1             ? params.n_rank_w1             : params.lora_r;
    int n_rank_w2                      = params.custom_n_rank_w2             ? params.n_rank_w2             : params.lora_r;
    int n_rank_w3                      = params.custom_n_rank_w3             ? params.n_rank_w3             : params.lora_r;
    int n_rank_tok_embeddings          = params.custom_n_rank_tok_embeddings ? params.n_rank_tok_embeddings : params.lora_r;
    int n_rank_norm                    = params.custom_n_rank_norm           ? params.n_rank_norm           : 1;
    int n_rank_output                  = params.custom_n_rank_output         ? params.n_rank_output         : params.lora_r;
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
    if (params.use_adam) {
        opt->params = ggml_opt_default_params(GGML_OPT_ADAM);
        opt->params.print_forward_graph     = false;
        opt->params.print_backward_graph    = false;
        opt->params.n_threads               = params.n_threads;
        opt->params.past                    = params.opt_past;
        opt->params.delta                   = params.opt_delta;
        opt->params.max_no_improvement      = params.opt_max_no_improvement;
        opt->params.n_gradient_accumulation = params.n_gradient_accumulation;
        opt->params.adam.n_iter             = params.adam_n_iter;
        opt->params.adam.sched              = 1.0f;
        opt->params.adam.alpha              = params.adam_alpha;
        opt->params.adam.decay              = params.adam_decay;
        opt->params.adam.decay_min_ndim     = params.adam_decay_min_ndim;
        opt->params.adam.beta1              = params.adam_beta1;
        opt->params.adam.beta2              = params.adam_beta2;
        opt->params.adam.gclip              = params.adam_gclip;
        opt->params.adam.eps_f              = params.adam_eps_f;
    } else {
        opt->params = ggml_opt_default_params(GGML_OPT_LBFGS);
        opt->params.print_forward_graph     = false;
        opt->params.print_backward_graph    = false;
        opt->params.n_threads               = params.n_threads;
        opt->params.past                    = params.opt_past;
        opt->params.delta                   = params.opt_delta;
        opt->params.max_no_improvement      = params.opt_max_no_improvement;
        opt->params.n_gradient_accumulation = params.n_gradient_accumulation;
        opt->params.lbfgs.n_iter            = params.lbfgs_n_iter;
    }

    ggml_allocr * alloc = NULL;

    printf("%s: init model\n", __func__);
    bool existed = load_checkpoint_lora_file(params.fn_checkpoint_in, &model, &lora, opt);

    if (existed) {
        // overwrite last n_ctx with user provided n_ctx
        if (params.custom_n_ctx) {
            model.hparams.n_ctx = params.n_ctx;
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

        const bool opt_past_changed = opt->params.past != params.opt_past;

        if (opt_param_count_changed) {
            print_lora_params(&lora.hparams);
            GGML_ASSERT(!"Provided rank differs from checkpoint file. To use different rank start finetune from scratch with empty input checkpoint, e.g --checkpoint-in ''. Aborting.");
            // need to discard previous optimizer gradient statistics and opt_init with new shapes
            // TODO
        }
        if (opt_past_changed) {
            GGML_ASSERT(!"Optimizer parameter '--opt-past N' differs from checkpoint file. To use different value finetune from scratch with empty input checkpoint, e.g --checkpoint-in ''. Aborting");
            // need to discard previous optimizer past function value statistics and opt_init with new shapes
            // TODO
        }
    } else { // existed == false
        init_lora(&model, &lora);
        randomize_lora(&lora, params.seed, 0.0f, 1.0f, -1.0f, +1.0f);
        if (!params.only_write_lora) {
            ggml_opt_init(opt->ctx, opt, opt->params, get_parameter_count(&lora));
        }
    }

    print_params(&model.hparams);
    print_lora_params(&lora.hparams);
    printf("%s: total train_iterations %llu\n", __func__, (long long unsigned) lora.train_its);
    printf("%s: seen train_samples     %llu\n", __func__, (long long unsigned) lora.train_samples);
    printf("%s: seen train_tokens      %llu\n", __func__, (long long unsigned) lora.train_tokens);
    printf("%s: completed train_epochs %llu\n", __func__, (long long unsigned) lora.train_epochs);
    printf("%s: max_lora_size = %zu bytes (%.1f MB)\n", __func__, lora.data.size(), (float) lora.data.size() / (1024.0f*1024.0f));
    printf("%s: max_opt_size  = %zu bytes (%.1f MB)\n", __func__, ggml_get_mem_size(opt->ctx), (float) ggml_get_mem_size(opt->ctx) / (1024.0f*1024.0f));
    opt->iter = lora.train_its;

    if (params.only_write_lora) {
        if (strlen(params.fn_lora_out) > 0) {
            save_as_llama_lora(&lora, params.fn_lora_out, params.pattern_fn_it, opt->iter, params.fn_latest);
            save_as_llama_lora(&lora, params.fn_lora_out, params.pattern_fn_it, -1, params.fn_latest);
        }
        ggml_free(lora.ctx);
        llama_free(lctx);
        llama_free_model(lmodel);
        return 0;
    }

    int n_tokens = model.hparams.n_ctx;
    int n_vocab  = model.hparams.n_vocab;
    int n_batch  = params.n_batch;

    printf("%s: opt iter %d\n", __func__, opt->iter);

    printf("used_mem model: %zu bytes\n", ggml_used_mem(lora.ctx));

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
    alloc = ggml_allocr_new_measure(tensor_alignment);
    ggml_allocr_alloc(alloc, tokens_input);
    ggml_allocr_alloc(alloc, target_probs);
    size_t max_input_size = ggml_allocr_max_size(alloc) + tensor_alignment;
    ggml_allocr_free(alloc);
    printf("%s: max_input_size = %zu bytes (%.1f MB)\n", __func__, max_input_size, (float) max_input_size / (1024.0f*1024.0f));

    // allocate input tensors
    mem_input_data.resize(max_input_size);
    alloc = ggml_allocr_new(mem_input_data.data(), mem_input_data.size(), tensor_alignment);
    ggml_allocr_alloc(alloc, tokens_input);
    ggml_allocr_alloc(alloc, target_probs);
    ggml_allocr_free(alloc);

    // context for compute tensors without their data
    size_t estimated_compute_size_wo_data = (
        ggml_tensor_overhead()*GGML_MAX_NODES*2
      + (GGML_OBJECT_SIZE+GGML_GRAPH_SIZE)*(
            params.use_checkpointing ? 3 : 2
        )
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
        gf = ggml_new_graph(ctx_compute);
        gf->order = (enum ggml_cgraph_eval_order) order;
        gb = ggml_new_graph(ctx_compute);
        gb_tmp = params.use_checkpointing
            ? ggml_new_graph(ctx_compute)
            : NULL;
        loss = llama_build_lora_finetune_graphs(
            &model, &lora, alloc, ctx_compute,
            gf, gb, gb_tmp,
            &logits, tokens_input, target_probs,
            n_tokens, n_batch,
            params.use_flash,
            params.use_checkpointing
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
    printf("%s: max_compute_size = %zu bytes (%.1f MB)\n", __func__, max_compute_size, (float) max_compute_size / (1024.0f*1024.0f));
    printf("%s: evaluation order = %s\n", __func__,
        (best_order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? "LEFT_TO_RIGHT" :
        (best_order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? "RIGHT_TO_LEFT" :
        "invalid");

    // allocate compute tensors
    mem_compute_data.resize(max_compute_size);
    ctx_compute = ggml_init(ctx_compute_params);
    alloc = ggml_allocr_new(mem_compute_data.data(), mem_compute_data.size(), tensor_alignment);
    gf = ggml_new_graph(ctx_compute);
    gf->order = best_order;
    gb = ggml_new_graph(ctx_compute);
    gb_tmp = params.use_checkpointing
        ? ggml_new_graph(ctx_compute)
        : NULL;
    loss = llama_build_lora_finetune_graphs(
        &model, &lora, alloc, ctx_compute,
        gf, gb, gb_tmp,
        &logits, tokens_input, target_probs,
        n_tokens, n_batch,
        params.use_flash,
        params.use_checkpointing
    );
    ggml_allocr_free(alloc);

    // tokenize data
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

    size_t shuffle_samples_hash = compute_samples_hash(params.fn_train_data, train_samples_begin.data(), train_samples_size.data(), train_samples_size.size());
    const bool changed_train_data = (shuffle_samples_hash != lora.shuffle_samples_hash) || (lora.shuffle_sample_count != train_samples_size.size());
    if (changed_train_data) {
        printf("%s: train data seems to have changed. restarting shuffled epoch.\n", __func__);
    }
    if (params.force_reshuffle) {
        printf("%s: forced reshuffling of data. restarting with newly shuffled epoch.\n", __func__);
    }
    if ((lora.shuffle_rng_state_current == "") || changed_train_data || params.force_reshuffle) {
        lora.shuffle_rng_state_current = mt19937_seed_to_state(params.seed);
        lora.shuffle_sample_count = train_samples_size.size();
        lora.shuffle_next_sample = 0;
        lora.shuffle_samples_hash = shuffle_samples_hash;
    }
    lora.shuffle_rng_state_next = shuffle_samples(
        lora.shuffle_rng_state_current,
        train_samples_begin.data(),
        train_samples_size.data(),
        train_samples_size.size());

    printf("%s: begin training\n", __func__);

    struct opt_callback_data opt_cb_data;
    opt_cb_data.params = &params;
    opt_cb_data.opt    = opt;
    opt_cb_data.model  = &model;
    opt_cb_data.lora   = &lora;
    opt_cb_data.lctx   = lctx;
    opt_cb_data.last_save_iter         = opt->iter;
    opt_cb_data.tokens_data            = train_tokens.data();
    opt_cb_data.tokens_size            = train_tokens.size();
    opt_cb_data.samples_begin          = train_samples_begin.data();
    opt_cb_data.samples_size           = train_samples_size.data();
    opt_cb_data.samples_count          = train_samples_size.size();
    opt_cb_data.tokens_input           = tokens_input;
    opt_cb_data.target_probs           = target_probs;
    opt_cb_data.first_iter             = opt->iter;
    opt_cb_data.last_time              = ggml_time_ms();
    opt_cb_data.millis_per_iter        = 0.0;

    // measure required memory for work buffer
    size_t max_work_size = ggml_graph_plan(gb, params.n_threads).work_size + GGML_OBJECT_SIZE;
    printf("%s: max_work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float) max_work_size / (1024.0f*1024.0f));

    // context for work buffer
    struct ggml_init_params ctx_work_params = {
        max_work_size, // mem_size
        NULL,          // mem_buffer
        false,         // no_alloc
    };
    struct ggml_context * ctx_work = ggml_init(ctx_work_params);

    int64_t t0 = ggml_time_ms();

    ggml_opt_resume_g(ctx_work, opt, loss, gf, gb, &opt_callback, (void *) &opt_cb_data);

    ggml_free(ctx_work);
    ggml_free(ctx_compute);
    ggml_free(ctx_input);

    int64_t t1 = ggml_time_ms();
    printf("%s: total training time: ", __func__);
    print_duration((double) (t1 - t0));
    printf("\n");

    int new_iters = opt->iter - opt_cb_data.last_save_iter;
    if (new_iters > 0) {
        lora.train_its += new_iters;
        lora.train_samples += new_iters * opt->params.n_gradient_accumulation * n_batch;
        lora.train_tokens  += new_iters * opt->params.n_gradient_accumulation * n_batch * n_tokens;

        if (strlen(params.fn_checkpoint_out) > 0) {
            save_checkpoint_lora_file(params.fn_checkpoint_out, &model, &lora, opt, params.pattern_fn_it, opt->iter, params.fn_latest);
            save_checkpoint_lora_file(params.fn_checkpoint_out, &model, &lora, opt, params.pattern_fn_it, -1, params.fn_latest);
        }
        if (strlen(params.fn_lora_out) > 0) {
            save_as_llama_lora(&lora, params.fn_lora_out, params.pattern_fn_it, opt->iter, params.fn_latest);
            save_as_llama_lora(&lora, params.fn_lora_out, params.pattern_fn_it, -1, params.fn_latest);
        }
        opt_cb_data.last_save_iter = opt->iter;
    }

    ggml_free(opt->ctx);
    ggml_free(lora.ctx);
    llama_free(lctx);
    llama_free_model(lmodel);
    return 0;
}
