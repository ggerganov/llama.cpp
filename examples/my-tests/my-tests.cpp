#include "ggml.h"
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

// static const float rms_norm_eps = 1e-6f;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float* wq; // (layer, dim, dim)
    float* wk; // (layer, dim, dim)
    float* wv; // (layer, dim, dim)
    float* wo; // (layer, dim, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
} TransformerWeights;

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

void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
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

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

struct my_llama_hparams {
    uint32_t n_vocab = 32000;
    uint32_t n_ctx   = 512;   // this is provided as user input?
    uint32_t n_embd  = 4096;
    uint32_t n_mult  = 4;
    uint32_t n_head  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;

    bool operator!=(const my_llama_hparams& other) const {
        return memcmp(this, &other, sizeof(my_llama_hparams));
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

struct my_llama_kv_cache {
    struct ggml_context * ctx = NULL;

    struct ggml_tensor * k;
    struct ggml_tensor * v;

    // llama_ctx_buffer buf;

    int n; // number of tokens currently in the cache
};

struct my_llama_model {
    struct ggml_context * ctx = NULL;

    my_llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<my_llama_layer> layers;

    uint32_t train_its = 0;
    uint32_t train_samples = 0;
    uint32_t train_tokens = 0;
};

uint32_t get_n_ff(const struct my_llama_hparams* hparams) {
    const uint32_t n_ff = ((2*(4*hparams->n_embd)/3 + hparams->n_mult - 1)/hparams->n_mult)*hparams->n_mult;
    return n_ff;
}

void print_params(struct my_llama_hparams * params) {
    printf("%s: n_vocab: %d\n", __func__, params->n_vocab);
    printf("%s: n_ctx:   %d\n", __func__, params->n_ctx);
    printf("%s: n_embd:  %d\n", __func__, params->n_embd);
    printf("%s: n_mult:  %d\n", __func__, params->n_mult);
    printf("%s: n_head:  %d\n", __func__, params->n_head);
    printf("%s: n_ff:    %d\n", __func__, get_n_ff(params));
    printf("%s: n_layer: %d\n", __func__, params->n_layer);
    printf("%s: n_rot:   %d\n", __func__, params->n_rot);
}

void init_model(struct my_llama_model * model) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;

    const uint32_t n_ff = get_n_ff(&hparams);

    struct ggml_context * ctx = model->ctx;

    model->train_its = 0;
    model->train_samples = 0;
    model->train_tokens = 0;
    // printf("FROM INIT_MODEL BHAI...\n\n\n");
    // print_params(&model->hparams);
    model->tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    printf("[%s:GG] Allocating [%d] x [%d] = [%d] float space for model->tok_embeddings\n",__func__,n_embd , n_vocab, n_embd * n_vocab);

    model->norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    printf("[%s:GG] Allocating [%d] float space for model->norm\n",__func__,n_embd);

    model->output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for model->output\n",__func__,n_embd, n_vocab, n_embd * n_vocab);

    // printing the per-layer allocations here so we dont print in the for loop.
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wq for [%d] layers\n",__func__,n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wk for [%d] layers\n",__func__,n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wv for [%d] layers\n",__func__,n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wo for [%d] layers\n",__func__,n_embd, n_embd, n_embd * n_embd, n_layer);

    printf("[%s:GG] Allocating [%d] float space for layer.ffn_norm for [%d] layers\n",__func__,n_embd, n_layer);

    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.w1 for [%d] layers\n",__func__,n_embd, n_ff, n_embd * n_ff, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.w2 for [%d] layers\n",__func__,n_ff, n_embd, n_ff * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.w3 for [%d] layers\n",__func__,n_embd, n_ff, n_embd * n_ff, n_layer);
    

    ggml_set_name(model->tok_embeddings, "tok_embeddings.weight");
    ggml_set_name(model->norm,           "norm.weight");
    ggml_set_name(model->output,         "output.weight");

    model->layers.resize(n_layer);
    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        std::string layers_i = "layers." + std::to_string(i);

        layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        layer.wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);

        layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        layer.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);
        layer.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff, n_embd);
        layer.w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);

        ggml_set_name(layer.attention_norm, (layers_i + ".attention_norm.weight").c_str());

        ggml_set_name(layer.wq, (layers_i + ".attention.wq.weight").c_str());
        ggml_set_name(layer.wk, (layers_i + ".attention.wk.weight").c_str());
        ggml_set_name(layer.wv, (layers_i + ".attention.wv.weight").c_str());
        ggml_set_name(layer.wo, (layers_i + ".attention.wo.weight").c_str());

        ggml_set_name(layer.ffn_norm, (layers_i + ".ffn_norm.weight").c_str());

        ggml_format_name(layer.w1, "%s.feed_forward.w1.weight", layers_i.c_str());
        ggml_format_name(layer.w2, "%s.feed_forward.w2.weight", layers_i.c_str());
        ggml_format_name(layer.w3, "%s.feed_forward.w3.weight", layers_i.c_str());
    }
}

void set_param_model(struct my_llama_model * model) {
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


bool init_kv_cache(struct my_llama_kv_cache* cache, struct my_llama_model * model, int n_batch) {
    const auto & hparams = model->hparams;

    const uint32_t n_ctx   = hparams.n_ctx;
    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx*n_batch;
    const int64_t n_elements = n_embd*n_mem;

    // cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);

    // struct ggml_init_params params;
    // params.mem_size   = cache.buf.size;
    // params.mem_buffer = cache.buf.addr;
    // params.no_alloc   = false;
    if (!cache->ctx) {
        struct ggml_init_params params;
        params.mem_size   = 2u*n_elements*ggml_type_size(GGML_TYPE_F32) + 2u*1024*1024;
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        cache->ctx = ggml_init(params);

        if (!cache->ctx) {
            fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
            return false;
        }
    }

    cache->k = ggml_new_tensor_1d(cache->ctx, GGML_TYPE_F32, n_elements);
    cache->v = ggml_new_tensor_1d(cache->ctx, GGML_TYPE_F32, n_elements);

    return true;
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

void set_f32_3d(struct ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2, float value) {
    float * ptr = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
    *ptr = value;
}

void set_f32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1, float value) {
    float * ptr = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    *ptr = value;
}

void set_i32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1, int32_t value) {
    int32_t * ptr = (int32_t *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    *ptr = value;
}

float get_f32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1) {
    float * ptr = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    return *ptr;
}

int32_t get_i32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1) {
    int32_t * ptr = (int32_t *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    return *ptr;
}

void print_row(struct ggml_tensor * probs, int i) {
    for (int k = 0; k < probs->ne[0]; ++k) {
        float p = get_f32_2d(probs, k, i);
        printf(" %.2f", p);
    }
    printf("\n");
}

void print_matrix(struct ggml_tensor * probs) {
    assert(probs->n_dims == 2);
    for (int i = 0; i < probs->ne[1]; ++i) {
        for (int k = 0; k < probs->ne[0]; ++k) {
            float p = get_f32_2d(probs, k, i);
            printf(" %.2f", p);
        }
        printf("\n");
    }
}


void print_token(struct llama_context * ctx, llama_token token) {
    printf("%s", llama_token_to_str(ctx, token));
}

void print_tokens(struct llama_context* ctx, struct ggml_tensor * tokens) {
    for (int i=0; i<tokens->ne[0]; ++i) {
        int token = ggml_get_i32_1d(tokens, i);
        print_token(ctx, token);
    }
}

void print_tokens_batch(struct llama_context* ctx, struct ggml_tensor * tokens) {
    for (int i1=0; i1<tokens->ne[1]; ++i1) {
        //int num_newline = 0;
        for (int i0=0; i0<tokens->ne[0]; ++i0) {
            int token = get_i32_2d(tokens, i0, i1);
            print_token(ctx, token);
            // bool isnl = (token == llama_token_nl());
            // if (isnl) {
            //     ++num_newline;
            // }
            // if (isnl) {
            //     if (num_newline < 2) {
            //         print_token(ctx, token);
            //     } else {
            //         printf("\\n");
            //     }
            // } else {
            //     print_token(ctx, token);
            // }
        }
        printf("\n--\n");
    }
}

void get_example_targets(const int * train_samples, size_t n_train_samples, const llama_token * train_data, size_t n_train_data, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * target_logits, struct ggml_tensor * target_probs) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab  = target_logits->ne[0];

    size_t sample = train_samples[example_id % n_train_samples];
    GGML_ASSERT(sample+n_tokens-1 < n_train_data);

    ggml_set_f32(target_logits, -1.0f/n_vocab);
    ggml_set_f32(target_probs, 0.0f);
    ggml_set_i32_1d(tokens_input, 0, llama_token_bos());
    for (int i=1; i<n_tokens+1; ++i) {
        int token = clamp(train_data[sample+i-1], 0, n_vocab-1);
        set_f32_2d(target_logits, token, i-1, +1.0f);
        set_f32_2d(target_probs,  token, i-1, +1.0f);
        if (i<n_tokens) {
            ggml_set_i32_1d(tokens_input, i, token);
        }
    }
}

void get_example_targets_batch(struct llama_context * /*lctx*/, const int * train_samples, size_t n_train_samples, const llama_token * train_data, size_t n_train_data, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * target_logits, struct ggml_tensor * target_probs) {
    GGML_ASSERT(tokens_input->n_dims  == 2);
    GGML_ASSERT(target_logits->n_dims == 3);
    GGML_ASSERT(target_probs->n_dims  == 3);
    int n_vocab  = target_logits->ne[0];
    int n_tokens = tokens_input->ne[0];
    int n_batch  = tokens_input->ne[1];
    GGML_ASSERT(n_tokens == target_logits->ne[1]);
    GGML_ASSERT(n_batch  == target_logits->ne[2]);
    GGML_ASSERT(n_vocab  == target_probs->ne[0]);
    GGML_ASSERT(n_tokens == target_probs->ne[1]);
    GGML_ASSERT(n_batch  == target_probs->ne[2]);

    ggml_set_f32(target_logits, -1.0f/n_vocab);
    ggml_set_f32(target_probs, 0.0f);
    for (int k=0; k<n_batch; ++k) {
        // printf("%s: batch %d\n", __func__, k);
        size_t sample = train_samples[(example_id*n_batch + k) % n_train_samples];
        GGML_ASSERT(sample+n_tokens-1 < n_train_data);

        set_i32_2d(tokens_input, 0, k, llama_token_bos());
        for (int i=1; i<n_tokens+1; ++i) {
            int token = clamp(train_data[sample+i-1], 0, n_vocab-1);
            // print_token(lctx, token);
            set_f32_3d(target_logits, token, i-1, k, +1.0f);
            set_f32_3d(target_probs,  token, i-1, k, +1.0f);
            if (i<n_tokens) {
                set_i32_2d(tokens_input, i, k, token);
            }
        }
        // printf("\n=\n");
        // for (int i=0; i<n_tokens; ++i) {
        //     int token = get_i32_2d(tokens_input, i, k);
        //     print_token(lctx, token);
        // }
        // printf("\n-\n");
    }
}


void lshift_examples(struct ggml_tensor * tokens_input, struct ggml_tensor * target_logits, struct ggml_tensor * target_probs, int n_shift) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab = target_logits->ne[0];
    for (int i=0; i<n_tokens-n_shift; ++i) {
        ggml_set_i32_1d(tokens_input, i, ggml_get_i32_1d(tokens_input, i + n_shift));
        for (int k=0; k<n_vocab; ++k) {
            ggml_set_f32_1d(target_logits, i*n_vocab + k, ggml_get_f32_1d(target_logits, (i + n_shift)*n_vocab + k));
            ggml_set_f32_1d(target_probs, i*n_vocab + k,  ggml_get_f32_1d(target_probs,  (i + n_shift)*n_vocab + k));
        }
    }
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

int tokenize_file(struct llama_context * lctx, const char * filename, std::vector<llama_token>& out) {
    struct llama_file f(filename, "rb");

    std::vector<char> buf;
    buf.resize(f.size+1);

    f.read_raw(buf.data(), f.size);
    buf[f.size] = '\0';

    out.resize(buf.size());

    int n_tokens = llama_tokenize(lctx, buf.data(), out.data(), buf.size(), false);
    if (n_tokens >= 0) {
        out.resize(n_tokens);
    }

    bool verify = false;
    if (verify) {
        const char * in  = buf.data();
        const char * end = buf.data() + buf.size();
        for (int i = 0; i < (int) out.size(); ++i) {
            const char * s = llama_token_to_str(lctx, out[i]);
            int len = strlen(s);
            if (in >= end) {
                printf("%s: unexpected end of original text.\n", __func__);
                break;
            }
            const bool matches = (strncmp(in, s, len) == 0);
            if (matches) {
                in += len;
            } else {
                printf("%s: mismatch: expected '%s', but got '%s'\n", __func__, std::string(in, len).c_str(), s);
            }
        }
    }

    return n_tokens;
}

void shuffle_ints(int * begin, int * end) {
    if (end <= begin) return;
    int max=begin[0];
    for (int i=1; i<end-begin; ++i) {
        if (begin[i] > max) {
            max = begin[i];
        }
    }
    std::vector<float> vals;
    vals.resize(max+1);
    for (int i=0; i<max+1; ++i) {
       vals[i] = frand();
    }
    std::sort(begin, end, [&vals](int a, int b){
       return vals.at(a) < vals.at(b);
    });
}

struct my_llama_sampler_params {
    float temp            = 0.0f;  // <= 0.0 disabled
    int   top_k           = 20;    // <= 0 to use vocab size
    float top_p           = 0.95f; // 1.0 = disabled
    float tfs_z           = 1.00f; // 1.0 = disabled
    float typical_p       = 1.00f; // 1.0 = disabled
    int   repeat_last_n   = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float repeat_penalty  = 1.0f;  // 1.0 = disabled
    float alpha_presence  = 0.0f;  // 0.0 = disabled
    float alpha_frequency = 0.0f;  // 0.0 = disabled
    int   mirostat        = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float mirostat_tau    = 5.00f; // target entropy
    float mirostat_eta    = 0.10f; // learning rate
    bool  penalize_nl     = true;  // consider newlines as a repeatable token
};

struct my_llama_sampler {
    struct llama_context * ctx = NULL;
    my_llama_sampler_params params;

    int n_vocab = 0;
    int n_ctx = 0;

    float mirostat_mu;

    std::vector<llama_token_data> candidates;
    llama_token_data_array candidates_p;

};

void init_sampler(struct my_llama_sampler * sampler, struct llama_context * ctx) {
    sampler->ctx = ctx;
    sampler->n_vocab = llama_n_vocab(sampler->ctx);
    sampler->n_ctx   = llama_n_ctx(sampler->ctx);
    sampler->mirostat_mu = 2.0f * sampler->params.mirostat_tau;
}

llama_token sample(struct my_llama_sampler * sampler, float * logits, const llama_token * last_tokens, int n_last_tokens) {
    GGML_ASSERT(sampler->ctx != NULL);

    struct llama_context * ctx = sampler->ctx;

    sampler->candidates.resize(sampler->n_vocab);
    for (llama_token token_id = 0; token_id < sampler->n_vocab; ++token_id) {
        sampler->candidates[token_id].id = token_id;
        sampler->candidates[token_id].logit = logits[token_id];
        sampler->candidates[token_id].p = 0.0;
    }

    llama_token_data_array * candidates_p = & sampler->candidates_p;

    candidates_p->data = sampler->candidates.data();
    candidates_p->size = sampler->candidates.size();
    candidates_p->sorted = false;

    const auto params = sampler->params;

    // Apply penalties
    const float nl_logit = logits[llama_token_nl()];

    const int n_last = std::min(std::min(n_last_tokens, params.repeat_last_n), sampler->n_ctx);

    llama_sample_repetition_penalty(
        ctx,
        candidates_p,
        last_tokens + n_last_tokens - n_last,
        n_last,
        params.repeat_penalty);
    llama_sample_frequency_and_presence_penalties(
        ctx,
        candidates_p,
        last_tokens + n_last_tokens - n_last,
        n_last,
        params.alpha_frequency,
        params.alpha_presence);

    if (!params.penalize_nl) {
        logits[llama_token_nl()] = nl_logit;
    }

    llama_token token = 0;
    if (params.temp <= 0) {
        // Greedy sampling
        token = llama_sample_token_greedy(ctx, candidates_p);
    } else {
        if (params.mirostat == 1) {
            int mirostat_m = 100;
            llama_sample_temperature(ctx, candidates_p, params.temp);
            token = llama_sample_token_mirostat(ctx, candidates_p, params.mirostat_tau, params.mirostat_eta, mirostat_m, &sampler->mirostat_mu);
        } else if (params.mirostat == 2) {
            llama_sample_temperature(ctx, candidates_p, params.temp);
            token = llama_sample_token_mirostat_v2(ctx, candidates_p, params.mirostat_tau, params.mirostat_eta, &sampler->mirostat_mu);
        } else {
            // Temperature sampling
            llama_sample_top_k        (ctx, candidates_p, params.top_k, 1);
            llama_sample_tail_free    (ctx, candidates_p, params.tfs_z, 1);
            llama_sample_typical      (ctx, candidates_p, params.typical_p, 1);

            llama_sample_top_p        (ctx, candidates_p, params.top_p, 1);
            llama_sample_temperature  (ctx, candidates_p, params.temp);
            token = llama_sample_token(ctx, candidates_p);
        }
    }
    return token;
}

void set_logits_masked(struct ggml_tensor * logits, std::vector<bool>& mask, float value) {
    GGML_ASSERT(logits->ne[0] == (int64_t) mask.size());
    for (int i2 = 0; i2 < logits->ne[2]; ++i2) {
        for (int i1 = 0; i1 < logits->ne[1]; ++i1) {
            for (int i0 = 0; i0 < logits->ne[0]; ++i0) {
                if (!mask[i0]) continue;
                float * ptr = (float *) ((char *) logits->data + i2*logits->nb[2] + i1*logits->nb[1] + i0*logits->nb[0]);
                *ptr = value;
            }
        }
    }
}

void write_tensor(struct llama_file * file, struct ggml_tensor * tensor) {
    if (tensor == NULL) {
        file->write_u32(0);
        file->write_u32(0);
        file->write_u32(GGML_TYPE_F32);
        file->seek((0-file->tell()) & 31, SEEK_CUR);
        return;
    }
    const char * name = ggml_get_name(tensor);
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

void read_tensor(struct llama_file * file, struct ggml_tensor * tensor) {
    int32_t nd = file->read_u32();
    GGML_ASSERT(nd == tensor->n_dims);

    uint32_t name_len       = file->read_u32();
    enum     ggml_type type = (enum ggml_type) file->read_u32();
    GGML_ASSERT(type == tensor->type);

    uint32_t ne[4];
    file->read_raw(ne, sizeof(ne[0]) * nd);
    for (int i=0; i<nd; ++i) {
        GGML_ASSERT(ne[i] == tensor->ne[i]);
    }

    std::string name = file->read_string(name_len);
    GGML_ASSERT(strncmp(ggml_get_name(tensor), name.c_str(), sizeof(tensor->name)-1) == 0);

    file->seek((0-file->tell()) & 31, SEEK_CUR);
    file->read_raw(tensor->data, ggml_nbytes(tensor));
}

void write_opt_context(struct llama_file * file, struct ggml_opt_context * opt) {
    const uint32_t version = 0;
    GGML_ASSERT(opt->nx   >= 0);
    GGML_ASSERT(opt->iter >= 0);
    file->write_u32(version);
    file->write_raw(&opt->params, sizeof(opt->params));
    file->write_raw(&opt->nx,     sizeof(opt->nx));
    file->write_raw(&opt->iter,   sizeof(opt->iter));
    file->write_u32((uint32_t)  opt->just_initialized);
    switch (opt->params.type) {
        case GGML_OPT_ADAM:
            {
                GGML_ASSERT(opt->adam.x != NULL);
                write_tensor(file, opt->adam.x);
                write_tensor(file, opt->adam.g1);
                write_tensor(file, opt->adam.g2);
                write_tensor(file, opt->adam.m);
                write_tensor(file, opt->adam.v);
                write_tensor(file, opt->adam.mh);
                write_tensor(file, opt->adam.vh);
                write_tensor(file, opt->adam.pf);
                file->write_raw(&opt->adam.fx_best,          sizeof(opt->adam.fx_best));
                file->write_raw(&opt->adam.fx_prev,          sizeof(opt->adam.fx_prev));
                file->write_raw(&opt->adam.n_no_improvement, sizeof(opt->adam.n_no_improvement));
            } break;
        case GGML_OPT_LBFGS:
            {
                GGML_ASSERT(opt->adam.x != NULL);
                write_tensor(file, opt->lbfgs.x);
                write_tensor(file, opt->lbfgs.xp);
                write_tensor(file, opt->lbfgs.g);
                write_tensor(file, opt->lbfgs.gp);
                write_tensor(file, opt->lbfgs.d);
                write_tensor(file, opt->lbfgs.pf);
                write_tensor(file, opt->lbfgs.lmal);
                write_tensor(file, opt->lbfgs.lmys);
                write_tensor(file, opt->lbfgs.lms);
                write_tensor(file, opt->lbfgs.lmy);
                file->write_raw(&opt->lbfgs.fx_best,          sizeof(opt->lbfgs.fx_best));
                file->write_raw(&opt->lbfgs.step,             sizeof(opt->lbfgs.step));
                file->write_raw(&opt->lbfgs.j,                sizeof(opt->lbfgs.j));
                file->write_raw(&opt->lbfgs.k,                sizeof(opt->lbfgs.k));
                file->write_raw(&opt->lbfgs.end,              sizeof(opt->lbfgs.end));
                file->write_raw(&opt->lbfgs.n_no_improvement, sizeof(opt->lbfgs.n_no_improvement));
            } break;
    }
}

void read_opt_context(struct llama_file * file, struct ggml_context * ctx, struct ggml_opt_context * opt) {
    uint32_t version = file->read_u32();
    GGML_ASSERT(version == 0);

    file->read_raw(&opt->params, sizeof(opt->params));
    file->read_raw(&opt->nx,     sizeof(opt->nx));
    ggml_opt_init(ctx, opt, opt->params, opt->nx);

    file->read_raw(&opt->iter,   sizeof(opt->iter));
    opt->just_initialized = (bool) file->read_u32();

    switch (opt->params.type) {
        case GGML_OPT_ADAM:
            {
                read_tensor(file, opt->adam.x);
                read_tensor(file, opt->adam.g1);
                read_tensor(file, opt->adam.g2);
                read_tensor(file, opt->adam.m);
                read_tensor(file, opt->adam.v);
                read_tensor(file, opt->adam.mh);
                read_tensor(file, opt->adam.vh);
                if (opt->adam.pf) { read_tensor(file, opt->adam.pf); }
                file->read_raw(&opt->adam.fx_best,          sizeof(opt->adam.fx_best));
                file->read_raw(&opt->adam.fx_prev,          sizeof(opt->adam.fx_prev));
                file->read_raw(&opt->adam.n_no_improvement, sizeof(opt->adam.n_no_improvement));
            } break;
        case GGML_OPT_LBFGS:
            {
                GGML_ASSERT(opt->adam.x != NULL);
                read_tensor(file, opt->lbfgs.x);
                read_tensor(file, opt->lbfgs.xp);
                read_tensor(file, opt->lbfgs.g);
                read_tensor(file, opt->lbfgs.gp);
                read_tensor(file, opt->lbfgs.d);
                if (opt->lbfgs.pf) { read_tensor(file, opt->lbfgs.pf); }
                read_tensor(file, opt->lbfgs.lmal);
                read_tensor(file, opt->lbfgs.lmys);
                read_tensor(file, opt->lbfgs.lms);
                read_tensor(file, opt->lbfgs.lmy);
                file->read_raw(&opt->lbfgs.fx_best,          sizeof(opt->lbfgs.fx_best));
                file->read_raw(&opt->lbfgs.step,             sizeof(opt->lbfgs.step));
                file->read_raw(&opt->lbfgs.j,                sizeof(opt->lbfgs.j));
                file->read_raw(&opt->lbfgs.k,                sizeof(opt->lbfgs.k));
                file->read_raw(&opt->lbfgs.end,              sizeof(opt->lbfgs.end));
                file->read_raw(&opt->lbfgs.n_no_improvement, sizeof(opt->lbfgs.n_no_improvement));
            } break;
    }
}

bool load_checkpoint(struct my_llama_model * model, struct ggml_opt_context * opt, const char * filename, bool init) {
    struct llama_file file(filename, "rb");

    uint32_t magic;
    uint32_t version;

    uint32_t train_its = 0;
    uint32_t train_samples = 0;
    uint32_t train_tokens = 0;

    if (file.fp) {
        printf("%s: Loading model from '%s'.\n", __func__, filename);
        magic                  = file.read_u32();
        GGML_ASSERT(magic     == 'ggcp');
        version                = file.read_u32();
        GGML_ASSERT(version   == 0);
        train_its              = file.read_u32();
        train_samples          = file.read_u32();
        train_tokens           = file.read_u32();
        model->hparams.n_vocab = file.read_u32();
        model->hparams.n_embd  = file.read_u32();
        model->hparams.n_mult  = file.read_u32();
        model->hparams.n_head  = file.read_u32();
        model->hparams.n_layer = file.read_u32();
        model->hparams.n_rot   = file.read_u32();
        print_params(&model->hparams);
    }

    if (init) {
        init_model(model);
    }

    if (file.fp) {
        model->train_its = train_its;
        model->train_samples = train_samples;
        model->train_tokens = train_tokens;
    }

    printf("%s: Training iterations: %u.\n", __func__, model->train_its);
    printf("%s: Training samples:    %u.\n", __func__, model->train_samples);
    printf("%s: Training tokens:     %u.\n", __func__, model->train_tokens);

    if (file.fp) {
        read_tensor(&file, model->tok_embeddings);
        read_tensor(&file, model->norm);
        read_tensor(&file, model->output);

        for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
            auto & layer = model->layers[i];

            read_tensor(&file, layer.attention_norm);
            read_tensor(&file, layer.wq);
            read_tensor(&file, layer.wk);
            read_tensor(&file, layer.wv);
            read_tensor(&file, layer.wo);
            read_tensor(&file, layer.ffn_norm);
            read_tensor(&file, layer.w1);
            read_tensor(&file, layer.w2);
            read_tensor(&file, layer.w3);
        }

        read_opt_context(&file, model->ctx, opt);
    }

    return (file.fp != NULL);
}

void print_sample_weights(TransformerWeights *w){
    printf("----- Quick print of first of the weight vales of all the variables\n");
    printf("%f\n", w->token_embedding_table[0]);
    printf("%f\n", w->rms_att_weight[0]);
    printf("%f\n", w->rms_ffn_weight[0]);

    printf("%f\n", w->wq[0]);
    printf("%f\n", w->wk[0]);
    printf("%f\n", w->wv[0]);
    printf("%f\n", w->wo[0]);
    printf("%f\n", w->w1[0]);
    printf("%f\n", w->w2[0]);
    printf("%f\n", w->w3[0]);
    printf("%f\n", w->rms_att_weight[0]);
    printf("%f\n", w->freq_cis_real[0]);
    printf("%f\n", w->freq_cis_imag[0]);
    printf("------------------------------------------------------------------\n");

    
}

void stuff_karpathy_weights_into_gg(struct ggml_tensor * gg_weights, float * karpathy_weights){
    
    int ct;
    switch (gg_weights->n_dims){
        case 1:
            ct = 0;
            for (int i0 = 0; i0 < gg_weights->ne[0]; i0++){
                float * ptr = (float *) ((char *) gg_weights->data + i0*gg_weights->nb[0]);
                *ptr = karpathy_weights[ct];
            }
        case 2:
            ct = 0;
            for (int i1 = 0; i1 < gg_weights->ne[1]; i1++) {
                for (int i0 = 0; i0 < gg_weights->ne[0]; i0++) {
                    // set_f32_2d(gg_weights, k, i, karpathy_weights[ct]);
                    float * ptr = (float *) ((char *) gg_weights->data + i0*gg_weights->nb[0] + i1*gg_weights->nb[1]);
                    *ptr = karpathy_weights[ct];
                    ct++;
                }
            }
            break;
        case 3:
            ct = 0;
            for (int i2 = 0; i2 < gg_weights->ne[2]; i2++) {
                for (int i1 = 0; i1 < gg_weights->ne[1]; i1++) {
                    for (int i0 = 0; i0 < gg_weights->ne[0]; i0++) {
                        // set_f32_3d(gg_weights, k, j, i, karpathy_weights[ct]);
                        float * ptr = (float *) ((char *) gg_weights->data + i0*gg_weights->nb[0] + i1*gg_weights->nb[1] + i2*gg_weights->nb[2]);
                        *ptr = karpathy_weights[ct];
                        ct++;
                    }
                }
            }
            break;    
    }

    // void set_f32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1, float value)
    // set_f32_2d(gg_weights, 142.0, 0, 0);

    // float p = get_f32_2d(gg_weights, 0, 0);
    // print_row(gg_weights, 0);
    // print_matrix(gg_weights);
}

void save_as_llama_model(struct llama_vocab * vocab, struct my_llama_model * model, TransformerWeights* w, const char * filename) {
    struct llama_file file(filename, "wb");
    if (file.fp == NULL) {
        return;
    }
    // print_sample_weights(w);
    // write_magic
    file.write_u32(LLAMA_FILE_MAGIC);   // magic
    file.write_u32(LLAMA_FILE_VERSION); // version
    // write_hparams
    file.write_u32(model->hparams.n_vocab);
    file.write_u32(model->hparams.n_embd);
    file.write_u32(model->hparams.n_mult);
    file.write_u32(model->hparams.n_head);
    file.write_u32(model->hparams.n_layer);
    file.write_u32(model->hparams.n_rot);
    file.write_u32(LLAMA_FTYPE_ALL_F32);
    // write_vocab
    uint32_t n_vocab = model->hparams.n_vocab;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const auto & token_score = vocab->id_to_token.at(i);
        file.write_u32((uint32_t) token_score.tok.size());
        file.write_raw(token_score.tok.data(), token_score.tok.size());
        file.write_raw(&token_score.score, sizeof(token_score.score));
    }

    // stuff AK weights into GG weights one by one.
    // w->token_embedding_table -> model->tok_embeddings
    // float*                   -> struct ggml_tensor
    stuff_karpathy_weights_into_gg(model->tok_embeddings, w->token_embedding_table);
    print_row(model->tok_embeddings, 0);

    // stuff_karpathy_weights_into_gg(model->norm, w->rms_final_weight);
    // stuff_karpathy_weights_into_gg(model->norm, w->freq_cis_real);               // <<<<<<<<<< mostly wrong
    // stuff_karpathy_weights_into_gg(model->norm, w->freq_cis_imag);               // <<<<<<<<<< mostly wrong

    // for rms-att-weight 
    int row_length = model->hparams.n_embd;
    for (uint32_t i = 0; i < model->hparams.n_layer; ++i){
        auto & layer = model->layers[i];
        // 2d        
        stuff_karpathy_weights_into_gg(layer.attention_norm, &w->rms_att_weight[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.ffn_norm      , &w->rms_ffn_weight[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.wq            , &w->wq[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.wk            , &w->wk[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.wv            , &w->wv[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.wo            , &w->wo[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.w1            , &w->w1[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.w2            , &w->w2[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.w3            , &w->w3[i*row_length]);
    }
    
    // write tensors
    write_tensor(&file, model->tok_embeddings);
    write_tensor(&file, model->norm);
    write_tensor(&file, model->output);
    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        printf(" testing new here %d\n", i);
        auto & layer = model->layers[i];

        write_tensor(&file, layer.attention_norm);
        write_tensor(&file, layer.wq);
        write_tensor(&file, layer.wk);
        write_tensor(&file, layer.wv);
        write_tensor(&file, layer.wo);
        write_tensor(&file, layer.ffn_norm);
        write_tensor(&file, layer.w1);
        write_tensor(&file, layer.w2);
        write_tensor(&file, layer.w3);
    }
}



struct train_params {
    const char * fn_vocab_model;
    const char * fn_train_data;
    const char * fn_checkpoint_in;
    const char * fn_checkpoint_out;
    const char * fn_model_out;

    uint32_t seed;

    int n_ctx;
    int n_embd;
    int n_mult;
    int n_head;
    int n_layer;
    int n_rotmax;

    int n_threads;
    int n_batch;
    int n_examples;
    int n_predict;

    int print_info_interval;
    int print_details_interval;

    bool samples_start_after_nl;
    bool use_adam;
    bool use_flash;
    bool use_scratch;

    // only adam
    int   warmup;
    int   cos_decay_steps;
    float cos_decay_restart;
    float cos_decay_alpha;

    int   lbfgs_n_iter;
    int   adam_n_iter;
    float adam_alpha;
    float adam_decay;

    int mem_model_gb;
    int mem_compute_gb;
    int mem_compute0_gb;
    int mem_compute1_gb;
};

struct train_params get_default_train_params() {
    struct train_params params;
    params.fn_vocab_model    = "ggml-vic7b-uncensored-q4_0.bin";
    params.fn_train_data     = "shakespeare.txt";
    params.fn_checkpoint_in  = "checkpoint.bin";
    params.fn_checkpoint_out = "checkpoint.bin";
    params.fn_model_out      = "ggml-checkpoint-f32.bin";

    params.seed       =   -1;

    params.n_ctx      =  128;
    params.n_embd     =  256;
    params.n_mult     =  256;
    params.n_head     =    8;
    params.n_layer    =   16;
    params.n_rotmax   =   64;

    params.n_threads  =    6;
    params.n_batch    =    8;
    params.n_examples =    8;
    params.n_predict  = 1024;

    params.print_info_interval    = 1;
    params.print_details_interval = 2;

    params.samples_start_after_nl = false;
    params.use_adam               = true;
    params.use_flash              = true;
    params.use_scratch            = true;

    // only adam
    params.warmup            =  100;
    params.cos_decay_steps   = 1000;
    params.cos_decay_restart = 1.1f;
    params.cos_decay_alpha   = 0.0f;

    params.lbfgs_n_iter      = 16;
    params.adam_n_iter       = 16;
    params.adam_alpha        = 1e-3f;
    params.adam_decay        = 1e-3f;

    params.mem_model_gb   = 2;
    params.mem_compute_gb = 24;
    params.mem_compute0_gb = 8;
    params.mem_compute1_gb = 2;

    return params;
}

void train_print_usage(int /*argc*/, char ** argv, const struct train_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                 show this help message and exit\n");
    fprintf(stderr, "  --vocab-model FNAME        model path from which to load vocab (default '%s')\n", params->fn_vocab_model);
    fprintf(stderr, "  --train-data FNAME         path from which to load training data (default '%s')\n", params->fn_train_data);
    fprintf(stderr, "  --checkpoint-in FNAME      path from which to load training checkpoint (default '%s')\n", params->fn_checkpoint_in);
    fprintf(stderr, "  --checkpoint-out FNAME     path to save training checkpoint (default '%s')\n", params->fn_checkpoint_out);
    fprintf(stderr, "  --model-out FNAME          path to save ggml model (default '%s')\n", params->fn_model_out);
    fprintf(stderr, "  -s SEED, --seed SEED       RNG seed (default: -1, use random seed for -1)\n");
    fprintf(stderr, "  -c N, --ctx N              Context size used during training (default %d)\n", params->n_ctx);
    fprintf(stderr, "  --embd N                   Embedding size used for new models (default %d)\n", params->n_embd);
    fprintf(stderr, "  --mult N                   Mult size used for new models, influences feedforward size. (default %d)\n", params->n_mult);
    fprintf(stderr, "  --head N                   Number of heads for new models (default %d)\n", params->n_head);
    fprintf(stderr, "  --layer N                  Number of layers for new models (default %d)\n", params->n_layer);
    fprintf(stderr, "  --rotmax N                 Maximal number Rope dimensions for new models (default %d)\n", params->n_rotmax);
    fprintf(stderr, "  -t N, --threads N          Number of threads (default %d)\n", params->n_threads);
    fprintf(stderr, "  -b N, --batch N            Parallel batch size (default %d)\n", params->n_batch);
    fprintf(stderr, "  -n N, --examples N         Number of examples to train (default %d)\n", params->n_examples);
    fprintf(stderr, "  --predict N                Number of tokens to generate after training (default %d)\n", params->n_predict);
    fprintf(stderr, "  --print-info-interval N    Print infos during training each N examples (default %d)\n", params->print_info_interval);
    fprintf(stderr, "  --print-details-interval N Print details during training each N examples (default %d)\n", params->print_details_interval);
    fprintf(stderr, "  --samples-after-nl         Training samples start after newlines. (default %s)\n", params->samples_start_after_nl ? "on" : "off");
    fprintf(stderr, "  --use-lbfgs                Use LBFGS optimizer instead of default Adam\n");
    fprintf(stderr, "  --use-adam                 Use Adam optimizer (default)\n");
    fprintf(stderr, "  --no-flash                 Don't use flash attention.\n");
    fprintf(stderr, "  --use-flash                Use flash attention (default)\n");
    fprintf(stderr, "  --no-scratch               Don't use scratch buffers\n");
    fprintf(stderr, "  --use-scratch              Use scratch buffers (default)\n");
    fprintf(stderr, "  --warmup N                 Number of warmup steps (default %d)\n", params->warmup);
    fprintf(stderr, "  --cos-decay-steps N        Number of cosine decay steps (default %d)\n", params->cos_decay_steps);
    fprintf(stderr, "  --cos-decay-restart N      Increase of cosine decay steps after restart (default %f)\n", params->cos_decay_restart);
    fprintf(stderr, "  --cos-decay-alpha N        Cosine decay alpha (default %f)\n", params->cos_decay_alpha);
    fprintf(stderr, "  --lbfgs-iter N             Maximum number of LBFGS optimization iterations for each batch (default %d)\n", params->lbfgs_n_iter);
    fprintf(stderr, "  --adam-iter N              Maximum number of Adam optimization iterations for each batch (default %d)\n", params->adam_n_iter);
    fprintf(stderr, "  --adam-alpha N             Adam learning rate alpha (default %f)\n", params->adam_alpha);
    fprintf(stderr, "  --adam-decay N             AdamW weight decay. Values greater zero enable AdamW instead of regular Adam. (default %f)\n", params->adam_decay);
    fprintf(stderr, "  --mem-model N              Memory to allocate for model and cache in gigabytes. (default %d)\n", params->mem_model_gb);
    fprintf(stderr, "  --mem-compute N            Memory to allocate for compute in gigabytes. (default %d)\n", params->mem_compute_gb);
    fprintf(stderr, "  --mem-compute0 N           Memory to allocate for compute in gigabytes. (default %d)\n", params->mem_compute0_gb);
    fprintf(stderr, "  --mem-compute1 N           Memory to allocate for compute in gigabytes. (default %d)\n", params->mem_compute1_gb);
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
        } else if (arg == "--mult") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_mult = std::stoi(argv[i]);
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
        } else if (arg == "--rotmax") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_rotmax = std::stoi(argv[i]);
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
        } else if (arg == "-n" || arg == "--examples") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_examples = std::stoi(argv[i]);
        } else if (arg == "--predict") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_predict = std::stoi(argv[i]);
        } else if (arg == "--print-info-interval") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->print_info_interval = std::stoi(argv[i]);
        } else if (arg == "--print-details-interval") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->print_details_interval = std::stoi(argv[i]);
        } else if (arg == "--samples-after-nl") {
            params->samples_start_after_nl = true;
        } else if (arg == "--use-lbfgs") {
            params->use_adam = false;
        } else if (arg == "--use-adam") {
            params->use_adam = true;
        } else if (arg == "--no-flash") {
            params->use_flash = false;
        } else if (arg == "--use-flash") {
            params->use_flash = true;
        } else if (arg == "--no-scratch") {
            params->use_scratch = false;
        } else if (arg == "--use-scratch") {
            params->use_scratch = true;
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
        } else if (arg == "--cos-decay-alpha") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->cos_decay_alpha = std::stof(argv[i]);
        } else if (arg == "--lbfgs-iter") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->lbfgs_n_iter = std::stoi(argv[i]);
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
        } else if (arg == "--adam-decay") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->adam_decay = std::stof(argv[i]);
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
        } else if (arg == "--mem-compute1") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->mem_compute1_gb = std::stoi(argv[i]);
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

    return true;
}

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;



void malloc_weights(TransformerWeights* w, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    w->token_embedding_table = new float[p->vocab_size * p->dim]();//calloc(p->vocab_size * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->token_embedding_table\n",__func__,p->vocab_size , p->dim, p->vocab_size * p->dim);
    
    w->rms_att_weight = new float[p->n_layers * p->dim](); //calloc(p->n_layers * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->rms_att_weight\n",__func__,p->n_layers, p->dim, p->n_layers * p->dim);

    w->rms_ffn_weight = new float[p->n_layers * p->dim](); //calloc(p->n_layers * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->rms_ffn_weight\n",__func__,p->n_layers , p->dim, p->n_layers * p->dim);

    w->wq = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wq\n",__func__,p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->wk = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wk\n",__func__,p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->wv = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wv\n",__func__, p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->wo = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wo\n",__func__,p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->w1 = new float[p->n_layers * p->hidden_dim * p->dim](); //calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->w1\n",__func__,p->n_layers, p->hidden_dim, p->dim, p->n_layers * p->hidden_dim * p->dim);

    w->w2 = new float[p->n_layers * p->hidden_dim * p->dim](); //calloc(p->n_layers * p->dim * p->hidden_dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->w2\n",__func__,p->n_layers, p->hidden_dim, p->dim, p->n_layers * p->hidden_dim * p->dim);

    w->w3 = new float[p->n_layers * p->hidden_dim * p->dim](); //calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->w3\n",__func__,p->n_layers, p->hidden_dim, p->dim, p->n_layers * p->hidden_dim * p->dim);

    w->rms_final_weight = new float[p->dim](); //calloc(p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] float space for w->rms_final_weight\n",__func__,p->dim);

    w->freq_cis_real = new float[p->seq_len * p->dim / 2](); //calloc(p->seq_len * p->dim / 2, sizeof(float));
    printf("[%s:AK] Allocating [%d] float space for w->freq_cis_real\n",__func__,p->seq_len * p->dim / 2);

    w->freq_cis_imag = new float[p->seq_len * p->dim / 2](); //calloc(p->seq_len * p->dim / 2, sizeof(float));
    printf("[%s:AK] Allocating [%d] float space for w->freq_cis_imag\n\n",__func__,p->seq_len * p->dim / 2);

    // ensure all mallocs went fine
    // if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight 
    //  || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 || 
    //     !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag) {
    //     printf("malloc failed!\n");
    //     exit(1);
    // }
}

int checkpoint_init_weights(TransformerWeights *w, Config* p, FILE* f) {
    if (fread(w->token_embedding_table, sizeof(float), p->vocab_size * p->dim, f) != static_cast<size_t>(p->vocab_size * p->dim)) return 1;
    if (fread(w->rms_att_weight, sizeof(float), p->n_layers * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim)) return 1;
    if (fread(w->wq, sizeof(float), p->n_layers * p->dim * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->dim)) return 1;
    if (fread(w->wk, sizeof(float), p->n_layers * p->dim * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->dim)) return 1;
    if (fread(w->wv, sizeof(float), p->n_layers * p->dim * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->dim)) return 1;
    if (fread(w->wo, sizeof(float), p->n_layers * p->dim * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->dim)) return 1;
    if (fread(w->rms_ffn_weight, sizeof(float), p->n_layers * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim)) return 1;
    if (fread(w->w1, sizeof(float), p->n_layers * p->dim * p->hidden_dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->hidden_dim)) return 1;
    if (fread(w->w2, sizeof(float), p->n_layers * p->hidden_dim * p->dim, f) != static_cast<size_t>(p->n_layers * p->hidden_dim * p->dim)) return 1;
    if (fread(w->w3, sizeof(float), p->n_layers * p->dim * p->hidden_dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->hidden_dim)) return 1;
    if (fread(w->rms_final_weight, sizeof(float), p->dim, f) != static_cast<size_t>(p->dim)) return 1;
    int head_size = p->dim / p->n_heads;
    if (fread(w->freq_cis_real, sizeof(float), p->seq_len * head_size / 2, f) != static_cast<size_t>(p->seq_len * head_size / 2)) return 1;
    if (fread(w->freq_cis_imag, sizeof(float), p->seq_len * head_size / 2, f) != static_cast<size_t>(p->seq_len * head_size / 2)) return 1;
    return 0;
}

void free_weights(TransformerWeights* w) {
    free(w->token_embedding_table);
    free(w->rms_att_weight);
    free(w->rms_ffn_weight);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->w1);
    free(w->w2);
    free(w->w3);
    free(w->rms_final_weight);
    free(w->freq_cis_real);
    free(w->freq_cis_imag);
}


void randomize_model(struct my_llama_model * model, int seed, float mean, float std, float min, float max) {
    const auto & hparams = model->hparams;

    const uint32_t n_layer = hparams.n_layer;

    struct random_normal_distribution rnd;
    init_random_normal_distribution(&rnd, seed, mean, std, min, max);

    randomize_tensor_normal(model->tok_embeddings, &rnd);
    randomize_tensor_normal(model->norm,           &rnd);
    randomize_tensor_normal(model->output,         &rnd);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];
        randomize_tensor_normal(layer.attention_norm, &rnd);

        randomize_tensor_normal(layer.wq, &rnd);
        randomize_tensor_normal(layer.wk, &rnd);
        randomize_tensor_normal(layer.wv, &rnd);
        randomize_tensor_normal(layer.wo, &rnd);

        randomize_tensor_normal(layer.ffn_norm, &rnd);

        randomize_tensor_normal(layer.w1, &rnd);
        randomize_tensor_normal(layer.w2, &rnd);
        randomize_tensor_normal(layer.w3, &rnd);
    }
}

int main(int argc, char ** argv) {
    Config config;
    TransformerWeights weights;
    {
        FILE *file = fopen("/Users/aniket/Projects/karpathy/llama2.c/out/model.bin", "rb");
        if (!file) {
            printf("Unable to open the checkpoint file %s!\n", "/Users/aniket/Projects/karpathy/llama2.c/out/model.bin");
            return 1;
        }
        else{
            printf("model file opened for reading...\n");
        }
        // read in the config header
        if(fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
        printf("config file read..\n");

        // read in the Transformer weights
        malloc_weights(&weights, &config);
        printf("reading the opened model file...\n");
        if(checkpoint_init_weights(&weights, &config, file)) { return 1; }

        fclose(file);

    }
    ////////////// Loads default train parameters ///////////////////////////
    struct train_params params = get_default_train_params();
    printf("params.n_ctx %d\n", params.n_ctx);
    printf("params.n_embd %d\n", params.n_embd);
    printf("params.fn_vocab_model %s\n", params.fn_vocab_model);
    
    if (!train_params_parse(argc, argv, &params)) {
        return 1;
    }

    // Seed not needed here.
    // if (params.seed == LLAMA_DEFAULT_SEED) {
    //     params.seed = time(NULL);
    // }
    // printf("[%s]: seed: %u\n", __func__, params.seed);
    // srand(params.seed);
    ////////////////////////////////////////////////////////////////////////////////////

    struct llama_context_params llama_params = llama_context_default_params();
    llama_params.vocab_only = true;

    struct llama_model * lmodel = llama_load_model_from_file(params.fn_vocab_model, llama_params);
    struct llama_context * lctx = llama_new_context_with_model(lmodel, llama_params);

    struct llama_vocab vocab;
    {
        std::vector<const char *> strings;
        std::vector<float> scores;
        int n_vocab = llama_n_vocab(lctx);
        printf("nvocab = %d\n", n_vocab);
        strings.resize(n_vocab, NULL);
        scores.resize(n_vocab, 0);
        n_vocab = llama_get_vocab(lctx, strings.data(), scores.data(), n_vocab);
        GGML_ASSERT(n_vocab == llama_n_vocab(lctx));
        vocab.id_to_token.resize(n_vocab);
        for (int i=0; i<n_vocab; ++i) {
            std::string tok   = std::string(strings[i]);
            float       score = scores[i];
            // printf("%s - %f\n", tok.c_str(), score);
            vocab.id_to_token[i].tok   = tok;
            vocab.id_to_token[i].score = score;
            vocab.token_to_id.emplace(tok, i);
        }
    }

    printf("%s: tokenize training data\n", __func__);
    std::vector<llama_token> train_tokens;
    if (tokenize_file(lctx, params.fn_train_data, train_tokens) < 0) {
        fprintf(stderr, "%s: failed to tokenize file '%s'\n", __func__, params.fn_train_data);
    }
    printf("%s: number of training tokens: %d\n", __func__, (int) train_tokens.size());

    struct my_llama_model model;
    
    model.hparams.n_vocab = config.vocab_size; //llama_n_vocab(lctx);
    model.hparams.n_ctx   = params.n_ctx;
    model.hparams.n_embd  = config.dim; //params.n_embd;
    model.hparams.n_mult  = params.n_mult; 
    model.hparams.n_head  = config.n_heads; //params.n_head;
    model.hparams.n_layer = config.n_layers; //params.n_layer;
    model.hparams.n_rot   = std::min((uint32_t)params.n_rotmax, model.hparams.n_embd / model.hparams.n_head);

    print_params(&model.hparams);
    struct ggml_init_params lcparams;
    lcparams.mem_size   = 1024ll*1024ll*1024ll*((size_t) params.mem_model_gb);
    lcparams.mem_buffer = NULL;
    lcparams.no_alloc   = false;

    model.ctx = ggml_init(lcparams);

    init_model(&model);
    // randomize_model(&model, params.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    save_as_llama_model(&vocab, &model, &weights, "ak_model.bin");

    // llama_free(lctx);
    llama_free_model(lmodel);
    ggml_free(model.ctx);
    // free(&weights);
    return 0;
}
