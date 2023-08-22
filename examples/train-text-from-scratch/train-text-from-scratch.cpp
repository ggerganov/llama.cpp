#include "ggml.h"
#include "common.h"
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

static const float rms_norm_eps = 1e-5f;

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
    using ttype = llama_token_type;

    struct token_data {
        token text;
        float score;
        ttype type;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data> id_to_token;
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

    model->tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    model->norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    model->output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);

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

struct ggml_tensor * forward(
        struct my_llama_model    * model,
        struct my_llama_kv_cache * cache,
        struct ggml_context   * ctx0,
        struct ggml_cgraph    * gf,
        struct ggml_tensor    * tokens_input,
        const  int              n_tokens,
        const  int              n_past) {

    const int N = n_tokens;

    struct my_llama_kv_cache& kv_self = *cache;
    const auto & hparams = model->hparams;
    const int n_ctx   = hparams.n_ctx;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;

    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(tokens->data, tokens_input->data, N*ggml_element_size(tokens));

    struct ggml_tensor * kc = kv_self.k;
    struct ggml_tensor * vc = kv_self.v;

    // inpL shape [n_embd,N,1,1]
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model->tok_embeddings, tokens);
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // lctx.use_buf(ctx0, 0);

        // norm
        {
            // cur shape [n_embd,N,1,1]
            cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].attention_norm, cur),
                        cur);
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            // wq   shape [n_embd, n_embd, 1, 1]
            // wk   shape [n_embd, n_embd, 1, 1]
            // Qcur shape [n_embd/n_head, n_head, N, 1]
            // Kcur shape [n_embd/n_head, n_head, N, 1]
            struct ggml_tensor * Qcur = ggml_rope_inplace(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wq, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0, 0);
            struct ggml_tensor * Kcur = ggml_rope_inplace(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wk, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0, 0);

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix
                // wv   shape [n_embd, n_embd, 1, 1]
                // Vcur shape [n_embd, N, 1, 1]
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wv, cur), n_embd, N)));

                // kv_self.k shape [n_embd * n_ctx * n_layer, 1]
                // kv_self.v shape [n_embd * n_ctx * n_layer, 1]
                // k         shape [n_embd * N, 1]   == kv_self.k[:,n_past:n_past+N,il,0]
                // v         shape [N, n_embd, 1, 1] == kv_self.v[:,n_past:n_past+N,il,0]

                /* {
                    struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                    struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd,
                            (   n_ctx)*ggml_element_size(kv_self.v),
                            (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));

                    // important: storing RoPE-ed version of K in the KV cache!
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
                } //*/

                kc = ggml_set_1d_inplace(ctx0, kc, ggml_reshape_1d(ctx0, Kcur, n_embd*N), (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                vc = ggml_set_2d_inplace(ctx0, vc, Vcur, (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));
            }

            // Qcur shape [n_embd/n_head, n_head, N, 1]
            // Q shape    [n_embd/n_head, N, n_head, 1]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);

            // kv_self.k shape [n_embd * n_ctx * n_layer, 1]
            // K shape [n_embd/n_head, n_past + N, n_head, 1]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, kc, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kc)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            // KQ shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head)));

            // KQ_masked = mask_past(KQ_scaled)
            // KQ_masked shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // KQ_soft_max shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // split cached V into n_head heads
            //// V shape [n_past + N, n_embd/n_head, n_head, 1]
            // V shape [n_past + N, n_embd/n_head, n_head, 1] == kv_self.v[:,:(n_past+N),il,1]
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, vc,
                        n_past + N, n_embd/n_head, n_head,
                        n_ctx*ggml_element_size(vc),
                        n_ctx*ggml_element_size(vc)*n_embd/n_head,
                        il*n_ctx*ggml_element_size(vc)*n_embd);

            // KQV shape [n_embd/n_head, N, n_head, 1]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // KQV_merged shape [n_embd/n_head, n_head, N, 1]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            // KQV_merged shape

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // cur shape [n_embd,N,1,1]
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, KQV_merged), n_embd, N);
            // cur = ggml_cpy(ctx0,
            //         KQV_merged,
            //         ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            // cur shape [n_embd,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].wo,
                    cur);
        }

        // lctx.use_buf(ctx0, 1);

        // inpFF shape [n_embd,N,1,1]
        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                // cur shape [n_embd,N,1,1]
                cur = ggml_rms_norm(ctx0, inpFF, rms_norm_eps);

                // cur = ffn_norm*cur
                // cur shape [n_embd,N,1,1]
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].ffn_norm, cur),
                        cur);
            }

            // tmp shape [n_ff,N,1,1]
            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model->layers[il].w3,
                    cur);

            // cur shape [n_ff,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w1,
                    cur);

            // SILU activation
            // cur shape [n_ff,N,1,1]
            cur = ggml_silu(ctx0, cur);

            // cur shape [n_ff,N,1,1]
            cur = ggml_mul(ctx0, cur, tmp);

            // cur shape [n_embd,N,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w2,
                    cur);
        }

        // cur shape [n_embd,N,1,1]
        cur = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        // inpL shape [n_embd,N,1,1]
        inpL = cur;
    }

    // norm
    {

        // inpL shape [n_embd,N,1,1]
        inpL = ggml_rms_norm(ctx0, inpL, rms_norm_eps);

        // inpL = norm*inpL
        // inpL shape [n_embd,N,1,1]
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model->norm, inpL),
                    inpL);

        //embeddings = inpL;
    }

    // lm_head
    // inpL shape [n_vocab,N,1,1]
    inpL = ggml_mul_mat(ctx0, model->output, inpL);

    // run the computation
    ggml_build_forward_expand(gf, inpL);

    return inpL;
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

struct ggml_tensor * forward_batch(
        struct my_llama_model    * model,
        struct my_llama_kv_cache * cache,
        struct ggml_context   * ctx0,
        struct ggml_cgraph    * gf,
        struct ggml_tensor    * tokens_input,
        const  int              n_tokens,
        const  int              n_past,
        const  int              n_batch) {

    const int N = n_tokens;

    struct my_llama_kv_cache& kv_self = *cache;
    const auto & hparams = model->hparams;
    const int n_ctx   = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;
    const int n_ff    = get_n_ff(&hparams);

    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N*n_batch);
    memcpy(tokens->data, tokens_input->data, ggml_element_size(tokens)*N*n_batch);

    struct ggml_tensor * kc = kv_self.k;
    struct ggml_tensor * vc = kv_self.v;

    // inpL shape [n_embd,N*n_batch,1]
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model->tok_embeddings, tokens);
    assert_shape_2d(inpL, n_embd, N*n_batch);
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // lctx.use_buf(ctx0, 0);

        // norm
        {
            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
            assert_shape_2d(cur, n_embd, N*n_batch);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].attention_norm, cur),
                        cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            // wq   shape [n_embd, n_embd, 1, 1]
            // wk   shape [n_embd, n_embd, 1, 1]
            // Qcur shape [n_embd/n_head, n_head, N, n_batch]
            // Kcur shape [n_embd/n_head, n_head, N, n_batch]
            struct ggml_tensor * Qcur = ggml_rope_inplace(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wq, cur), n_embd/n_head, n_head, N, n_batch), n_past, n_rot, 0, 0);
            struct ggml_tensor * Kcur = ggml_rope_inplace(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wk, cur), n_embd/n_head, n_head, N, n_batch), n_past, n_rot, 0, 0);
            assert_shape_4d(Qcur, n_embd/n_head, n_head, N, n_batch);
            assert_shape_4d(Kcur, n_embd/n_head, n_head, N, n_batch);

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix
                // wv   shape [n_embd, n_embd, 1, 1]
                // Vcur shape [N, n_embd, n_batch, 1]
                struct ggml_tensor * Vcur = ggml_cont(ctx0,
                    ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_mul_mat(ctx0,
                                model->layers[il].wv,
                                cur),
                        n_embd, N, n_batch),
                        1, 0, 2, 3));
                assert_shape_3d(Vcur, N, n_embd, n_batch);

                // kv_self.k shape [n_embd * n_ctx * n_batch * n_layer]
                // kv_self.v shape [n_ctx * n_embd * n_batch * n_layer]
                // k         shape [n_embd * N, n_batch]   == kv_self.k[:,n_past:n_past+N,:,il]
                // v         shape [N, n_embd, n_batch, 1] == kv_self.v[:,n_past:n_past+N,:,il]

                /* {
                    struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                    struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd,
                            (   n_ctx)*ggml_element_size(kv_self.v),
                            (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));

                    // important: storing RoPE-ed version of K in the KV cache!
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
                } //*/

                kc = ggml_set_2d_inplace(ctx0, kc,
                        ggml_reshape_2d(ctx0, Kcur, n_embd*N, n_batch),
                        ggml_element_size(kc)*n_embd*n_ctx,
                        (ggml_element_size(kc)*n_embd)*(il*n_batch*n_ctx + n_past));
                vc = ggml_set_2d_inplace(ctx0, vc,
                        ggml_reshape_2d(ctx0, Vcur, N*n_embd, n_batch),
                        ggml_element_size(vc)*n_ctx*n_embd,
                        ggml_element_size(vc)*(n_past + il*n_embd*n_batch*n_ctx));

                assert_shape_1d(kc, n_embd * n_ctx * n_batch * n_layer);
                assert_shape_1d(vc, n_embd * n_ctx * n_batch * n_layer);
            }

            // Qcur shape [n_embd/n_head, n_head, N, n_batch]
            // Q shape    [n_embd/n_head, N, n_head, n_batch]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);
            assert_shape_4d(Q, n_embd/n_head, N, n_head, n_batch);

            // kv_self.k shape [n_embd * n_ctx * n_batch * n_layer]
            // K shape [n_embd/n_head, n_past + N, n_head, n_batch]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0,
                            ggml_view_3d(ctx0,
                                kc,
                                n_embd,
                                (n_past + N),
                                n_batch,
                                n_embd*ggml_element_size(kc),
                                n_ctx*n_embd*ggml_element_size(kc),
                                il*n_batch*n_ctx*n_embd*ggml_element_size(kc)),
                            n_embd/n_head, n_head, n_past + N, n_batch),
                        0, 2, 1, 3);
            assert_shape_4d(K, n_embd/n_head, n_past + N, n_head, n_batch);

            // K * Q
            // KQ shape [n_past + N, N, n_head, n_batch]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            assert_shape_4d(KQ, n_past + N, N, n_head, n_batch);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // KQ_scaled shape [n_past + N, N, n_head, n_batch]
            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head)));
            assert_shape_4d(KQ_scaled, n_past + N, N, n_head, n_batch);

            // KQ_masked = mask_past(KQ_scaled)
            // KQ_masked shape [n_past + N, N, n_head, n_batch]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
            assert_shape_4d(KQ_masked, n_past + N, N, n_head, n_batch);

            // KQ = soft_max(KQ_masked)
            // KQ_soft_max shape [n_past + N, N, n_head, n_batch]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
            assert_shape_4d(KQ_soft_max, n_past + N, N, n_head, n_batch);

            // split cached V into n_head heads
            // kv_self.v shape [n_ctx * n_embd * n_batch * n_layer]
            // V shape [n_past + N, n_embd/n_head, n_head, n_batch] == kv_self.v[:(n_past+N),:,:,il]
            struct ggml_tensor * V =
                ggml_view_4d(ctx0, vc,
                        n_past + N, n_embd/n_head, n_head, n_batch,
                        ggml_element_size(vc)*n_ctx,
                        ggml_element_size(vc)*n_ctx*n_embd/n_head,
                        ggml_element_size(vc)*n_ctx*n_embd,
                        il*n_batch*n_ctx*n_embd*ggml_element_size(vc));
            assert_shape_4d(V, n_past + N, n_embd/n_head, n_head, n_batch);

            // KQV shape [n_embd/n_head, N, n_head, n_batch]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            assert_shape_4d(KQV, n_embd/n_head, N, n_head, n_batch);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // KQV_merged shape [n_embd/n_head, n_head, N, n_batch]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            assert_shape_4d(KQV_merged, n_embd/n_head, n_head, N, n_batch);
            // KQV_merged shape

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, KQV_merged), n_embd, N*n_batch);
            assert_shape_2d(cur, n_embd, N*n_batch);
            // cur = ggml_cpy(ctx0,
            //         KQV_merged,
            //         ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].wo,
                    cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // lctx.use_buf(ctx0, 1);

        // inpFF shape [n_embd,N*n_batch,1,1]
        struct ggml_tensor * inpFF = ggml_add_inplace(ctx0, cur, inpSA);
        assert_shape_2d(inpFF, n_embd, N*n_batch);

        // feed-forward network
        {
            // norm
            {
                // cur shape [n_embd,N*n_batch,1,1]
                cur = ggml_rms_norm(ctx0, inpFF, rms_norm_eps);
                assert_shape_2d(cur, n_embd, N*n_batch);

                // cur = ffn_norm*cur
                // cur shape [n_embd,N*n_batch,1,1]
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].ffn_norm, cur),
                        cur);
                assert_shape_2d(cur, n_embd, N*n_batch);
            }

            // tmp shape [n_ff,N*n_batch,1,1]
            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model->layers[il].w3,
                    cur);
            assert_shape_2d(tmp, n_ff, N*n_batch);

            // cur shape [n_ff,N*n_batch,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w1,
                    cur);
            assert_shape_2d(cur, n_ff, N*n_batch);

            // SILU activation
            // cur shape [n_ff,N*n_batch,1,1]
            cur = ggml_silu(ctx0, cur);
            assert_shape_2d(cur, n_ff, N*n_batch);

            // cur shape [n_ff,N*n_batch,1,1]
            cur = ggml_mul(ctx0, cur, tmp);
            assert_shape_2d(cur, n_ff, N*n_batch);

            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w2,
                    cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // cur shape [n_embd,N*n_batch,1,1]
        cur = ggml_add_inplace(ctx0, cur, inpFF);
        assert_shape_2d(cur, n_embd, N*n_batch);

        // input for next layer
        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = cur;
        assert_shape_2d(inpL, n_embd, N*n_batch);
    }

    // norm
    {

        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
        assert_shape_2d(inpL, n_embd, N*n_batch);

        // inpL = norm*inpL
        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model->norm, inpL),
                    inpL);

        assert_shape_2d(inpL, n_embd, N*n_batch);

        //embeddings = inpL;
    }

    // lm_head
    // inpL shape [n_vocab,N*n_batch,1,1]
    inpL = ggml_mul_mat(ctx0, model->output, inpL);
    assert_shape_2d(inpL, n_vocab, N*n_batch);

    {
        // inpL shape [n_vocab,N,n_batch,1]
        inpL = ggml_reshape_3d(ctx0,
                        inpL,
                        n_vocab, N, n_batch);
        assert_shape_3d(inpL, n_vocab, N, n_batch);
    }

    // run the computation
    ggml_build_forward_expand(gf, inpL);

    return inpL;
}

struct ggml_tensor * forward_batch_wo_cache(
        struct my_llama_model * model,
        struct ggml_context   * ctx0,
        struct ggml_cgraph    * gf,
        struct ggml_tensor    * tokens_input,
        const  int              n_tokens,
        const  int              n_batch) {

    const int n_past = 0;
    const int N = n_tokens;

    const auto & hparams = model->hparams;
    //const int n_ctx   = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;
    const int n_ff    = get_n_ff(&hparams);

    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N*n_batch);
    memcpy(tokens->data, tokens_input->data, ggml_element_size(tokens)*N*n_batch);

    // inpL shape [n_embd,N*n_batch,1]
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model->tok_embeddings, tokens);
    assert_shape_2d(inpL, n_embd, N*n_batch);
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // lctx.use_buf(ctx0, 0);

        // norm
        {
            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
            assert_shape_2d(cur, n_embd, N*n_batch);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].attention_norm, cur),
                        cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            // wq   shape [n_embd, n_embd, 1, 1]
            // wk   shape [n_embd, n_embd, 1, 1]
            // Qcur shape [n_embd/n_head, n_head, N, n_batch]
            // Kcur shape [n_embd/n_head, n_head, N, n_batch]
            struct ggml_tensor * Qcur = ggml_rope_inplace(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wq, cur), n_embd/n_head, n_head, N, n_batch), n_past, n_rot, 0, 0);
            struct ggml_tensor * Kcur = ggml_rope_inplace(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wk, cur), n_embd/n_head, n_head, N, n_batch), n_past, n_rot, 0, 0);
            assert_shape_4d(Qcur, n_embd/n_head, n_head, N, n_batch);
            assert_shape_4d(Kcur, n_embd/n_head, n_head, N, n_batch);

            // Vcur shape [N, n_batch, n_embd/n_head, n_head]
            struct ggml_tensor * Vcur = ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, cur, model->layers[il].wv), N, n_batch, n_embd/n_head, n_head);
            assert_shape_4d(Vcur, N, n_batch, n_embd/n_head, n_head);

            // Qcur shape [n_embd/n_head, n_head, N, n_batch]
            // Q shape    [n_embd/n_head, N, n_head, n_batch]
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);
            assert_shape_4d(Q, n_embd/n_head, N, n_head, n_batch);

            // kv_self.k shape [n_embd * n_ctx * n_batch * n_layer]
            // K shape [n_embd/n_head, N, n_head, n_batch]
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        Kcur,
                        0, 2, 1, 3);
            assert_shape_4d(K, n_embd/n_head, N, n_head, n_batch);

            // K * Q
            // KQ shape [N, N, n_head, n_batch]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            assert_shape_4d(KQ, N, N, n_head, n_batch);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // KQ_scaled shape [N, N, n_head, n_batch]
            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head)));
            assert_shape_4d(KQ_scaled, N, N, n_head, n_batch);

            // KQ_masked = mask_past(KQ_scaled)
            // KQ_masked shape [N, N, n_head, n_batch]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
            assert_shape_4d(KQ_masked, N, N, n_head, n_batch);

            // KQ = soft_max(KQ_masked)
            // KQ_soft_max shape [N, N, n_head, n_batch]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
            assert_shape_4d(KQ_soft_max, N, N, n_head, n_batch);

            // Vcur shape [N, n_batch, n_embd/n_head, n_head]
            // V shape    [N, n_embd/n_head, n_head, n_batch]
            struct ggml_tensor * V =
                ggml_permute(ctx0,
                    Vcur,
                    0, 3, 1, 2);
            assert_shape_4d(V, N, n_embd/n_head, n_head, n_batch);

            // KQV shape [n_embd/n_head, N, n_head, n_batch]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            assert_shape_4d(KQV, n_embd/n_head, N, n_head, n_batch);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // KQV_merged shape [n_embd/n_head, n_head, N, n_batch]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            assert_shape_4d(KQV_merged, n_embd/n_head, n_head, N, n_batch);
            // KQV_merged shape

            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, KQV_merged), n_embd, N*n_batch);
            assert_shape_2d(cur, n_embd, N*n_batch);

            // projection (no bias)
            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].wo,
                    cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // lctx.use_buf(ctx0, 1);

        // inpFF shape [n_embd,N*n_batch,1,1]
        struct ggml_tensor * inpFF = ggml_add_inplace(ctx0, cur, inpSA);
        assert_shape_2d(inpFF, n_embd, N*n_batch);

        // feed-forward network
        {
            // norm
            {
                // cur shape [n_embd,N*n_batch,1,1]
                cur = ggml_rms_norm(ctx0, inpFF, rms_norm_eps);
                assert_shape_2d(cur, n_embd, N*n_batch);

                // cur = ffn_norm*cur
                // cur shape [n_embd,N*n_batch,1,1]
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].ffn_norm, cur),
                        cur);
                assert_shape_2d(cur, n_embd, N*n_batch);
            }

            // tmp shape [n_ff,N*n_batch,1,1]
            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model->layers[il].w3,
                    cur);
            assert_shape_2d(tmp, n_ff, N*n_batch);

            // cur shape [n_ff,N*n_batch,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w1,
                    cur);
            assert_shape_2d(cur, n_ff, N*n_batch);

            // SILU activation
            // cur shape [n_ff,N*n_batch,1,1]
            cur = ggml_silu(ctx0, cur);
            assert_shape_2d(cur, n_ff, N*n_batch);

            // cur shape [n_ff,N*n_batch,1,1]
            cur = ggml_mul(ctx0, cur, tmp);
            assert_shape_2d(cur, n_ff, N*n_batch);

            // cur shape [n_embd,N*n_batch,1,1]
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w2,
                    cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // cur shape [n_embd,N*n_batch,1,1]
        cur = ggml_add_inplace(ctx0, cur, inpFF);
        assert_shape_2d(cur, n_embd, N*n_batch);

        // input for next layer
        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = cur;
        assert_shape_2d(inpL, n_embd, N*n_batch);
    }

    // norm
    {

        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
        assert_shape_2d(inpL, n_embd, N*n_batch);

        // inpL = norm*inpL
        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model->norm, inpL),
                    inpL);

        assert_shape_2d(inpL, n_embd, N*n_batch);

        //embeddings = inpL;
    }

    // lm_head
    // inpL shape [n_vocab,N*n_batch,1,1]
    inpL = ggml_mul_mat(ctx0, model->output, inpL);
    assert_shape_2d(inpL, n_vocab, N*n_batch);

    {
        // inpL shape [n_vocab,N,n_batch,1]
        inpL = ggml_reshape_3d(ctx0,
                        inpL,
                        n_vocab, N, n_batch);
        assert_shape_3d(inpL, n_vocab, N, n_batch);
    }

    // run the computation
    ggml_build_forward_expand(gf, inpL);

    return inpL;
}

struct ggml_tensor * forward_batch_wo_cache_flash_attn(
        struct my_llama_model * model,
        struct ggml_context   * ctx0,
        struct ggml_cgraph    * gf,
        struct ggml_tensor    * tokens_input,
        const  int              n_tokens,
        const  int              n_batch) {

    const int n_past = 0;
    const int N = n_tokens;

    const auto & hparams = model->hparams;
    //const int n_ctx   = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;
    const int n_ff    = get_n_ff(&hparams);

    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N*n_batch);
    memcpy(tokens->data, tokens_input->data, ggml_element_size(tokens)*N*n_batch);

    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model->tok_embeddings, tokens);
    assert_shape_2d(inpL, n_embd, N*n_batch);
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // norm
        {
            cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
            assert_shape_2d(cur, n_embd, N*n_batch);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].attention_norm, cur),
                        cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            // wq   shape [n_embd, n_embd, 1, 1]
            // wk   shape [n_embd, n_embd, 1, 1]
            struct ggml_tensor * Qcur = ggml_rope_inplace(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wq, cur), n_embd/n_head, n_head, N, n_batch), n_past, n_rot, 0, 0);
            struct ggml_tensor * Kcur = ggml_rope_inplace(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wk, cur), n_embd/n_head, n_head, N, n_batch), n_past, n_rot, 0, 0);
            assert_shape_4d(Qcur, n_embd/n_head, n_head, N, n_batch);
            assert_shape_4d(Kcur, n_embd/n_head, n_head, N, n_batch);

            struct ggml_tensor * Vcur = ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, cur, model->layers[il].wv), N, n_batch, n_embd/n_head, n_head);
            assert_shape_4d(Vcur, N, n_batch, n_embd/n_head, n_head);

            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);
            assert_shape_4d(Q, n_embd/n_head, N, n_head, n_batch);

            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        Kcur,
                        0, 2, 1, 3);
            assert_shape_4d(K, n_embd/n_head, N, n_head, n_batch);

            struct ggml_tensor * V =
                ggml_permute(ctx0,
                    Vcur,
                    0, 3, 1, 2);
            assert_shape_4d(V, N, n_embd/n_head, n_head, n_batch);

            bool masked = true;
            struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, masked);
            assert_shape_4d(KQV, n_embd/n_head, N, n_head, n_batch);

            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            assert_shape_4d(KQV_merged, n_embd/n_head, n_head, N, n_batch);
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, KQV_merged), n_embd, N*n_batch);
            assert_shape_2d(cur, n_embd, N*n_batch);

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].wo,
                    cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        struct ggml_tensor * inpFF = ggml_add_inplace(ctx0, cur, inpSA);
        assert_shape_2d(inpFF, n_embd, N*n_batch);

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF, rms_norm_eps);
                assert_shape_2d(cur, n_embd, N*n_batch);

                // cur = ffn_norm*cur
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].ffn_norm, cur),
                        cur);
                assert_shape_2d(cur, n_embd, N*n_batch);
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model->layers[il].w3,
                    cur);
            assert_shape_2d(tmp, n_ff, N*n_batch);

            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w1,
                    cur);
            assert_shape_2d(cur, n_ff, N*n_batch);

            // SILU activation
            cur = ggml_silu(ctx0, cur);
            assert_shape_2d(cur, n_ff, N*n_batch);

            cur = ggml_mul(ctx0, cur, tmp);
            assert_shape_2d(cur, n_ff, N*n_batch);

            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w2,
                    cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        cur = ggml_add_inplace(ctx0, cur, inpFF);
        assert_shape_2d(cur, n_embd, N*n_batch);

        // input for next layer
        inpL = cur;
        assert_shape_2d(inpL, n_embd, N*n_batch);
    }

    // norm
    {

        inpL = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
        assert_shape_2d(inpL, n_embd, N*n_batch);

        // inpL = norm*inpL
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model->norm, inpL),
                    inpL);

        assert_shape_2d(inpL, n_embd, N*n_batch);
    }

    // lm_head
    inpL = ggml_mul_mat(ctx0, model->output, inpL);
    assert_shape_2d(inpL, n_vocab, N*n_batch);

    {
        inpL = ggml_reshape_3d(ctx0,
                        inpL,
                        n_vocab, N, n_batch);
        assert_shape_3d(inpL, n_vocab, N, n_batch);
    }

    // run the computation
    ggml_build_forward_expand(gf, inpL);

    return inpL;
}

// expand the graph nodes without creating leafs.
struct ggml_tensor * expand(struct ggml_cgraph * g, struct ggml_tensor * t) {
    // check if already visited
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i] == t) {
            return t;
        }
    }

    for (int i = 0; i < g->n_leafs; i++) {
        if (g->leafs[i] == t) {
            return t;
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i]) {
            expand(g, t->src[i]);
        }
    }

    GGML_ASSERT(g->n_nodes < GGML_MAX_NODES);

    if (strlen(t->name) == 0) {
        snprintf(t->name, sizeof(t->name), "node_%d", g->n_nodes);
    }

    g->nodes[g->n_nodes] = t;
    g->grads[g->n_nodes] = t->grad;
    g->n_nodes++;
    return t;
}

void graph_set_leafs_grads(struct ggml_cgraph * g) {
    // moves leaf nodes to g->leafs.
    // i.e. g->n_nodes might change.
    int n_nodes = 0;
    for (int i = 0; i < g->n_nodes; ++i) {
        struct ggml_tensor * node = g->nodes[i];
        const bool is_leaf = node->op == GGML_OP_NONE && node->grad == NULL;
        if (is_leaf) {
            GGML_ASSERT(g->n_leafs < GGML_MAX_NODES);

            if (strlen(node->name) == 0) {
                snprintf(node->name, sizeof(node->name), "leaf_%d", g->n_leafs);
            }

            g->leafs[g->n_leafs] = node;
            g->n_leafs++;
        } else {
            GGML_ASSERT(n_nodes < GGML_MAX_NODES);

            if (strlen(node->name) == 0) {
                snprintf(node->name, sizeof(node->name), "node_%d", n_nodes);
            }

            g->nodes[n_nodes] = node;
            g->grads[n_nodes] = node->grad;
            n_nodes++;
        }
    }
    for (int i=n_nodes; i < g->n_nodes; ++i) {
        g->nodes[n_nodes] = NULL;
        g->grads[n_nodes] = NULL;
    }
    g->n_nodes = n_nodes;
}

struct ggml_tensor * forward_batch_wo_cache_flash_attn_train(
        struct my_llama_model * model,
        struct ggml_context   * ctx0,
        struct ggml_cgraph    * gf,
        struct ggml_cgraph    * gb,
        struct ggml_tensor  * * logits,
        struct ggml_tensor    * tokens_input,
        struct ggml_tensor    * targets,
        void                  * compute_buf_0,
        void                  * compute_buf_1,
        size_t                  size_buf_0,
        size_t                  size_buf_1,
        const  int              n_tokens,
        const  int              n_batch) {

    ggml_set_scratch(ctx0, { 0, 0, nullptr, });

    const int n_past = 0;
    const int N = n_tokens;

    gf->n_nodes = 0;
    gf->n_leafs = 0;
    gf->perf_runs = 0;
    gf->perf_cycles = 0;
    gf->perf_time_us = 0;

    const auto & hparams = model->hparams;
    const int n_ctx      = hparams.n_ctx;
    const int n_vocab    = hparams.n_vocab;
    const int n_embd     = hparams.n_embd;
    const int n_layer    = hparams.n_layer;
    const int n_head     = hparams.n_head;
    const int n_rot      = hparams.n_rot;
    const int n_ff       = get_n_ff(&hparams);
    const int rope_mode  = 0;

    int last_buf = -1;
    size_t buf_offs[2] = { 0, 0 };
    size_t buf_size[2] = { size_buf_0,
                           size_buf_1 };
    void * buf_data[2] = { compute_buf_0,
                           compute_buf_1 };
    auto use_buf = [ctx0, &last_buf, &buf_offs, &buf_size, &buf_data] (int buf) {
        size_t last_offs = 0;
        last_offs = ggml_set_scratch(ctx0, { 0, 0, nullptr, });
        if (last_buf >= 0) {
            buf_offs[last_buf] = last_offs;
        }
        if (buf >= 0) {
            size_t offs = buf_offs[buf];
            size_t size = buf_size[buf];
            void * data = buf_data[buf];
            ggml_set_scratch(ctx0, { offs, size, data, });
        }
        last_buf = buf;
    };

    bool track_max_mem = false;
    size_t buf_maxs[2] = { 0, 0 };

    auto clr_buf = [ctx0, &last_buf, &buf_offs, &buf_size, &buf_data, &buf_maxs, track_max_mem] (int buf) {
        if (buf < 0) return;
        if (track_max_mem) {
            size_t last_offs = 0;
            last_offs = ggml_set_scratch(ctx0, { 0, 0, nullptr, });
            if (last_buf >= 0) {
                buf_offs[last_buf] = last_offs;
                buf_maxs[last_buf] = std::max(buf_maxs[last_buf], buf_offs[last_buf]);
            }
        }
        buf_offs[buf] = 0;
        if (track_max_mem && last_buf >= 0) {
            size_t offs = buf_offs[last_buf];
            size_t size = buf_size[last_buf];
            void * data = buf_data[last_buf];
            ggml_set_scratch(ctx0, { offs, size, data, });
        }
    };


    auto view__q = [ctx0, n_embd, n_head, N, n_batch] (struct ggml_tensor * t) -> struct ggml_tensor * {
        int64_t ne0 = n_embd/n_head;
        int64_t ne1 = N;
        int64_t ne2 = n_head;
        int64_t ne3 = n_batch;
        size_t  nb0 = ggml_element_size(t);
        size_t  nb1 = nb0*ne0;
        size_t  nb2 = nb1*ne1;
        size_t  nb3 = nb2*ne2;
        size_t offset = 0;
        return ggml_view_4d(ctx0, t, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
    };

    auto view__k = [ctx0, n_embd, n_head, N, n_batch] (struct ggml_tensor * t) -> struct ggml_tensor * {
        int64_t ne0 = n_embd/n_head;
        int64_t ne1 = N;
        int64_t ne2 = n_head;
        int64_t ne3 = n_batch;
        size_t  nb0 = ggml_element_size(t);
        size_t  nb1 = nb0*ne0;
        size_t  nb2 = nb1*ne1;
        size_t  nb3 = nb2*ne2;
        size_t offset = nb3*ne3;
        return ggml_view_4d(ctx0, t, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
    };

    auto view__v = [ctx0, n_embd, n_head, N, n_batch] (struct ggml_tensor * t) -> struct ggml_tensor * {
        int64_t ne0 = N;
        int64_t ne1 = n_embd/n_head;
        int64_t ne2 = n_head;
        int64_t ne3 = n_batch;
        size_t  nb0 = ggml_element_size(t);
        size_t  nb1 = nb0*ne0;
        size_t  nb2 = nb1*ne1;
        size_t  nb3 = nb2*ne2;
        size_t offset = 2*nb3*ne3;
        return ggml_view_4d(ctx0, t, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
    };

    auto add_or_set = [ctx0] (struct ggml_tensor * a, struct ggml_tensor * b) -> struct ggml_tensor * {
        if (a == NULL) {
            return b;
        } else {
            return ggml_add_inplace(ctx0, a, b);
        }
    };

    use_buf(-1);

    model->tok_embeddings->grad    = NULL;
    model->norm->grad              = NULL;
    model->output->grad            = NULL;

    for (int il = 0; il < n_layer; ++il) {
        struct my_llama_layer & layer = model->layers[il];
        layer.attention_norm->grad = NULL;
        layer.wq->grad             = NULL;
        layer.wk->grad             = NULL;
        layer.wv->grad             = NULL;
        layer.wo->grad             = NULL;
        layer.ffn_norm->grad       = NULL;
        layer.w1->grad             = NULL;
        layer.w2->grad             = NULL;
        layer.w3->grad             = NULL;
    }

    clr_buf(0);
    clr_buf(1);

    use_buf(-1);

    struct ggml_tensor * t00 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N*n_batch); assert_shape_1d(t00, N*n_batch);
    memcpy(t00->data, tokens_input->data, ggml_element_size(t00)*N*n_batch);

    use_buf(-1);

    struct ggml_tensor * t01 = expand(gf, ggml_get_rows(ctx0, model->tok_embeddings, t00)); assert_shape_2d(t01, n_embd, N*n_batch);

    // need to remember these for the backward pass
    std::vector<struct ggml_tensor *> t02L; t02L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t03L; t03L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t04L; t04L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t05L; t05L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t06L; t06L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t07L; t07L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t08L; t08L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t09L; t09L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t10L; t10L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t11L; t11L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t12L; t12L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t13L; t13L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t14L; t14L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t15L; t15L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t16L; t16L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t17L; t17L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t18L; t18L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t19L; t19L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t20L; t20L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t21L; t21L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t22L; t22L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t23L; t23L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t24L; t24L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t25L; t25L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t26L; t26L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t27L; t27L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t28L; t28L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t29L; t29L.resize(n_layer, NULL);
    std::vector<struct ggml_tensor *> t30L; t30L.resize(n_layer, NULL);

    struct ggml_tensor * cur = t01;

    for (int il = 0; il < n_layer; ++il) {
        clr_buf(0);
        struct my_llama_layer & layer = model->layers[il];
        // tensors with values necessary for backward pass are in persistent buf(-1)
        // other tensors with buf(0) and buf(1) are only temporary needed, and their memory reused after layer is completed.
        use_buf(-1); struct ggml_tensor * t02 = expand(gf, ggml_rms_norm     (ctx0, cur, rms_norm_eps));                      assert_shape_2d(t02, n_embd, N*n_batch);
        use_buf( 0); struct ggml_tensor * t03 = expand(gf, ggml_repeat       (ctx0, layer.attention_norm, t02));              assert_shape_2d(t03, n_embd, N*n_batch);
        use_buf(-1); struct ggml_tensor * t04 = expand(gf, ggml_mul          (ctx0, t02, t03));                               assert_shape_2d(t04, n_embd, N*n_batch);
        use_buf(-1); struct ggml_tensor * t05 = expand(gf, ggml_mul_mat      (ctx0, layer.wq, t04));                          assert_shape_2d(t05, n_embd, N*n_batch);
        use_buf(-1); struct ggml_tensor * t06 = expand(gf, ggml_reshape_4d   (ctx0, t05, n_embd/n_head, n_head, N, n_batch)); assert_shape_4d(t06, n_embd/n_head, n_head, N, n_batch);
        use_buf(-1); struct ggml_tensor * t07 = expand(gf, ggml_rope_inplace (ctx0, t06, n_past, n_rot, rope_mode, 0));       assert_shape_4d(t07, n_embd/n_head, n_head, N, n_batch);
        use_buf(-1); struct ggml_tensor * t08 = expand(gf, ggml_mul_mat      (ctx0, layer.wk, t04));                          assert_shape_2d(t08, n_embd, N*n_batch);
        use_buf(-1); struct ggml_tensor * t09 = expand(gf, ggml_reshape_4d   (ctx0, t08, n_embd/n_head, n_head, N, n_batch)); assert_shape_4d(t09, n_embd/n_head, n_head, N, n_batch);
        use_buf(-1); struct ggml_tensor * t10 = expand(gf, ggml_rope_inplace (ctx0, t09, n_past, n_rot, rope_mode, 0));       assert_shape_4d(t10, n_embd/n_head, n_head, N, n_batch);
        use_buf(-1); struct ggml_tensor * t11 = expand(gf, ggml_mul_mat      (ctx0, t04, layer.wv));                          assert_shape_2d(t11, N*n_batch, n_embd);
        use_buf(-1); struct ggml_tensor * t12 = expand(gf, ggml_reshape_4d   (ctx0, t11, N, n_batch, n_embd/n_head, n_head)); assert_shape_4d(t12, N, n_batch, n_embd/n_head, n_head);
        use_buf(-1); struct ggml_tensor * t13 = expand(gf, ggml_permute      (ctx0, t07, 0, 2, 1, 3));                        assert_shape_4d(t13, n_embd/n_head, N, n_head, n_batch);
        use_buf(-1); struct ggml_tensor * t14 = expand(gf, ggml_permute      (ctx0, t10, 0, 2, 1, 3));                        assert_shape_4d(t14, n_embd/n_head, N, n_head, n_batch);
        use_buf(-1); struct ggml_tensor * t15 = expand(gf, ggml_permute      (ctx0, t12, 0, 3, 1, 2));                        assert_shape_4d(t15, N, n_embd/n_head, n_head, n_batch);
        use_buf(-1); struct ggml_tensor * t16 = expand(gf, ggml_flash_attn   (ctx0, t13, t14, t15, true));                    assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);
        use_buf( 0); struct ggml_tensor * t17 = expand(gf, ggml_permute      (ctx0, t16, 0, 2, 1, 3));                        assert_shape_4d(t17, n_embd/n_head, n_head, N, n_batch);
        use_buf(-1); struct ggml_tensor * t18 = expand(gf, ggml_cont         (ctx0, t17));                                    assert_shape_4d(t18, n_embd/n_head, n_head, N, n_batch);
        use_buf(-1); struct ggml_tensor * t19 = expand(gf, ggml_reshape_2d   (ctx0, t18, n_embd, N*n_batch));                 assert_shape_2d(t19, n_embd, N*n_batch);
        use_buf( 0); struct ggml_tensor * t20 = expand(gf, ggml_mul_mat      (ctx0, layer.wo, t19));                          assert_shape_2d(t20, n_embd, N*n_batch);
        use_buf(-1); struct ggml_tensor * t21 = expand(gf, ggml_add          (ctx0, t20, cur));                               assert_shape_2d(t21, n_embd, N*n_batch);
        use_buf(-1); struct ggml_tensor * t22 = expand(gf, ggml_rms_norm     (ctx0, t21, rms_norm_eps));                      assert_shape_2d(t22, n_embd, N*n_batch);
        use_buf( 0); struct ggml_tensor * t23 = expand(gf, ggml_repeat       (ctx0, layer.ffn_norm, t22));                    assert_shape_2d(t23, n_embd, N*n_batch);
        use_buf(-1); struct ggml_tensor * t24 = expand(gf, ggml_mul          (ctx0, t23, t22));                               assert_shape_2d(t24, n_embd, N*n_batch);
        use_buf(-1); struct ggml_tensor * t25 = expand(gf, ggml_mul_mat      (ctx0, layer.w3, t24));                          assert_shape_2d(t25, n_ff, N*n_batch);
        use_buf(-1); struct ggml_tensor * t26 = expand(gf, ggml_mul_mat      (ctx0, layer.w1, t24));                          assert_shape_2d(t26, n_ff, N*n_batch);
        use_buf(-1); struct ggml_tensor * t27 = expand(gf, ggml_silu         (ctx0, t26));                                    assert_shape_2d(t27, n_ff, N*n_batch);
        use_buf(-1); struct ggml_tensor * t28 = expand(gf, ggml_mul          (ctx0, t27, t25));                               assert_shape_2d(t28, n_ff, N*n_batch);
        use_buf( 0); struct ggml_tensor * t29 = expand(gf, ggml_mul_mat      (ctx0, layer.w2, t28));                          assert_shape_2d(t29, n_embd, N*n_batch);
        use_buf(-1); struct ggml_tensor * t30 = expand(gf, ggml_add          (ctx0, t21, t29));                               assert_shape_2d(t30, n_embd, N*n_batch);
        t02L[il] = t02;
        t03L[il] = t03;
        t04L[il] = t04;
        t05L[il] = t05;
        t06L[il] = t06;
        t07L[il] = t07;
        t08L[il] = t08;
        t09L[il] = t09;
        t10L[il] = t10;
        t11L[il] = t11;
        t12L[il] = t12;
        t13L[il] = t13;
        t14L[il] = t14;
        t15L[il] = t15;
        t16L[il] = t16;
        t17L[il] = t17;
        t18L[il] = t18;
        t19L[il] = t19;
        t20L[il] = t20;
        t21L[il] = t21;
        t22L[il] = t22;
        t23L[il] = t23;
        t24L[il] = t24;
        t25L[il] = t25;
        t26L[il] = t26;
        t27L[il] = t27;
        t28L[il] = t28;
        t29L[il] = t29;
        t30L[il] = t30;

        cur      = t30;
    }
    clr_buf(0);
    use_buf(0);
    struct ggml_tensor * t31   = expand(gf, ggml_rms_norm  (ctx0, cur, rms_norm_eps));         assert_shape_2d(t31, n_embd, N*n_batch);
    struct ggml_tensor * t32   = expand(gf, ggml_repeat    (ctx0, model->norm, t31));          assert_shape_2d(t32, n_embd, N*n_batch);
    struct ggml_tensor * t33   = expand(gf, ggml_mul       (ctx0, t32, t31));                  assert_shape_2d(t33, n_embd, N*n_batch);
    use_buf(-1);
    struct ggml_tensor * t34   = expand(gf, ggml_mul_mat   (ctx0, model->output, t33));        assert_shape_2d(t34, n_vocab, N*n_batch);
    struct ggml_tensor * t35   = expand(gf, ggml_reshape_3d(ctx0, t34, n_vocab, N, n_batch));  assert_shape_3d(t35, n_vocab, N, n_batch);
    struct ggml_tensor * t36   = expand(gf, ggml_cross_entropy_loss(ctx0, t35, targets));      assert_shape_1d(t36, 1);

    {
        /*
        tok_embeddings                                                        | grad_tok_embeddings = ggml_get_rows_back(grad_t01, t00)
        L0_att_norm                                                           | grad_L0_att_norm    = ggml_repeat_back(grad_t03L0, L0_att_norm.shape)
        L0_wq                                                                 | grad_L0_wq          = ggml_out_prod(t04L0, grad_t05L0)
        L0_wk                                                                 | grad_L0_wk          = ggml_out_prod(t04L0, grad_t08L0)
        L0_wv                                                                 | grad_L0_wv          = ggml_out_prod(t04L0, ggml_transpose(grad_t11L0))
        L0_wo                                                                 | grad_L0_wo          = ggml_out_prod(t19L0, grad_t20L0)
        L0_ffn_norm                                                           | grad_L0_ffn_norm    = ggml_repeat_back(grad_t23L0, L0_ffn_norm.shape)
        L0_w1                                                                 | grad_L0_w1          = ggml_out_prod(t24L0, grad_t26L0)
        L0_w2                                                                 | grad_L0_w2          = ggml_out_prod(t28L0, grad_t29L0)
        L0_w3                                                                 | grad_L0_w3          = ggml_out_prod(t24L0, grad_t25L0)
        L1_att_norm                                                           | grad_L1_att_norm    = ggml_repeat_back(grad_t03L1, L1_att_norm.shape)
        L1_wq                                                                 | grad_L1_wq          = ggml_out_prod(t04L1, grad_t05L1)
        L1_wk                                                                 | grad_L1_wk          = ggml_out_prod(t04L1, grad_t08L1)
        L1_wv                                                                 | grad_L1_wv          = ggml_out_prod(t04L1, ggml_transpose(grad_t11L1))
        L1_wo                                                                 | grad_L1_wo          = ggml_out_prod(t19L1, grad_t20L1)
        L1_ffn_norm                                                           | grad_L1_ffn_norm    = ggml_repeat_back(grad_t23L1, L1_ffn_norm.shape)
        L1_w1                                                                 | grad_L1_w1          = ggml_out_prod(t24L1, grad_t26L1)
        L1_w2                                                                 | grad_L1_w2          = ggml_out_prod(t28L1, grad_t29L1)
        L1_w3                                                                 | grad_L1_w3          = ggml_out_prod(t24L1, grad_t25L1)
        norm                                                                  | grad_norm           = ggml_repeat_back(grad_t32, norm.shape)
        output                                                                | grad_output         = ggml_out_prod(t33, grad_t34)
                                                                              |
        t01 = ggml_get_rows(tok_embeddings, t00)                              | grad_t01   = grad_t21L0 + ggml_rms_norm_back(t01, grad_t02L0)
        for layer:                                                            |
        t02L0*= ggml_rms_norm     (t01)                                       | grad_t02L0 = ggml_mul(grad_t04L0, t03L0)
        t03L0 = ggml_repeat       (L0_att_norm, t02L0_shape)                  | grad_t03L0 = ggml_mul(grad_t04L0, t02L0)
        t04L0*= ggml_mul          (t02L0, t03L0)                              | grad_t04L0 = ggml_out_prod(L0_wv, grad_t11L0) + ggml_out_prod(L0_wk, ggml_transpose(grad_t08L0)) + ggml_out_prod(L0_wq, ggml_transpose(grad_t05L0))
        t05L0 = ggml_mul_mat      (L0_wq, t04L0)                              | grad_t05L0 = ggml_reshape(grad_t06L0, t05L0_shape)
        t06L0 = ggml_reshape_4d   (t05L0, n_embd/n_head, n_head, N, n_batch)  | grad_t06L0 = ggml_rope_back(grad_t07L0)
        t07L0 = ggml_rope_inplace (t06L0)                                     | grad_t07L0 = ggml_permute_back(grad_t13L0, 0, 2, 1, 3) = ggml_permute(grad_t13L0, 0, 2, 1, 3)
        t08L0 = ggml_mul_mat      (L0_wk, t04L0)                              | grad_t08L0 = ggml_reshape(grad_t09L0, t08L0_shape)
        t09L0 = ggml_reshape_4d   (t08L0, n_embd/n_head, n_head, N, n_batch)  | grad_t09L0 = ggml_rope_back(grad_t10L0)
        t10L0 = ggml_rope_inplace (t09L0)                                     | grad_t10L0 = ggml_permute_back(grad_t14L0, 0, 2, 1, 3) = ggml_permute(grad_t14L0, 0, 2, 1, 3)
        t11L0 = ggml_mul_mat      (t04L0, L0_wv)                              | grad_t11L0 = ggml_reshape(grad_t12L0, t11L0_shape)
        t12L0 = ggml_reshape_4d   (t11L0, N, n_batch, n_embd/n_head, n_head)  | grad_t12L0 = ggml_permute_back(grad_t15L0, 0, 3, 1, 2) = ggml_permute(grad_t15L0, 0, 2, 3, 1)
        t13L0*= ggml_permute      (t07L0, 0, 2, 1, 3)                         | grad_t13L0 = view__q(ggml_flash_attn_back(t13L0, t14L0, t15L0, grad_t16L0))
        t14L0*= ggml_permute      (t10L0, 0, 2, 1, 3)                         | grad_t14L0 = view__k(ggml_flash_attn_back(t13L0, t14L0, t15L0, grad_t16L0))
        t15L0*= ggml_permute      (t12L0, 0, 3, 1, 2)                         | grad_t15L0 = view__v(ggml_flash_attn_back(t13L0, t14L0, t15L0, grad_t16L0))
        t16L0 = ggml_flash_attn   (t13L0, t14L0, t15L0)                       | grad_t16L0 = ggml_permute_back(grad_t17L0, 0, 2, 1, 3) = ggml_permute(grad_t17L0, 0, 2, 1, 3)
        t17L0 = ggml_permute      (t16L0, 0, 2, 1, 3)                         | grad_t17L0 = grad_t18L0
        t18L0 = ggml_cont         (t17L0)                                     | grad_t18L0 = ggml_reshape(grad_t19L0, t18L0_shape)
        t19L0*= ggml_reshape_2d   (t18L0, n_embd, N*n_batch)                  | grad_t19L0 = ggml_out_prod(L0_wo, ggml_transpose(grad_t20L0))
        t20L0 = ggml_mul_mat      (L0_wo, t19L0)                              | grad_t20L0 = grad_t21L0
        t21L0*= ggml_add          (t20L0, t01)                                | grad_t21L0 = grad_t30L0 + ggml_rms_norm_back(t21L0, grad_t22L0)
        t22L0*= ggml_rms_norm     (t21L0)                                     | grad_t22L0 = ggml_mul(grad_t24L0, t23L0)
        t23L0 = ggml_repeat       (L0_ffn_norm, t22L0_shape)                  | grad_t23L0 = ggml_mul(grad_t24L0, t22L0)
        t24L0*= ggml_mul          (t23L0, t22L0)                              | grad_t24L0 = ggml_out_prod(L0_w1, ggml_transpose(grad_t26L0)) + ggml_out_prod(L0_w3, ggml_transpose(grad_t25L0))
        t25L0*= ggml_mul_mat      (L0_w3, t24L0)                              | grad_t25L0 = ggml_mul(grad_t28L0, t27L0)
        t26L0*= ggml_mul_mat      (L0_w1, t24L0)                              | grad_t26L0 = ggml_silu_back(t26L0, grad_t27L0)
        t27L0*= ggml_silu         (t26L0)                                     | grad_t27L0 = ggml_mul(grad_t28L0, t25L0)
        t28L0*= ggml_mul          (t27L0, t25L0)                              | grad_t28L0 = ggml_out_prod(L0_w2, ggml_transpose(grad_t29L0))
        t29L0 = ggml_mul_mat      (L0_w2, t28L0)                              | grad_t29L0 = grad_t30L0
        t30L0*= ggml_add          (t21L0, t29L0)                              | grad_t30L0 = ggml_rms_norm_back(t30L0, grad_t02L1) + grad_t21L1
                                                                              ^
        t02L1*= ggml_rms_norm     (t30L0)                                     | grad_t02L1 = ggml_mul(grad_t04L1, t03L1)
        t03L1 = ggml_repeat       (L1_att_norm, t02L1_shape)                  | grad_t03L1 = ggml_mul(grad_t04L1, t02L1)
        t04L1*= ggml_mul          (t02L1, t03L1)                              | grad_t04L1 = ggml_out_prod(L1_wv, grad_t11L1) + ggml_out_prod(L1_wk, ggml_transpose(grad_t08L1)) + ggml_out_prod(L1_wq, ggml_transpose(grad_t05L1))
        t05L1 = ggml_mul_mat      (L1_wq, t04L1)                              | grad_t05L1 = ggml_reshape(grad_t06L1, t05L1_shape)
        t06L1 = ggml_reshape_4d   (t05L1, n_embd/n_head, n_head, N, n_batch)  | grad_t06L1 = ggml_rope_back(grad_t07L1)
        t07L1 = ggml_rope_inplace (t06L1)                                     | grad_t07L1 = ggml_permute_back(grad_t13L1, 0, 2, 1, 3) = ggml_permute(grad_t13L1, 0, 2, 1, 3)
        t08L1 = ggml_mul_mat      (L1_wk, t04L1)                              | grad_t08L1 = ggml_reshape(grad_t09L1, t08L1_shape)
        t09L1 = ggml_reshape_4d   (t08L1, n_embd/n_head, n_head, N, n_batch)  | grad_t09L1 = ggml_rope_back(grad_t10L1)
        t10L1 = ggml_rope_inplace (t09L1)                                     | grad_t10L1 = ggml_permute_back(grad_t14L1, 0, 2, 1, 3) = ggml_permute(grad_t14L1, 0, 2, 1, 3)
        t11L1 = ggml_mul_mat      (t04L1, L1_wv)                              | grad_t11L1 = ggml_reshape(grad_t12L1, t11L1_shape)
        t12L1 = ggml_reshape_4d   (t11L1, N, n_batch, n_embd/n_head, n_head)  | grad_t12L1 = ggml_permute_back(grad_t15L1, 0, 3, 1, 2) = ggml_permute(grad_t15L1, 0, 2, 3, 1)
        t13L1*= ggml_permute      (t07L1, 0, 2, 1, 3)                         | grad_t13L1 = view__q(ggml_flash_attn_back(t13L1, t14L1, t15L1, grad_t16L1))
        t14L1*= ggml_permute      (t10L1, 0, 2, 1, 3)                         | grad_t14L1 = view__k(ggml_flash_attn_back(t13L1, t14L1, t15L1, grad_t16L1))
        t15L1*= ggml_permute      (t12L1, 0, 3, 1, 2)                         | grad_t15L1 = view__v(ggml_flash_attn_back(t13L1, t14L1, t15L1, grad_t16L1))
        t16L1 = ggml_flash_attn   (t13L1, t14L1, t15L1)                       | grad_t16L1 = ggml_permute_back(grad_t17L1, 0, 2, 1, 3) = ggml_permute(grad_t17L1, 0, 2, 1, 3)
        t17L1 = ggml_permute      (t16L1, 0, 2, 1, 3)                         | grad_t17L1 = grad_t18L1
        t18L1 = ggml_cont         (t17L1)                                     | grad_t18L1 = ggml_reshape(grad_t19L1, t18L1_shape)
        t19L1*= ggml_reshape_2d   (t18L1, n_embd, N*n_batch)                  | grad_t19L1 = ggml_out_prod(L1_wo, ggml_transpose(grad_t20L1))
        t20L1 = ggml_mul_mat      (L1_wo, t19L1)                              | grad_t20L1 = grad_t21L1
        t21L1*= ggml_add          (t20L1, t30L0)                              | grad_t21L1 = grad_t30L1 + ggml_rms_norm_back(t21L1, grad_t22L1)
        t22L1*= ggml_rms_norm     (t21L1)                                     | grad_t22L1 = ggml_mul(grad_t24L1, t23L1)
        t23L1 = ggml_repeat       (L1_ffn_norm, t22L1_shape)                  | grad_t23L1 = ggml_mul(grad_t24L1, t22L1)
        t24L1*= ggml_mul          (t23L1, t22L1)                              | grad_t24L1 = ggml_out_prod(L1_w1, ggml_transpose(grad_t26L1)) + ggml_out_prod(L1_w3, ggml_transpose(grad_t25L1))
        t25L1*= ggml_mul_mat      (L1_w3, t24L1)                              | grad_t25L1 = ggml_mul(grad_t28L1, t27L1)
        t26L1*= ggml_mul_mat      (L1_w1, t24L1)                              | grad_t26L1 = ggml_silu_back(t26L1, grad_t27L1)
        t27L1*= ggml_silu         (t26L1)                                     | grad_t27L1 = ggml_mul(grad_t28L1, t25L1)
        t28L1*= ggml_mul          (t27L1, t25L1)                              | grad_t28L1 = ggml_out_prod(L1_w2, ggml_transpose(grad_t29L1))
        t29L1 = ggml_mul_mat      (L1_w2, t28L1)                              | grad_t29L1 = grad_t30L1
        t30L1*= ggml_add          (t21L1, t29L1)                              | grad_t30L1 = ggml_rms_norm_back(t30L1, grad_t31)
                                                                              ^
        t31   = ggml_rms_norm     (t30L1)                                     | grad_t31   = ggml_mul(grad_t33, t32)
        t32   = ggml_repeat       (norm, t31.shape)                           | grad_t32   = ggml_mul(grad_t33, t31)
        t33   = ggml_mul          (t32, t31)                                  | grad_t33   = ggml_out_prod(output, ggml_transpose(grad_t34))
        t34   = ggml_mul_mat      (output, t33)                               | grad_t34   = ggml_reshape(grad_t35, t34.shape)
        t35   = ggml_reshape_3d   (t34, n_vocab, N, n_batch)                  | grad_t35   = ggml_cross_entropy_loss_back(t35, targets, grad_t36)
        t36   = ggml_cross_entropy_loss(t35, targets)                         | grad_t36   = 1 (optimizer)
        tensors marked with * need to be stored until grad computation
        tensors during grad computation are all temporary
        */
    }

    *gb = *gf;

    // t36->grad gets set to one by optimizer, so we need the tensor.
    // initialize it with 1.0f to make sure.
    use_buf(-1);
    t36->grad = expand(gb, ggml_new_f32(ctx0, 1.0f));

    use_buf(0);
    t35->grad = expand(gb, ggml_cross_entropy_loss_back(ctx0, t35, targets, t36->grad));              assert_shape_3d(t35->grad, n_vocab, N, n_batch);
    t34->grad = expand(gb, ggml_reshape_2d (ctx0, t35->grad, n_vocab, N*n_batch));                    assert_shape_2d(t34->grad, n_vocab, N*n_batch);
    t33->grad = expand(gb, ggml_out_prod   (ctx0, model->output, ggml_transpose(ctx0, t34->grad)));   assert_shape_2d(t33->grad, n_embd, N*n_batch);
    t32->grad = expand(gb, ggml_mul        (ctx0, t33->grad, t31));                                   assert_shape_2d(t32->grad, n_embd, N*n_batch);

    use_buf(-1);

    model->norm->grad   = expand(gb, add_or_set(model->norm->grad,   ggml_repeat_back(ctx0, t32->grad, model->norm))); assert_shape_1d(model->norm->grad, n_embd);
    model->output->grad = expand(gb, add_or_set(model->output->grad, ggml_out_prod(ctx0, t33, t34->grad)));            assert_shape_2d(model->output->grad, n_embd, n_vocab);

    clr_buf(1);
    use_buf(1);
    t31->grad = expand(gb, ggml_mul(ctx0, t33->grad, t32));  assert_shape_2d(t31->grad, n_embd, N*n_batch);

    struct ggml_tensor * back_layer_inp = t31;
    struct ggml_tensor * grad_layer_inp = NULL;

    for (int k = 0; k < n_layer; ++k) {
        int il = n_layer-1-k;
        struct my_llama_layer & layer = model->layers[il];

        struct ggml_tensor * t02 = t02L[il];
        struct ggml_tensor * t03 = t03L[il];
        struct ggml_tensor * t04 = t04L[il];
        struct ggml_tensor * t05 = t05L[il];
        struct ggml_tensor * t06 = t06L[il];
        struct ggml_tensor * t07 = t07L[il];
        struct ggml_tensor * t08 = t08L[il];
        struct ggml_tensor * t09 = t09L[il];
        struct ggml_tensor * t10 = t10L[il];
        struct ggml_tensor * t11 = t11L[il];
        struct ggml_tensor * t12 = t12L[il];
        struct ggml_tensor * t13 = t13L[il];
        struct ggml_tensor * t14 = t14L[il];
        struct ggml_tensor * t15 = t15L[il];
        struct ggml_tensor * t16 = t16L[il];
        struct ggml_tensor * t17 = t17L[il];
        struct ggml_tensor * t18 = t18L[il];
        struct ggml_tensor * t19 = t19L[il];
        struct ggml_tensor * t20 = t20L[il];
        struct ggml_tensor * t21 = t21L[il];
        struct ggml_tensor * t22 = t22L[il];
        struct ggml_tensor * t23 = t23L[il];
        struct ggml_tensor * t24 = t24L[il];
        struct ggml_tensor * t25 = t25L[il];
        struct ggml_tensor * t26 = t26L[il];
        struct ggml_tensor * t27 = t27L[il];
        struct ggml_tensor * t28 = t28L[il];
        struct ggml_tensor * t29 = t29L[il];
        struct ggml_tensor * t30 = t30L[il];

        clr_buf(0);
        use_buf(0);
        t30->grad = expand(gb, ggml_rms_norm_back(ctx0, t30, back_layer_inp->grad)); assert_shape_2d(t30->grad, n_embd, N*n_batch);
        if (grad_layer_inp) {
            t30->grad = expand(gb, ggml_add(ctx0, t30->grad, grad_layer_inp->grad)); assert_shape_2d(t30->grad, n_embd, N*n_batch);
        }
        clr_buf(1);
        t29->grad = t30->grad;                                                                                        assert_shape_2d(t29->grad, n_embd, N*n_batch);
        t28->grad = expand(gb, ggml_out_prod(ctx0, layer.w2, ggml_transpose(ctx0, t29->grad)));                       assert_shape_2d(t28->grad, n_ff, N*n_batch);
        t27->grad = expand(gb, ggml_mul(ctx0, t28->grad, t25));                                                       assert_shape_2d(t27->grad, n_ff, N*n_batch);
        t26->grad = expand(gb, ggml_silu_back(ctx0, t26, t27->grad));                                                 assert_shape_2d(t26->grad, n_ff, N*n_batch);
        t25->grad = expand(gb, ggml_mul(ctx0, t28->grad, t27));                                                       assert_shape_2d(t25->grad, n_ff, N*n_batch);
        t24->grad = expand(gb, ggml_add_inplace(ctx0,
                        ggml_out_prod(ctx0, layer.w1, ggml_transpose(ctx0, t26->grad)),
                        ggml_out_prod(ctx0, layer.w3, ggml_transpose(ctx0, t25->grad))));                             assert_shape_2d(t24->grad, n_embd, N*n_batch);
        t23->grad = expand(gb, ggml_mul(ctx0, t24->grad, t22));                                                       assert_shape_2d(t23->grad, n_embd, N*n_batch);
        t22->grad = expand(gb, ggml_mul(ctx0, t24->grad, ggml_repeat(ctx0, layer.ffn_norm, t24->grad)));              assert_shape_2d(t22->grad, n_embd, N*n_batch);
        use_buf(1);
        t21->grad = expand(gb, ggml_add(ctx0, t30->grad, ggml_rms_norm_back(ctx0, t21, t22->grad)));                  assert_shape_2d(t21->grad, n_embd, N*n_batch);
        grad_layer_inp = t21;
        use_buf(0);
        t20->grad = t21->grad;                                                                                        assert_shape_2d(t20->grad, n_embd, N*n_batch);
        t19->grad = expand(gb, ggml_out_prod(ctx0, layer.wo, ggml_transpose(ctx0, t20->grad)));                       assert_shape_2d(t19->grad, n_embd, N*n_batch);
        t18->grad = expand(gb, ggml_reshape_4d(ctx0, t19->grad, n_embd/n_head, n_head, N, n_batch));                  assert_shape_4d(t18->grad, n_embd/n_head, n_head, N, n_batch);
        t17->grad = t18->grad;                                                                                        assert_shape_4d(t17->grad, n_embd/n_head, n_head, N, n_batch);
        t16->grad = expand(gb, ggml_permute(ctx0, t17->grad, 0, 2, 1, 3));                                            assert_shape_4d(t16->grad, n_embd/n_head, N, n_head, n_batch);
        struct ggml_tensor * flash_attn = expand(gb, ggml_flash_attn_back(ctx0, t13, t14, t15, t16->grad, true));     assert_shape_4d(flash_attn, n_embd/n_head, N*3, n_head, n_batch);
        t15->grad = expand(gb, view__v(flash_attn));                                                                  assert_shape_4d(t15->grad, N, n_embd/n_head, n_head, n_batch);
        t14->grad = expand(gb, view__k(flash_attn));                                                                  assert_shape_4d(t14->grad, n_embd/n_head, N, n_head, n_batch);
        t13->grad = expand(gb, view__q(flash_attn));                                                                  assert_shape_4d(t13->grad, n_embd/n_head, N, n_head, n_batch);
        t12->grad = expand(gb, ggml_permute(ctx0, t15->grad, 0, 2, 3, 1));                                            assert_shape_4d(t12->grad, N, n_batch, n_embd/n_head, n_head);
        t11->grad = expand(gb, ggml_reshape_2d(ctx0, ggml_cont(ctx0, t12->grad), N*n_batch, n_embd));                 assert_shape_2d(t11->grad, N*n_batch, n_embd);
        t10->grad = expand(gb, ggml_permute(ctx0, t14->grad, 0, 2, 1, 3));                                            assert_shape_4d(t10->grad, n_embd/n_head, n_head, N, n_batch);
        t09->grad = expand(gb, ggml_rope_back(ctx0, t10->grad, n_past, n_rot, rope_mode, n_ctx));                     assert_shape_4d(t09->grad, n_embd/n_head, n_head, N, n_batch);
        t08->grad = expand(gb, ggml_reshape_2d(ctx0, t09->grad, n_embd, N*n_batch));                                  assert_shape_2d(t08->grad, n_embd, N*n_batch);
        t07->grad = expand(gb, ggml_permute(ctx0, t13->grad, 0, 2, 1, 3));                                            assert_shape_4d(t07->grad, n_embd/n_head, n_head, N, n_batch);
        t06->grad = expand(gb, ggml_rope_back(ctx0, t07->grad, n_past, n_rot, rope_mode, n_ctx));                     assert_shape_4d(t06->grad, n_embd/n_head, n_head, N, n_batch);
        t05->grad = expand(gb, ggml_reshape_2d(ctx0, t06->grad, n_embd, N*n_batch));                                  assert_shape_2d(t05->grad, n_embd, N*n_batch);
        t04->grad = expand(gb, ggml_add_inplace(ctx0,
                        ggml_add_inplace(ctx0,
                            ggml_out_prod(ctx0, layer.wv, t11->grad),
                            ggml_out_prod(ctx0, layer.wk, ggml_transpose(ctx0, t08->grad))),
                        ggml_out_prod(ctx0, layer.wq, ggml_transpose(ctx0, t05->grad))));                             assert_shape_2d(t04->grad, n_embd, N*n_batch);
        t03->grad = expand(gb, ggml_mul(ctx0, t04->grad, t02));                                                       assert_shape_2d(t04->grad, n_embd, N*n_batch);
        use_buf(1);
        t02->grad = expand(gb, ggml_mul(ctx0, t04->grad, ggml_repeat(ctx0, layer.attention_norm, t02)));              assert_shape_2d(t02->grad, n_embd, N*n_batch);
        back_layer_inp = t02;
        // use_buf(0);

        use_buf(-1);
        layer.attention_norm->grad = expand(gb, add_or_set(layer.attention_norm->grad, ggml_repeat_back(ctx0, t03->grad, layer.attention_norm)));   assert_shape_1d(layer.attention_norm->grad, n_embd);
        layer.wq->grad             = expand(gb, add_or_set(layer.wq->grad,             ggml_out_prod(ctx0, t04, t05->grad)));                       assert_shape_2d(layer.wq->grad,             n_embd, n_embd);
        layer.wk->grad             = expand(gb, add_or_set(layer.wk->grad,             ggml_out_prod(ctx0, t04, t08->grad)));                       assert_shape_2d(layer.wk->grad,             n_embd, n_embd);
        layer.wv->grad             = expand(gb, add_or_set(layer.wv->grad,             ggml_out_prod(ctx0, t04, ggml_transpose(ctx0, t11->grad)))); assert_shape_2d(layer.wv->grad,             n_embd, n_embd);
        layer.wo->grad             = expand(gb, add_or_set(layer.wo->grad,             ggml_out_prod(ctx0, t19, t20->grad)));                       assert_shape_2d(layer.wo->grad,             n_embd, n_embd);
        layer.ffn_norm->grad       = expand(gb, add_or_set(layer.ffn_norm->grad,       ggml_repeat_back(ctx0, t23->grad, layer.ffn_norm)));         assert_shape_1d(layer.ffn_norm->grad,       n_embd);
        layer.w1->grad             = expand(gb, add_or_set(layer.w1->grad,             ggml_out_prod(ctx0, t24, t26->grad)));                       assert_shape_2d(layer.w1->grad,             n_embd, n_ff);
        layer.w2->grad             = expand(gb, add_or_set(layer.w2->grad,             ggml_out_prod(ctx0, t28, t29->grad)));                       assert_shape_2d(layer.w2->grad,             n_ff, n_embd);
        layer.w3->grad             = expand(gb, add_or_set(layer.w3->grad,             ggml_out_prod(ctx0, t24, t25->grad)));                       assert_shape_2d(layer.w3->grad,             n_embd, n_ff);
        // use_buf(0);
    }
    clr_buf(0);
    use_buf(0);
    t01->grad = expand(gb, ggml_add_inplace(ctx0, grad_layer_inp->grad, ggml_rms_norm_back(ctx0, t01, back_layer_inp->grad)));  assert_shape_2d(t01->grad, n_embd, N*n_batch);
    use_buf(-1);
    model->tok_embeddings->grad = expand(gb, ggml_get_rows_back(ctx0, t01->grad, t00, model->tok_embeddings));                  assert_shape_2d(model->tok_embeddings->grad, n_embd, n_vocab);
    // clr_buf(1);
    // clr_buf(0);

    *logits = t35;

    if (track_max_mem) {
        printf("%s: max size compute buf0: %zu\n", __func__, buf_maxs[0]);
        printf("%s: max size compute buf1: %zu\n", __func__, buf_maxs[1]);
    }

    // now that all grads are created, set the graph leafs and grads
    graph_set_leafs_grads(gf);
    graph_set_leafs_grads(gb);

    return t36;
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
    printf("%s", llama_token_to_str(ctx, token).c_str());
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

void get_example_targets(struct llama_context * lctx, const int * train_samples, size_t n_train_samples, const llama_token * train_data, size_t n_train_data, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * target_logits, struct ggml_tensor * target_probs) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab  = target_logits->ne[0];

    size_t sample = train_samples[example_id % n_train_samples];
    GGML_ASSERT(sample+n_tokens-1 < n_train_data);

    ggml_set_f32(target_logits, -1.0f/n_vocab);
    ggml_set_f32(target_probs, 0.0f);
    ggml_set_i32_1d(tokens_input, 0, llama_token_bos(lctx));
    for (int i=1; i<n_tokens+1; ++i) {
        int token = clamp(train_data[sample+i-1], 0, n_vocab-1);
        set_f32_2d(target_logits, token, i-1, +1.0f);
        set_f32_2d(target_probs,  token, i-1, +1.0f);
        if (i<n_tokens) {
            ggml_set_i32_1d(tokens_input, i, token);
        }
    }
}

void get_example_targets_batch(struct llama_context * lctx, const int * train_samples, size_t n_train_samples, const llama_token * train_data, size_t n_train_data, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * target_logits, struct ggml_tensor * target_probs) {
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

        set_i32_2d(tokens_input, 0, k, llama_token_bos(lctx));
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

struct ggml_tensor * square_error_loss(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * target) {
    return ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, target, a)));
}

struct ggml_tensor * cross_entropy_loss(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * probs) {
    return ggml_cross_entropy_loss(ctx, a, probs);
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

    int n_tokens = llama_tokenize(lctx, buf.data(), out.data(), out.size(), false);
    if (n_tokens < 0) {
        out.resize(-n_tokens);
        llama_tokenize(lctx, buf.data(), out.data(), out.size(), false);
    }

    bool verify = false;
    if (verify) {
        const char * in  = buf.data();
        const char * end = buf.data() + buf.size();
        for (int i = 0; i < (int) out.size(); ++i) {
            std::string s = llama_token_to_str(lctx, out[i]);
            int len = s.length();
            if (in >= end) {
                printf("%s: unexpected end of original text.\n", __func__);
                break;
            }
            const bool matches = (strncmp(in, s.c_str(), len) == 0);
            if (matches) {
                in += len;
            } else {
                printf("%s: mismatch: expected '%s', but got '%s'\n", __func__, std::string(in, len).c_str(), s.c_str());
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
    const float nl_logit = logits[llama_token_nl(ctx)];

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
        logits[llama_token_nl(ctx)] = nl_logit;
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

void save_checkpoint(struct my_llama_model * model, struct ggml_opt_context * opt, const char * filename) {
    struct llama_file file(filename, "wb");
    if (file.fp == NULL) {
        return;
    }

    const uint32_t magic   = 'ggcp';
    const uint32_t version = 0;

    file.write_u32(magic);
    file.write_u32(version);
    file.write_u32(model->train_its);
    file.write_u32(model->train_samples);
    file.write_u32(model->train_tokens);
    file.write_u32(model->hparams.n_vocab);
    file.write_u32(model->hparams.n_embd);
    file.write_u32(model->hparams.n_mult);
    file.write_u32(model->hparams.n_head);
    file.write_u32(model->hparams.n_layer);
    file.write_u32(model->hparams.n_rot);

    write_tensor(&file, model->tok_embeddings);
    write_tensor(&file, model->norm);
    write_tensor(&file, model->output);

    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
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

    write_opt_context(&file, opt);
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

void save_as_llama_model(struct llama_vocab * vocab, struct my_llama_model * model, const char * filename) {
    struct llama_file file(filename, "wb");
    if (file.fp == NULL) {
        return;
    }

#pragma message("TODO: implement file saving using gguf")
    (void) vocab;
    (void) model;
//    // write_magic
//    file.write_u32(LLAMA_FILE_MAGIC);   // magic
//    file.write_u32(LLAMA_FILE_VERSION); // version
//    // write_hparams
//    file.write_u32(model->hparams.n_vocab);
//    file.write_u32(model->hparams.n_embd);
//    file.write_u32(model->hparams.n_mult);
//    file.write_u32(model->hparams.n_head);
//    file.write_u32(model->hparams.n_layer);
//    file.write_u32(model->hparams.n_rot);
//    file.write_u32(LLAMA_FTYPE_ALL_F32);
//    // write_vocab
//    uint32_t n_vocab = model->hparams.n_vocab;
//    for (uint32_t i = 0; i < n_vocab; i++) {
//        const auto & token_data = vocab->id_to_token.at(i);
//        file.write_u32((uint32_t) token_data.tok.size());
//        file.write_raw(token_data.tok.data(), token_data.tok.size());
//        file.write_raw(&token_data.score, sizeof(token_data.score));
//    }
//    // write tensors
//    write_tensor(&file, model->tok_embeddings);
//    write_tensor(&file, model->norm);
//    write_tensor(&file, model->output);
//    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
//        auto & layer = model->layers[i];
//
//        write_tensor(&file, layer.attention_norm);
//        write_tensor(&file, layer.wq);
//        write_tensor(&file, layer.wk);
//        write_tensor(&file, layer.wv);
//        write_tensor(&file, layer.wo);
//        write_tensor(&file, layer.ffn_norm);
//        write_tensor(&file, layer.w1);
//        write_tensor(&file, layer.w2);
//        write_tensor(&file, layer.w3);
//    }
}

float cosine_decay(const int decay_steps, const float alpha, int step) {
    if (step > decay_steps) {
        step = decay_steps;
    }
    const float cosine_decay = 0.50f*(1.0f + cosf(3.14159265359f*step/decay_steps));
    const float decay = (1 - alpha)*cosine_decay + alpha;
    return decay;
}

float cosine_decay_restart(int decay_steps, const float alpha, int step, float restart_step_mult) {
    while (step > decay_steps) {
        step -= decay_steps;
        decay_steps = (int) restart_step_mult * decay_steps;
    }
    return cosine_decay(decay_steps, alpha, step);
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

    struct llama_vocab vocab;
    {
        const int n_vocab = llama_n_vocab(lctx);
        vocab.id_to_token.resize(n_vocab);
        for (int i=0; i<n_vocab; ++i) {
            vocab.id_to_token[i].text  = llama_token_get_text(lctx, i);
            vocab.id_to_token[i].score = llama_token_get_score(lctx, i);
            vocab.id_to_token[i].type  = llama_token_get_type(lctx, i);
            vocab.token_to_id.emplace(vocab.id_to_token[i].text, i);
        }
    }

    printf("%s: tokenize training data\n", __func__);
    std::vector<llama_token> train_tokens;
    if (tokenize_file(lctx, params.fn_train_data, train_tokens) < 0) {
        fprintf(stderr, "%s: failed to tokenize file '%s'\n", __func__, params.fn_train_data);
    }
    printf("%s: number of training tokens: %d\n", __func__, (int) train_tokens.size());

    struct my_llama_model model;
    model.hparams.n_vocab = llama_n_vocab(lctx);
    model.hparams.n_ctx   = params.n_ctx;
    model.hparams.n_embd  = params.n_embd;
    model.hparams.n_mult  = params.n_mult;
    model.hparams.n_head  = params.n_head;
    model.hparams.n_layer = params.n_layer;
    model.hparams.n_rot   = std::min((uint32_t)params.n_rotmax, model.hparams.n_embd / model.hparams.n_head);

    print_params(&model.hparams);

    std::vector<size_t> token_noccurs;
    std::vector<bool>   token_notavail;
    token_noccurs.resize(model.hparams.n_vocab, 0);
    token_notavail.resize(model.hparams.n_vocab, true);
    for (int i = 0; i < (int) train_tokens.size(); ++i) {
        ++token_noccurs[train_tokens[i]];
        token_notavail[train_tokens[i]] = false;
    }

    std::vector<float> token_freq;
    token_freq.resize(model.hparams.n_vocab, 0);
    int n_unique_tokens = 0;
    for (int i = 0; i < (int) token_noccurs.size(); ++i) {
        token_freq[i] = (float) token_noccurs[i] / (float) train_tokens.size();
        n_unique_tokens += (token_noccurs[i] > 0) ? 1 : 0;
    }
    printf("%s: number of unique tokens: %d\n", __func__, n_unique_tokens);

    struct my_llama_kv_cache kv_self;


    struct ggml_init_params lcparams;
    lcparams.mem_size   = 1024ll*1024ll*1024ll*((size_t) params.mem_model_gb);
    lcparams.mem_buffer = NULL;
    lcparams.no_alloc   = false;

    model.ctx = ggml_init(lcparams);
    kv_self.ctx = model.ctx;

    my_llama_sampler sampler;


    int n_tokens = model.hparams.n_ctx;
    int n_vocab  = model.hparams.n_vocab;
    int n_batch  = params.n_batch;

    struct ggml_opt_context * opt = (struct ggml_opt_context *) alloca(sizeof(struct ggml_opt_context));
    memset(opt, 0, sizeof(struct ggml_opt_context));

    struct ggml_opt_params opt_params_adam = ggml_opt_default_params(GGML_OPT_ADAM);
    struct ggml_opt_params opt_params_lbfgs = ggml_opt_default_params(GGML_OPT_LBFGS);
    opt_params_adam.print_forward_graph = false;
    opt_params_adam.print_backward_graph = false;
    opt_params_adam.n_threads   = params.n_threads;
    opt_params_adam.adam.n_iter = params.adam_n_iter;
    opt_params_adam.adam.sched  = 1.0f;
    opt_params_adam.adam.alpha  = params.adam_alpha;
    opt_params_adam.adam.decay  = params.adam_decay;

    opt_params_lbfgs.print_forward_graph = false;
    opt_params_lbfgs.print_backward_graph = false;
    opt_params_lbfgs.n_threads    = params.n_threads;
    opt_params_lbfgs.lbfgs.n_iter = params.lbfgs_n_iter;

    opt->ctx = model.ctx;
    opt->params = params.use_adam ? opt_params_adam : opt_params_lbfgs;

    printf("%s: init model\n", __func__);
    bool existed = load_checkpoint(&model, opt, params.fn_checkpoint_in, true);
    set_param_model(&model);

    opt->params = params.use_adam ? opt_params_adam : opt_params_lbfgs;

    opt->iter = model.train_its;
    printf("%s: opt iter %d\n", __func__, opt->iter);

    bool from_scratch = !existed;
    if (from_scratch) {
        randomize_model(&model, params.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    }

    init_kv_cache(&kv_self, &model, 1);
    // init_kv_cache(&kv_self, &model, n_batch);
    init_sampler(&sampler, lctx);

    printf("used_mem model+cache: %zu bytes\n", ggml_used_mem(model.ctx));
    // ggml_print_tensor_objects(model.ctx);

    // TODO: use std::vector<uint8_t> intead of "new"
    size_t    compute_size = 1024ll*1024ll*1024ll*((size_t) params.mem_compute_gb);
    uint8_t * compute_addr = new uint8_t[compute_size];

    size_t size_buf_0 = 1024ll*1024ll*1024ll*((size_t) params.mem_compute0_gb);
    size_t size_buf_1 = 1024ll*1024ll*1024ll*((size_t) params.mem_compute1_gb);
    uint8_t * compute_buf_0 = new uint8_t[size_buf_0];
    uint8_t * compute_buf_1 = new uint8_t[size_buf_1];

    GGML_ASSERT(n_tokens < (int) train_tokens.size());
    std::vector<int> train_samples;
    train_samples.push_back(0);
    for (int i = 1; i < (int) train_tokens.size() - n_tokens; ++i) {
        if (!params.samples_start_after_nl || (train_tokens[i-1] == llama_token_nl(lctx))) {
            train_samples.push_back(i);
        }
    }
    shuffle_ints(train_samples.data(), train_samples.data() + train_samples.size());
    for (int i = 0; i < (int) train_samples.size(); ++i) {
        GGML_ASSERT(train_samples[i]+n_tokens-1 < (int) train_tokens.size());
    }

    std::vector<uint8_t> work_buffer;

    printf("%s: begin training\n", __func__);

    for (int ex = 0; ex < params.n_examples; ++ex) {
        if (ex*n_batch >= (int) train_samples.size()) {
            shuffle_ints(train_samples.data(), train_samples.data() + train_samples.size());
            for (int i = 0; i < (int) train_samples.size(); ++i) {
                GGML_ASSERT(train_samples[i]+n_tokens-1 < (int) train_tokens.size());
            }
        }

        struct ggml_init_params cparams = {
            /*.mem_size   =*/ compute_size,
            /*.mem_buffer =*/ compute_addr,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context * ctx0 = ggml_init(cparams);

        struct ggml_tensor * after_opt_best_samples = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        //struct ggml_tensor * after_opt_probs        = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
        struct ggml_tensor * tokens_input           = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        struct ggml_tensor * target_logits          = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
        struct ggml_tensor * target_probs           = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);

        int n_past = 0;

        struct ggml_tensor * gfbuf = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, sizeof(struct ggml_cgraph) / ggml_type_size(GGML_TYPE_I32) + (sizeof(struct ggml_cgraph) % ggml_type_size(GGML_TYPE_I32) ? 1 : 0));
        struct ggml_tensor * gbbuf = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, sizeof(struct ggml_cgraph) / ggml_type_size(GGML_TYPE_I32) + (sizeof(struct ggml_cgraph) % ggml_type_size(GGML_TYPE_I32) ? 1 : 0));

        memset(gfbuf->data, 0, ggml_nbytes(gfbuf));
        memset(gbbuf->data, 0, ggml_nbytes(gbbuf));

        struct ggml_cgraph * gf = (struct ggml_cgraph *) gfbuf->data;
        struct ggml_cgraph * gb = (struct ggml_cgraph *) gbbuf->data;


        get_example_targets_batch(lctx, train_samples.data(), train_samples.size(), train_tokens.data(), train_tokens.size(), ex,  tokens_input, target_logits, target_probs);

        GGML_ASSERT(n_past == 0);

        struct ggml_tensor * loss   = NULL;
        struct ggml_tensor * logits = NULL;

        if (params.use_scratch) {
            loss = forward_batch_wo_cache_flash_attn_train(
                    &model, ctx0,
                    gf, gb,
                    &logits, tokens_input, target_probs,
                    compute_buf_0, compute_buf_1,
                    size_buf_0, size_buf_1,
                    n_tokens, n_batch);
        } else if (params.use_flash) {
            logits = forward_batch_wo_cache_flash_attn(&model, ctx0, gf, tokens_input, n_tokens, n_batch);
            loss   = cross_entropy_loss(ctx0, logits, target_probs);
            ggml_build_forward_expand(gf, loss);
            *gb = ggml_build_backward(ctx0, gf, true);
        } else {
            logits = forward_batch_wo_cache(&model, ctx0, gf, tokens_input, n_tokens, n_batch);
            loss   = cross_entropy_loss(ctx0, logits, target_probs);
            ggml_build_forward_expand(gf, loss);
            *gb = ggml_build_backward(ctx0, gf, true);
        }

        ggml_graph_compute_helper(work_buffer, gf, params.n_threads);

        size_t used_mem_before_opt = ggml_used_mem(ctx0);

        float error_before_opt = ggml_get_f32_1d(loss, 0);

        opt->params.adam.sched = (opt->iter < params.warmup)
            ? (float) opt->iter / (float) params.warmup
            : cosine_decay_restart(
                params.cos_decay_steps,
                params.cos_decay_alpha,
                opt->iter - params.warmup,
                params.cos_decay_restart);

        printf("%s: opt->params.adam.sched %.5f\n", __func__, opt->params.adam.sched);

        ggml_opt_resume_g(ctx0, opt, loss, gf, gb);

        size_t used_mem_after_opt = ggml_used_mem(ctx0);

        model.train_its = opt->iter;
        model.train_samples += n_batch;
        model.train_tokens  += n_batch * n_tokens;

        ggml_graph_compute_helper(work_buffer, gf, params.n_threads);

        float error_after_opt = ggml_get_f32_1d(loss, 0);

        if (params.print_info_interval > 0 && ex % params.print_info_interval == 0) {
            printf("Example %d, opt iter %d\n", ex, opt->iter);
            printf("error_before_opt: %.6f\n", error_before_opt);
            printf("error_after_opt:  %.6f\n", error_after_opt);
            printf("used_mem_before_opt: %zu bytes\n", used_mem_before_opt);
            printf("used_mem_after_opt:  %zu bytes\n", used_mem_after_opt);
        }

        if (params.print_details_interval > 0 && ex % params.print_details_interval == 0) {
            // set_logits_masked(logits, token_notavail, -1e9);
            for (int i=0; i<n_batch; ++i) {
                init_sampler(&sampler, lctx);
                for (int k=0; k<n_tokens; ++k) {
                    int32_t token = sample(&sampler,
                        (float *)       ((char *) logits->data + i*logits->nb[2] + k*logits->nb[1]),
                        (llama_token *) ((char *) tokens_input->data + i*tokens_input->nb[1]),
                        k);
                    * ((int32_t *) ((char *) after_opt_best_samples->data + i*after_opt_best_samples->nb[1] + k*after_opt_best_samples->nb[0])) = token;
                }
            }

            // printf("probabilities after optimization:\n");
            // print_matrix(after_opt_probs);
            printf("Example:\n---\n");
            print_tokens_batch(lctx, tokens_input);
            printf("\n---\n");

            // printf("best samples after optimization:\n---\n");
            printf("samples after optimization:\n---\n");
            print_tokens_batch(lctx, after_opt_best_samples);
            printf("\n---\n");
        }

        ggml_free(ctx0);
    }

    if (params.n_examples > 0) {
        save_checkpoint(&model, opt, params.fn_checkpoint_out);
    }

    if (strlen(params.fn_model_out) > 0) {
        save_as_llama_model(&vocab, &model, params.fn_model_out);
    }

    {
        int n_gen = params.n_predict;
        int sample_ctx = n_tokens - n_tokens/8;

        sampler.params.temp = 0.2f;
        sampler.params.repeat_penalty = 1.1f;
        sampler.params.mirostat = 2;
        init_sampler(&sampler, lctx);

        printf("Generating %d tokens.\n", n_gen);

        struct ggml_tensor * tokens_input  = ggml_new_tensor_1d(model.ctx, GGML_TYPE_I32, n_tokens);
        struct ggml_tensor * target_logits = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_vocab,  n_tokens);
        struct ggml_tensor * target_probs  = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_vocab,  n_tokens);

        get_example_targets(lctx, train_samples.data(), train_samples.size(), train_tokens.data(), train_tokens.size(), rand()%train_samples.size(), tokens_input, target_logits, target_probs);
        for (int i=sample_ctx; i<n_tokens; ++i) {
            ggml_set_i32_1d(tokens_input, i, n_vocab/2);
        }

        for (int i=0; i<sample_ctx-1; ++i) {
            print_token(lctx, ggml_get_i32_1d(tokens_input, i));
        }

        printf("---\n");
        for (int i=0; i<n_gen; ++i) {
            struct ggml_init_params cparams = {
                /*.mem_size   =*/ compute_size,
                /*.mem_buffer =*/ compute_addr,
                /*.no_alloc   =*/ false,
            };
            struct ggml_context * ctx0 = ggml_init(cparams);

            ggml_cgraph gf = {};

            int n_past = 0;
            struct ggml_tensor * logits = forward(&model, &kv_self, ctx0, &gf, tokens_input, sample_ctx, n_past);

            ggml_build_forward_expand(&gf, logits);
            ggml_graph_compute_helper(work_buffer, &gf, params.n_threads);

            //struct ggml_tensor * best_samples = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, sample_ctx);
            //struct ggml_tensor * probs        = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, sample_ctx);

            // set_logits_masked(logits, token_notavail, -1e9);
            int token = sample(&sampler,
                (float *) ((char *) logits->data + (sample_ctx-1)*logits->nb[1]),
                (llama_token *) tokens_input->data,
                sample_ctx-1);
            //int token = ggml_get_i32_1d(best_samples, sample_ctx-1);

            // print_row(probs, sample_at);
            print_token(lctx, token);

            lshift_examples(tokens_input, target_logits, target_probs, 1);
            ggml_set_i32_1d(tokens_input, 0, 0);
            ggml_set_i32_1d(tokens_input, sample_ctx-1, token);

            ggml_free(ctx0);
        }
    }

    delete[] compute_addr;
    delete[] compute_buf_0;
    delete[] compute_buf_1;

    llama_free(lctx);
    llama_free_model(lmodel);
    ggml_free(model.ctx);

    return 0;
}
