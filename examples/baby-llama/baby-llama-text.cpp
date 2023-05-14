#include "ggml.h"
#include "llama.h"
#include <unordered_map>
#include <vector>
#include <cassert>
#include <random>
#include <cstring>
#include <stdexcept>
#include <cstdarg>


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

float frand_normal(struct random_normal_distribution * rnd) {
    return fclamp(rnd->rd(rnd->gen), rnd->min, rnd->max);
}

float frand_uniform(struct random_uniform_distribution * rnd) {
    return rnd->rd(rnd->gen);
}

struct ggml_tensor * randomize_tensor_normal(struct ggml_tensor * tensor, struct random_normal_distribution * rnd) {
    switch (tensor->n_dims) {
        case 1:
            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0]);
                *dst = frand_normal(rnd);
            }
            break;
        case 2:
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
                    *dst = frand_normal(rnd);
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
                        *dst = frand_normal(rnd);
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
                            *dst = frand_normal(rnd);
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

uint32_t get_n_ff(const struct my_llama_hparams* hparams) {
    const uint32_t n_ff = ((2*(4*hparams->n_embd)/3 + hparams->n_mult - 1)/hparams->n_mult)*hparams->n_mult;
    return n_ff;
}

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
};

void init_model(struct my_llama_model * model) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;

    const uint32_t n_ff = get_n_ff(&hparams);

    struct ggml_context * ctx = model->ctx;

    model->tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab); // ("tok_embeddings.weight", {n_embd, n_vocab});
    model->norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);          // ("norm.weight",           {n_embd});
    model->output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab); // ("output.weight",         {n_embd, n_vocab});

    model->layers.resize(n_layer);
    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        // std::string layers_i = "layers." + std::to_string(i);

        layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd); // (layers_i + ".attention_norm.weight", {n_embd});

        layer.wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);     // (layers_i + ".attention.wq.weight", {n_embd, n_embd});
        layer.wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);     // (layers_i + ".attention.wk.weight", {n_embd, n_embd});
        layer.wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);     // (layers_i + ".attention.wv.weight", {n_embd, n_embd});
        layer.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);     // (layers_i + ".attention.wo.weight", {n_embd, n_embd});

        layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);       // (layers_i + ".ffn_norm.weight", {n_embd});

        layer.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);     // (layers_i + ".feed_forward.w1.weight", {n_embd,   n_ff});
        layer.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff, n_embd);     // (layers_i + ".feed_forward.w2.weight", {  n_ff,   n_embd});
        layer.w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);     // (layers_i + ".feed_forward.w3.weight", {n_embd,   n_ff});
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
            cur = ggml_rms_norm(ctx0, inpL);

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
            struct ggml_tensor * Qcur = ggml_rope(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wq, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0);
            struct ggml_tensor * Kcur = ggml_rope(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wk, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0);

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

                kc = ggml_set_1d(ctx0, kc, ggml_reshape_1d(ctx0, Kcur, n_embd*N), (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                vc = ggml_set_2d(ctx0, vc, Vcur, (   n_ctx)*ggml_element_size(kv_self.v),
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
                cur = ggml_rms_norm(ctx0, inpFF);

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
        inpL = ggml_rms_norm(ctx0, inpL);

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
            cur = ggml_rms_norm(ctx0, inpL);
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
            struct ggml_tensor * Qcur = ggml_rope(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wq, cur), n_embd/n_head, n_head, N, n_batch), n_past, n_rot, 0);
            struct ggml_tensor * Kcur = ggml_rope(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wk, cur), n_embd/n_head, n_head, N, n_batch), n_past, n_rot, 0);
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

                kc = ggml_set_2d(ctx0, kc,
                        ggml_reshape_2d(ctx0, Kcur, n_embd*N, n_batch),
                        ggml_element_size(kc)*n_embd*n_ctx,
                        (ggml_element_size(kc)*n_embd)*(il*n_batch*n_ctx + n_past));
                vc = ggml_set_2d(ctx0, vc,
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
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head)));
            assert_shape_4d(KQ_scaled, n_past + N, N, n_head, n_batch);

            // KQ_masked = mask_past(KQ_scaled)
            // KQ_masked shape [n_past + N, N, n_head, n_batch]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);
            assert_shape_4d(KQ_masked, n_past + N, N, n_head, n_batch);

            // KQ = soft_max(KQ_masked)
            // KQ_soft_max shape [n_past + N, N, n_head, n_batch]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
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
        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
        assert_shape_2d(inpFF, n_embd, N*n_batch);

        // feed-forward network
        {
            // norm
            {
                // cur shape [n_embd,N*n_batch,1,1]
                cur = ggml_rms_norm(ctx0, inpFF);
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
        cur = ggml_add(ctx0, cur, inpFF);
        assert_shape_2d(cur, n_embd, N*n_batch);

        // input for next layer
        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = cur;
        assert_shape_2d(inpL, n_embd, N*n_batch);
    }

    // norm
    {

        // inpL shape [n_embd,N*n_batch,1,1]
        inpL = ggml_rms_norm(ctx0, inpL);
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

void sample_softmax(struct ggml_tensor * logits, struct ggml_tensor * probs, struct ggml_tensor * best_samples) {
    assert(logits->n_dims == 2);
    assert(probs->n_dims == 2);
    assert(best_samples->n_dims == 1);
    assert(logits->ne[1] == best_samples->ne[0]);
    assert(logits->ne[0] == probs->ne[0]);
    assert(logits->ne[1] == probs->ne[1]);
    for (int i = 0; i < logits->ne[1]; ++i) {
        float max_logit = ggml_get_f32_1d(logits, i * logits->ne[0]);
        ggml_set_i32_1d(best_samples, i, 0);
        for (int k = 0; k < logits->ne[0]; ++k) {
            float logit = ggml_get_f32_1d(logits, i * logits->ne[0] + k);
            if (logit > max_logit) {
                max_logit = logit;
                ggml_set_i32_1d(best_samples, i, k);
            }
        }
        float psum = 0;
        for (int k = 0; k < logits->ne[0]; ++k) {
            float logit = ggml_get_f32_1d(logits, i * logits->ne[0] + k);
            float p = (logit == -INFINITY) ? 0 : expf(logit - max_logit);
            psum += p;
            ggml_set_f32_1d(probs, i * probs->ne[0] + k, p);
        }
        for (int k = 0; k < logits->ne[0]; ++k) {
            float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
            ggml_set_f32_1d(probs, i * probs->ne[0] + k, p / psum);
        }
    }
}

void sample_softmax_batch(struct ggml_context * ctx, struct ggml_tensor * logits, struct ggml_tensor * probs, struct ggml_tensor * best_samples) {
    GGML_ASSERT(best_samples->n_dims == 2);
    GGML_ASSERT(logits->n_dims == 3);
    GGML_ASSERT(probs->n_dims == 3);
    int n_tokens = best_samples->ne[0];
    int n_batch  = best_samples->ne[1];
    int n_vocab  = logits->ne[0];
    GGML_ASSERT(n_tokens == logits->ne[1]);
    GGML_ASSERT(n_batch  == logits->ne[2]);
    GGML_ASSERT(n_vocab  == probs->ne[0]);
    GGML_ASSERT(n_tokens == probs->ne[1]);
    GGML_ASSERT(n_batch  == probs->ne[2]);

    for (int k = 0; k < n_batch; ++k) {
        struct ggml_tensor * best_samples_k = ggml_view_1d(ctx,
                                                best_samples,
                                                best_samples->ne[0],
                                                k*best_samples->nb[1]);
        struct ggml_tensor * logits_k       = ggml_view_2d(ctx,
                                                logits,
                                                logits->ne[0],
                                                logits->ne[1],
                                                logits->nb[1],
                                                k*logits->nb[2]);
        struct ggml_tensor * probs_k        = ggml_view_2d(ctx,
                                                probs,
                                                probs->ne[0],
                                                probs->ne[1],
                                                probs->nb[1],
                                                k*probs->nb[2]);
        sample_softmax(logits_k, probs_k, best_samples_k);
    }
}


void print_row(struct ggml_tensor * probs, int i) {
    for (int k = 0; k < probs->ne[0]; ++k) {
        float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
        printf(" %.2f", p);
    }
    printf("\n");
}

void print_matrix(struct ggml_tensor * probs) {
    assert(probs->n_dims == 2);
    for (int i = 0; i < probs->ne[1]; ++i) {
        for (int k = 0; k < probs->ne[0]; ++k) {
            float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
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
        int num_newline = 0;
        for (int i0=0; i0<tokens->ne[0]; ++i0) {
            int token = ggml_get_i32_1d(tokens, i0 + i1*tokens->ne[0]);
            bool isnl = (token == llama_token_nl());
            if (isnl) {
                ++num_newline;
            }
            if (!isnl || (num_newline < 2)) {
                print_token(ctx, token);
            }
        }
        printf("\n--\n");
    }
}

void get_example_targets(const llama_token * train_data, size_t n_train_data, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * targets) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab  = targets->ne[0];

    int n_examples = (n_train_data / (size_t) n_tokens);
    int begin = (example_id % n_examples) * n_tokens;
    GGML_ASSERT(begin+n_tokens-1 < n_train_data);

    ggml_set_f32(targets, -1.0f);
    ggml_set_i32_1d(tokens_input, 0, llama_token_bos());
    for (int i=1; i<n_tokens+1; ++i) {
        int token = clamp(train_data[begin+i-1], 0, n_vocab-1);
        ggml_set_f32_1d(targets, (i-1)*n_vocab + token, +1.0f);
        if (i<n_tokens) {
            ggml_set_i32_1d(tokens_input, i, token);
        }
    }
}

void get_example_targets_batch(struct ggml_context * ctx, const llama_token * train_data, size_t n_train_data, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * targets) {
    GGML_ASSERT(tokens_input->n_dims == 2);
    GGML_ASSERT(     targets->n_dims == 3);
    int n_tokens = tokens_input->ne[0];
    int n_batch  = tokens_input->ne[1];
    GGML_ASSERT(n_tokens == targets->ne[1]);
    GGML_ASSERT(n_batch  == targets->ne[2]);

    for (int k=0; k<n_batch; ++k) {
        struct ggml_tensor * tokens_input_k = ggml_view_1d(ctx,
                                                tokens_input,
                                                tokens_input->ne[0],
                                                k*tokens_input->nb[1]);
        struct ggml_tensor * targets_k    = ggml_view_2d(ctx,
                                                targets,
                                                targets->ne[0],
                                                targets->ne[1],
                                                targets->nb[1],
                                                k*targets->nb[2]);
        
        get_example_targets(train_data, n_train_data,
            example_id*n_batch + k, tokens_input_k, targets_k);
    }
}


void lshift_examples(struct ggml_tensor * tokens_input, struct ggml_tensor * targets, int n_shift) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab = targets->ne[0];
    for (int i=0; i<n_tokens-n_shift; ++i) {
        ggml_set_i32_1d(tokens_input, i, ggml_get_i32_1d(tokens_input, i + n_shift));
        for (int k=0; k<n_vocab; ++k) {
            ggml_set_f32_1d(targets, i*n_vocab + k, ggml_get_f32_1d(targets, (i + n_shift)*n_vocab + k));
        }
    }
}

struct ggml_tensor * square_error_loss(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    // todo: instead of a-b: a[1:]-b[:-1]
    return ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, a, b)));
}

struct ggml_tensor * cross_entropy_loss(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b) {
    const float eps = 1e-3;
    return
        ggml_sum(ctx,
            ggml_neg(ctx,
                ggml_sum_rows(ctx,
                    ggml_mul(ctx,
                        ggml_soft_max(ctx, a),
                        ggml_log(ctx,
                            ggml_add1(ctx,
                                ggml_soft_max(ctx, b),
                                ggml_new_f32(ctx, eps)))))));
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
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
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
    buf.resize(f.size);

    f.read_raw(buf.data(), f.size);
    
    out.resize(buf.size());

    int n_tokens = llama_tokenize(lctx, buf.data(), out.data(), buf.size(), false);
    
    if (n_tokens >= 0) { 
        out.resize(n_tokens);
    }

    return n_tokens;
}

int main(int argc, char ** argv) {
    const char * default_model = "ggml-vic7b-uncensored-q4_0.bin";
    const char * default_train = "shakespeare.txt";
    const char * default_argv[3] = {argv[0], default_model, default_train};
    
    if (argc < 3) {
        fprintf(stderr, "usage: %s model training_data\n", argv[0]);
        //return 1;
    }

    const char * fn_model = (argc >= 2) ? argv[1] : default_argv[1];
    const char * fn_train = (argc >= 3) ? argv[2] : default_argv[2];

    struct llama_context_params llama_params = llama_context_default_params();
    llama_params.vocab_only = true;

    struct llama_context * lctx = llama_init_from_file(fn_model, llama_params);

    std::vector<llama_token> train_tokens;
    if (tokenize_file(lctx, fn_train, train_tokens) < 0) {
        fprintf(stderr, "%s: failed to tokenize file '%s'\n", __func__, fn_train);
    }

    struct my_llama_model model;
    model.hparams.n_vocab = llama_n_vocab(lctx);
    model.hparams.n_ctx   = 16;
    model.hparams.n_embd  = 64;
    model.hparams.n_mult  = 8;
    model.hparams.n_head  = 8;
    model.hparams.n_layer = 1;
    model.hparams.n_rot   = std::min(16u, model.hparams.n_embd / model.hparams.n_head);

    struct my_llama_kv_cache kv_self;

    int n_batch = 8;

    struct ggml_init_params lcparams;
    lcparams.mem_size   = 1024ll*1024ll*1024ll;
    lcparams.mem_buffer = NULL;
    lcparams.no_alloc   = false;

    model.ctx = ggml_init(lcparams);
    kv_self.ctx = model.ctx;

    printf("init model\n");
    init_model(&model);
    set_param_model(&model);
    randomize_model(&model, 1337, 0.0f, 1.0f, -1.0f, +1.0f);
    init_kv_cache(&kv_self, &model, n_batch);

    size_t    compute_size = 1024ll*1024ll*1024ll*32ll;
    uint8_t * compute_addr = new uint8_t[compute_size];
    
    int n_examples = 256;
    int n_tokens = model.hparams.n_ctx;
    int n_vocab  = model.hparams.n_vocab;

    for (int ex=0; ex<n_examples; ++ex) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ compute_size,
            /*.mem_buffer =*/ compute_addr,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * after_opt_best_samples  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        struct ggml_tensor * after_opt_probs         = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
        struct ggml_tensor * tokens_input            = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        struct ggml_tensor * targets                 = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);

        int n_past = 0;

        ggml_cgraph gf = {};
        gf.n_threads = 1;

        get_example_targets_batch(ctx0, train_tokens.data(), train_tokens.size(), ex,  tokens_input, targets);

        struct ggml_tensor * logits = forward_batch(&model, &kv_self, ctx0, &gf, tokens_input, n_tokens, n_past, n_batch);
        struct ggml_tensor * e = square_error_loss(ctx0, targets, logits);
        // struct ggml_tensor * e = cross_entropy_loss(ctx0, targets, logits);

        ggml_build_forward_expand(&gf, e);
        ggml_graph_compute(ctx0, &gf);

        float error_before_opt = ggml_get_f32_1d(e, 0);
        
        struct ggml_opt_params opt_params_adam = ggml_opt_default_params(GGML_OPT_ADAM);
        struct ggml_opt_params opt_params_lbfgs = ggml_opt_default_params(GGML_OPT_LBFGS);
        opt_params_adam.print_forward_graph = false;
        opt_params_adam.print_backward_graph = false;
        opt_params_lbfgs.print_forward_graph = false;
        opt_params_lbfgs.print_backward_graph = false;
        opt_params_adam.adam.n_iter = 16;
        opt_params_lbfgs.lbfgs.n_iter = 16;
        ggml_opt(ctx0, opt_params_adam, e);
        // ggml_opt(ctx0, opt_params_lbfgs, e);

        ggml_build_forward_expand(&gf, e);
        ggml_graph_compute(ctx0, &gf);

        float error_after_opt = ggml_get_f32_1d(e, 0);

        if (ex % 1 == 0) {
            printf("Example %d\n", ex);
            printf("error_before_opt: %.2f\n", error_before_opt);
            printf("error_after_opt:  %.2f\n", error_after_opt);
        }

        if (ex % 2 == 0) {
            sample_softmax_batch(ctx0, logits, after_opt_probs, after_opt_best_samples);
            // printf("probabilities after optimization:\n");
            // print_matrix(after_opt_probs);
            printf("Example:\n---\n");
            print_tokens_batch(lctx, tokens_input);
            printf("\n---\n");

            printf("best samples after optimization:\n---\n");
            print_tokens_batch(lctx, after_opt_best_samples);
            printf("\n---\n");
        }

        ggml_free(ctx0);
    }

    {
        int n_gen = 128;
        int sample_ctx = n_tokens - n_tokens/8;

        printf("Generating %d tokens.\n", n_gen);

        struct ggml_tensor * tokens_input = ggml_new_tensor_1d(model.ctx, GGML_TYPE_I32, n_tokens);
        struct ggml_tensor * targets      = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_vocab, n_tokens);

        get_example_targets(train_tokens.data(), train_tokens.size(), 137, tokens_input, targets);
        for (int i=sample_ctx; i<n_tokens; ++i) {
            ggml_set_i32_1d(tokens_input, i, n_vocab/2);
        }

        for (int i=0; i<sample_ctx-1; ++i) {
            print_token(lctx, ggml_get_i32_1d(tokens_input, i));
        }

        printf("---\n");
        for (int i=0; i<n_gen; ++i) {
            struct ggml_init_params params = {
                /*.mem_size   =*/ compute_size,
                /*.mem_buffer =*/ compute_addr,
                /*.no_alloc   =*/ false,
            };
            struct ggml_context * ctx0 = ggml_init(params);

            ggml_cgraph gf = {};
            gf.n_threads = 1;

            int n_past = 0;
            struct ggml_tensor * logits = forward(&model, &kv_self, ctx0, &gf, tokens_input, sample_ctx, n_past);

            ggml_build_forward_expand(&gf, logits);
            ggml_graph_compute(ctx0, &gf);

            struct ggml_tensor * best_samples = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, sample_ctx);
            struct ggml_tensor * probs        = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, sample_ctx);

            sample_softmax(logits, probs, best_samples);

            int token = ggml_get_i32_1d(best_samples, sample_ctx-1);

            // print_row(probs, sample_at);
            print_token(lctx, token);

            lshift_examples(tokens_input, targets, 1);
            ggml_set_i32_1d(tokens_input, 0, 0);
            ggml_set_i32_1d(tokens_input, sample_ctx-1, token);

            ggml_free(ctx0);
        }        
    }

    free(compute_addr);
    ggml_free(model.ctx);

    return 0;
}
