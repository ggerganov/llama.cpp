#include "ggml.h"
#include <vector>
#include <assert.h>
#include <random>

float frand() {
    return (float)rand()/(float)RAND_MAX;
}

struct random_normal_distribution {
    std::mt19937 gen;
    std::normal_distribution<float> nd;
    float min;
    float max;
};

void init_random_normal_distribution(struct random_normal_distribution * rnd, int seed, float mean, float std, float min, float max) {
    rnd->gen = std::mt19937(seed);
    rnd->nd = std::normal_distribution<float>{mean, std};
    rnd->min = min;
    rnd->max = max;
}

float frand_normal(struct random_normal_distribution * rnd) {
    const float r = rnd->nd(rnd->gen);
    return ((r < rnd->min) ? (rnd->min) : (r > rnd->max) ? (rnd->max) : r);
}

struct ggml_tensor * randomize_tensor(
        struct ggml_tensor * tensor,
        int ndims,
        int64_t ne[],
        float fmin,
        float fmax) {

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)tensor->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)tensor->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)tensor->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)tensor->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
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

struct ggml_tensor * randomize_tensor_normal(
        struct ggml_tensor * tensor,
        int ndims,
        int64_t ne[],
        struct random_normal_distribution * rnd) {
    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)tensor->data)[i0] = frand_normal(rnd);
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)tensor->data)[i1*ne[0] + i0] = frand_normal(rnd);
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)tensor->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand_normal(rnd);
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)tensor->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand_normal(rnd);
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

struct llama_hparams {
    uint32_t n_vocab = 32000;
    uint32_t n_ctx   = 512;   // this is provided as user input?
    uint32_t n_embd  = 4096;
    uint32_t n_mult  = 4;
    uint32_t n_head  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;

    bool operator!=(const llama_hparams & other) const {
        return memcmp(this, &other, sizeof(llama_hparams));
    }
};

struct llama_layer {
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


struct llama_kv_cache {
    struct ggml_context * ctx = NULL;

    struct ggml_tensor * k;
    struct ggml_tensor * v;

    // llama_ctx_buffer buf;

    int n; // number of tokens currently in the cache
};

struct llama_model {
    struct ggml_context * ctx = NULL;

    llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;
};

void init_model(struct llama_model * model) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;

    uint32_t n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;

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
        layer.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff,   n_embd);   // (layers_i + ".feed_forward.w2.weight", {  n_ff,   n_embd});
        layer.w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);     // (layers_i + ".feed_forward.w3.weight", {n_embd,   n_ff});
    }
}

void set_param_model(struct llama_model * model) {
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

void randomize_model(struct llama_model * model, int seed, float mean, float std, float min, float max) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;

    uint32_t n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;

    struct random_normal_distribution rnd;
    init_random_normal_distribution(&rnd, seed, mean, std, min, max);
    randomize_tensor_normal(model->tok_embeddings, model->tok_embeddings->n_dims, model->tok_embeddings->ne, &rnd);
    randomize_tensor_normal(model->norm,           model->norm->n_dims,           model->norm->ne,           &rnd);
    randomize_tensor_normal(model->output,         model->output->n_dims,         model->output->ne,         &rnd);
    
    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];
        randomize_tensor_normal(layer.attention_norm, layer.attention_norm->n_dims, layer.attention_norm->ne, &rnd);

        randomize_tensor_normal(layer.wq, layer.wq->n_dims, layer.wq->ne, &rnd);
        randomize_tensor_normal(layer.wk, layer.wk->n_dims, layer.wk->ne, &rnd);
        randomize_tensor_normal(layer.wv, layer.wv->n_dims, layer.wv->ne, &rnd);
        randomize_tensor_normal(layer.wo, layer.wo->n_dims, layer.wo->ne, &rnd);

        randomize_tensor_normal(layer.ffn_norm, layer.ffn_norm->n_dims, layer.ffn_norm->ne, &rnd);

        randomize_tensor_normal(layer.w1, layer.w1->n_dims, layer.w1->ne, &rnd);
        randomize_tensor_normal(layer.w2, layer.w2->n_dims, layer.w2->ne, &rnd);
        randomize_tensor_normal(layer.w3, layer.w3->n_dims, layer.w3->ne, &rnd);
    }
}

bool init_kv_cache(struct llama_kv_cache* cache, struct llama_model * model) {
    const auto & hparams = model->hparams;
    const int n_ctx   = hparams.n_ctx;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx;
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
        struct llama_model    * model, 
        struct llama_kv_cache * cache, 
        struct ggml_context   * ctx0,
        struct ggml_cgraph    * gf,
        struct ggml_tensor    * tokens_input,
        const  int              n_tokens,
        const  int              n_past) {
    
    const int N = n_tokens;

    struct llama_kv_cache& kv_self = *cache;
    const auto & hparams = model->hparams;
    const int n_ctx   = hparams.n_ctx;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;

    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(tokens->data, tokens_input->data, N*ggml_element_size(tokens));

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
            struct ggml_tensor * Qcur = ggml_rope_inplace(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wq, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0);
            struct ggml_tensor * Kcur = ggml_rope_inplace(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wk, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0);

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix
                // wv   shape [n_embd, n_embd, 1, 1]
                // Vcur shape [n_embd, N, 1, 1]
                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, ggml_mul_mat(ctx0, model->layers[il].wv, cur), n_embd, N));

                // kv_self.k shape [n_embd * n_ctx * n_layer, 1] 
                // kv_self.v shape [n_embd * n_ctx * n_layer, 1] 
                // k         shape [n_embd * N, 1]   == kv_self.k[:,n_past:n_past+N,il,0]
                // v         shape [N, n_embd, 1, 1] == kv_self.v[:,n_past:n_past+N,il,0]
                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));

                // important: storing RoPE-ed version of K in the KV cache!
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
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
                            ggml_view_1d(ctx0, kv_self.k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kv_self.k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            // KQ shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head)));

            // KQ_masked = mask_past(KQ_scaled)
            // KQ_masked shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // KQ_soft_max shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

            // split cached V into n_head heads
            //// V shape [n_past + N, n_embd/n_head, n_head, 1]
            // V shape [n_past + N, n_embd/n_head, n_head, 1] == kv_self.v[:,:(n_past+N),il,1]
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_past + N, n_embd/n_head, n_head,
                        n_ctx*ggml_element_size(kv_self.v),
                        n_ctx*ggml_element_size(kv_self.v)*n_embd/n_head,
                        il*n_ctx*ggml_element_size(kv_self.v)*n_embd);

            // KQV shape [n_embd/n_head, N, n_head, 1]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // KQV_merged shape [n_embd/n_head, n_head, N, 1]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            // KQV_merged shape 

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // cur shape [n_embd,N,1,1]
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model->layers[il].wo,
                    cur);
        }

        // lctx.use_buf(ctx0, 1);

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF);

                // cur = ffn_norm*cur
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model->layers[il].ffn_norm, cur),
                        cur);
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model->layers[il].w3,
                    cur);

            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w1,
                    cur);

            // SILU activation
            cur = ggml_silu(ctx0, cur);

            cur = ggml_mul(ctx0, cur, tmp);

            cur = ggml_mul_mat(ctx0,
                    model->layers[il].w2,
                    cur);
        }

        cur = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    // norm
    {

        inpL = ggml_rms_norm(ctx0, inpL);

        // inpL = norm*inpL
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model->norm, inpL),
                    inpL);

        //embeddings = inpL;
    }

    // lm_head
    inpL = ggml_mul_mat(ctx0, model->output, inpL);

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
    for (int i=0; i< logits->ne[1]; ++i) {
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

void print_probs1(struct ggml_tensor * probs, int i) {
    for (int k = 0; k < probs->ne[0]; ++k) {
        float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
        printf(" %.2f", p);
    }
    printf("\n");
}

void print_probs(struct ggml_tensor * probs) {
    assert(probs->n_dims == 2);
    for (int i=0; i<probs->ne[1]; ++i) {
        for (int k = 0; k < probs->ne[0]; ++k) {
            float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
            printf(" %.2f", p);
        }
        printf("\n");
    }
}

void print_token(int token, int n_vocab) {
    for (int k = 0; k < token; ++k) {
        printf(" ");
    }
    printf("X");
    for (int k = token+1; k < n_vocab; ++k) {
        printf(" ");
    }
    printf("\n");
}

void print_tokens(struct ggml_tensor * tokens, int n_vocab) {
    for (int i=0; i<tokens->ne[0]; ++i) {
        int token = ggml_get_i32_1d(tokens, i);
        print_token(token, n_vocab);
    }
}

void get_example_targets(int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * targets) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab = targets->ne[0];
    float randomness = 0.0f;
    ggml_set_zero(targets);
    for (int i=0; i<n_tokens; ++i) {
        float x = example_id + i * 3.14159f * 2.0f * 4.0f / n_tokens;
        float y = sinf(x);//*cosf(x*1.1f+1.0f);
        float z = (y+1.0f)*0.5f; // scale to [0..1]
        z += (frand()-0.5f)*(randomness/n_tokens);
        z = (z < 0.0f) ? 0.0f : (z > 1.0f) ? 1.0f : z; // clamp to [0..1]
        int token = (int)(z*(float)(n_vocab-1));
        ggml_set_f32_1d(targets, i*n_vocab + token, +1.0f);
        ggml_set_i32_1d(tokens_input, i, token);
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

int main(int argc, char ** argv) {
    struct ggml_init_params lcparams;
    lcparams.mem_size   = 1024*1024*1024;
    lcparams.mem_buffer = NULL;
    lcparams.no_alloc   = false;

    struct llama_model model;
    model.hparams.n_vocab = 8;
    model.hparams.n_ctx   = 32;
    model.hparams.n_embd  = 32;
    model.hparams.n_mult  = 2;
    model.hparams.n_head  = 8;
    model.hparams.n_layer = 8;
    model.hparams.n_rot   = model.hparams.n_embd / model.hparams.n_head;

    // model.hparams.n_embd  = 32;
    // model.hparams.n_mult  = 2;
    // model.hparams.n_head  = 4;
    // model.hparams.n_layer = 8;
    // model.hparams.n_rot   = 8;

    model.ctx = ggml_init(lcparams);
    printf("init model\n");
    init_model(&model);
    set_param_model(&model);

    randomize_model(&model, 1337, 0.0f, 1.0f, -1.0f, +1.0f);

    // key + value cache for the self attention
    struct llama_kv_cache kv_self;    
    printf("init_kv_cache\n");
    kv_self.ctx = model.ctx;
    init_kv_cache(&kv_self, &model);


    size_t    compute_size = 1024*1024*1024;
    uint8_t * compute_addr = new uint8_t[compute_size];

    int n_examples = 32;
    int n_tokens = model.hparams.n_ctx;

    for (int ex=0; ex<n_examples; ++ex) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ compute_size,
            /*.mem_buffer =*/ compute_addr,
            /*.no_alloc   =*/ false,
        };

        struct ggml_context * ctx0 = ggml_init(params);


        // struct ggml_tensor * before_opt_best_samples = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        // struct ggml_tensor * before_opt_probs        = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, model.hparams.n_vocab, n_tokens);
        // struct ggml_tensor * after_opt_best_samples  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        // struct ggml_tensor * after_opt_probs         = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, model.hparams.n_vocab, n_tokens);
        struct ggml_tensor * tokens_input            = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        struct ggml_tensor * targets                 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, model.hparams.n_vocab, n_tokens);

        int n_past = 0;

        ggml_cgraph gf = {};
        gf.n_threads = 1;

        get_example_targets(ex, tokens_input, targets);
        printf("Example %d\n", (ex+1));
        // print_probs(targets);
        // print_tokens(tokens_input, model.hparams.n_vocab);

        struct ggml_tensor * logits = forward(&model, &kv_self, ctx0, &gf, tokens_input, n_tokens, n_past);
        struct ggml_tensor * e = ggml_sum(ctx0, ggml_sqr(ctx0, ggml_sub(ctx0, targets, logits)));

        ggml_build_forward_expand(&gf, e);
        ggml_graph_compute(ctx0, &gf);

        float error_before_opt = ggml_get_f32_1d(e, 0);
        // sample_softmax(logits, before_opt_probs, before_opt_best_samples);

        // printf("probabilities before optimization:\n");
        // print_probs(before_opt_probs);
        // printf("best samples before optimization:\n");
        // print_tokens(before_opt_best_samples, model.hparams.n_vocab);

        struct ggml_opt_params opt_params_adam = ggml_opt_default_params(GGML_OPT_ADAM);
        struct ggml_opt_params opt_params_lbfgs = ggml_opt_default_params(GGML_OPT_LBFGS);
        opt_params_adam.print_forward_graph = false;
        opt_params_adam.print_backward_graph = false;
        opt_params_lbfgs.print_forward_graph = false;
        opt_params_lbfgs.print_backward_graph = false;
        opt_params_adam.adam.n_iter = 16;
        opt_params_lbfgs.lbfgs.n_iter = 16;
        // ggml_opt(ctx0, opt_params_adam, e);
        ggml_opt(ctx0, opt_params_lbfgs, e);
        // 
        ggml_build_forward_expand(&gf, e);
        ggml_graph_compute(ctx0, &gf);

        float error_after_opt = ggml_get_f32_1d(e, 0);
        // sample_softmax(logits, after_opt_probs, after_opt_best_samples);

        printf("error_before_opt: %.2f\n", error_before_opt);
        printf("error_after_opt:  %.2f\n", error_after_opt);

        // printf("probabilities after optimization:\n");
        // print_probs(after_opt_probs);
        // printf("best samples after optimization:\n");
        // print_tokens(after_opt_best_samples, model.hparams.n_vocab);

        ggml_free(ctx0);
    }

    {
        int n_gen = 64;
        int sample_ctx = n_tokens/2;

        printf("Generating %d tokens.\n", n_gen);

        struct ggml_tensor * tokens_input = ggml_new_tensor_1d(model.ctx, GGML_TYPE_I32, n_tokens);
        struct ggml_tensor * targets      = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, model.hparams.n_vocab, n_tokens);
        
        get_example_targets(137, tokens_input, targets);
        for (int i=sample_ctx; i<n_tokens; ++i) {
            ggml_set_i32_1d(tokens_input, i, model.hparams.n_vocab/2);
        }

        for (int i=0; i<sample_ctx-1; ++i) {
            print_token(ggml_get_i32_1d(tokens_input, i), model.hparams.n_vocab);
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
            struct ggml_tensor * probs        = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, model.hparams.n_vocab, sample_ctx);

            sample_softmax(logits, probs, best_samples);

            // int sample_at = n_tokens-1;
            int token = ggml_get_i32_1d(best_samples, sample_ctx-1);
            
            // print_probs1(probs, sample_at);
            print_token(token, model.hparams.n_vocab);

            lshift_examples(tokens_input, targets, 1);
            ggml_set_i32_1d(tokens_input, sample_ctx-1, token);
            
            // printf("---\n");
            // for (int i=0; i<sample_ctx-1; ++i) {
            //     print_token(ggml_get_i32_1d(tokens_input, i), model.hparams.n_vocab);
            // }
            // printf("--\n");

            ggml_free(ctx0);
        }
    }

    printf("done\n");
    // ggml_free(kv_self.ctx);
    // ggml_free(model.ctx);
    return 0;
}
