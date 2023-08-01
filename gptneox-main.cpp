#include "ggml.h"

#include "gptneox-common.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// default hparams
struct gpt_neox_hparams {
    size_t n_merges = 0;
    size_t n_vocab  = 0;
    int32_t n_ctx    = 0;
    int32_t n_embd   = 0;
    int32_t n_head   = 0;
    int32_t n_layer  = 0;
    int32_t n_rot    = 0; // rotary_pct * (n_embd / n_head)
    bool par_res = true;
    float norm_eps = 1e-5;
};

struct gpt_neox_layer {
    // pre normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    // attention
    struct ggml_tensor * c_attn_attn_w;
    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    // post normalization
    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    // ff
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gpt_neox_model {
    gpt_neox_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte; // position embedding

    struct ggml_tensor * lmh_g; // language model head

    std::vector<gpt_neox_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct gguf_context * ggufctx;
    struct ggml_context * ctx;
    struct ggml_context * kvctx;

    std::map<std::string, struct ggml_tensor *> tensors;
};

struct ggml_tensor * get_tensor_ex( struct ggml_context * ctx, std::string name){

    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if( cur == NULL ) {
        fprintf(stdout, "%s: tensor '%s' not found!\n", __func__, name.c_str());
    } else {
//        fprintf(stdout, "%s: n_dims = %d, name = '%s'\n", __func__, cur->n_dims, cur->name);
    }

    return cur;
}

// load the model's weights from a file
bool gpt_neox_model_load(const std::string & fname, gpt_neox_model & model, gpt_vocab & vocab) {
    printf("%s: loading model from '%s'..\n", __func__, fname.c_str());

    model.ctx = NULL;

    struct gguf_init_params ggufparams = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &model.ctx,
    };

    auto & ggufctx = model.ggufctx;

    ggufctx  = gguf_init_from_file(fname.c_str(), ggufparams);

    if (!ggufctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    fprintf(stdout, "%s: gguf version     = %d\n", __func__, gguf_get_version(ggufctx));
    fprintf(stdout, "%s: gguf alignment   = %zu\n", __func__, gguf_get_alignment(ggufctx));
    fprintf(stdout, "%s: gguf data offset = %zu\n", __func__, gguf_get_data_offset(ggufctx));

    // print all kv
    if( false )
    {
        const int n_kv = gguf_get_n_kv(ggufctx);

        fprintf(stdout, "%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ggufctx, i);

            fprintf(stdout, "%s: kv[%d]: key = %s\n", __func__, i, key);
        }
    }

    // print some standard metadata
    {
        int keyidx;

        keyidx = gguf_find_key(ggufctx, "general.name");
        if (keyidx != -1) { fprintf(stdout, "%s: model name         = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "general.description");
        if (keyidx != -1) { fprintf(stdout, "%s: model description  = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "general.author");
        if (keyidx != -1) { fprintf(stdout, "%s: model author       = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "general.license");
        if (keyidx != -1) { fprintf(stdout, "%s: model license      = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "general.architecture");
        if (keyidx != -1) { fprintf(stdout, "%s: model architecture = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
    }

    // check required metadata
    {
        int keyidx;

        keyidx = gguf_find_key(ggufctx, "general.architecture");
        if (keyidx != -1) {
            if ( strcmp(gguf_get_val_str(ggufctx, keyidx), "gptneox") != 0) {
                fprintf(stdout, "%s: model architecture not supported!\n", __func__);
                return false;
            }
        } else {
            fprintf(stdout, "%s: gguf model architecture not found!\n", __func__);
            return false;
        }

    }

    // load hparams
    {
        auto & hparams = model.hparams;

        bool ok = true;
        int keyidx;

        if (ok) { keyidx = gguf_find_key(ggufctx, "gptneox.context_length");
                  if (keyidx != -1) { hparams.n_ctx = gguf_get_val_u32(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "gptneox.embedding_length");
                  if (keyidx != -1) { hparams.n_embd = gguf_get_val_u32(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "gptneox.attention.head_count");
                  if (keyidx != -1) { hparams.n_head = gguf_get_val_u32(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "gptneox.layer_count");
                  if (keyidx != -1) { hparams.n_layer = gguf_get_val_u32(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "gptneox.rope.dimension_count");
                  if (keyidx != -1) { hparams.n_rot = gguf_get_val_u32(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "gptneox.use_parallel_residual");
                  if (keyidx != -1) { hparams.par_res = gguf_get_val_bool(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "gptneox.attention.layer_norm_epsilon");
                  if (keyidx != -1) { hparams.norm_eps= gguf_get_val_f32(ggufctx, keyidx); } else { ok = false; }  }

        if (!ok) {
            fprintf(stderr, "%s: required hparam missing!\n", __func__);
            return false;
        }

        printf("%s: n_ctx    = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd   = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head   = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer  = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot    = %d\n", __func__, hparams.n_rot);
        printf("%s: par_res  = %d\n", __func__, hparams.par_res);
        printf("%s: norm_eps = %g\n", __func__, hparams.norm_eps);

    }

    // load vocab
    {

        // TODO: implement a better bpe tokenizer, utilizing merges and handles unicode

        auto & hparams = model.hparams;

        int keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.model");

        if (keyidx != -1) {
            if ( strcmp(gguf_get_val_str(ggufctx, keyidx), "gpt2") != 0) {
                fprintf(stdout, "%s: tokenizer model not supported!\n", __func__);
                return false;
            }
        } else {
            fprintf(stdout, "%s: tokenizer model not found!\n", __func__);
            return false;
        }


        int tokens_keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.tokens");

        if (tokens_keyidx == -1) {
            fprintf(stdout, "%s: gpt2 tokenizer vocab not found!\n", __func__);
            return false;
        }

        int merges_keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.merges");

        if (merges_keyidx == -1) {
            fprintf(stdout, "%s: gpt2 tokenizer merges not found!\n", __func__);
            return false;
        }

        hparams.n_vocab = gguf_get_arr_n(ggufctx,tokens_keyidx);
        hparams.n_merges = gguf_get_arr_n(ggufctx,merges_keyidx);

        fprintf(stdout, "%s: gpt2 tokenizer vocab  = %zu\n", __func__, hparams.n_vocab);
        fprintf(stdout, "%s: gpt2 tokenizer merges = %zu\n", __func__, hparams.n_merges);

        for (size_t i = 0; i < hparams.n_vocab; i++) {
            std::string word = gguf_get_arr_str(ggufctx, tokens_keyidx, i);


            // TEMP until a better bpe tokenizer is implemented
            word = replace(word, "Ġ", " ");
            word = replace(word, "Ċ", "\n");


            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }

        keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.bos_token_id"); if( keyidx != -1 ) {       printf("bos id = %d\n", gguf_get_val_u32(ggufctx, keyidx) ); }
        keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.eos_token_id"); if( keyidx != -1 ) {       printf("eos id = %d\n", gguf_get_val_u32(ggufctx, keyidx) ); }
        keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.unknown_token_id"); if( keyidx != -1 ) {   printf("unk id = %d\n", gguf_get_val_u32(ggufctx, keyidx) ); }
        keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.separator_token_id"); if( keyidx != -1 ) { printf("sep id = %d\n", gguf_get_val_u32(ggufctx, keyidx) ); }
        keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.padding_token_id"); if( keyidx != -1 ) {   printf("pad id = %d\n", gguf_get_val_u32(ggufctx, keyidx) ); }

    }


    auto & ctx = model.ctx;
    size_t ctx_size = ggml_get_mem_size(ctx);

    printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));

    // print tensor info
    if( false )
    {
        const int n_tensors = gguf_get_n_tensors(ggufctx);

        fprintf(stdout, "%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ggufctx, i);
            const size_t offset = gguf_get_tensor_offset(ggufctx, i);

            fprintf(stdout, "%s: tensor[%d]: name = %s, offset = %zu\n", __func__, i, name, offset);
        }
    }


    // prepare memory for the weights
    {
        const int n_layer = model.hparams.n_layer;

        model.layers.resize(n_layer);

        model.wte    = ggml_get_tensor(ctx, "gpt_neox.embed_in.weight");
        model.ln_f_g = ggml_get_tensor(ctx, "gpt_neox.final_layer_norm.weight");
        model.ln_f_b = ggml_get_tensor(ctx, "gpt_neox.final_layer_norm.bias");
        model.lmh_g  = ggml_get_tensor(ctx, "embed_out.weight");

        // map by name
        model.tensors["gpt_neox.embed_in.weight"] = model.wte;
        model.tensors["gpt_neox.final_layer_norm.weight"] = model.ln_f_g;
        model.tensors["gpt_neox.final_layer_norm.bias"]   = model.ln_f_b;
        model.tensors["embed_out.weight"] = model.lmh_g;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.ln_1_g          = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".input_layernorm.weight" );
            layer.ln_1_b          = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".input_layernorm.bias" );

            layer.c_attn_attn_w   = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.weight" );
            layer.c_attn_attn_b   = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.bias" );

            layer.c_attn_proj_w   = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".attention.dense.weight" );
            layer.c_attn_proj_b   = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".attention.dense.bias" );

            layer.ln_2_g          = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.weight" );
            layer.ln_2_b          = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.bias");

            layer.c_mlp_fc_w      = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.weight" );
            layer.c_mlp_fc_b      = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.bias" );

            layer.c_mlp_proj_w    = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.weight" );
            layer.c_mlp_proj_b    = get_tensor_ex(ctx, "gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.bias" );

            // map by name
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.weight"] = layer.ln_1_g;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.bias"]   = layer.ln_1_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.weight"] = layer.c_attn_attn_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.bias"]   = layer.c_attn_attn_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.weight"] = layer.c_attn_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.bias"]   = layer.c_attn_proj_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.weight"] = layer.ln_2_g;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.bias"]   = layer.ln_2_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.weight"] = layer.c_mlp_fc_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.bias"]   = layer.c_mlp_fc_b;

            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.weight"] = layer.c_mlp_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.bias"]   = layer.c_mlp_proj_b;
        }
    }

    // key + value memory
    {
        const auto & kvctx = model.kvctx;
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int64_t n_mem      = n_layer*n_ctx;
        const int64_t n_elements = n_embd*n_mem;

        // create the ggml context
        {
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_t(n_elements*4+ggml_tensor_overhead()*2),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ false,
            };

            model.kvctx = ggml_init(params);
            if (!model.kvctx) {
                fprintf(stderr, "%s: kv ggml_init() failed\n", __func__);
                return false;
            }

        }


        model.memory_k = ggml_new_tensor_1d(kvctx, GGML_TYPE_F16, n_elements);
        model.memory_v = ggml_new_tensor_1d(kvctx, GGML_TYPE_F16, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %" PRId64 "\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    return true;
}


// feed-forward network
ggml_tensor * gpt_neox_ff(
        const gpt_neox_layer &layer,
        ggml_context * ctx0,
        ggml_tensor * inp) {
    ggml_tensor * cur = ggml_norm(ctx0, inp);

    cur = ggml_add(ctx0,
        ggml_mul(ctx0,
            ggml_repeat(ctx0, layer.ln_2_g, cur),
            cur),
        ggml_repeat(ctx0, layer.ln_2_b, cur));

    cur = ggml_mul_mat(ctx0,
            layer.c_mlp_fc_w,
            cur);

    cur = ggml_add(ctx0,
            ggml_repeat(ctx0, layer.c_mlp_fc_b, cur),
            cur);

    // GELU activation
    cur = ggml_gelu(ctx0, cur);

    // projection
    // cur = proj_w*cur + proj_b
    cur = ggml_mul_mat(ctx0,
            layer.c_mlp_proj_w,
            cur);

    cur = ggml_add(ctx0,
            ggml_repeat(ctx0, layer.c_mlp_proj_b, cur),
            cur);
    return cur;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool gpt_neox_eval(
        const gpt_neox_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_rot;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    // use 2 scratch buffers
    // TODO: very hacky solution - reimplement in a more elegant way
    static size_t scr0_size = 256u*1024*1024;
    static void * scr0 = malloc(scr0_size);

    static size_t scr1_size = 256u*1024*1024;
    static void * scr1 = malloc(scr1_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));


    // wte
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.wte, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        ggml_set_scratch(ctx0, { 0, scr0_size, scr0, });

        // self-attention
        {
            {
                cur = ggml_norm(ctx0, inpL);

                cur = ggml_add(ctx0,
                        ggml_mul(ctx0,
                            ggml_repeat(ctx0, model.layers[il].ln_1_g, cur),
                            cur),
                        ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
            }

            // compute QKV
            {

                cur = ggml_mul_mat(ctx0,
                        model.layers[il].c_attn_attn_w,
                        cur);

                cur = ggml_add(ctx0,
                        ggml_repeat(ctx0, model.layers[il].c_attn_attn_b, cur),
                        cur);
            }

            struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd/n_head, n_head, N, cur->nb[1]/n_head, cur->nb[1], 0*sizeof(float)*n_embd/n_head));
            struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd/n_head, n_head, N, cur->nb[1]/n_head, cur->nb[1], 1*sizeof(float)*n_embd/n_head));
            struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd/n_head, n_head, N, cur->nb[1]/n_head, cur->nb[1], 2*sizeof(float)*n_embd/n_head));

            // using mode = 2 for GPT-NeoX mode
            Qcur = ggml_rope_inplace(ctx0, Qcur, n_past, n_rot, 2, 0);
            Kcur = ggml_rope_inplace(ctx0, Kcur, n_past, n_rot, 2, 0);

            // store key and value to memory
            {
                Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd, N));

                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_2d(ctx0, model.memory_v, N, n_embd,
                        (   n_ctx)*ggml_element_size(model.memory_v),
                        (il*n_ctx)*ggml_element_size(model.memory_v)*n_embd + n_past*ggml_element_size(model.memory_v));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, model.memory_v,
                        n_past + N, n_embd/n_head, n_head,
                        n_ctx*ggml_element_size(model.memory_v),
                        n_ctx*ggml_element_size(model.memory_v)*n_embd/n_head,
                        il*n_ctx*ggml_element_size(model.memory_v)*n_embd);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection
            {
                cur = ggml_mul_mat(ctx0,
                        model.layers[il].c_attn_proj_w,
                        cur);

                cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_attn_proj_b, cur), cur);
            }
        }

        ggml_set_scratch(ctx0, { 0, scr1_size, scr1, });

        if (hparams.par_res == 0) {
            struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpL);

            cur = gpt_neox_ff(model.layers[il], ctx0, inpFF);

            // input for next layer
            inpL = ggml_add(ctx0, cur, inpFF);
        } else {
            struct ggml_tensor * inpFF = cur;

            // this is independent of the self-attention result, so it could be done in parallel to the self-attention
            // note here we pass inpL instead of cur
            cur = gpt_neox_ff(model.layers[il], ctx0, inpL);

            // layer input + FF
            cur  = ggml_add(ctx0, cur, inpFF);

            // input for next layer
            inpL = ggml_add(ctx0, cur, inpL);
        }
    }

    ggml_set_scratch(ctx0, { 0, scr0_size, scr0, });

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model.ln_f_b, inpL));
    }

    ggml_set_scratch(ctx0, { 0, 0, nullptr, });

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.lmh_g, inpL);

        //inpL = ggml_add(ctx0,
        //        ggml_repeat(ctx0, model.lmh_b, inpL),
        //        inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max_inplace(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    gpt_neox_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!gpt_neox_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;

    }

    uint32_t eos_token_id = 0;
    int keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.eos_token_id"); if( keyidx != -1 ) {  eos_token_id = gguf_get_val_u32(ggufctx, keyidx); }    

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    for (int i = 0; i < embd_inp.size(); i++) {
        printf("%s: token[%d] = %6d, %s\n", __func__, i, embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    }
    printf("\n");

    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    gpt_neox_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!gpt_neox_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == eos_token_id) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
