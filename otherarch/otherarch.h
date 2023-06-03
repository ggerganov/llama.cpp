#pragma once

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "utils.h"
#include "model_adapter.h"


// default hparams (GPT-J 6B)
struct gptj_hparams {
    int32_t n_vocab = 50400;
    int32_t n_ctx   = 2048;
    int32_t n_embd  = 4096;
    int32_t n_head  = 16;
    int32_t n_layer = 28;
    int32_t n_rot   = 64;
    int32_t ftype   = 1;
};

struct gptj_layer {
    // normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    // attention
    struct ggml_tensor * c_attn_q_proj_w;
    struct ggml_tensor * c_attn_k_proj_w;
    struct ggml_tensor * c_attn_v_proj_w;

    struct ggml_tensor * c_attn_proj_w;

    // ff
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_w_trans; //for backwards compatibility
    struct ggml_tensor * c_mlp_proj_b;
};
struct gptj_layer_v2 {
    // normalization
    struct ggml_v2_tensor * ln_1_g;
    struct ggml_v2_tensor * ln_1_b;

    // attention
    struct ggml_v2_tensor * c_attn_q_proj_w;
    struct ggml_v2_tensor * c_attn_k_proj_w;
    struct ggml_v2_tensor * c_attn_v_proj_w;

    struct ggml_v2_tensor * c_attn_proj_w;

    // ff
    struct ggml_v2_tensor * c_mlp_fc_w;
    struct ggml_v2_tensor * c_mlp_fc_b;

    struct ggml_v2_tensor * c_mlp_proj_w;
    struct ggml_v2_tensor * c_mlp_proj_w_trans; //for backwards compatibility
    struct ggml_v2_tensor * c_mlp_proj_b;
};
struct gptj_layer_v1 {
    // normalization
    struct ggml_v1_tensor * ln_1_g;
    struct ggml_v1_tensor * ln_1_b;

    // attention
    struct ggml_v1_tensor * c_attn_q_proj_w;
    struct ggml_v1_tensor * c_attn_k_proj_w;
    struct ggml_v1_tensor * c_attn_v_proj_w;

    struct ggml_v1_tensor * c_attn_proj_w;

    // ff
    struct ggml_v1_tensor * c_mlp_fc_w;
    struct ggml_v1_tensor * c_mlp_fc_b;

    struct ggml_v1_tensor * c_mlp_proj_w;
    struct ggml_v1_tensor * c_mlp_proj_w_trans; //for backwards compatibility
    struct ggml_v1_tensor * c_mlp_proj_b;
};

struct gptj_v1_model {
    gptj_hparams hparams;

    // normalization
    struct ggml_v1_tensor * ln_f_g;
    struct ggml_v1_tensor * ln_f_b;

    struct ggml_v1_tensor * wte; // position embedding

    struct ggml_v1_tensor * lmh_g; // language model head
    struct ggml_v1_tensor * lmh_b; // language model bias

    std::vector<gptj_layer_v1> layers;

    // key + value memory
    struct ggml_v1_tensor * memory_k;
    struct ggml_v1_tensor * memory_v;

    //
    struct ggml_v1_context * ctx;
    std::map<std::string, struct ggml_v1_tensor *> tensors;
};

struct gptj_v2_model {
    gptj_hparams hparams;

    // normalization
    struct ggml_v2_tensor * ln_f_g;
    struct ggml_v2_tensor * ln_f_b;

    struct ggml_v2_tensor * wte; // position embedding

    struct ggml_v2_tensor * lmh_g; // language model head
    struct ggml_v2_tensor * lmh_b; // language model bias

    std::vector<gptj_layer_v2> layers;

    // key + value memory
    struct ggml_v2_tensor * memory_k;
    struct ggml_v2_tensor * memory_v;

    //
    struct ggml_v2_context * ctx;
    std::map<std::string, struct ggml_v2_tensor *> tensors;
};

struct gptj_model {
    gptj_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte; // position embedding

    struct ggml_tensor * lmh_g; // language model head
    struct ggml_tensor * lmh_b; // language model bias

    std::vector<gptj_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// default hparams (GPT-2 117M)
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 1024;
    int32_t n_embd  = 768;
    int32_t n_head  = 12;
    int32_t n_layer = 12;
    int32_t ftype     = 1;
};

struct gpt2_v1_layer {
    // normalization
    struct ggml_v1_tensor * ln_1_g;
    struct ggml_v1_tensor * ln_1_b;

    struct ggml_v1_tensor * ln_2_g;
    struct ggml_v1_tensor * ln_2_b;

    // attention
    struct ggml_v1_tensor * c_attn_attn_w;
    struct ggml_v1_tensor * c_attn_attn_b;

    struct ggml_v1_tensor * c_attn_proj_w;
    struct ggml_v1_tensor * c_attn_proj_b;

    // mlp
    struct ggml_v1_tensor * c_mlp_fc_w;
    struct ggml_v1_tensor * c_mlp_fc_b;

    struct ggml_v1_tensor * c_mlp_proj_w_trans; // transposed for efficiency
    struct ggml_v1_tensor * c_mlp_proj_b;
};

struct gpt2_v1_model {
    gpt2_hparams hparams;

    // normalization
    struct ggml_v1_tensor * ln_f_g;
    struct ggml_v1_tensor * ln_f_b;

    struct ggml_v1_tensor * wte; // position embedding
    struct ggml_v1_tensor * wpe; //    token embedding

    std::vector<gpt2_v1_layer> layers;

    // key + value memory
    struct ggml_v1_tensor * memory_k;
    struct ggml_v1_tensor * memory_v;

    //
    struct ggml_v1_context * ctx;
    std::map<std::string, struct ggml_v1_tensor *> tensors;
};

struct gpt2_layer_v2 {
    // normalization
    struct ggml_v2_tensor * ln_1_g;
    struct ggml_v2_tensor * ln_1_b;

    struct ggml_v2_tensor * ln_2_g;
    struct ggml_v2_tensor * ln_2_b;

    // attention
    struct ggml_v2_tensor * c_attn_attn_w;
    struct ggml_v2_tensor * c_attn_attn_b;

    struct ggml_v2_tensor * c_attn_proj_w;
    struct ggml_v2_tensor * c_attn_proj_b;

    // mlp
    struct ggml_v2_tensor * c_mlp_fc_w;
    struct ggml_v2_tensor * c_mlp_fc_b;

    struct ggml_v2_tensor * c_mlp_proj_w;
    struct ggml_v2_tensor * c_mlp_proj_b;
};

struct gpt2_v2_model {
    gpt2_hparams hparams;

    // normalization
    struct ggml_v2_tensor * ln_f_g;
    struct ggml_v2_tensor * ln_f_b;

    struct ggml_v2_tensor * wte;     // position embedding
    struct ggml_v2_tensor * wpe;     //    token embedding
    struct ggml_v2_tensor * lm_head; // language model head

    std::vector<gpt2_layer_v2> layers;

    // key + value memory
    struct ggml_v2_tensor * memory_k;
    struct ggml_v2_tensor * memory_v;

    //
    struct ggml_v2_context * ctx;
    std::map<std::string, struct ggml_v2_tensor *> tensors;
};

struct gpt2_layer {
    // normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    // attention
    struct ggml_tensor * c_attn_attn_w;
    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    // mlp
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gpt2_model {
    gpt2_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte;     // position embedding
    struct ggml_tensor * wpe;     //    token embedding
    struct ggml_tensor * lm_head; // language model head

    std::vector<gpt2_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// default hparams (StableLM 3B)
struct gpt_neox_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 4096;
    int32_t n_embd  = 4096;
    int32_t n_head  = 32;
    int32_t n_layer = 16;
    int32_t n_rot   = 32; // rotary_pct * (n_embd / n_head)
    int32_t par_res = 1; // 1 = true, 0 = false
    int32_t ftype   = 1;
};

struct gpt_neox_layer_v2 {
    // pre normalization
    struct ggml_v2_tensor * ln_1_g;
    struct ggml_v2_tensor * ln_1_b;

    // attention
    struct ggml_v2_tensor * c_attn_attn_w;
    struct ggml_v2_tensor * c_attn_attn_b;

    struct ggml_v2_tensor * c_attn_proj_w;
    struct ggml_v2_tensor * c_attn_proj_b;

    // post normalization
    struct ggml_v2_tensor * ln_2_g;
    struct ggml_v2_tensor * ln_2_b;

    // ff
    struct ggml_v2_tensor * c_mlp_fc_w;
    struct ggml_v2_tensor * c_mlp_fc_b;

    struct ggml_v2_tensor * c_mlp_proj_w;
    struct ggml_v2_tensor * c_mlp_proj_b;
};

struct gpt_neox_v2_model {
    gpt_neox_hparams hparams;

    // normalization
    struct ggml_v2_tensor * ln_f_g;
    struct ggml_v2_tensor * ln_f_b;

    struct ggml_v2_tensor * wte; // position embedding

    struct ggml_v2_tensor * lmh_g; // language model head
    //struct ggml_tensor * lmh_b; // language model bias

    std::vector<gpt_neox_layer_v2> layers;

    // key + value memory
    struct ggml_v2_tensor * memory_k;
    struct ggml_v2_tensor * memory_v;

    //
    struct ggml_v2_context * ctx;
    std::map<std::string, struct ggml_v2_tensor *> tensors;
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
    //struct ggml_tensor * lmh_b; // language model bias

    std::vector<gpt_neox_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};


// no defaults for now
struct mpt_hparams {
    int32_t d_model      = 0;
    int32_t max_seq_len  = 0;
    int32_t n_heads      = 0;
    int32_t n_layers     = 0;
    int32_t n_vocab      = 0;
    float alibi_bias_max = 0;
    float clip_qkv       = 0;
    int32_t ftype        = 0;
    int32_t n_ctx        = 0;

};

struct mpt_layer {
    // pre normalization
    struct ggml_tensor * norm_1_weight;

    // attention
    struct ggml_tensor * c_attn_wqkv_weight;
    struct ggml_tensor * c_attn_out_proj_weight;

    // post normalization
    struct ggml_tensor * norm_2_weight;

    // ff
    struct ggml_tensor * ffn_up_proj;
    struct ggml_tensor * ffn_down_proj;
};

struct mpt_model {
    mpt_hparams hparams;

    struct ggml_tensor * wte_weight;    // position embedding
    struct ggml_tensor * norm_f_weight; // language model head

    std::vector<mpt_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};


