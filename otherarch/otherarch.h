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
    int32_t f16     = 1;
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

struct gptj_model_v1 {
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
    int32_t f16     = 1;
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
struct stablelm_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 4096;
    int32_t n_embd  = 4096;
    int32_t n_head  = 32;
    int32_t n_layer = 16;
    int32_t n_rot   = 32; // rotary_pct * (n_embd / n_head)
    int32_t ftype   = 1;
};

struct stablelm_layer {
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

struct stablelm_model {
    stablelm_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte; // position embedding

    struct ggml_tensor * lmh_g; // language model head
    //struct ggml_tensor * lmh_b; // language model bias

    std::vector<stablelm_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct rwkv_layer {
    struct ggml_rwkv_tensor * ln1_weight;
    struct ggml_rwkv_tensor * ln1_bias;

    // RWKV, also called "attention" by the author.
    struct ggml_rwkv_tensor * att_time_mix_k;
    struct ggml_rwkv_tensor * att_time_mix_v;
    struct ggml_rwkv_tensor * att_time_mix_r;
    struct ggml_rwkv_tensor * att_time_first;
    struct ggml_rwkv_tensor * att_time_decay;
    struct ggml_rwkv_tensor * att_key;
    struct ggml_rwkv_tensor * att_value;
    struct ggml_rwkv_tensor * att_receptance;
    struct ggml_rwkv_tensor * att_output;

    struct ggml_rwkv_tensor * ln2_weight;
    struct ggml_rwkv_tensor * ln2_bias;

    // FFN.
    struct ggml_rwkv_tensor * ffn_time_mix_k;
    struct ggml_rwkv_tensor * ffn_time_mix_r;
    struct ggml_rwkv_tensor * ffn_key;
    struct ggml_rwkv_tensor * ffn_value;
    struct ggml_rwkv_tensor * ffn_receptance;
};

struct rwkv_model {
    int32_t n_vocab;
    int32_t n_layer;
    int32_t n_embed;
    // 0 for float32, 1 for float16.
    int32_t data_type;

    struct ggml_rwkv_tensor * emb;

    struct ggml_rwkv_tensor * ln0_weight;
    struct ggml_rwkv_tensor * ln0_bias;

    std::vector<rwkv_layer> layers;

    struct ggml_rwkv_tensor * ln_out_weight;
    struct ggml_rwkv_tensor * ln_out_bias;

    struct ggml_rwkv_tensor * head;
};
struct rwkv_context {
    struct rwkv_model * model;
    struct ggml_rwkv_tensor * token_index;
    struct ggml_rwkv_tensor * state;
    struct ggml_rwkv_tensor ** state_parts;
    struct ggml_rwkv_tensor * logits;
    struct ggml_rwkv_context * ctx;
    struct ggml_rwkv_cgraph * graph;
    bool freed;
    float * state_in = 0; //stores input state, or use null for a new state
    float * state_out = 0; //stores address of output state buffer
    float * logits_out = 0; //stores address of output logit buffer
};
