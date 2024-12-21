#pragma once

#include "llama.h"
#include "llama-arch.h"
#include "llama-vocab.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <array>
#include <vector>
#include <cassert>

// bump if necessary
#define LLAMA_MAX_LAYERS  512
#define LLAMA_MAX_EXPERTS 160  // DeepSeekV2

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_14M,
    MODEL_17M,
    MODEL_22M,
    MODEL_33M,
    MODEL_60M,
    MODEL_70M,
    MODEL_80M,
    MODEL_109M,
    MODEL_137M,
    MODEL_160M,
    MODEL_220M,
    MODEL_250M,
    MODEL_270M,
    MODEL_335M,
    MODEL_410M,
    MODEL_450M,
    MODEL_770M,
    MODEL_780M,
    MODEL_0_5B,
    MODEL_1B,
    MODEL_1_3B,
    MODEL_1_4B,
    MODEL_1_5B,
    MODEL_1_6B,
    MODEL_2B,
    MODEL_2_8B,
    MODEL_3B,
    MODEL_4B,
    MODEL_6B,
    MODEL_6_9B,
    MODEL_7B,
    MODEL_8B,
    MODEL_9B,
    MODEL_11B,
    MODEL_12B,
    MODEL_13B,
    MODEL_14B,
    MODEL_15B,
    MODEL_16B,
    MODEL_20B,
    MODEL_30B,
    MODEL_32B,
    MODEL_34B,
    MODEL_35B,
    MODEL_40B,
    MODEL_65B,
    MODEL_70B,
    MODEL_236B,
    MODEL_314B,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
    MODEL_XL,
    MODEL_A1_7B,
    MODEL_A2_7B,
    MODEL_8x7B,
    MODEL_8x22B,
    MODEL_16x12B,
    MODEL_10B_128x3_66B,
    MODEL_57B_A14B,
    MODEL_27B,
};

static const char * llama_model_type_name(e_model type) {
    switch (type) {
        case MODEL_14M:           return "14M";
        case MODEL_17M:           return "17M";
        case MODEL_22M:           return "22M";
        case MODEL_33M:           return "33M";
        case MODEL_60M:           return "60M";
        case MODEL_70M:           return "70M";
        case MODEL_80M:           return "80M";
        case MODEL_109M:          return "109M";
        case MODEL_137M:          return "137M";
        case MODEL_160M:          return "160M";
        case MODEL_220M:          return "220M";
        case MODEL_250M:          return "250M";
        case MODEL_270M:          return "270M";
        case MODEL_335M:          return "335M";
        case MODEL_410M:          return "410M";
        case MODEL_450M:          return "450M";
        case MODEL_770M:          return "770M";
        case MODEL_780M:          return "780M";
        case MODEL_0_5B:          return "0.5B";
        case MODEL_1B:            return "1B";
        case MODEL_1_3B:          return "1.3B";
        case MODEL_1_4B:          return "1.4B";
        case MODEL_1_5B:          return "1.5B";
        case MODEL_1_6B:          return "1.6B";
        case MODEL_2B:            return "2B";
        case MODEL_2_8B:          return "2.8B";
        case MODEL_3B:            return "3B";
        case MODEL_4B:            return "4B";
        case MODEL_6B:            return "6B";
        case MODEL_6_9B:          return "6.9B";
        case MODEL_7B:            return "7B";
        case MODEL_8B:            return "8B";
        case MODEL_9B:            return "9B";
        case MODEL_11B:           return "11B";
        case MODEL_12B:           return "12B";
        case MODEL_13B:           return "13B";
        case MODEL_14B:           return "14B";
        case MODEL_15B:           return "15B";
        case MODEL_16B:           return "16B";
        case MODEL_20B:           return "20B";
        case MODEL_30B:           return "30B";
        case MODEL_32B:           return "32B";
        case MODEL_34B:           return "34B";
        case MODEL_35B:           return "35B";
        case MODEL_40B:           return "40B";
        case MODEL_65B:           return "65B";
        case MODEL_70B:           return "70B";
        case MODEL_236B:          return "236B";
        case MODEL_314B:          return "314B";
        case MODEL_SMALL:         return "0.1B";
        case MODEL_MEDIUM:        return "0.4B";
        case MODEL_LARGE:         return "0.8B";
        case MODEL_XL:            return "1.5B";
        case MODEL_A1_7B:         return "A1.7B";
        case MODEL_A2_7B:         return "A2.7B";
        case MODEL_8x7B:          return "8x7B";
        case MODEL_8x22B:         return "8x22B";
        case MODEL_16x12B:        return "16x12B";
        case MODEL_10B_128x3_66B: return "10B+128x3.66B";
        case MODEL_57B_A14B:      return "57B.A14B";
        case MODEL_27B:           return "27B";
        default:                  return "?B";
    }
}

struct llama_hparams_posnet {
    uint32_t n_embd;
    uint32_t n_layer;
};

struct llama_hparams_convnext {
    uint32_t n_embd;
    uint32_t n_layer;
};

struct llama_hparams {
    bool vocab_only;
    bool rope_finetuned;
    bool use_par_res;
    bool swin_norm;

    uint32_t n_vocab = 0;
    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_embd_features = 0;
    uint32_t n_layer;
    uint32_t n_rot;
    uint32_t n_swa = 0; // sliding window attention (SWA)
    uint32_t n_embd_head_k; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v; // dimension of values (d_v) aka n_embd_head
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_vocab_type = 0; // for BERT-style token types
    uint32_t n_rel_attn_bkts = 0;

    // for WavTokenizer
    struct llama_hparams_posnet   posnet;
    struct llama_hparams_convnext convnext;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    uint32_t n_layer_dense_lead = 0;
    uint32_t n_lora_q = 0;
    uint32_t n_lora_kv = 0;
    uint32_t n_ff_exp = 0;
    uint32_t n_ff_shexp = 0;
    uint32_t n_expert_shared = 0;
    float    expert_weights_scale = 0.0;

    float f_norm_eps;
    float f_norm_rms_eps;
    float f_norm_group_eps;

    uint32_t n_norm_groups;

    float f_attn_logit_softcapping = 50.0f;
    float f_final_logit_softcapping = 30.0f;

    // for RWKV
    uint32_t rescale_every_n_layers = 0;
    uint32_t time_mix_extra_dim = 0;
    uint32_t time_decay_extra_dim = 0;
    uint32_t wkv_head_size = 0;

    float     rope_attn_factor = 1.0f;
    float     rope_freq_base_train;
    float     rope_freq_scale_train;
    uint32_t  n_ctx_orig_yarn;
    float     rope_yarn_log_mul;
    int       rope_sections[4];

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    bool ssm_dt_b_c_rms = false;

    float f_clamp_kqv      = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_logit_scale    = 0.0f;

    // Additional scale factors (Granite/Granite MoE)
    float f_residual_scale  = 0.0f;
    float f_embedding_scale = 0.0f;
    float f_attention_scale = 0.0f;

    bool causal_attn   = true;
    bool use_alibi     = false;
    bool attn_soft_cap = false;

    // needed by encoder-decoder models (e.g. T5, FLAN-T5)
    // ref: https://github.com/ggerganov/llama.cpp/pull/8141
    llama_token dec_start_token_id = LLAMA_TOKEN_NULL;

    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    uint32_t n_head(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_head_kv(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_kv_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_ff(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_ff_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_gqa(uint32_t il = 0) const {
        const uint32_t n_head    = this->n_head(il);
        const uint32_t n_head_kv = this->n_head_kv(il);

        if (n_head_kv == 0) {
            return 0;
        }

        return n_head/n_head_kv;
    }

    uint32_t n_embd_k_gqa(uint32_t il = 0) const { // dimension of key embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);

        return n_embd_head_k * n_head_kv;
    }

    uint32_t n_embd_v_gqa(uint32_t il = 0) const { // dimension of value embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);

        return n_embd_head_v * n_head_kv;
    }

    uint32_t n_embd_k_s() const { // dimension of the rolling state embeddings
        // corresponds to Mamba's conv_states size or RWKV's token_shift states size
        if (wkv_head_size != 0) {
            // for RWKV models
            return 2 * n_embd;
        }

        // TODO: maybe support other convolution strides than 1
        // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
        return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * ssm_d_inner;
    }

    uint32_t n_embd_v_s() const { // dimension of the recurrent state embeddings
        if (wkv_head_size != 0) {
            // corresponds to RWKV's wkv_states size
            return n_embd * wkv_head_size;
        }

        // corresponds to Mamba's ssm_states size
        return ssm_d_state * ssm_d_inner;
    }
};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");

struct llama_layer_posnet {
    // resnet
    struct ggml_tensor * norm1   = nullptr;
    struct ggml_tensor * norm1_b = nullptr;

    struct ggml_tensor * conv1   = nullptr;
    struct ggml_tensor * conv1_b = nullptr;

    struct ggml_tensor * norm2   = nullptr;
    struct ggml_tensor * norm2_b = nullptr;

    struct ggml_tensor * conv2   = nullptr;
    struct ggml_tensor * conv2_b = nullptr;

    // attention
    struct ggml_tensor * attn_norm   = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;

    struct ggml_tensor * attn_q   = nullptr;
    struct ggml_tensor * attn_q_b = nullptr;

    struct ggml_tensor * attn_k   = nullptr;
    struct ggml_tensor * attn_k_b = nullptr;

    struct ggml_tensor * attn_v   = nullptr;
    struct ggml_tensor * attn_v_b = nullptr;

    struct ggml_tensor * attn_o   = nullptr;
    struct ggml_tensor * attn_o_b = nullptr;

    // normalize
    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;
};

struct llama_layer_convnext {
    struct ggml_tensor * dw   = nullptr;
    struct ggml_tensor * dw_b = nullptr;

    struct ggml_tensor * norm   = nullptr;
    struct ggml_tensor * norm_b = nullptr;

    struct ggml_tensor * pw1   = nullptr;
    struct ggml_tensor * pw1_b = nullptr;

    struct ggml_tensor * pw2   = nullptr;
    struct ggml_tensor * pw2_b = nullptr;

    struct ggml_tensor * gamma = nullptr;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm       = nullptr;
    struct ggml_tensor * attn_norm_b     = nullptr;
    struct ggml_tensor * attn_norm_2     = nullptr;
    struct ggml_tensor * attn_norm_2_b   = nullptr;
    struct ggml_tensor * attn_q_norm     = nullptr;
    struct ggml_tensor * attn_q_norm_b   = nullptr;
    struct ggml_tensor * attn_k_norm     = nullptr;
    struct ggml_tensor * attn_k_norm_b   = nullptr;
    struct ggml_tensor * attn_out_norm   = nullptr;
    struct ggml_tensor * attn_out_norm_b = nullptr;
    struct ggml_tensor * attn_q_a_norm   = nullptr;
    struct ggml_tensor * attn_kv_a_norm  = nullptr;
    struct ggml_tensor * attn_sub_norm   = nullptr;
    struct ggml_tensor * attn_post_norm  = nullptr;
    struct ggml_tensor * ffn_sub_norm    = nullptr;
    struct ggml_tensor * attn_norm_cross = nullptr;
    struct ggml_tensor * attn_norm_enc   = nullptr;

    // attention
    struct ggml_tensor * wq        = nullptr;
    struct ggml_tensor * wk        = nullptr;
    struct ggml_tensor * wv        = nullptr;
    struct ggml_tensor * wo        = nullptr;
    struct ggml_tensor * wqkv      = nullptr;
    struct ggml_tensor * wq_a      = nullptr;
    struct ggml_tensor * wq_b      = nullptr;
    struct ggml_tensor * wkv_a_mqa = nullptr;
    struct ggml_tensor * wkv_b     = nullptr;
    struct ggml_tensor * wq_cross  = nullptr;
    struct ggml_tensor * wk_cross  = nullptr;
    struct ggml_tensor * wv_cross  = nullptr;
    struct ggml_tensor * wo_cross  = nullptr;
    struct ggml_tensor * wq_enc    = nullptr;
    struct ggml_tensor * wk_enc    = nullptr;
    struct ggml_tensor * wv_enc    = nullptr;
    struct ggml_tensor * wo_enc    = nullptr;

    // attention bias
    struct ggml_tensor * bq   = nullptr;
    struct ggml_tensor * bk   = nullptr;
    struct ggml_tensor * bv   = nullptr;
    struct ggml_tensor * bo   = nullptr;
    struct ggml_tensor * bqkv = nullptr;

    // relative position bias
    struct ggml_tensor * attn_rel_b       = nullptr;
    struct ggml_tensor * attn_rel_b_enc   = nullptr;
    struct ggml_tensor * attn_rel_b_cross = nullptr;

    // normalization
    struct ggml_tensor * ffn_norm         = nullptr;
    struct ggml_tensor * ffn_norm_b       = nullptr;
    struct ggml_tensor * ffn_post_norm    = nullptr;
    struct ggml_tensor * layer_out_norm   = nullptr;
    struct ggml_tensor * layer_out_norm_b = nullptr;
    struct ggml_tensor * ffn_norm_exps    = nullptr;
    struct ggml_tensor * ffn_norm_enc     = nullptr;

    // ff
    struct ggml_tensor * ffn_gate     = nullptr; // w1
    struct ggml_tensor * ffn_down     = nullptr; // w2
    struct ggml_tensor * ffn_up       = nullptr; // w3
    struct ggml_tensor * ffn_gate_enc = nullptr;
    struct ggml_tensor * ffn_down_enc = nullptr;
    struct ggml_tensor * ffn_up_enc   = nullptr;

    // ff MoE
    struct ggml_tensor * ffn_gate_inp  = nullptr;
    struct ggml_tensor * ffn_gate_exps = nullptr;
    struct ggml_tensor * ffn_down_exps = nullptr;
    struct ggml_tensor * ffn_up_exps   = nullptr;

    // ff shared expert (shexp)
    struct ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct ggml_tensor * ffn_gate_shexp     = nullptr;
    struct ggml_tensor * ffn_down_shexp     = nullptr;
    struct ggml_tensor * ffn_up_shexp       = nullptr;

    // ff bias
    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr; // b2
    struct ggml_tensor * ffn_up_b   = nullptr; // b3
    struct ggml_tensor * ffn_act    = nullptr;

    // mamba proj
    struct ggml_tensor * ssm_in  = nullptr;
    struct ggml_tensor * ssm_x   = nullptr;
    struct ggml_tensor * ssm_dt  = nullptr;
    struct ggml_tensor * ssm_out = nullptr;

    // mamba
    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_a      = nullptr;
    struct ggml_tensor * ssm_d      = nullptr;

    // mamba bias
    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b     = nullptr;

    // rwkv
    struct ggml_tensor * time_mix_w1         = nullptr;
    struct ggml_tensor * time_mix_w2         = nullptr;
    struct ggml_tensor * time_mix_lerp_x     = nullptr;
    struct ggml_tensor * time_mix_lerp_w     = nullptr;
    struct ggml_tensor * time_mix_lerp_k     = nullptr;
    struct ggml_tensor * time_mix_lerp_v     = nullptr;
    struct ggml_tensor * time_mix_lerp_r     = nullptr;
    struct ggml_tensor * time_mix_lerp_g     = nullptr;

    struct ggml_tensor * time_mix_first      = nullptr;
    struct ggml_tensor * time_mix_decay      = nullptr;
    struct ggml_tensor * time_mix_decay_w1   = nullptr;
    struct ggml_tensor * time_mix_decay_w2   = nullptr;
    struct ggml_tensor * time_mix_key        = nullptr;
    struct ggml_tensor * time_mix_value      = nullptr;
    struct ggml_tensor * time_mix_receptance = nullptr;
    struct ggml_tensor * time_mix_gate       = nullptr;

    struct ggml_tensor * time_mix_ln     = nullptr;
    struct ggml_tensor * time_mix_ln_b   = nullptr;
    struct ggml_tensor * time_mix_output = nullptr;

    struct ggml_tensor * channel_mix_lerp_k = nullptr;
    struct ggml_tensor * channel_mix_lerp_r = nullptr;

    struct ggml_tensor * channel_mix_key        = nullptr;
    struct ggml_tensor * channel_mix_receptance = nullptr;
    struct ggml_tensor * channel_mix_value      = nullptr;

    // long rope factors
    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;

    // bitnet scale
    struct ggml_tensor * wq_scale       = nullptr;
    struct ggml_tensor * wk_scale       = nullptr;
    struct ggml_tensor * wv_scale       = nullptr;
    struct ggml_tensor * wo_scale       = nullptr;
    struct ggml_tensor * ffn_gate_scale = nullptr;
    struct ggml_tensor * ffn_up_scale   = nullptr;
    struct ggml_tensor * ffn_down_scale = nullptr;

    struct llama_layer_posnet posnet;

    struct llama_layer_convnext convnext;
};

struct llama_model {
    e_model     type  = MODEL_UNKNOWN;
    llm_arch    arch  = LLM_ARCH_UNKNOWN;
    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    struct ggml_tensor * tok_embd = nullptr;
    struct ggml_tensor * type_embd = nullptr;
    struct ggml_tensor * pos_embd = nullptr;
    struct ggml_tensor * tok_norm = nullptr;
    struct ggml_tensor * tok_norm_b = nullptr;

    struct ggml_tensor * output_norm = nullptr;
    struct ggml_tensor * output_norm_b = nullptr;
    struct ggml_tensor * output = nullptr;
    struct ggml_tensor * output_b = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;

    // classifier
    struct ggml_tensor * cls = nullptr;
    struct ggml_tensor * cls_b = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;

    struct ggml_tensor * conv1d = nullptr;
    struct ggml_tensor * conv1d_b = nullptr;

    std::vector<llama_layer> layers;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    llama_split_mode split_mode;
    int main_gpu;
    int n_gpu_layers;

    std::vector<std::string> rpc_servers;

    // list of devices used in this model
    std::vector<ggml_backend_dev_t> devices;


    // lists of buffer types used for each layer
    using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;
    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };
    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    // contexts where the model tensors metadata is stored
    std::vector<ggml_context_ptr> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_ptr> bufs;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    // total number of parameters in the model
    uint64_t n_elements = 0;

    // total size of all the tensors in the model in bytes
    size_t  n_bytes     = 0;

    // keep track of loaded lora adapters
    std::set<struct llama_lora_adapter *> lora_adapters;

    ~llama_model() {
       while (!lora_adapters.empty()) {
            llama_lora_adapter_free(*lora_adapters.begin());
        }
    }
};

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const llama_model::buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }
    throw std::runtime_error(format("no suitable buffer type found"));
}

