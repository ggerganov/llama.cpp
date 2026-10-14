#include "models.h"

struct llama_model_vibevoice::tok_tensors {
    llm_tensor conv;
    llm_tensor head;
    llm_tensor norm;
    llm_tensor dwconv;
    llm_tensor gamma;
    llm_tensor ffn_norm;
    llm_tensor ffn_up;
    llm_tensor ffn_down;
    llm_tensor ffn_gamma;
};

// there is one stage per depth, and one ratio per resample conv (all convs but the stem)
static uint32_t vv_get_stages(llama_model_loader & ml, llm_kv kid_ratios, llm_kv kid_depths,
        std::array<uint32_t, LLAMA_MAX_VV_STAGES> & ratios, std::array<uint32_t, LLAMA_MAX_VV_STAGES> & depths) {
    std::vector<uint32_t> r;
    std::vector<uint32_t> d;
    ml.get_arr(kid_ratios, r);
    ml.get_arr(kid_depths, d);

    if (d.empty() || d.size() > LLAMA_MAX_VV_STAGES || r.size() + 1 != d.size()) {
        throw std::runtime_error(format("vibevoice: invalid tokenizer layout, %zu ratios and %zu depths", r.size(), d.size()));
    }
    for (uint32_t depth : d) {
        if (depth == 0 || depth > LLAMA_MAX_LAYERS) {
            throw std::runtime_error(format("vibevoice: invalid tokenizer stage depth %u", depth));
        }
    }

    ratios.fill(0);
    depths.fill(0);
    std::copy(r.begin(), r.end(), ratios.begin());
    std::copy(d.begin(), d.end(), depths.begin());

    return d.size();
}

void llama_model_vibevoice::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen2::load_arch_hparams(ml);

    auto & vv = hparams.vibevoice;

    ml.get_key(LLM_KV_VIBEVOICE_SPEECH_SCALING_FACTOR, vv.speech_scaling_factor);
    ml.get_key(LLM_KV_VIBEVOICE_SPEECH_BIAS_FACTOR,    vv.speech_bias_factor);

    ml.get_key(LLM_KV_VIBEVOICE_ACOUSTIC_VAE_DIM,       vv.acoustic.vae_dim);
    ml.get_key(LLM_KV_VIBEVOICE_ACOUSTIC_N_FILTERS,     vv.acoustic.n_filters);
    ml.get_key(LLM_KV_VIBEVOICE_ACOUSTIC_FIX_STD,       vv.acoustic.fix_std);
    ml.get_key(LLM_KV_VIBEVOICE_ACOUSTIC_LAYERNORM_EPS, vv.acoustic.f_norm_rms_eps);
    vv.acoustic.n_enc_stages = vv_get_stages(ml, LLM_KV_VIBEVOICE_ACOUSTIC_ENCODER_RATIOS, LLM_KV_VIBEVOICE_ACOUSTIC_ENCODER_DEPTHS, vv.acoustic.enc_ratios, vv.acoustic.enc_depths);
    vv.acoustic.n_dec_stages = vv_get_stages(ml, LLM_KV_VIBEVOICE_ACOUSTIC_DECODER_RATIOS, LLM_KV_VIBEVOICE_ACOUSTIC_DECODER_DEPTHS, vv.acoustic.dec_ratios, vv.acoustic.dec_depths);

    ml.get_key(LLM_KV_VIBEVOICE_SEMANTIC_VAE_DIM,       vv.semantic.vae_dim);
    ml.get_key(LLM_KV_VIBEVOICE_SEMANTIC_N_FILTERS,     vv.semantic.n_filters);
    ml.get_key(LLM_KV_VIBEVOICE_SEMANTIC_LAYERNORM_EPS, vv.semantic.f_norm_rms_eps);
    vv.semantic.fix_std      = 0.0f;
    vv.semantic.n_enc_stages = vv_get_stages(ml, LLM_KV_VIBEVOICE_SEMANTIC_ENCODER_RATIOS, LLM_KV_VIBEVOICE_SEMANTIC_ENCODER_DEPTHS, vv.semantic.enc_ratios, vv.semantic.enc_depths);
    vv.semantic.n_dec_stages = 0;

    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_BLOCK_COUNT,      vv.diff_n_layer);
    if (vv.diff_n_layer == 0 || vv.diff_n_layer > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("vibevoice: invalid diffusion block count %u", vv.diff_n_layer));
    }
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_FEED_FORWARD_LEN, vv.diff_n_ff);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_LATENT_SIZE,      vv.diff_latent_size);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_LAYERNORM_EPS,    vv.diff_f_norm_rms_eps);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_TRAIN_STEPS,      vv.diff_train_steps);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_INFERENCE_STEPS,  vv.diff_inference_steps);

    // all released checkpoints use the same scheduler, it is the only one we will implement
    std::string diff_type;
    std::string diff_pred_type;
    std::string diff_beta_schedule;
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_TYPE,            diff_type);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_PREDICTION_TYPE, diff_pred_type);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_BETA_SCHEDULE,   diff_beta_schedule);
    if (diff_type != "ddpm" || diff_pred_type != "v_prediction" || diff_beta_schedule != "cosine") {
        throw std::runtime_error(format("vibevoice: unsupported diffusion head %s/%s/%s",
                diff_type.c_str(), diff_pred_type.c_str(), diff_beta_schedule.c_str()));
    }
}

// the shape comes from the file, it is not checked against hparams yet
ggml_tensor * llama_model_vibevoice::create_tensor_from_file(llama_model_loader & ml, const LLM_TN_IMPL & tn) {
    const ggml_tensor * meta = ml.require_tensor_meta(tn.str());
    const int64_t * ne = meta->ne;

    switch (ggml_n_dims(meta)) {
        case 1:  return create_tensor(ml, tn, {ne[0]}, 0);
        case 2:  return create_tensor(ml, tn, {ne[0], ne[1]}, 0);
        case 3:  return create_tensor(ml, tn, {ne[0], ne[1], ne[2]}, 0);
        default: return create_tensor(ml, tn, {ne[0], ne[1], ne[2], ne[3]}, 0);
    }
}

void llama_model_vibevoice::load_connector(llama_model_loader & ml, connector & conn, llm_tensor fc1, llm_tensor norm, llm_tensor fc2) {
    conn.fc1   = create_tensor_from_file(ml, tn(fc1,  "weight"));
    conn.fc1_b = create_tensor_from_file(ml, tn(fc1,  "bias"));
    conn.norm  = create_tensor_from_file(ml, tn(norm, "weight"));
    conn.fc2   = create_tensor_from_file(ml, tn(fc2,  "weight"));
    conn.fc2_b = create_tensor_from_file(ml, tn(fc2,  "bias"));
}

void llama_model_vibevoice::load_tok_part(llama_model_loader & ml, tok_part & part, const tok_tensors & ids, uint32_t n_stages, const std::array<uint32_t, LLAMA_MAX_VV_STAGES> & depths) {
    part.stages.resize(n_stages);

    for (uint32_t s = 0; s < n_stages; ++s) {
        auto & stage = part.stages[s];

        stage.conv   = create_tensor_from_file(ml, tn(ids.conv, "weight", s));
        stage.conv_b = create_tensor_from_file(ml, tn(ids.conv, "bias",   s));

        if (depths[s] == 0 || depths[s] > LLAMA_MAX_LAYERS) {
            throw std::runtime_error(format("vibevoice: invalid tokenizer stage depth %u", depths[s]));
        }
        stage.blocks.resize(depths[s]);

        for (uint32_t b = 0; b < depths[s]; ++b) {
            auto & blk = stage.blocks[b];

            blk.norm       = create_tensor_from_file(ml, tn(ids.norm,      "weight", s, b));
            blk.dwconv     = create_tensor_from_file(ml, tn(ids.dwconv,    "weight", s, b));
            blk.dwconv_b   = create_tensor_from_file(ml, tn(ids.dwconv,    "bias",   s, b));
            blk.gamma      = create_tensor_from_file(ml, tn(ids.gamma,     "weight", s, b));
            blk.ffn_norm   = create_tensor_from_file(ml, tn(ids.ffn_norm,  "weight", s, b));
            blk.ffn_up     = create_tensor_from_file(ml, tn(ids.ffn_up,    "weight", s, b));
            blk.ffn_up_b   = create_tensor_from_file(ml, tn(ids.ffn_up,    "bias",   s, b));
            blk.ffn_down   = create_tensor_from_file(ml, tn(ids.ffn_down,  "weight", s, b));
            blk.ffn_down_b = create_tensor_from_file(ml, tn(ids.ffn_down,  "bias",   s, b));
            blk.ffn_gamma  = create_tensor_from_file(ml, tn(ids.ffn_gamma, "weight", s, b));
        }
    }

    part.head   = create_tensor_from_file(ml, tn(ids.head, "weight"));
    part.head_b = create_tensor_from_file(ml, tn(ids.head, "bias"));
}

void llama_model_vibevoice::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen2::load_arch_tensors(ml);

    const auto & vv = hparams.vibevoice;

    load_connector(ml, acoustic_conn, LLM_TENSOR_VV_ACOUSTIC_CONN_FC1, LLM_TENSOR_VV_ACOUSTIC_CONN_NORM, LLM_TENSOR_VV_ACOUSTIC_CONN_FC2);
    load_connector(ml, semantic_conn, LLM_TENSOR_VV_SEMANTIC_CONN_FC1, LLM_TENSOR_VV_SEMANTIC_CONN_NORM, LLM_TENSOR_VV_SEMANTIC_CONN_FC2);

    // diffusion head
    {
        diff.noisy_proj = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_NOISY_PROJ, "weight"));
        diff.cond_proj  = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_COND_PROJ,  "weight"));
        diff.t_embd_fc1 = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_T_EMBD_FC1, "weight"));
        diff.t_embd_fc2 = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_T_EMBD_FC2, "weight"));

        diff.layers.resize(vv.diff_n_layer);

        for (uint32_t i = 0; i < vv.diff_n_layer; ++i) {
            auto & layer = diff.layers[i];

            layer.norm     = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_NORM,     "weight", i));
            layer.adaln    = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_ADALN,    "weight", i));
            layer.ffn_gate = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_FFN_GATE, "weight", i));
            layer.ffn_up   = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_FFN_UP,   "weight", i));
            layer.ffn_down = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_FFN_DOWN, "weight", i));
        }

        diff.final_adaln = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_FINAL_ADALN, "weight"));
        diff.final_proj  = create_tensor_from_file(ml, tn(LLM_TENSOR_VV_DIFF_FINAL_PROJ,  "weight"));
    }

    // tokenizers
    {
        const tok_tensors acoustic_enc_ids = {
            LLM_TENSOR_VV_ACOUSTIC_ENC_CONV, LLM_TENSOR_VV_ACOUSTIC_ENC_HEAD,
            LLM_TENSOR_VV_ACOUSTIC_ENC_NORM, LLM_TENSOR_VV_ACOUSTIC_ENC_DWCONV, LLM_TENSOR_VV_ACOUSTIC_ENC_GAMMA,
            LLM_TENSOR_VV_ACOUSTIC_ENC_FFN_NORM, LLM_TENSOR_VV_ACOUSTIC_ENC_FFN_UP, LLM_TENSOR_VV_ACOUSTIC_ENC_FFN_DOWN, LLM_TENSOR_VV_ACOUSTIC_ENC_FFN_GAMMA,
        };
        const tok_tensors acoustic_dec_ids = {
            LLM_TENSOR_VV_ACOUSTIC_DEC_CONV, LLM_TENSOR_VV_ACOUSTIC_DEC_HEAD,
            LLM_TENSOR_VV_ACOUSTIC_DEC_NORM, LLM_TENSOR_VV_ACOUSTIC_DEC_DWCONV, LLM_TENSOR_VV_ACOUSTIC_DEC_GAMMA,
            LLM_TENSOR_VV_ACOUSTIC_DEC_FFN_NORM, LLM_TENSOR_VV_ACOUSTIC_DEC_FFN_UP, LLM_TENSOR_VV_ACOUSTIC_DEC_FFN_DOWN, LLM_TENSOR_VV_ACOUSTIC_DEC_FFN_GAMMA,
        };
        const tok_tensors semantic_enc_ids = {
            LLM_TENSOR_VV_SEMANTIC_ENC_CONV, LLM_TENSOR_VV_SEMANTIC_ENC_HEAD,
            LLM_TENSOR_VV_SEMANTIC_ENC_NORM, LLM_TENSOR_VV_SEMANTIC_ENC_DWCONV, LLM_TENSOR_VV_SEMANTIC_ENC_GAMMA,
            LLM_TENSOR_VV_SEMANTIC_ENC_FFN_NORM, LLM_TENSOR_VV_SEMANTIC_ENC_FFN_UP, LLM_TENSOR_VV_SEMANTIC_ENC_FFN_DOWN, LLM_TENSOR_VV_SEMANTIC_ENC_FFN_GAMMA,
        };

        load_tok_part(ml, acoustic_enc, acoustic_enc_ids, vv.acoustic.n_enc_stages, vv.acoustic.enc_depths);
        load_tok_part(ml, acoustic_dec, acoustic_dec_ids, vv.acoustic.n_dec_stages, vv.acoustic.dec_depths);
        load_tok_part(ml, semantic_enc, semantic_enc_ids, vv.semantic.n_enc_stages, vv.semantic.enc_depths);
    }
}
