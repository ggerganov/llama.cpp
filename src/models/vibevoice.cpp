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

//
// acoustic decoder
//

// causal conv1d with optional stride, zero left padding. x: [T, IC], w: [K, IC, OC]
// padding_total = K - stride (upstream SConv1d); caller should pass a T that needs no extra right pad
// returns [ceil-ish T/stride, OC]; with T % stride == 0 and K == 2*stride this is exactly T/stride
static ggml_tensor * vv_causal_conv1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, int stride = 1) {
    GGML_ASSERT(stride >= 1 && w->ne[0] >= stride);
    const int padding_total = (int) w->ne[0] - stride;
    x = ggml_pad_ext(ctx, x, padding_total, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, 0, 1);
    return ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
}

// depthwise variant of vv_causal_conv1d. x: [T, C], w: [K, 1, C], returns [T, C]
static ggml_tensor * vv_causal_conv1d_dw(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    x = ggml_pad_ext(ctx, x, w->ne[0] - 1, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor * y = ggml_conv_1d_dw(ctx, w, x, 1, 0, 1);
    return ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
}

// causal ConvTranspose1d, the (K - stride) overlap is trimmed on the right. x: [T, IC], w: [K, OC, IC], returns [T * stride, OC]
static ggml_tensor * vv_causal_conv_transpose1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, int stride) {
    GGML_ASSERT(w->ne[0] >= stride);
    ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    return ggml_view_2d(ctx, y, x->ne[0] * stride, y->ne[1], y->nb[1], 0);
}

// the decoder stream is [C, T], the convs run on the transposed [T, C]
static ggml_tensor * vv_transpose(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

// ConvNeXt-style Block1D: RMSNorm -> depthwise conv -> gamma -> residual, RMSNorm -> FFN -> ffn_gamma -> residual
static ggml_tensor * vv_build_tok_block(ggml_context * ctx, ggml_tensor * x, const llama_model_vibevoice::tok_block & blk, float eps) {
    ggml_tensor * cur = ggml_rms_norm(ctx, x, eps);
    cur = ggml_mul(ctx, cur, blk.norm);
    cur = vv_transpose(ctx, vv_causal_conv1d_dw(ctx, vv_transpose(ctx, cur), blk.dwconv));
    cur = ggml_add(ctx, cur, blk.dwconv_b);
    cur = ggml_mul(ctx, cur, blk.gamma);
    x = ggml_add(ctx, x, cur);

    cur = ggml_rms_norm(ctx, x, eps);
    cur = ggml_mul(ctx, cur, blk.ffn_norm);
    cur = ggml_mul_mat(ctx, blk.ffn_up, cur);
    cur = ggml_add(ctx, cur, blk.ffn_up_b);
    cur = ggml_gelu_erf(ctx, cur);
    cur = ggml_mul_mat(ctx, blk.ffn_down, cur);
    cur = ggml_add(ctx, cur, blk.ffn_down_b);
    cur = ggml_mul(ctx, cur, blk.ffn_gamma);

    return ggml_add(ctx, x, cur);
}

ggml_tensor * llama_model_vibevoice::build_acoustic_dec(ggml_context * ctx, ggml_tensor * latents) const {
    const auto & hp = hparams.vibevoice.acoustic;

    GGML_ASSERT(ggml_is_matrix(latents) && latents->ne[0] == hp.vae_dim);

    ggml_tensor * cur = latents;

    for (uint32_t s = 0; s < hp.n_dec_stages; ++s) {
        const auto & stage = acoustic_dec.stages[s];

        // the first conv is the stem, the others upsample by the decoder ratios in config order
        ggml_tensor * x = vv_transpose(ctx, cur);
        if (s == 0) {
            x = vv_causal_conv1d(ctx, x, stage.conv);
        } else {
            x = vv_causal_conv_transpose1d(ctx, x, stage.conv, hp.dec_ratios[s - 1]);
        }
        cur = ggml_add(ctx, vv_transpose(ctx, x), stage.conv_b);

        for (const auto & blk : stage.blocks) {
            cur = vv_build_tok_block(ctx, cur, blk, hp.f_norm_rms_eps);
        }
    }

    // disable_last_norm: there is no norm before the head
    cur = vv_causal_conv1d(ctx, vv_transpose(ctx, cur), acoustic_dec.head);
    cur = ggml_add(ctx, cur, ggml_reshape_2d(ctx, acoustic_dec.head_b, 1, acoustic_dec.head_b->ne[0]));

    return cur;
}

//
// tokenizer encoder (semantic / acoustic) + speech connector
//

// mono waveform [T, 1] -> latents [vae_dim, T / hop]
// encoder ratios are stored in HF config order; upstream TokenizerEncoder reverses them at build time
static ggml_tensor * vv_build_tok_enc(
        ggml_context * ctx,
        ggml_tensor * wav,
        const llama_model_vibevoice::tok_part & enc,
        const llama_hparams_vv_tokenizer & hp) {
    GGML_ASSERT(ggml_is_matrix(wav) && wav->ne[1] == 1);
    GGML_ASSERT(hp.n_enc_stages > 0);

    // channel-first stream like the decoder: [C, T]
    ggml_tensor * cur = vv_transpose(ctx, wav);

    const uint32_t n_ratios = hp.n_enc_stages - 1;

    for (uint32_t s = 0; s < hp.n_enc_stages; ++s) {
        const auto & stage = enc.stages[s];

        ggml_tensor * x = vv_transpose(ctx, cur);
        if (s == 0) {
            x = vv_causal_conv1d(ctx, x, stage.conv, 1);
        } else {
            // reversed config ratios: first downsample uses the last config ratio
            const uint32_t stride = hp.enc_ratios[n_ratios - s];
            x = vv_causal_conv1d(ctx, x, stage.conv, (int) stride);
        }
        cur = ggml_add(ctx, vv_transpose(ctx, x), stage.conv_b);

        for (const auto & blk : stage.blocks) {
            cur = vv_build_tok_block(ctx, cur, blk, hp.f_norm_rms_eps);
        }
    }

    // disable_last_norm: there is no norm before the head
    cur = vv_causal_conv1d(ctx, vv_transpose(ctx, cur), enc.head, 1);
    cur = ggml_add(ctx, cur, ggml_reshape_2d(ctx, enc.head_b, 1, enc.head_b->ne[0]));

    // [T', vae_dim] -> [vae_dim, T']
    return vv_transpose(ctx, cur);
}

ggml_tensor * llama_model_vibevoice::build_semantic_enc(ggml_context * ctx, ggml_tensor * wav) const {
    return vv_build_tok_enc(ctx, wav, semantic_enc, hparams.vibevoice.semantic);
}

ggml_tensor * llama_model_vibevoice::build_acoustic_enc(ggml_context * ctx, ggml_tensor * wav) const {
    return vv_build_tok_enc(ctx, wav, acoustic_enc, hparams.vibevoice.acoustic);
}

// speech connector: Linear -> RMSNorm -> Linear (no activation), matches upstream SpeechConnector
// features [in_dim, N] -> [n_embd, N]
ggml_tensor * llama_model_vibevoice::build_connector(ggml_context * ctx, ggml_tensor * features, const connector & conn) const {
    GGML_ASSERT(ggml_is_matrix(features));
    GGML_ASSERT(features->ne[0] == conn.fc1->ne[0]);

    ggml_tensor * cur = ggml_mul_mat(ctx, conn.fc1, features);
    cur = ggml_add(ctx, cur, conn.fc1_b);

    cur = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
    cur = ggml_mul(ctx, cur, conn.norm);

    cur = ggml_mul_mat(ctx, conn.fc2, cur);
    cur = ggml_add(ctx, cur, conn.fc2_b);

    return cur;
}

//
// diffusion head
//

// x * (1 + scale) + shift
static ggml_tensor * vv_modulate(ggml_context * ctx, ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) {
    return ggml_add(ctx, ggml_add(ctx, x, ggml_mul(ctx, x, scale)), shift);
}

// i-th of n equal chunks along the features, x: [n * C, N], returns [C, N]
static ggml_tensor * vv_chunk(ggml_context * ctx, ggml_tensor * x, int n, int i) {
    const int64_t c = x->ne[0] / n;
    return ggml_view_2d(ctx, x, c, x->ne[1], x->nb[1], i * c * x->nb[0]);
}

ggml_tensor * llama_model_vibevoice::build_diffusion_head(ggml_context * ctx, ggml_tensor * noisy, ggml_tensor * cond, ggml_tensor * t) const {
    const auto & vv = hparams.vibevoice;

    GGML_ASSERT(ggml_is_matrix(noisy) && noisy->ne[0] == vv.diff_latent_size);
    GGML_ASSERT(ggml_is_matrix(cond)  && cond->ne[1] == noisy->ne[1]);
    GGML_ASSERT(ggml_is_vector(t)     && (t->ne[0] == noisy->ne[1] || t->ne[0] == 1));

    // the frequency embedding size comes from fc1, it is 256 for all released checkpoints
    ggml_tensor * t_emb = ggml_timestep_embedding(ctx, t, diff.t_embd_fc1->ne[0], 10000);
    t_emb = ggml_mul_mat(ctx, diff.t_embd_fc1, t_emb);
    t_emb = ggml_silu(ctx, t_emb);
    t_emb = ggml_mul_mat(ctx, diff.t_embd_fc2, t_emb);

    // every adaLN modulation starts with a SiLU of the condition
    ggml_tensor * c = ggml_add(ctx, ggml_mul_mat(ctx, diff.cond_proj, cond), t_emb);
    c = ggml_silu(ctx, c);

    ggml_tensor * x = ggml_mul_mat(ctx, diff.noisy_proj, noisy);

    for (const auto & layer : diff.layers) {
        // shift, scale, gate
        ggml_tensor * mod = ggml_mul_mat(ctx, layer.adaln, c);

        ggml_tensor * cur = ggml_rms_norm(ctx, x, vv.diff_f_norm_rms_eps);
        cur = ggml_mul(ctx, cur, layer.norm);
        cur = vv_modulate(ctx, cur, vv_chunk(ctx, mod, 3, 0), vv_chunk(ctx, mod, 3, 1));

        cur = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, layer.ffn_gate, cur), ggml_mul_mat(ctx, layer.ffn_up, cur));
        cur = ggml_mul_mat(ctx, layer.ffn_down, cur);

        x = ggml_add(ctx, x, ggml_mul(ctx, cur, vv_chunk(ctx, mod, 3, 2)));
    }

    // shift, scale, the final norm has no weight
    ggml_tensor * mod = ggml_mul_mat(ctx, diff.final_adaln, c);

    x = ggml_rms_norm(ctx, x, vv.diff_f_norm_rms_eps);
    x = vv_modulate(ctx, x, vv_chunk(ctx, mod, 2, 0), vv_chunk(ctx, mod, 2, 1));

    return ggml_mul_mat(ctx, diff.final_proj, x);
}

// the schedule follows the float32 math of the reference scheduler, its cumprod accumulates in double
llama_model_vibevoice::diff_sampler::diff_sampler(uint32_t n_train_steps, uint32_t n_steps) {
    GGML_ASSERT(n_steps > 0 && n_steps <= n_train_steps);

    // squaredcos_cap_v2 betas
    const auto alpha_bar = [](double t) {
        const double c = std::cos((t + 0.008) / 1.008 * std::acos(-1.0) / 2);
        return c * c;
    };

    std::vector<float> sigmas_train(n_train_steps);
    double alphas_cumprod = 1.0;
    for (uint32_t t = 0; t < n_train_steps; ++t) {
        const float beta = std::min(1.0 - alpha_bar((t + 1.0) / n_train_steps) / alpha_bar((double) t / n_train_steps), 0.999);
        alphas_cumprod *= 1.0f - beta;
        const float ac = alphas_cumprod;
        sigmas_train[t] = std::sqrt((1.0f - ac) / ac);
    }

    // linspace(0, n_train_steps - 1, n_steps + 1) rounded half to even like numpy, descending, without the 0
    const double dt = (double) (n_train_steps - 1) / n_steps;
    for (uint32_t k = n_steps; k > 0; --k) {
        const uint32_t t = k == n_steps ? n_train_steps - 1 : (uint32_t) std::nearbyint(k * dt);
        timesteps.push_back(t);
        sigmas.push_back(sigmas_train[t]);
    }
    sigmas.push_back(0.0f);
}

void llama_model_vibevoice::diff_sampler::step(const float * v, float * x, size_t n) {
    GGML_ASSERT(i < timesteps.size());
    GGML_ASSERT(x0_prev.empty() || x0_prev.size() == n);

    const auto alpha_sigma = [](double sigma) {
        const double alpha = 1.0 / std::sqrt(sigma * sigma + 1.0);
        return std::make_pair(alpha, sigma * alpha);
    };

    const auto [alpha_s0, sigma_s0] = alpha_sigma(sigmas[i]);
    const auto [alpha_t,  sigma_t]  = alpha_sigma(sigmas[i + 1]);

    // v_prediction -> data prediction
    std::vector<float> x0(n);
    for (size_t j = 0; j < n; ++j) {
        x0[j] = alpha_s0 * x[j] - sigma_s0 * v[j];
    }

    if (sigma_t == 0.0) {
        // the last step is first order, with a final sigma of 0 it lands on the data prediction
        std::copy(x0.begin(), x0.end(), x);
    } else {
        const double lambda_t  = std::log(alpha_t)  - std::log(sigma_t);
        const double lambda_s0 = std::log(alpha_s0) - std::log(sigma_s0);
        const double h = lambda_t - lambda_s0;

        const double c_x  = sigma_t / sigma_s0;
        const double c_x0 = alpha_t * (std::exp(-h) - 1.0);

        if (x0_prev.empty()) {
            // first order (DDIM) for the first step
            for (size_t j = 0; j < n; ++j) {
                x[j] = c_x * x[j] - c_x0 * x0[j];
            }
        } else {
            const auto [alpha_s1, sigma_s1] = alpha_sigma(sigmas[i - 1]);
            const double lambda_s1 = std::log(alpha_s1) - std::log(sigma_s1);
            const double r0 = (lambda_s0 - lambda_s1) / h;

            for (size_t j = 0; j < n; ++j) {
                const double d1 = (x0[j] - x0_prev[j]) / r0;
                x[j] = c_x * x[j] - c_x0 * x0[j] - 0.5 * c_x0 * d1;
            }
        }
    }

    x0_prev = std::move(x0);
    i++;
}

//
// generation wiring: CFG + speech feedback embeds
//

void llama_model_vibevoice::cfg_guide(const float * v_cond, const float * v_uncond, float * guided, size_t n, float scale) {
    for (size_t i = 0; i < n; ++i) {
        guided[i] = v_uncond[i] + scale * (v_cond[i] - v_uncond[i]);
    }
}

void llama_model_vibevoice::unscale_speech_latents(float * latents, size_t n) const {
    const float scale = hparams.vibevoice.speech_scaling_factor;
    const float bias  = hparams.vibevoice.speech_bias_factor;
    for (size_t i = 0; i < n; ++i) {
        latents[i] = latents[i] / scale - bias;
    }
}

// acoustic_conn(scaled latents) + semantic_conn(semantic_enc(wav)), matches upstream feedback embeds
ggml_tensor * llama_model_vibevoice::build_speech_feedback(ggml_context * ctx, ggml_tensor * scaled_latents, ggml_tensor * wav) const {
    const auto & vv = hparams.vibevoice;

    GGML_ASSERT(ggml_is_matrix(scaled_latents) && scaled_latents->ne[0] == vv.diff_latent_size);
    GGML_ASSERT(ggml_is_matrix(wav) && wav->ne[1] == 1);

    ggml_tensor * acoustic_emb = build_connector(ctx, scaled_latents, acoustic_conn);

    ggml_tensor * semantic_latents = build_semantic_enc(ctx, wav);
    GGML_ASSERT(semantic_latents->ne[1] == scaled_latents->ne[1]);

    ggml_tensor * semantic_emb = build_connector(ctx, semantic_latents, semantic_conn);

    return ggml_add(ctx, acoustic_emb, semantic_emb);
}
