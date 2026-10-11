#include "models.h"

void llama_model_clef_omni::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen3vlmoe::load_arch_hparams(ml);
    head.load_hparams(ml, hparams);
}

void llama_model_clef_omni::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen3vlmoe::load_arch_tensors(ml);
    head.load_tensors(ml, *this);
}

std::unique_ptr<llm_graph_context> llama_model_clef_omni::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// the backbone is copied from llama_model_qwen3vlmoe::graph, without the memory module
llama_model_clef_omni::graph::graph(const llama_model & model_base, const llm_graph_params & params) : llm_graph_context(params) {
    const auto & model = static_cast<const llama_model_clef_omni &>(model_base);

    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = llama_clef_head::build_attn_inp_causal(*this);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);

        Qcur = ggml_rope_multi(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        Kcur = ggml_rope_multi(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cur = build_attn(inp_attn,
                model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        if (il < (int) hparams.n_deepstack_layers) {
            ggml_tensor * ds = ggml_view_2d(ctx0, res->t_inp_embd, n_embd, n_tokens, res->t_inp_embd->nb[1], (il + 1) * n_embd * sizeof(float));
            cur = ggml_add(ctx0, cur, ds);
            cb(cur, "deepstack_out", il);
        }

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // the head is always evaluated, so that the graph has the same nodes for every batch
    cur = model.head.build(*this, cur, model.output);

    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
