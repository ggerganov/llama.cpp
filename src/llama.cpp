#include "llama-impl.h"

#include "llama-chat.h"
#include "llama-mmap.h"
#include "llama-context.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// llm_build
//

using llm_build_cb = std::function<void(struct ggml_tensor * cur, const char * name, int nl)>;

enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

struct llm_build_context {
          llama_context      & lctx;
    const llama_model        & model;
    const llama_hparams      & hparams;
    const llama_cparams      & cparams;
    const llama_ubatch       & ubatch;
    const llama_adapter_cvec & cvec;
    const llama_loras        & loras;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int32_t n_tokens;
    const int32_t n_outputs;
    const int32_t n_outputs_enc;
    const int32_t n_ctx_orig;

    const bool worst_case;
    const bool flash_attn;

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    const llm_build_cb & cb;

    std::vector<uint8_t> & buf_compute_meta;

    struct ggml_context * ctx0 = nullptr;

    // TODO: consider making the entire interface noexcept
    llm_build_context(
            llama_context & lctx,
       const llama_ubatch & ubatch,
       const llm_build_cb & cb,
                     bool   worst_case) :
        lctx             (lctx),
        model            (lctx.model),
        hparams          (model.hparams),
        cparams          (lctx.cparams),
        ubatch           (ubatch),
        cvec             (lctx.cvec),
        loras            (lctx.loras),
        n_embd           (hparams.n_embd),
        n_layer          (hparams.n_layer),
        n_rot            (hparams.n_rot),
        n_ctx            (cparams.n_ctx),
        n_head           (hparams.n_head()),
        n_head_kv        (hparams.n_head_kv()),
        n_embd_head_k    (hparams.n_embd_head_k),
        n_embd_k_gqa     (hparams.n_embd_k_gqa()),
        n_embd_head_v    (hparams.n_embd_head_v),
        n_embd_v_gqa     (hparams.n_embd_v_gqa()),
        n_expert         (hparams.n_expert),
        n_expert_used    (hparams.n_expert_used),
        freq_base        (cparams.rope_freq_base),
        freq_scale       (cparams.rope_freq_scale),
        ext_factor       (cparams.yarn_ext_factor),
        attn_factor      (cparams.yarn_attn_factor),
        beta_fast        (cparams.yarn_beta_fast),
        beta_slow        (cparams.yarn_beta_slow),
        norm_eps         (hparams.f_norm_eps),
        norm_rms_eps     (hparams.f_norm_rms_eps),
        n_tokens         (ubatch.n_tokens),
        n_outputs        (worst_case ? n_tokens : lctx.n_outputs),
        n_outputs_enc    (worst_case ? n_tokens : lctx.embd_enc.size() / hparams.n_embd),
        n_ctx_orig       (cparams.n_ctx_orig_yarn),
        worst_case       (worst_case),
        flash_attn       (cparams.flash_attn),
        pooling_type     (cparams.pooling_type),
        rope_type        (hparams.rope_type),
        cb               (cb),
        buf_compute_meta (lctx.buf_compute_meta) {
            // all initializations should be done in init()
        }

    void init() {
        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_compute_meta.size(),
            /*.mem_buffer =*/ buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };

        ctx0 = ggml_init(params);

        lctx.reset();
    }

    void free() {
        ggml_free(ctx0);
        ctx0 = nullptr;
    }

    struct ggml_tensor * build_inp_embd(struct ggml_tensor * tok_embd) {
        struct ggml_tensor * inpL;

        if (ubatch.token) {
            lctx.inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
            cb(lctx.inp_tokens, "inp_tokens", -1);
            ggml_set_input(lctx.inp_tokens);

            inpL = ggml_get_rows(ctx0, tok_embd, lctx.inp_tokens);

            // apply lora for embedding tokens if needed
            for (const auto & lora : loras) {
                struct llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
                if (lw == nullptr) {
                    continue;
                }

                const float adapter_scale = lora.second;
                const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

                struct ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                            ctx0, lw->b, // non-transposed lora_b
                            ggml_get_rows(ctx0, lw->a, lctx.inp_tokens)
                            ), scale);

                inpL = ggml_add(ctx0, inpL, inpL_delta);
            }
        } else {
            lctx.inp_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ubatch.n_tokens);
            inpL = lctx.inp_embd;
            ggml_set_input(lctx.inp_embd);
        }

        // For Granite architecture
        if (hparams.f_embedding_scale != 0.0f) {
            inpL = ggml_scale(ctx0, inpL, hparams.f_embedding_scale);
        }

        cb(inpL, "inp_embd", -1);

        return inpL;
    }

    // do mat_mul, while optionally apply lora
    struct ggml_tensor * build_lora_mm(
              struct ggml_tensor * w,
              struct ggml_tensor * cur) {
        struct ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

        for (const auto & lora : loras) {
            struct llama_adapter_lora_weight * lw = lora.first->get_weight(w);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            struct ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
            );

            ab_cur = ggml_scale(ctx0, ab_cur, scale);
            res = ggml_add(ctx0, res, ab_cur);
        }

        return res;
    }

    // do mat_mul_id, while optionally apply lora
    struct ggml_tensor * build_lora_mm_id(
              struct ggml_tensor * w,   // struct ggml_tensor * as
              struct ggml_tensor * cur, // struct ggml_tensor * b
              struct ggml_tensor * ids) {
        struct ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
        for (const auto & lora : loras) {
            struct llama_adapter_lora_weight * lw = lora.first->get_weight(w);
            if (lw == nullptr) {
                continue;
            }

            const float alpha = lora.first->alpha;
            const float rank  = (float) lw->b->ne[0];
            const float scale = alpha ? lora.second * alpha / rank : lora.second;

            struct ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
            );

            ab_cur = ggml_scale(ctx0, ab_cur, scale);
            res = ggml_add(ctx0, res, ab_cur);
        }

        return res;
    }

    struct ggml_tensor * build_norm(
             struct ggml_tensor * cur,
             struct ggml_tensor * mw,
             struct ggml_tensor * mb,
                  llm_norm_type   type,
                            int   il) {
        switch (type) {
            case LLM_NORM:       cur = ggml_norm      (ctx0, cur, hparams.f_norm_eps);     break;
            case LLM_NORM_RMS:   cur = ggml_rms_norm  (ctx0, cur, hparams.f_norm_rms_eps); break;
            case LLM_NORM_GROUP:
                {
                    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                    cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
                } break;
        }

        if (mw || mb) {
            cb(cur, "norm", il);
        }

        if (mw) {
            cur = ggml_mul(ctx0, cur, mw);
            if (mb) {
                cb(cur, "norm_w", il);
            }
        }

        if (mb) {
            cur = ggml_add(ctx0, cur, mb);
        }

        return cur;
    }

    struct ggml_tensor * build_ffn(
             struct ggml_tensor * cur,
             struct ggml_tensor * up,
             struct ggml_tensor * up_b,
             struct ggml_tensor * up_s,
             struct ggml_tensor * gate,
             struct ggml_tensor * gate_b,
             struct ggml_tensor * gate_s,
             struct ggml_tensor * down,
             struct ggml_tensor * down_b,
             struct ggml_tensor * down_s,
             struct ggml_tensor * act_scales,
                llm_ffn_op_type   type_op,
              llm_ffn_gate_type   type_gate,
             const llm_build_cb & cb,
                            int   il) {
        struct ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
        cb(tmp, "ffn_up", il);

        if (up_b) {
            tmp = ggml_add(ctx0, tmp, up_b);
            cb(tmp, "ffn_up_b", il);
        }

        if (up_s) {
            tmp = ggml_mul(ctx0, tmp, up_s);
            cb(tmp, "ffn_up_s", il);
        }

        if (gate) {
            switch (type_gate) {
                case LLM_FFN_SEQ:
                    {
                        cur = build_lora_mm(gate, tmp);
                        cb(cur, "ffn_gate", il);
                    } break;
                case LLM_FFN_PAR:
                    {
                        cur = build_lora_mm(gate, cur);
                        cb(cur, "ffn_gate", il);
                    } break;
            }

            if (gate_b) {
                cur = ggml_add(ctx0, cur, gate_b);
                cb(cur, "ffn_gate_b", il);
            }

            if (gate_s) {
                cur = ggml_mul(ctx0, cur, gate_s);
                cb(cur, "ffn_gate_s", il);
            }

        } else {
            cur = tmp;
        }

        switch (type_op) {
            case LLM_FFN_SILU:
                {
                    cur = ggml_silu(ctx0, cur);
                    cb(cur, "ffn_silu", il);
                } break;
            case LLM_FFN_GELU:
                {
                    cur = ggml_gelu(ctx0, cur);
                    cb(cur, "ffn_gelu", il);
                    if (act_scales != NULL) {
                        cur = ggml_div(ctx0, cur, act_scales);
                        cb(cur, "ffn_act", il);
                    }
                } break;
            case LLM_FFN_RELU:
                {
                    cur = ggml_relu(ctx0, cur);
                    cb(cur, "ffn_relu", il);
                } break;
            case LLM_FFN_RELU_SQR:
                {
                    cur = ggml_relu(ctx0, cur);
                    cb(cur, "ffn_relu", il);

                    cur = ggml_sqr(ctx0, cur);
                    cb(cur, "ffn_sqr(relu)", il);
                } break;
            case LLM_FFN_SWIGLU:
                {
                    // Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
                    int64_t split_point = cur->ne[0] / 2;
                    struct ggml_tensor * x0 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, split_point, cur->ne[1], cur->nb[1], 0));
                    struct ggml_tensor * x1 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, split_point, cur->ne[1], cur->nb[1], split_point * ggml_element_size(cur)));

                    x0 = ggml_silu(ctx0, x0);
                    cb(cur, "ffn_silu", il);

                    cur = ggml_mul(ctx0, x0, x1);
                    cb(cur, "ffn_mul", il);
                } break;
        }

        if (type_gate == LLM_FFN_PAR) {
            cur = ggml_mul(ctx0, cur, tmp);
            cb(cur, "ffn_gate_par", il);
        }

        if (down) {
            cur = build_lora_mm(down, cur);
        }

        if (down_b) {
            cb(cur, "ffn_down", il);
        }

        if (down_b) {
            cur = ggml_add(ctx0, cur, down_b);
        }

        if (down_s) {
            cur = ggml_mul(ctx0, cur, down_s);
            cb(cur, "ffn_down_s", il);
        }

        return cur;
    }

    struct ggml_tensor * build_moe_ffn(
             struct ggml_tensor * cur,
             struct ggml_tensor * gate_inp,
             struct ggml_tensor * up_exps,
             struct ggml_tensor * gate_exps,
             struct ggml_tensor * down_exps,
             struct ggml_tensor * exp_probs_b,
                        int64_t   n_expert,
                        int64_t   n_expert_used,
                llm_ffn_op_type   type_op,
                           bool   norm_w,
                           bool   scale_w,
                          float   w_scale,
  llama_expert_gating_func_type   gating_op,
             const llm_build_cb & cb,
                            int   il) {
        int64_t n_embd = cur->ne[0];
        int64_t n_tokens = cur->ne[1];

        ggml_tensor * logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);

        ggml_tensor * probs = nullptr;
        switch (gating_op) {
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
                {
                    probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
                } break;
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
                {
                    probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
                } break;
            default:
                GGML_ABORT("fatal error");
        }
        cb(probs, "ffn_moe_probs", il);

        // add experts selection bias - introduced in DeepSeek V3
        // leave probs unbiased as it's later used to get expert weights
        ggml_tensor * selection_probs = probs;
        if (exp_probs_b != nullptr) {
            selection_probs = ggml_add(ctx0, probs, exp_probs_b);
            cb(selection_probs, "ffn_moe_probs_biased", il);
        }

        // select experts
        ggml_tensor * selected_experts = ggml_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
        cb(selected_experts, "ffn_moe_topk", il);

        ggml_tensor * weights = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights", il);

        if (norm_w) {
            weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

            ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
            cb(weights_sum, "ffn_moe_weights_sum", il);

            weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
            cb(weights, "ffn_moe_weights_norm", il);

            weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        }
        if (scale_w) {
            weights = ggml_scale(ctx0, weights, w_scale);
            cb(weights, "ffn_moe_weights_scaled", il);
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        ggml_tensor * up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        ggml_tensor * gate = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(gate, "ffn_moe_gate", il);

        switch (type_op) {
            case LLM_FFN_SILU:
                {
                    gate = ggml_silu(ctx0, gate);
                    cb(gate, "ffn_moe_silu", il);
                } break;
            case LLM_FFN_GELU:
                {
                    gate = ggml_gelu(ctx0, gate);
                    cb(gate, "ffn_moe_gelu", il);
                } break;
            default:
                GGML_ABORT("fatal error");
        }

        ggml_tensor * par = ggml_mul(ctx0, up, gate); // [n_ff, n_expert_used, n_tokens]
        cb(par, "ffn_moe_gate_par", il);

        ggml_tensor * experts = build_lora_mm_id(down_exps, par, selected_experts); // [n_embd, n_expert_used, n_tokens]
        cb(experts, "ffn_moe_down", il);

        experts = ggml_mul(ctx0, experts, weights);

        // aggregate experts
        ggml_tensor * moe_out = nullptr;
        for (int i = 0; i < n_expert_used; ++i) {
            ggml_tensor * cur_expert = ggml_view_2d(ctx0, experts, n_embd, n_tokens,
                    experts->nb[2], i*experts->nb[1]);

            if (i == 0) {
                moe_out = cur_expert;
            } else {
                moe_out = ggml_add(ctx0, moe_out, cur_expert);
            }
        }

        if (n_expert_used == 1) {
            // avoid returning a non-contiguous tensor
            moe_out = ggml_cont(ctx0, moe_out);
        }

        return moe_out;
    }

    struct ggml_tensor * build_attn(
             struct ggml_cgraph * graph,
             struct ggml_tensor * wo,
             struct ggml_tensor * wo_b,
             struct ggml_tensor * k_cur,
             struct ggml_tensor * v_cur,
             struct ggml_tensor * q_cur,
                        int32_t   n_tokens,
                        float     kq_scale,
             const llm_build_cb & cb,
                        int       il) {
        // these nodes are added to the graph together so that they are not reordered
        // by doing so, the number of splits in the graph is reduced
        ggml_build_forward_expand(graph, q_cur);
        ggml_build_forward_expand(graph, k_cur);
        ggml_build_forward_expand(graph, v_cur);

        //build_kv_store(graph, k_cur, v_cur, il);
        lctx.build_attn_kv_store(ctx0, graph, k_cur, v_cur, n_tokens, il, worst_case);

        struct ggml_tensor * cur;

        //cur = build_kqv(graph, wo, wo_b, q_cur, kq_mask, kq_scale, il);
        cur = lctx.build_attn_qkv(ctx0, graph, wo, wo_b, q_cur, n_tokens, kq_scale, il, worst_case);
        cb(cur, "kqv_out", il);

        return cur;
    }

    //struct ggml_tensor * build_rwkv6_time_mix(
    //        const struct llama_layer * layer,
    //        struct ggml_tensor * cur,
    //        struct ggml_tensor * x_prev,
    //        struct ggml_tensor ** wkv_state,
    //        size_t wkv_head_size,
    //        size_t head_count_kv) {
    //    size_t n_embd       = cur->ne[0];
    //    size_t n_seq_tokens = cur->ne[1];
    //    size_t n_seqs       = cur->ne[2];

    //    size_t head_size  = wkv_head_size;
    //    size_t head_count = n_embd / head_size;

    //    size_t n_tokens = n_seqs * n_seq_tokens;

    //    bool is_qrwkv = layer->time_mix_first == nullptr;

    //    struct ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);

    //    sx  = ggml_reshape_2d(ctx0, sx,  n_embd, n_tokens);
    //    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);

    //    struct ggml_tensor * xxx = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->time_mix_lerp_x), cur);

    //    xxx = ggml_reshape_4d(
    //        ctx0,
    //        ggml_tanh(
    //            ctx0,
    //            ggml_mul_mat(ctx0, layer->time_mix_w1, xxx)
    //        ),
    //        layer->time_mix_w1->ne[1] / 5, 1, 5, n_tokens
    //    );

    //    xxx = ggml_cont(ctx0, ggml_permute(ctx0, xxx, 0, 1, 3, 2));

    //    xxx = ggml_mul_mat(
    //        ctx0,
    //        ggml_reshape_4d(
    //            ctx0,
    //            layer->time_mix_w2,
    //            layer->time_mix_w2->ne[0], layer->time_mix_w2->ne[1], 1, 5
    //        ),
    //        xxx
    //    );

    //    struct ggml_tensor *xw, *xk, *xv, *xr, *xg;
    //    if (layer->time_mix_lerp_fused) {
    //        // fusing these weights makes some performance improvement
    //        sx  = ggml_reshape_3d(ctx0, sx,  n_embd, 1, n_tokens);
    //        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    //        xxx = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xxx, layer->time_mix_lerp_fused), sx), cur);
    //        xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
    //        xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
    //        xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
    //        xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
    //        xg = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));
    //    } else {
    //        // for backward compatibility
    //        xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
    //        xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
    //        xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
    //        xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
    //        xg = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));

    //        xw = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xw, layer->time_mix_lerp_w), sx), cur);
    //        xk = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xk, layer->time_mix_lerp_k), sx), cur);
    //        xv = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xv, layer->time_mix_lerp_v), sx), cur);
    //        xr = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xr, layer->time_mix_lerp_r), sx), cur);
    //        xg = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xg, layer->time_mix_lerp_g), sx), cur);
    //    }

    //    struct ggml_tensor * r = build_lora_mm(layer->time_mix_receptance, xr);
    //    struct ggml_tensor * k = build_lora_mm(layer->time_mix_key,        xk);
    //    struct ggml_tensor * v = build_lora_mm(layer->time_mix_value,      xv);
    //    if (layer->time_mix_receptance_b) {
    //        r = ggml_add(ctx0, r, layer->time_mix_receptance_b);
    //    }
    //    if (layer->time_mix_key_b) {
    //        k = ggml_add(ctx0, k, layer->time_mix_key_b);
    //    }
    //    if (layer->time_mix_value_b) {
    //        v = ggml_add(ctx0, v, layer->time_mix_value_b);
    //    }

    //    struct ggml_tensor * g = build_lora_mm(layer->time_mix_gate, xg);
    //    if (is_qrwkv) {
    //        g = ggml_sigmoid(ctx0, g);
    //    } else {
    //        g = ggml_silu(ctx0, g);
    //    }

    //    if (head_count_kv != head_count) {
    //        GGML_ASSERT(head_count % head_count_kv == 0);
    //        k = ggml_reshape_4d(ctx0, k, head_size, 1, head_count_kv, n_tokens);
    //        v = ggml_reshape_4d(ctx0, v, head_size, 1, head_count_kv, n_tokens);
    //        struct ggml_tensor * tmp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_size, head_count / head_count_kv, head_count_kv, n_tokens);
    //        k = ggml_repeat(ctx0, k, tmp);
    //        v = ggml_repeat(ctx0, v, tmp);
    //    }

    //    k = ggml_reshape_3d(ctx0, k, head_size, head_count, n_tokens);
    //    v = ggml_reshape_3d(ctx0, v, head_size, head_count, n_tokens);
    //    r = ggml_reshape_3d(ctx0, r, head_size, head_count, n_tokens);

    //    struct ggml_tensor * w = ggml_mul_mat(
    //        ctx0,
    //        layer->time_mix_decay_w2,
    //        ggml_tanh(
    //            ctx0,
    //            ggml_mul_mat(ctx0, layer->time_mix_decay_w1, xw)
    //        )
    //    );

    //    w = ggml_add(ctx0, w, layer->time_mix_decay);
    //    w = ggml_exp(ctx0, ggml_neg(ctx0, ggml_exp(ctx0, w)));
    //    w = ggml_reshape_3d(ctx0, w, head_size, head_count, n_tokens);

    //    if (is_qrwkv) {
    //        // k = k * (1 - w)
    //        k = ggml_sub(ctx0, k, ggml_mul(ctx0, k, w));
    //    }

    //    struct ggml_tensor * wkv_output;
    //    if (!layer->time_mix_first) {
    //        wkv_output = ggml_gated_linear_attn(ctx0, k, v, r, w, *wkv_state, pow(head_size, -0.5f));
    //    } else {
    //        wkv_output = ggml_rwkv_wkv6(ctx0, k, v, r, layer->time_mix_first, w, *wkv_state);
    //    }
    //    cur = ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
    //    *wkv_state = ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));

    //    if (!is_qrwkv) {
    //        // group norm with head_count groups
    //        cur = ggml_reshape_3d(ctx0, cur, n_embd / head_count, head_count, n_tokens);
    //        cur = ggml_norm(ctx0, cur, 64e-5f);

    //        // Convert back to regular vectors.
    //        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    //        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer->time_mix_ln), layer->time_mix_ln_b);
    //    } else {
    //        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    //    }

    //    cur = ggml_mul(ctx0, cur, g);
    //    cur = build_lora_mm(layer->time_mix_output, cur);

    //    return ggml_reshape_3d(ctx0, cur, n_embd, n_seq_tokens, n_seqs);
    //}

    //struct ggml_tensor * build_rwkv6_channel_mix(
    //    const struct llama_layer * layer,
    //    struct ggml_tensor * cur,
    //    struct ggml_tensor * x_prev) {
    //    struct ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
    //    struct ggml_tensor * xk = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);
    //    struct ggml_tensor * xr = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_r), cur);

    //    struct ggml_tensor * r = ggml_sigmoid(ctx0, build_lora_mm(layer->channel_mix_receptance, xr));
    //    struct ggml_tensor * k = ggml_sqr(
    //            ctx0,
    //            ggml_relu(
    //                ctx0,
    //                build_lora_mm(layer->channel_mix_key, xk)
    //                )
    //            );

    //    return ggml_mul(ctx0, r, build_lora_mm(layer->channel_mix_value, k));
    //}

    struct ggml_cgraph * build_k_shift() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        lctx.build_k_shift(ctx0, gf);

        return gf;
    }

    struct ggml_cgraph * build_defrag() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        lctx.build_defrag(ctx0, gf);

        return gf;
    }

    struct ggml_tensor * build_inp_pos() {
        lctx.inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        cb(lctx.inp_pos, "inp_pos", -1);
        ggml_set_input(lctx.inp_pos);
        return lctx.inp_pos;
    }

    struct ggml_tensor * build_inp_out_ids() {
        lctx.inp_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
        cb(lctx.inp_out_ids, "inp_out_ids", -1);
        ggml_set_input(lctx.inp_out_ids);
        return lctx.inp_out_ids;
    }

    struct ggml_tensor * build_inp_mean() {
        lctx.inp_mean = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
        cb(lctx.inp_mean, "inp_mean", -1);
        ggml_set_input(lctx.inp_mean);
        return lctx.inp_mean;
    }

    struct ggml_tensor * build_inp_cls() {
        lctx.inp_cls = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        cb(lctx.inp_cls, "inp_cls", -1);
        ggml_set_input(lctx.inp_cls);
        return lctx.inp_cls;
    }

    struct ggml_cgraph * append_pooling(struct ggml_cgraph * gf) {
        // find result_norm tensor for input
        struct ggml_tensor * inp = nullptr;
        for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
            inp = ggml_graph_node(gf, i);
            if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
                break;
            }

            inp = nullptr;
        }
        GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

        struct ggml_tensor * cur;

        switch (pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    cur = inp;
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
                {
                    struct ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } break;
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    struct ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    struct ggml_tensor * inp_cls = build_inp_cls();
                    inp = ggml_get_rows(ctx0, inp, inp_cls);

                    // classification head
                    // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                    GGML_ASSERT(model.cls       != nullptr);
                    GGML_ASSERT(model.cls_b     != nullptr);

                    cur = ggml_add (ctx0, ggml_mul_mat(ctx0, model.cls, inp), model.cls_b);
                    cur = ggml_tanh(ctx0, cur);

                    // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                    // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                    if (model.cls_out) {
                        GGML_ASSERT(model.cls_out_b != nullptr);

                        cur = ggml_add (ctx0, ggml_mul_mat(ctx0, model.cls_out, cur), model.cls_out_b);
                    }
                } break;
            default:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }

        cb(cur, "result_embd_pooled", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    //struct ggml_tensor * build_pos_bucket(bool causal) {
    //    if (causal) {
    //        lctx.inp_pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv,     n_tokens);
    //    } else {
    //        lctx.inp_pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    //    }

    //    ggml_set_input(lctx.inp_pos_bucket);
    //    cb(lctx.inp_pos_bucket, "pos_bucket", -1);

    //    return lctx.inp_pos_bucket;
    //}

    //struct ggml_tensor * build_pos_bias(struct ggml_tensor * pos_bucket, struct ggml_tensor * attn_rel_b) {
    //    struct ggml_tensor * pos_bucket_1d = ggml_view_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1], 0);
    //    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    //    struct ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);
    //    cb(pos_bias, "pos_bias", -1);

    //    pos_bias = ggml_view_3d(ctx0, pos_bias, pos_bias->ne[0], lctx.inp_pos_bucket->ne[0], lctx.inp_pos_bucket->ne[1], ggml_element_size(pos_bias) * pos_bias->ne[0], ggml_element_size(pos_bias) * pos_bias->ne[0] * lctx.inp_pos_bucket->ne[0],  0);
    //    cb(pos_bias, "pos_bias", -1);

    //    pos_bias = ggml_permute(ctx0, pos_bias, 2, 0, 1, 3);
    //    cb(pos_bias, "pos_bias", -1);

    //    pos_bias = ggml_cont(ctx0, pos_bias);
    //    cb(pos_bias, "pos_bias", -1);

    //    return pos_bias;
    //}

    struct ggml_tensor * build_inp_embd_enc() {
        const int64_t n_embd = hparams.n_embd;
        lctx.inp_embd_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_outputs_enc);
        ggml_set_input(lctx.inp_embd_enc);
        cb(lctx.inp_embd_enc, "embd_enc", -1);
        return lctx.inp_embd_enc;
    }

    struct ggml_tensor * build_inp_KQ_mask_cross() {
        lctx.inp_KQ_mask_cross = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_outputs_enc, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        ggml_set_input(lctx.inp_KQ_mask_cross);
        cb(lctx.inp_KQ_mask_cross, "KQ_mask_cross", -1);
        return lctx.inp_KQ_mask_cross;
    }

    struct ggml_cgraph * build_llama() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor * rope_factors = lctx.get_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, kq_scale, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {

                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        cb, il);
                cb(cur, "ffn_moe_out", il);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // For Granite architecture
        if (hparams.f_logit_scale) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
        }

        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_deci() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_head    = hparams.n_head(il);

            if (n_head == 0) {
                // attention-free layer of Llama-3_1-Nemotron-51B
                cur = inpL;
            } else {
                // norm
                cur = build_norm(inpL,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_norm", il);
            }

            if (n_head > 0 && n_head_kv == 0) {
                // "linear attention" of Llama-3_1-Nemotron-51B
                cur = build_lora_mm(model.layers[il].wo, cur);
                cb(cur, "wo", il);
            } else if (n_head > 0) {
                // self-attention
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor * rope_factors = lctx.get_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, kq_scale, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            // modified to support attention-free layer of Llama-3_1-Nemotron-51B
            struct ggml_tensor * ffn_inp = cur;
            if (n_head > 0) {
                ffn_inp = ggml_add(ctx0, cur, inpSA);
                cb(ffn_inp, "ffn_inp", il);
            }

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // For Granite architecture
        if (hparams.f_logit_scale) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
        }

        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_baichuan() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = model.type == LLM_TYPE_7B ? build_inp_pos() : nullptr;

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                switch (model.type) {
                    case LLM_TYPE_7B:
                        Qcur = ggml_rope_ext(
                            ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                        );
                        Kcur = ggml_rope_ext(
                            ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                        );
                        break;
                    case LLM_TYPE_13B:
                        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd/n_head, n_head, n_tokens);
                        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd/n_head, n_head, n_tokens);
                        break;
                    default:
                        GGML_ABORT("fatal error");
                }
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_xverse() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);
                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_falcon() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * attn_norm;

            attn_norm = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(attn_norm, "attn_norm", il);

            // self-attention
            {
                if (model.layers[il].attn_norm_2) {
                    // Falcon-40B
                    cur = build_norm(inpL,
                            model.layers[il].attn_norm_2,
                            model.layers[il].attn_norm_2_b,
                            LLM_NORM, il);
                    cb(cur, "attn_norm_2", il);
                } else {
                    cur = attn_norm;
                }

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur       = ggml_get_rows(ctx0,       cur, inp_out_ids);
                inpL      = ggml_get_rows(ctx0,      inpL, inp_out_ids);
                attn_norm = ggml_get_rows(ctx0, attn_norm, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = cur;

            // feed forward
            {
                cur = build_ffn(attn_norm, // !! use the attn norm, not the result
                        model.layers[il].ffn_up,   NULL, NULL,
                        NULL,                      NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = ggml_add(ctx0, cur, inpL);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        // norm
        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_grok() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // multiply by embedding_multiplier_scale of 78.38367176906169
        inpL = ggml_scale(ctx0, inpL, 78.38367176906169f);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);


            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // Grok
            // if attn_out_norm is present then apply it before adding the input
            if (model.layers[il].attn_out_norm) {
                cur = build_norm(cur,
                        model.layers[il].attn_out_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_out_norm", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    cb, il);
            cb(cur, "ffn_moe_out", il);

            // Grok
            // if layer_out_norm is present then apply it before adding the input
            // Idea: maybe ffn_out_norm is a better name
            if (model.layers[il].layer_out_norm) {
                cur = build_norm(cur,
                        model.layers[il].layer_out_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "layer_out_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // Grok
        // multiply logits by output_multiplier_scale of 0.5773502691896257

        cur = ggml_scale(ctx0, cur, 0.5773502691896257f);

        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_dbrx() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = nullptr;
                struct ggml_tensor * Kcur = nullptr;
                struct ggml_tensor * Vcur = nullptr;

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(cur, "wqkv_clamped", il);

                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].attn_out_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_out_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    cb, il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_starcoder() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        struct ggml_tensor * pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_refact() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                cb(Kcur, "Kcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                cb(Qcur, "Qcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_bert() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;
        struct ggml_tensor * inp_pos = nullptr;

        if (model.arch != LLM_ARCH_JINA_BERT_V2) {
            inp_pos = build_inp_pos();
        }

        // construct input embeddings (token, type, position)
        inpL = build_inp_embd(model.tok_embd);

        // token types are hardcoded to zero ("Sentence A")
        struct ggml_tensor * type_row0 = ggml_view_1d(ctx0, model.type_embd, n_embd, 0);
        inpL = ggml_add(ctx0, inpL, type_row0);
        if (model.arch == LLM_ARCH_BERT) {
            inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
        }
        cb(inpL, "inp_embd", -1);

        // embed layer norm
        inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);
        cb(inpL, "inp_norm", -1);

        lctx.build_attn_inp(ctx0, n_tokens, false, false, worst_case);

        // iterate layers
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * cur = inpL;

            struct ggml_tensor * Qcur;
            struct ggml_tensor * Kcur;
            struct ggml_tensor * Vcur;

            // self-attention
            if (model.arch == LLM_ARCH_BERT || model.arch == LLM_ARCH_JINA_BERT_V2) {
                Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, cur), model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            model.layers[il].attn_q_norm_b,
                            LLM_NORM, il);
                }

                Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, cur), model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                if (model.layers[il].attn_k_norm) {
                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            model.layers[il].attn_k_norm_b,
                            LLM_NORM, il);
                }
                Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, cur), model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            } else {
                // compute Q and K and RoPE them
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);
            }

            struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            cb(kq, "kq", il);

            //kq = ggml_soft_max_ext(ctx0, kq, KQ_mask, 1.0f/sqrtf(float(n_embd_head)), hparams.f_max_alibi_bias);
            kq = lctx.build_soft_max_ext(ctx0, kq, 1.0f/sqrtf(float(n_embd_head)));
            cb(kq, "kq_soft_max_ext", il);

            struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_tokens)));
            cb(v, "v", il);

            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_tokens, n_embd_head, n_head_kv), kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
            cb(cur, "kqv_merged_cont", il);

            ggml_build_forward_expand(gf, cur);

            cur = build_lora_mm(model.layers[il].wo, cur);
            if (model.layers[il].bo) {
                cb(cur, "kqv_wo", il);
            }

            if (model.layers[il].bo) {
                cur = ggml_add(ctx0, cur, model.layers[il].bo);
            }
            cb(cur, "kqv_out", il);

            if (il == n_layer - 1 && pooling_type == LLAMA_POOLING_TYPE_NONE) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // re-add the layer input
            cur = ggml_add(ctx0, cur, inpL);

            // attention layer norm
            cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);

            if (model.layers[il].attn_norm_2 != nullptr) {
                cur = ggml_add(ctx0, cur, inpL); // re-add the layer input
                cur = build_norm(cur, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, il);
            }

            struct ggml_tensor * ffn_inp = cur;
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (model.arch == LLM_ARCH_BERT) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            } else if (model.arch == LLM_ARCH_JINA_BERT_V2) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL,                        NULL,
                        model.layers[il].ffn_gate, NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
            } else {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            }
            cb(cur, "ffn_out", il);

            // attentions bypass the intermediate layer
            cur = ggml_add(ctx0, cur, ffn_inp);

            // output layer norm
            cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cb(cur, "result_embd", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_bloom() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        inpL = build_norm(inpL,
                model.tok_norm,
                model.tok_norm_b,
                LLM_NORM, -1);
        cb(inpL, "inp_norm", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // Add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_mpt() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * pos;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        if (model.pos_embd) {
            // inp_pos - contains the positions
            struct ggml_tensor * inp_pos = build_inp_pos();
            pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
            cb(pos, "pos_embd", -1);

            inpL = ggml_add(ctx0, inpL, pos);
            cb(inpL, "inpL", -1);
        }

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * attn_norm;

            attn_norm = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(attn_norm, "attn_norm", il);

            // self-attention
            {
                cur = attn_norm;

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                if (model.layers[il].bqkv){
                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);
                }

                if (hparams.f_clamp_kqv > 0.0f) {
                    cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(cur, "wqkv_clamped", il);
                }

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // Q/K Layernorm
                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            model.layers[il].attn_q_norm_b,
                            LLM_NORM, il);
                    cb(Qcur, "Qcur", il);

                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            model.layers[il].attn_k_norm_b,
                            LLM_NORM, il);
                    cb(Kcur, "Kcur", il);

                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                    cur = build_attn(gf,
                            model.layers[il].wo, model.layers[il].bo,
                            Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                } else {
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                    cur = build_attn(gf,
                            model.layers[il].wo, model.layers[il].bo,
                            Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
                }
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // Add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // feed forward
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        model.layers[il].ffn_act,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_stablelm() {
        struct ggml_cgraph * gf = ggml_new_graph(ctx0);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {


            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            struct ggml_tensor * inpSA = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                cb(Qcur, "Qcur", il);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                cb(Kcur, "Kcur", il);

                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            NULL,
                            LLM_NORM, il);
                    cb(Qcur, "Qcur", il);
                }
                if (model.layers[il].attn_k_norm) {
                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            NULL,
                            LLM_NORM, il);
                    cb(Kcur, "Kcur", il);
                }


                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpL  = ggml_get_rows(ctx0,  inpL, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                if (model.layers[il].ffn_norm) {
                    cur = build_norm(ffn_inp,
                            model.layers[il].ffn_norm,
                            model.layers[il].ffn_norm_b,
                            LLM_NORM, il);
                    cb(cur, "ffn_norm", il);
                } else {
                    // parallel residual
                    cur = inpSA;
                }
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_qwen() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 2*sizeof(float)*(n_embd)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward forward
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_qwen2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_qwen2vl() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);
        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        lctx.inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * 4);
        cb(lctx.inp_pos, "inp_pos", -1);
        ggml_set_input(lctx.inp_pos);
        struct ggml_tensor * inp_pos = lctx.inp_pos;

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        int sections[4];
        std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_multi(
                    ctx0,
                    ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_multi(
                    ctx0,
                    ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_qwen2moe() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor * moe_out =
                    build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, false,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        cb, il);
            cb(cur, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * cur_gate_inp = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
                cb(cur_gate_inp, "ffn_shexp_gate_inp", il);

                // sigmoid
                ggml_tensor * cur_gate = ggml_div(ctx0, ggml_silu(ctx0, cur_gate_inp), cur_gate_inp);
                cb(cur_gate, "ffn_shexp_gate", il);

                ggml_tensor * cur_ffn = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur_ffn, "ffn_shexp", il);

                ggml_tensor * ffn_shexp_out = ggml_mul(ctx0, cur_ffn, cur_gate);
                cb(ffn_shexp_out, "ffn_shexp_out", il);

                moe_out = ggml_add(ctx0, moe_out, ffn_shexp_out);
                cb(moe_out, "ffn_out", il);

                cur = moe_out;
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_phi2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * attn_norm_output;
        struct ggml_tensor * ffn_output;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            attn_norm_output = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(attn_norm_output, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = nullptr;
                struct ggml_tensor * Kcur = nullptr;
                struct ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = build_lora_mm(model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);

                    Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                    Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
                } else {
                    Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                // with phi2, we scale the Q to avoid precision issues
                // ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
                Qcur = ggml_scale(ctx0, Qcur, 1.0f/sqrtf(float(n_embd_head)));
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur              = ggml_get_rows(ctx0,              cur, inp_out_ids);
                inpL             = ggml_get_rows(ctx0,             inpL, inp_out_ids);
                attn_norm_output = ggml_get_rows(ctx0, attn_norm_output, inp_out_ids);
            }

            // FF
            {
                ffn_output = build_ffn(attn_norm_output,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(ffn_output, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_output);
            cur = ggml_add(ctx0, cur, inpL);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output_no_bias", -1);

        cur = ggml_add(ctx0, cur, model.output_b);
        cb(cur, "result_output", -1);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    struct ggml_cgraph * build_phi3() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        lctx.build_attn_inp(ctx0, n_tokens, true, true, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            auto residual = inpL;

            // self-attention
            {
                // rope freq factors for 128k context
                struct ggml_tensor * rope_factors = lctx.get_rope_factors(il);

                struct ggml_tensor* attn_norm_output = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM_RMS, il);
                cb(attn_norm_output, "attn_norm", il);

                struct ggml_tensor * Qcur = nullptr;
                struct ggml_tensor * Kcur = nullptr;
                struct ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = build_lora_mm(model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0 * sizeof(float) * (n_embd)));
                    Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd)));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa)));
                } else {
                    Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }

            cur = ggml_add(ctx0, cur, residual);
            residual = cur;

            cur = build_norm(cur,
                model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        NULL,                      NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SWIGLU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                cur = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        cb, il);
                cb(cur, "ffn_moe_out", il);
            }

            cur = ggml_add(ctx0, residual, cur);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);

        if (model.output_b != nullptr) {
            cb(cur, "result_output_no_bias", -1);
            cur = ggml_add(ctx0, cur, model.output_b);
        }
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }


    struct ggml_cgraph * build_plamo() {
        struct ggml_cgraph * gf = ggml_new_graph(ctx0);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            struct ggml_tensor * attention_norm = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Qcur, n_rot, n_head,    n_tokens), inp_pos, nullptr,
                        n_embd_head, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Kcur, n_rot, n_head_kv, n_tokens), inp_pos, nullptr,
                        n_embd_head, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }
            struct ggml_tensor * sa_out = cur;

            cur = attention_norm;

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur    = ggml_get_rows(ctx0,    cur, inp_out_ids);
                sa_out = ggml_get_rows(ctx0, sa_out, inp_out_ids);
                inpL   = ggml_get_rows(ctx0,   inpL, inp_out_ids);
            }

            // feed-forward network
            {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, sa_out);
            cur = ggml_add(ctx0, cur, inpL);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_gpt2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * pos;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_codeshell() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * tmpq = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * tmpk = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(tmpq, "tmpq", il);
                cb(tmpk, "tmpk", il);
                cb(Vcur, "Vcur", il);

                struct ggml_tensor * Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_orion() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                // if (model.layers[il].bq) {
                //     Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                //     cb(Qcur, "Qcur", il);
                // }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                // if (model.layers[il].bk) {
                //     Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                //     cb(Kcur, "Kcur", il);
                // }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                // if (model.layers[il].bv) {
                //     Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                //     cb(Vcur, "Vcur", il);
                // }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_internlm2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_minicpm3() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        //TODO: if the model varies, these parameters need to be read from the model
        const int64_t n_embd_base = 256;
        const float scale_embd  = 12.0f;
        const float scale_depth = 1.4f;
        const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k));

        const uint32_t n_embd_head_qk_rope = hparams.n_rot;
        const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
        const uint32_t kv_lora_rank = hparams.n_lora_kv;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // scale the input embeddings
        inpL = ggml_scale(ctx0, inpL, scale_embd);
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            struct ggml_tensor * rope_factors = lctx.get_rope_factors(il);
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                struct ggml_tensor * q = NULL;
                // {n_embd, q_lora_rank} * {n_embd, n_tokens} -> {q_lora_rank, n_tokens}
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = build_norm(q,
                        model.layers[il].attn_q_a_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(q, "q", il);

                // {q_lora_rank, n_head * hparams.n_embd_head_k} * {q_lora_rank, n_tokens} -> {n_head * hparams.n_embd_head_k, n_tokens}
                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        0);
                cb(q_nope, "q_nope", il);

                // and {n_head * n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * kv_pe_compresseed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_pe_compresseed, "kv_pe_compresseed", il);

                // split into {kv_lora_rank, n_tokens}
                struct ggml_tensor * kv_compressed = ggml_view_2d(ctx0, kv_pe_compresseed, kv_lora_rank, n_tokens,
                        kv_pe_compresseed->nb[1],
                        0);
                cb(kv_compressed, "kv_compressed", il);

                // and {n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_pe_compresseed, n_embd_head_qk_rope, 1, n_tokens,
                        kv_pe_compresseed->nb[1],
                        kv_pe_compresseed->nb[1],
                        ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
                cb(k_pe, "k_pe", il);

                kv_compressed = ggml_cont(ctx0, kv_compressed); // TODO: the CUDA backend does not support non-contiguous norm
                kv_compressed = build_norm(kv_compressed,
                        model.layers[il].attn_kv_a_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(kv_compressed, "kv_compressed", il);

                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                struct ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                        ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                struct ggml_tensor * v_states = ggml_view_3d(ctx0, kv, hparams.n_embd_head_v, n_head, n_tokens,
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)*n_head),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                v_states = ggml_view_2d(ctx0, v_states, hparams.n_embd_head_v * n_head, n_tokens,
                    ggml_row_size(kv->type, hparams.n_embd_head_v * n_head),
                    0);
                cb(v_states, "v_states", il);

                q_pe = ggml_cont(ctx0, q_pe); // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                q_pe = ggml_rope_ext(
                    ctx0, q_pe, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_cont(ctx0, k_pe); // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                k_pe = ggml_rope_ext(
                    ctx0, k_pe, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(k_pe, "k_pe", il);

                struct ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(q_states, "q_states", il);

                struct ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(k_states, "k_states", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        k_states, v_states, q_states, n_tokens, kq_scale, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // scale_res - scale the hidden states for residual connection
            const float scale_res = scale_depth/sqrtf(float(n_layer));
            cur = ggml_scale(ctx0, cur, scale_res);
            cb(cur, "hidden_scaled", il);

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            // scale the hidden states for residual connection
            cur = ggml_scale(ctx0, cur, scale_res);
            cb(cur, "hidden_scaled_ffn", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head scaling
        const float scale_lmhead = float(n_embd_base)/float(n_embd);
        cur = ggml_scale(ctx0, cur, scale_lmhead);
        cb(cur, "lmhead_scaling", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_gemma() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head_k = hparams.n_embd_head_k;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head_k)));
                cb(Qcur, "Qcur_scaled", il);

                Kcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            struct ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
            cb(sa_out, "sa_out", il);

            cur = build_norm(sa_out,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, sa_out);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_gemma2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head_k = hparams.n_embd_head_k;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, true, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                // ref: https://github.com/google/gemma_pytorch/commit/03e657582d17cb5a8617ebf333c1c16f3694670e
                switch (model.type) {
                    case LLM_TYPE_2B:
                    case LLM_TYPE_9B:  Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head_k)));   break;
                    case LLM_TYPE_27B: Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd / n_head))); break;
                    default: GGML_ABORT("fatal error");
                };
                cb(Qcur, "Qcur_scaled", il);

                Kcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f, cb, il);
            }

            cur = build_norm(cur,
                    model.layers[il].attn_post_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            struct ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
            cb(sa_out, "sa_out", il);

            cur = build_norm(sa_out,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = build_norm(cur,
                model.layers[il].ffn_post_norm, NULL,
                LLM_NORM_RMS, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, sa_out);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // final logit soft-capping
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);

        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }


    struct ggml_cgraph * build_starcoder2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_mamba() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = build_inp_embd(model.tok_embd);

        struct ggml_tensor * state_copy = lctx.build_inp_s_copy(ctx0, worst_case);
        struct ggml_tensor * state_mask = lctx.build_inp_s_mask(ctx0, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            //cur = build_mamba_layer(gf, cur, state_copy, state_mask, il);
            cur = lctx.build_mamba_layer(ctx0, gf, cur, state_copy, state_mask, ubatch, il, worst_case);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // residual
            cur = ggml_add(ctx0, cur, inpL);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        // final rmsnorm
        cur = build_norm(inpL,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_command_r() {

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        const float f_logit_scale = hparams.f_logit_scale;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);
            struct ggml_tensor * ffn_inp = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                if (model.layers[il].attn_q_norm) {
                    Qcur = ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens,
                                ggml_element_size(Qcur) * n_embd_head,
                                ggml_element_size(Qcur) * n_embd_head * n_head,
                                0);
                    cb(Qcur, "Qcur", il);
                    Kcur = ggml_view_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens,
                                ggml_element_size(Kcur) * n_embd_head,
                                ggml_element_size(Kcur) * n_embd_head * n_head_kv,
                                0);
                    cb(Kcur, "Kcur", il);

                    Qcur = build_norm(Qcur,
                                model.layers[il].attn_q_norm,
                                NULL,
                                LLM_NORM, il);
                    cb(Qcur, "Qcur", il);

                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            NULL,
                            LLM_NORM, il);
                    cb(Kcur, "Kcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur     = ggml_get_rows(ctx0,     cur, inp_out_ids);
                inpL    = ggml_get_rows(ctx0,    inpL, inp_out_ids);
                ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            }

            struct ggml_tensor * attn_out = cur;

            // feed-forward network
            {
                cur = build_ffn(ffn_inp,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            // add together residual + FFN + self-attention
            cur = ggml_add(ctx0, cur, inpL);
            cur = ggml_add(ctx0, cur, attn_out);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        if (f_logit_scale) {
            cur = ggml_scale(ctx0, cur, f_logit_scale);
        }

        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;

    }

    struct ggml_cgraph * build_cohere2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        const float f_logit_scale = hparams.f_logit_scale;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, true, worst_case);

        // sliding window switch pattern
        const int32_t sliding_window_pattern = 4;

        for (int il = 0; il < n_layer; ++il) {
            // three layers sliding window attention (window size 4096) and ROPE
            // fourth layer uses global attention without positional embeddings
            const bool is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);

            // norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM, il);
            cb(cur, "attn_norm", il);
            struct ggml_tensor * ffn_inp = cur;

            // self-attention
            {
                // rope freq factors for 128k context
                struct ggml_tensor * rope_factors = lctx.get_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                if (is_sliding) {
                    Qcur = ggml_rope_ext(ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                                        beta_fast, beta_slow);
                    cb(Qcur, "Qcur", il);

                    Kcur = ggml_rope_ext(ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                                        rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                        attn_factor, beta_fast, beta_slow);
                    cb(Kcur, "Kcur", il);
                } else {
                    // For non-sliding layers, just reshape without applying RoPE
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                    cb(Qcur, "Qcur", il);

                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                    cb(Kcur, "Kcur", il);
                }

                cur = build_attn(gf, model.layers[il].wo, model.layers[il].bo, Kcur, Vcur, Qcur,
                                   n_tokens, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur                              = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL                             = ggml_get_rows(ctx0, inpL, inp_out_ids);
                ffn_inp                          = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            }

            struct ggml_tensor * attn_out = cur;

            // feed-forward network
            {
                cur = build_ffn(ffn_inp, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate,
                                    NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR,
                                    cb, il);
                cb(cur, "ffn_out", il);
            }

            // add together residual + FFN + self-attention
            cur = ggml_add(ctx0, cur, inpL);
            cur = ggml_add(ctx0, cur, attn_out);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        if (f_logit_scale) {
            cur = ggml_scale(ctx0, cur, f_logit_scale);
        }

        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    // ref: https://allenai.org/olmo
    // based on the original build_llama() function, changes:
    //   * non-parametric layer norm
    //   * clamp qkv
    //   * removed bias
    //   * removed MoE
    struct ggml_cgraph * build_olmo() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    NULL, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, nullptr,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    NULL, NULL,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                NULL, NULL,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_olmo2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = inpL;

            // self_attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur_rope", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            cur = build_norm(cur,
                    model.layers[il].attn_post_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_ffn(ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = build_norm(cur,
                model.layers[il].ffn_post_norm, NULL,
                LLM_NORM_RMS, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    // based on the build_qwen2moe() function, changes:
    //   * removed shared experts
    //   * removed bias
    //   * added q, k norm
    struct ggml_cgraph * build_olmoe() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur_rope", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, false,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    cb, il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_openelm() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;
        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            const int64_t n_head    = hparams.n_head(il);
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_head_qkv = 2*n_head_kv + n_head;

            cur = inpL;
            struct ggml_tensor * residual = cur;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_reshape_3d(ctx0, cur, n_embd_head_k, n_head_qkv, n_tokens);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, cur->nb[1], cur->nb[2], 0));
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*n_head));
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*(n_head+n_head_kv)));
                cb(Vcur, "Vcur", il);

                Qcur = build_norm(Qcur,
                        model.layers[il].attn_q_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Qcur, "Qcur", il);

                Kcur = build_norm(Kcur,
                        model.layers[il].attn_k_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Kcur, "Kcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, NULL, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, NULL, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                Vcur = ggml_reshape_2d(ctx0, Vcur, n_embd_head * n_head_kv, n_tokens);
                cb(Qcur, "Vcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, residual, cur);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        cur = inpL;

        // norm
        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_gptneox() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // ffn
            if (hparams.use_par_res) {
                // attention and ffn are computed in parallel
                // x = x + attn(ln1(x)) + ffn(ln2(x))

                struct ggml_tensor * attn_out = cur;

                cur = build_norm(inpL,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, inpL);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, attn_out);
                cur = cvec.apply_to(ctx0, cur, il);
                cb(cur, "l_out", il);

                // input for next layer
                inpL = cur;
            } else {
                // attention and ffn are computed sequentially
                // x = x + attn(ln1(x))
                // x = x + ffn(ln2(x))

                struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
                cb(ffn_inp, "ffn_inp", il);

                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, ffn_inp);
                cur = cvec.apply_to(ctx0, cur, il);
                cb(cur, "l_out", il);

                // input for next layer
                inpL = cur;
            }
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_arctic() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            struct ggml_tensor * ffn_out = ggml_add(ctx0, cur, ffn_inp);
            cb(ffn_out, "ffn_out", il);

            // MoE
            cur = build_norm(inpSA,
                    model.layers[il].ffn_norm_exps, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm_exps", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    cb, il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_out);
            cb(cur, "ffn_out", il);

            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_deepseek() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor * rope_factors = lctx.get_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, kq_scale, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }


            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if ((uint32_t) il < hparams.n_layer_dense_lead) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                ggml_tensor * moe_out =
                        build_moe_ffn(cur,
                            model.layers[il].ffn_gate_inp,
                            model.layers[il].ffn_up_exps,
                            model.layers[il].ffn_gate_exps,
                            model.layers[il].ffn_down_exps,
                            nullptr,
                            n_expert, n_expert_used,
                            LLM_FFN_SILU, false,
                            false, hparams.expert_weights_scale,
                            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                            cb, il);
                cb(moe_out, "ffn_moe_out", il);

                // FFN shared expert
                {
                    ggml_tensor * ffn_shexp = build_ffn(cur,
                            model.layers[il].ffn_up_shexp,   NULL, NULL,
                            model.layers[il].ffn_gate_shexp, NULL, NULL,
                            model.layers[il].ffn_down_shexp, NULL, NULL,
                            NULL,
                            LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                }
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_deepseek2() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        bool is_lite = (hparams.n_layer == 27);

        // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
        // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
        const float mscale = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
        const float kq_scale = 1.0f*mscale*mscale/sqrtf(float(hparams.n_embd_head_k));
        const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

        const uint32_t n_embd_head_qk_rope = hparams.n_rot;
        const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
        const uint32_t kv_lora_rank = hparams.n_lora_kv;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                struct ggml_tensor * q = NULL;
                if (!is_lite) {
                    // {n_embd, q_lora_rank} * {n_embd, n_tokens} -> {q_lora_rank, n_tokens}
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                    cb(q, "q", il);

                    q = build_norm(q,
                            model.layers[il].attn_q_a_norm, NULL,
                            LLM_NORM_RMS, il);
                    cb(q, "q", il);

                    // {q_lora_rank, n_head * hparams.n_embd_head_k} * {q_lora_rank, n_tokens} -> {n_head * hparams.n_embd_head_k, n_tokens}
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                    cb(q, "q", il);
                } else {
                    q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                    cb(q, "q", il);
                }

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        0);
                cb(q_nope, "q_nope", il);

                // and {n_head * n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * kv_pe_compresseed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_pe_compresseed, "kv_pe_compresseed", il);

                // split into {kv_lora_rank, n_tokens}
                struct ggml_tensor * kv_compressed = ggml_view_2d(ctx0, kv_pe_compresseed, kv_lora_rank, n_tokens,
                        kv_pe_compresseed->nb[1],
                        0);
                cb(kv_compressed, "kv_compressed", il);

                // and {n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_pe_compresseed, n_embd_head_qk_rope, 1, n_tokens,
                        kv_pe_compresseed->nb[1],
                        kv_pe_compresseed->nb[1],
                        ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
                cb(k_pe, "k_pe", il);

                kv_compressed = ggml_cont(ctx0, kv_compressed); // TODO: the CUDA backend does not support non-contiguous norm
                kv_compressed = build_norm(kv_compressed,
                        model.layers[il].attn_kv_a_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(kv_compressed, "kv_compressed", il);

                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                struct ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                        ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                struct ggml_tensor * v_states = ggml_view_3d(ctx0, kv, hparams.n_embd_head_v, n_head, n_tokens,
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)*n_head),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                v_states = ggml_view_2d(ctx0, v_states, hparams.n_embd_head_v * n_head, n_tokens,
                    ggml_row_size(kv->type, hparams.n_embd_head_v * n_head),
                    0);
                cb(v_states, "v_states", il);

                q_pe = ggml_cont(ctx0, q_pe); // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                q_pe = ggml_rope_ext(
                    ctx0, q_pe, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor_scaled, beta_fast, beta_slow
                );
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_cont(ctx0, k_pe); // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                k_pe = ggml_rope_ext(
                    ctx0, k_pe, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor_scaled, beta_fast, beta_slow
                );
                cb(k_pe, "k_pe", il);

                struct ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(q_states, "q_states", il);

                struct ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(k_states, "k_states", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        k_states, v_states, q_states, n_tokens, kq_scale, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if ((uint32_t) il < hparams.n_layer_dense_lead) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                ggml_tensor * moe_out =
                        build_moe_ffn(cur,
                            model.layers[il].ffn_gate_inp,
                            model.layers[il].ffn_up_exps,
                            model.layers[il].ffn_gate_exps,
                            model.layers[il].ffn_down_exps,
                            model.layers[il].ffn_exp_probs_b,
                            n_expert, n_expert_used,
                            LLM_FFN_SILU, hparams.expert_weights_norm,
                            true, hparams.expert_weights_scale,
                            (enum llama_expert_gating_func_type) hparams.expert_gating_func,
                            cb, il);
                cb(moe_out, "ffn_moe_out", il);

                // FFN shared expert
                {
                    ggml_tensor * ffn_shexp = build_ffn(cur,
                            model.layers[il].ffn_up_shexp,   NULL, NULL,
                            model.layers[il].ffn_gate_shexp, NULL, NULL,
                            model.layers[il].ffn_down_shexp, NULL, NULL,
                            NULL,
                            LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                }
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_bitnet() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                if (model.layers[il].wq_scale) {
                    Qcur = ggml_mul(ctx0, Qcur, model.layers[il].wq_scale);
                }
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                // B1.K
                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                if (model.layers[il].wk_scale) {
                    Kcur = ggml_mul(ctx0, Kcur, model.layers[il].wk_scale);
                }
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                // B1.V
                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                if (model.layers[il].wv_scale) {
                    Vcur = ggml_mul(ctx0, Vcur, model.layers[il].wv_scale);
                }
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        NULL, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);

                cur = build_norm(cur,
                        model.layers[il].attn_sub_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_sub_norm", il);

                cur = build_lora_mm(model.layers[il].wo, cur);
                if (model.layers[il].wo_scale) {
                    cur = ggml_mul(ctx0, cur, model.layers[il].wo_scale);
                }
                if (model.layers[il].bo) {
                    cur = ggml_add(ctx0, cur, model.layers[il].bo);
                }
                cb(cur, "attn_o_out", il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward forward
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_scale,
                    model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_scale,
                    NULL,                      NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_sub_out", il);

            cur = build_norm(cur,
                    model.layers[il].ffn_sub_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_sub_norm", il);

            cur = build_lora_mm(model.layers[il].ffn_down, cur);
            if (model.layers[il].ffn_down_scale) {
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_down_scale);
            }
            cb(cur, "ffn_down", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        // FIXME: do not use model.tok_embd directly, duplicate as model.output
        cur = build_lora_mm(model.tok_embd, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    //struct ggml_cgraph * build_t5_enc() {
    //    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    //    // mutable variable, needed during the last layer of the computation to skip unused tokens
    //    int32_t n_tokens = this->n_tokens;

    //    const int64_t n_embd_head = hparams.n_embd_head_v;
    //    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    //    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    //    struct ggml_tensor * cur;
    //    struct ggml_tensor * inpL;

    //    inpL = build_inp_embd(model.tok_embd);

    //    GGML_ASSERT(lctx.is_encoding);
    //    struct ggml_tensor * pos_bucket_enc = build_pos_bucket(false);

    //    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    //    struct ggml_tensor * KQ_mask_enc = build_inp_KQ_mask(false);

    //    for (int il = 0; il < n_layer; ++il) {
    //        struct ggml_tensor * inpSA = inpL;

    //        // norm
    //        cur = build_norm(inpL,
    //                model.layers[il].attn_norm_enc, NULL,
    //                LLM_NORM_RMS, il);
    //        cb(cur, "attn_norm", il);

    //        // self-attention
    //        {
    //            struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_enc, cur);
    //            cb(Qcur, "Qcur", il);

    //            struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_enc, cur);
    //            cb(Kcur, "Kcur", il);

    //            struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_enc, cur);
    //            cb(Vcur, "Vcur", il);

    //            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
    //            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

    //            struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
    //            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

    //            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    //            cb(kq, "kq", il);

    //            struct ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b_enc ? model.layers[il].attn_rel_b_enc : model.layers[0].attn_rel_b_enc;
    //            struct ggml_tensor * pos_bias = build_pos_bias(pos_bucket_enc, attn_rel_b);
    //            struct ggml_tensor * kq_b = ggml_add(ctx0, kq, pos_bias);
    //            cb(kq_b, "kq_b", il);

    //            kq = ggml_soft_max_ext(ctx0, kq_b, KQ_mask_enc, 1.0f, hparams.f_max_alibi_bias);
    //            cb(kq, "kq_soft_max_ext", il);

    //            struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_tokens)));
    //            cb(v, "v", il);

    //            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_tokens, n_embd_head, n_head_kv), kq);
    //            cb(kqv, "kqv", il);

    //            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    //            cb(kqv_merged, "kqv_merged", il);

    //            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
    //            cb(cur, "kqv_merged_cont", il);

    //            ggml_build_forward_expand(gf, cur);

    //            cur = build_lora_mm(model.layers[il].wo_enc, cur);
    //            cb(cur, "kqv_out", il);
    //        }

    //        if (il == n_layer - 1) {
    //            // skip computing output for unused tokens
    //            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
    //            n_tokens = n_outputs;
    //            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
    //            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
    //        }

    //        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    //        cb(ffn_inp, "ffn_inp", il);

    //        // feed-forward network
    //        {
    //            cur = build_norm(ffn_inp,
    //                    model.layers[il].ffn_norm_enc, NULL,
    //                    LLM_NORM_RMS, il);
    //            cb(cur, "ffn_norm", il);

    //            // T5 uses relu, flan-T5 uses gelu-gated
    //            cur = build_ffn(cur,
    //                    model.layers[il].ffn_up_enc,   NULL, NULL,
    //                    model.layers[il].ffn_gate_enc, NULL, NULL,
    //                    model.layers[il].ffn_down_enc, NULL, NULL,
    //                    NULL,
    //                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
    //                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR  : LLM_FFN_SEQ,
    //                    cb, il);
    //            cb(cur, "ffn_out", il);
    //        }

    //        cur = ggml_add(ctx0, cur, ffn_inp);
    //        cb(cur, "ffn_out", il);

    //        ggml_tensor * layer_dir = cvec.tensor_for(il);
    //        if (layer_dir != nullptr) {
    //            cur = ggml_add(ctx0, cur, layer_dir);
    //        }
    //        cb(cur, "l_out", il);

    //        // input for next layer
    //        inpL = cur;
    //    }

    //    cur = inpL;
    //    cb(cur, "result_embd", -1);

    //    cur = build_norm(cur,
    //            model.output_norm_enc, NULL,
    //            LLM_NORM_RMS, -1);
    //    cb(cur, "result_norm", -1);

    //    ggml_build_forward_expand(gf, cur);

    //    return gf;
    //}

    //struct ggml_cgraph * build_t5_dec() {
    //    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    //    // mutable variable, needed during the last layer of the computation to skip unused tokens
    //    int32_t n_tokens = this->n_tokens;

    //    const int64_t n_embd_head = hparams.n_embd_head_v;
    //    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    //    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    //    struct ggml_tensor * cur;
    //    struct ggml_tensor * inpL;

    //    inpL = build_inp_embd(model.tok_embd);

    //    GGML_ASSERT(!lctx.is_encoding);
    //    GGML_ASSERT(n_outputs_enc > 0 && "call llama_encode() first");

    //    struct ggml_tensor * embd_enc       = build_inp_embd_enc();
    //    struct ggml_tensor * pos_bucket_dec = build_pos_bucket(true);

    //    struct ggml_tensor * KQ_mask_dec   = build_inp_KQ_mask();
    //    struct ggml_tensor * KQ_mask_cross = build_inp_KQ_mask_cross();

    //    for (int il = 0; il < n_layer; ++il) {
    //        struct ggml_tensor * inpSA = inpL;

    //        // norm
    //        cur = build_norm(inpL,
    //                model.layers[il].attn_norm, NULL,
    //                LLM_NORM_RMS, il);
    //        cb(cur, "attn_norm", il);

    //        // self-attention
    //        {
    //            struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
    //            cb(Qcur, "Qcur", il);

    //            struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    //            cb(Kcur, "Kcur", il);

    //            struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    //            cb(Vcur, "Vcur", il);

    //            build_kv_store(gf, Kcur, Vcur, il);

    //            struct ggml_tensor * k =
    //                ggml_view_3d(ctx0, kv_self.k_l[il],
    //                        n_embd_head_k, n_kv, n_head_kv,
    //                        ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
    //                        ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
    //                        0);
    //            cb(k, "k", il);

    //            struct ggml_tensor * v =
    //                ggml_view_3d(ctx0, kv_self.v_l[il],
    //                        n_kv, n_embd_head_v, n_head_kv,
    //                        ggml_element_size(kv_self.v_l[il])*n_ctx,
    //                        ggml_element_size(kv_self.v_l[il])*n_ctx*n_embd_head_v,
    //                        0);
    //            cb(v, "v", il);

    //            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

    //            struct ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

    //            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    //            cb(kq, "kq", il);

    //            struct ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b ? model.layers[il].attn_rel_b : model.layers[0].attn_rel_b;
    //            struct ggml_tensor * pos_bias = build_pos_bias(pos_bucket_dec, attn_rel_b);
    //            struct ggml_tensor * kq_b = ggml_add(ctx0, kq, pos_bias);
    //            cb(kq_b, "kq_b", il);

    //            kq = ggml_soft_max_ext(ctx0, kq_b, KQ_mask_dec, 1.0f, hparams.f_max_alibi_bias);
    //            cb(kq, "kq_soft_max_ext", il);

    //            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
    //            cb(kqv, "kqv", il);

    //            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    //            cb(kqv_merged, "kqv_merged", il);

    //            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
    //            cb(cur, "kqv_merged_cont", il);

    //            ggml_build_forward_expand(gf, cur);

    //            cur = build_lora_mm(model.layers[il].wo, cur);
    //            cb(cur, "kqv_out", il);
    //        }

    //        cur = ggml_add(ctx0, cur, inpSA);
    //        cb(cur, "cross_inp", il);

    //        struct ggml_tensor * inpCA = cur;

    //        // norm
    //        cur = build_norm(cur,
    //                model.layers[il].attn_norm_cross, NULL,
    //                LLM_NORM_RMS, il);
    //        cb(cur, "attn_norm_cross", il);

    //        // cross-attention
    //        {
    //            struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_cross, cur);
    //            cb(Qcur, "Qcur", il);

    //            struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_cross, embd_enc);
    //            cb(Kcur, "Kcur", il);

    //            struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_cross, embd_enc);
    //            cb(Vcur, "Vcur", il);

    //            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
    //            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_outputs_enc);

    //            struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
    //            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

    //            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    //            cb(kq, "kq", il);

    //            kq = ggml_soft_max_ext(ctx0, kq, KQ_mask_cross, 1.0f, hparams.f_max_alibi_bias);
    //            cb(kq, "kq_soft_max_ext", il);

    //            struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_outputs_enc)));
    //            cb(v, "v", il);

    //            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_outputs_enc, n_embd_head, n_head_kv), kq);
    //            cb(kqv, "kqv", il);

    //            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    //            cb(kqv_merged, "kqv_merged", il);

    //            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
    //            cb(cur, "kqv_merged_cont", il);

    //            ggml_build_forward_expand(gf, cur);

    //            cur = build_lora_mm(model.layers[il].wo_cross, cur);
    //            cb(cur, "kqv_out", il);
    //        }

    //        if (il == n_layer - 1) {
    //            // skip computing output for unused tokens
    //            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
    //            n_tokens = n_outputs;
    //            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
    //            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
    //            inpCA = ggml_get_rows(ctx0, inpCA, inp_out_ids);
    //        }

    //        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpCA);
    //        cb(ffn_inp, "ffn_inp", il);

    //        // feed-forward network
    //        {
    //            cur = build_norm(ffn_inp,
    //                    model.layers[il].ffn_norm, NULL,
    //                    LLM_NORM_RMS, il);
    //            cb(cur, "ffn_norm", il);

    //            // T5 uses relu, flan-T5 uses gelu-gated
    //            cur = build_ffn(cur,
    //                    model.layers[il].ffn_up,   NULL, NULL,
    //                    model.layers[il].ffn_gate, NULL, NULL,
    //                    model.layers[il].ffn_down, NULL, NULL,
    //                    NULL,
    //                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
    //                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR : LLM_FFN_SEQ,
    //                    cb, il);
    //            cb(cur, "ffn_out", il);
    //        }

    //        cur = ggml_add(ctx0, cur, ffn_inp);
    //        cb(cur, "ffn_out", il);

    //        ggml_tensor * layer_dir = lctx.cvec.tensor_for(il);
    //        if (layer_dir != nullptr) {
    //            cur = ggml_add(ctx0, cur, layer_dir);
    //        }
    //        cb(cur, "l_out", il);

    //        // input for next layer
    //        inpL = cur;
    //    }

    //    cur = inpL;
    //    cb(cur, "result_embd", -1);

    //    cur = build_norm(cur,
    //            model.output_norm, NULL,
    //            LLM_NORM_RMS, -1);
    //    cb(cur, "result_norm", -1);

    //    // lm_head
    //    cur = build_lora_mm(model.output, cur);
    //    cb(cur, "result_output", -1);

    //    ggml_build_forward_expand(gf, cur);

    //    return gf;
    //}

    struct ggml_cgraph * build_jais() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*cur->nb[0]*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*cur->nb[0]*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*cur->nb[0]*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/float(n_embd_head), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_chatglm() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = nullptr;
                struct ggml_tensor * Kcur = nullptr;
                struct ggml_tensor * Vcur = nullptr;

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);
                //printf("freq_base: %f freq_scale: %f ext_factor: %f attn_factor: %f\n", freq_base, freq_scale, ext_factor, attn_factor);
                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur_rope", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);

            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // Add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        NULL,                      NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SWIGLU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);

            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = build_norm(inpL,
                model.output_norm,
                NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_nemotron() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        //GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_SEQ, cb, il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, model.output_norm_b,
                LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_exaone() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor * rope_factors = lctx.get_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    //ggml_cgraph * build_rwkv6() {
    //    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    //    // Token shift state dimensions should be 2 * n_emb
    //    GGML_ASSERT(n_embd == hparams.n_embd_k_s() / 2);

    //    const int64_t n_seqs = ubatch.n_seqs;
    //    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    //    const int64_t n_tokens = ubatch.n_tokens;
    //    GGML_ASSERT(n_seqs != 0);
    //    GGML_ASSERT(ubatch.equal_seqs);
    //    GGML_ASSERT(n_tokens == n_seq_tokens * n_seqs);

    //    struct ggml_tensor * cur;
    //    struct ggml_tensor * inpL;
    //    struct ggml_tensor * state_copy = build_inp_s_copy();
    //    struct ggml_tensor * state_mask = build_inp_s_mask();

    //    inpL = build_inp_embd(model.tok_embd);
    //    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);

    //    for (int il = 0; il < n_layer; ++il) {
    //        const llama_layer * layer = &model.layers[il];

    //        // (ab)using the KV cache to store the states
    //        struct ggml_tensor * token_shift = build_copy_mask_state(
    //                gf, kv_self.k_l[il], state_copy, state_mask,
    //                hparams.n_embd_k_s(), n_seqs);

    //        struct ggml_tensor * wkv_states = build_copy_mask_state(
    //                gf, kv_self.v_l[il], state_copy, state_mask,
    //                hparams.n_embd_v_s(), n_seqs);

    //        cur = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);
    //        token_shift = ggml_reshape_3d(ctx0, token_shift, n_embd, 2, n_seqs);

    //        struct ggml_tensor * att_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], 0);
    //        struct ggml_tensor * ffn_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], n_embd * ggml_element_size(token_shift));

    //        struct ggml_tensor * x_norm_att = build_norm(cur, layer->attn_norm, layer->attn_norm_b, LLM_NORM, il);
    //        struct ggml_tensor * x_prev = ggml_concat(
    //            ctx0,
    //            att_shift,
    //            ggml_view_3d(ctx0, x_norm_att, n_embd, n_seq_tokens - 1, n_seqs, x_norm_att->nb[1], x_norm_att->nb[2], 0),
    //            1
    //        );

    //        cur = ggml_add(ctx0, cur, build_rwkv6_time_mix(layer, x_norm_att, x_prev, &wkv_states, hparams.wkv_head_size, n_embd / hparams.wkv_head_size));
    //        ggml_build_forward_expand(gf, cur);
    //        ggml_build_forward_expand(
    //            gf,
    //            ggml_cpy(
    //                ctx0,
    //                wkv_states,
    //                ggml_view_1d(
    //                    ctx0,
    //                    kv_self.v_l[il],
    //                    hparams.n_embd_v_s() * n_seqs,
    //                    hparams.n_embd_v_s() * kv_head * ggml_element_size(kv_self.v_l[il])
    //                )
    //            )
    //        );

    //        struct ggml_tensor * x_norm_ffn = build_norm(cur, layer->attn_norm_2, layer->attn_norm_2_b, LLM_NORM, il);
    //        x_prev = ggml_concat(
    //            ctx0,
    //            ffn_shift,
    //            ggml_view_3d(ctx0, x_norm_ffn, n_embd, n_seq_tokens - 1, n_seqs, x_norm_ffn->nb[1], x_norm_ffn->nb[2], 0),
    //            1
    //        );
    //        cur = ggml_add(ctx0, cur, build_rwkv6_channel_mix(layer, x_norm_ffn, x_prev));
    //        ggml_build_forward_expand(gf, cur);

    //        struct ggml_tensor * last_norm_att = ggml_view_3d(ctx0, x_norm_att, n_embd, 1, n_seqs, x_norm_att->nb[1], x_norm_att->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(x_norm_att));
    //        struct ggml_tensor * last_norm_ffn = ggml_view_3d(ctx0, x_norm_ffn, n_embd, 1, n_seqs, x_norm_ffn->nb[1], x_norm_ffn->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(x_norm_ffn));

    //        token_shift = ggml_concat(ctx0, last_norm_att, last_norm_ffn, 1);

    //        ggml_build_forward_expand(
    //            gf,
    //            ggml_cpy(
    //                ctx0,
    //                ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * 2, 0),
    //                ggml_view_1d(ctx0, kv_self.k_l[il], hparams.n_embd_k_s() * n_seqs, hparams.n_embd_k_s() * kv_head * ggml_element_size(kv_self.k_l[il]))
    //            )
    //        );

    //        if (hparams.rescale_every_n_layers != 0 && (il + 1) % hparams.rescale_every_n_layers == 0) {
    //            cur = ggml_scale(ctx0, cur, 0.5F);
    //        }

    //        cur = lctx.cvec.apply_to(ctx0, cur, il);
    //        cb(cur, "l_out", il);

    //        // input for next layer
    //        inpL = cur;
    //    }

    //    cur = inpL;
    //    struct ggml_tensor * inp_out_ids = build_inp_out_ids();
    //    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    //    cur = ggml_get_rows(ctx0, cur, inp_out_ids);

    //    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);
    //    cb(cur, "result_norm", -1);

    //    cur = build_lora_mm(model.output, cur);
    //    cb(cur, "result_output", -1);

    //    ggml_build_forward_expand(gf, cur);

    //    return gf;
    //}

    // ref: https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1/blob/main/modeling_rwkv6qwen2.py
    //ggml_cgraph * build_rwkv6qwen2() {
    //    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

    //    GGML_ASSERT(n_embd == hparams.n_embd_k_s());

    //    const int64_t n_seqs = ubatch.n_seqs;
    //    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    //    const int64_t n_tokens = ubatch.n_tokens;
    //    GGML_ASSERT(n_seqs != 0);
    //    GGML_ASSERT(ubatch.equal_seqs);
    //    GGML_ASSERT(n_tokens == n_seq_tokens * n_seqs);

    //    struct ggml_tensor * cur;
    //    struct ggml_tensor * inpL;
    //    struct ggml_tensor * state_copy = build_inp_s_copy();
    //    struct ggml_tensor * state_mask = build_inp_s_mask();

    //    inpL = build_inp_embd(model.tok_embd);

    //    for (int il = 0; il < n_layer; ++il) {
    //        const llama_layer * layer = &model.layers[il];

    //        // (ab)using the KV cache to store the states
    //        struct ggml_tensor * token_shift = build_copy_mask_state(
    //                gf, kv_self.k_l[il], state_copy, state_mask,
    //                hparams.n_embd_k_s(), n_seqs);

    //        struct ggml_tensor * wkv_states = build_copy_mask_state(
    //                gf, kv_self.v_l[il], state_copy, state_mask,
    //                hparams.n_embd_v_s(), n_seqs);

    //        cur = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);
    //        token_shift = ggml_reshape_3d(ctx0, token_shift, n_embd, 1, n_seqs);

    //        struct ggml_tensor * x_norm_att = build_norm(cur, layer->attn_norm, layer->attn_norm_b, LLM_NORM_RMS, il);
    //        struct ggml_tensor * x_prev = ggml_concat(
    //            ctx0,
    //            token_shift,
    //            ggml_view_3d(ctx0, x_norm_att, n_embd, n_seq_tokens - 1, n_seqs, x_norm_att->nb[1], x_norm_att->nb[2], 0),
    //            1
    //        );

    //        ggml_build_forward_expand(
    //            gf,
    //            ggml_cpy(
    //                ctx0,
    //                wkv_states,
    //                ggml_view_1d(
    //                    ctx0,
    //                    kv_self.v_l[il],
    //                    hparams.n_embd_v_s() * n_seqs,
    //                    hparams.n_embd_v_s() * kv_head * ggml_element_size(kv_self.v_l[il])
    //                )
    //            )
    //        );

    //        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, build_rwkv6_time_mix(layer, x_norm_att, x_prev, &wkv_states, hparams.wkv_head_size, hparams.n_head_kv()));
    //        ggml_build_forward_expand(gf, ffn_inp);
    //        ggml_build_forward_expand(
    //            gf,
    //            ggml_cpy(
    //                ctx0,
    //                wkv_states,
    //                ggml_view_1d(
    //                    ctx0,
    //                    kv_self.v_l[il],
    //                    hparams.n_embd_v_s() * n_seqs,
    //                    hparams.n_embd_v_s() * kv_head * ggml_element_size(kv_self.v_l[il])
    //                )
    //            )
    //        );

    //        cb(ffn_inp, "ffn_inp", il);

    //        // feed-forward network
    //        cur = build_norm(ffn_inp,
    //                model.layers[il].ffn_norm, NULL,
    //                LLM_NORM_RMS, il);
    //        cb(cur, "ffn_norm", il);

    //        cur = build_ffn(cur,
    //                model.layers[il].ffn_up,   NULL, NULL,
    //                model.layers[il].ffn_gate, NULL, NULL,
    //                model.layers[il].ffn_down, NULL, NULL,
    //                NULL,
    //                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
    //        cb(cur, "ffn_out", il);

    //        cur = ggml_add(ctx0, cur, ffn_inp);
    //        cur = lctx.cvec.apply_to(ctx0, cur, il);
    //        cb(cur, "l_out", il);

    //        // input for next layer
    //        inpL = cur;
    //    }

    //    cur = inpL;
    //    struct ggml_tensor * inp_out_ids = build_inp_out_ids();
    //    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    //    cur = ggml_get_rows(ctx0, cur, inp_out_ids);

    //    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM_RMS, -1);
    //    cb(cur, "result_norm", -1);

    //    cur = build_lora_mm(model.output, cur);
    //    cb(cur, "result_output", -1);

    //    ggml_build_forward_expand(gf, cur);

    //    return gf;
    //}

    // ref: https://github.com/facebookresearch/chameleon
    // based on the original build_llama() function, changes:
    //   * qk-norm
    //   * swin-norm
    //   * removed bias
    //   * removed MoE
    struct ggml_cgraph * build_chameleon() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lctx.build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            if (hparams.swin_norm) {
                cur = inpL;
            } else {
                cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
                cb(cur, "attn_norm", il);
            }

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                if (model.layers[il].attn_q_norm) {
                    Qcur = ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens,
                                ggml_element_size(Qcur) * n_embd_head,
                                ggml_element_size(Qcur) * n_embd_head * n_head,
                                0);
                    cb(Qcur, "Qcur", il);

                    Qcur = build_norm(Qcur,
                                model.layers[il].attn_q_norm,
                                model.layers[il].attn_q_norm_b,
                                LLM_NORM, il);
                    cb(Qcur, "Qcur", il);
                }

                if (model.layers[il].attn_k_norm) {
                    Kcur = ggml_view_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens,
                                ggml_element_size(Kcur) * n_embd_head,
                                ggml_element_size(Kcur) * n_embd_head * n_head_kv,
                                0);
                    cb(Kcur, "Kcur", il);

                    Kcur = build_norm(Kcur,
                               model.layers[il].attn_k_norm,
                               model.layers[il].attn_k_norm_b,
                               LLM_NORM, il);
                    cb(Kcur, "Kcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, nullptr,
                        Kcur, Vcur, Qcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), cb, il);

                if (hparams.swin_norm) {
                    cur = build_norm(cur,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, il);
                }
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (!hparams.swin_norm) {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);
            }

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            if (hparams.swin_norm) {
                cur = build_norm(cur,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output_with_img_logits", -1);

        // TODO: this suppresses the output of image tokens, which is required to enable text-only outputs.
        // Needs to be removed once image outputs are supported.
        int img_token_end_idx = 8196;
        int img_token_start_idx = 4;
        int num_img_tokens = img_token_end_idx - img_token_start_idx;
        // creates 1d tensor of size num_img_tokens and values -FLT_MAX,
        // which ensures that text token values are always at least larger than image token values
        struct ggml_tensor * img_logits = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_img_tokens);
        img_logits = ggml_clamp(ctx0, img_logits, -FLT_MAX, -FLT_MAX);
        cb(img_logits, "img_logits", -1);
        cur = ggml_set_1d(ctx0, cur, img_logits, ggml_element_size(cur) * img_token_start_idx);
        cb(cur, "result_output", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    struct ggml_cgraph * build_wavtokenizer_dec() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));

        cur = ggml_conv_1d_ph(ctx0, model.conv1d, cur, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_b);

        // posnet
        for (uint32_t il = 0; il < hparams.posnet.n_layer; ++il) {
            const auto & layer = model.layers[il].posnet;

            inpL = cur;

            switch (il) {
                case 0:
                case 1:
                case 3:
                case 4:
                    {
                        cur = build_norm(cur,
                                layer.norm1,
                                layer.norm1_b,
                                LLM_NORM_GROUP, 0);

                        cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                        cur = ggml_conv_1d_ph(ctx0, layer.conv1, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.conv1_b);

                        cur = build_norm(cur,
                                layer.norm2,
                                layer.norm2_b,
                                LLM_NORM_GROUP, 0);

                        cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                        cur = ggml_conv_1d_ph(ctx0, layer.conv2, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.conv2_b);

                        cur = ggml_add(ctx0, cur, inpL);
                    } break;
                case 2:
                    {
                        cur = build_norm(cur,
                                layer.attn_norm,
                                layer.attn_norm_b,
                                LLM_NORM_GROUP, 0);

                        struct ggml_tensor * q;
                        struct ggml_tensor * k;
                        struct ggml_tensor * v;

                        q = ggml_conv_1d_ph(ctx0, layer.attn_q, cur, 1, 1);
                        k = ggml_conv_1d_ph(ctx0, layer.attn_k, cur, 1, 1);
                        v = ggml_conv_1d_ph(ctx0, layer.attn_v, cur, 1, 1);

                        q = ggml_add(ctx0, q, layer.attn_q_b);
                        k = ggml_add(ctx0, k, layer.attn_k_b);
                        v = ggml_add(ctx0, v, layer.attn_v_b);

                        q = ggml_cont(ctx0, ggml_transpose(ctx0, q));
                        k = ggml_cont(ctx0, ggml_transpose(ctx0, k));

                        struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);

                        kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f/sqrtf(float(hparams.posnet.n_embd)), 0.0f);

                        cur = ggml_mul_mat(ctx0, kq, v);

                        cur = ggml_conv_1d_ph(ctx0, layer.attn_o, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.attn_o_b);

                        cur = ggml_add(ctx0, cur, inpL);
                    } break;
                case 5:
                    {
                        cur = build_norm(cur,
                                layer.norm,
                                layer.norm_b,
                                LLM_NORM_GROUP, 0);
                    } break;
                default: GGML_ABORT("unknown posnet layer");
            };
        }

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = build_norm(cur,
                model.tok_norm,
                model.tok_norm_b,
                LLM_NORM, -1);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        inpL = cur;

        // convnext
        for (uint32_t il = 0; il < hparams.convnext.n_layer; ++il) {
            const auto & layer = model.layers[il].convnext;

            cur = inpL;

            cur = ggml_conv_1d_dw_ph(ctx0, layer.dw, cur, 1, 1);
            cur = ggml_add(ctx0, cur, layer.dw_b);

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            cur = build_norm(cur,
                    layer.norm,
                    layer.norm_b,
                    LLM_NORM, -1);

            cur = build_ffn(cur,
                    layer.pw1, layer.pw1_b, NULL,
                    NULL,      NULL,        NULL,
                    layer.pw2, layer.pw2_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);

            cur = ggml_mul(ctx0, cur, layer.gamma);

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            inpL = ggml_add(ctx0, cur, inpL);
        }

        cur = inpL;

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cur = ggml_add(ctx0, cur, model.output_b);
        cb(cur, "result_embd", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }
};

static struct ggml_cgraph * llama_build_graph(
         llama_context & lctx,
    const llama_ubatch & ubatch,
                  bool   worst_case) {
    const auto & model = lctx.model;

    // this callback allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
    llm_build_cb cb = [&](struct ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!lctx.cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(lctx.sched.get(), cur, lctx.backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = lctx.model.params.n_gpu_layers > (int) lctx.model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = lctx.model.dev_layer(il);
                for (auto & backend : lctx.backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(lctx.sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };

    struct ggml_cgraph * result = NULL;

    struct llm_build_context llm(lctx, ubatch, cb, worst_case);

    llm.init();

    switch (model.arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
            {
                result = llm.build_llama();
            } break;
        case LLM_ARCH_DECI:
            {
                result = llm.build_deci();
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                result = llm.build_baichuan();
            } break;
        case LLM_ARCH_FALCON:
            {
                result = llm.build_falcon();
            } break;
        case LLM_ARCH_GROK:
            {
                result = llm.build_grok();
            } break;
        case LLM_ARCH_STARCODER:
            {
                result = llm.build_starcoder();
            } break;
        case LLM_ARCH_REFACT:
            {
                result = llm.build_refact();
            } break;
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_NOMIC_BERT:
            {
                result = llm.build_bert();
            } break;
        case LLM_ARCH_BLOOM:
            {
                result = llm.build_bloom();
            } break;
        case LLM_ARCH_MPT:
            {
                result = llm.build_mpt();
            } break;
         case LLM_ARCH_STABLELM:
            {
                result = llm.build_stablelm();
            } break;
        case LLM_ARCH_QWEN:
            {
                result = llm.build_qwen();
            } break;
        case LLM_ARCH_QWEN2:
            {
                result = llm.build_qwen2();
            } break;
        case LLM_ARCH_QWEN2VL:
            {
                lctx.n_pos_per_token = 4;
                result = llm.build_qwen2vl();
            } break;
        case LLM_ARCH_QWEN2MOE:
            {
                result = llm.build_qwen2moe();
            } break;
        case LLM_ARCH_PHI2:
            {
                result = llm.build_phi2();
            } break;
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
            {
                result = llm.build_phi3();
            } break;
        case LLM_ARCH_PLAMO:
            {
                result = llm.build_plamo();
            } break;
        case LLM_ARCH_GPT2:
            {
                result = llm.build_gpt2();
            } break;
        case LLM_ARCH_CODESHELL:
            {
                result = llm.build_codeshell();
            } break;
        case LLM_ARCH_ORION:
            {
                result = llm.build_orion();
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                result = llm.build_internlm2();
            } break;
        case LLM_ARCH_MINICPM3:
            {
                result = llm.build_minicpm3();
            } break;
        case LLM_ARCH_GEMMA:
            {
                result = llm.build_gemma();
            } break;
        case LLM_ARCH_GEMMA2:
            {
                result = llm.build_gemma2();
            } break;
        case LLM_ARCH_STARCODER2:
            {
                result = llm.build_starcoder2();
            } break;
        case LLM_ARCH_MAMBA:
            {
                result = llm.build_mamba();
            } break;
        case LLM_ARCH_XVERSE:
            {
                result = llm.build_xverse();
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                result = llm.build_command_r();
            } break;
        case LLM_ARCH_COHERE2:
            {
                result = llm.build_cohere2();
            } break;
        case LLM_ARCH_DBRX:
            {
                result = llm.build_dbrx();
            } break;
        case LLM_ARCH_OLMO:
            {
                result = llm.build_olmo();
            } break;
        case LLM_ARCH_OLMO2:
            {
                result = llm.build_olmo2();
            } break;
        case LLM_ARCH_OLMOE:
            {
                result = llm.build_olmoe();
            } break;
        case LLM_ARCH_OPENELM:
            {
                result = llm.build_openelm();
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                result = llm.build_gptneox();
            } break;
        case LLM_ARCH_ARCTIC:
            {
                result = llm.build_arctic();
            } break;
        case LLM_ARCH_DEEPSEEK:
            {
                result = llm.build_deepseek();
            } break;
        case LLM_ARCH_DEEPSEEK2:
            {
                result = llm.build_deepseek2();
            } break;
        case LLM_ARCH_CHATGLM:
            {
                result = llm.build_chatglm();
            } break;
        case LLM_ARCH_BITNET:
            {
                result = llm.build_bitnet();
            } break;
        //case LLM_ARCH_T5:
        //    {
        //        if (lctx.is_encoding) {
        //            result = llm.build_t5_enc();
        //        } else {
        //            result = llm.build_t5_dec();
        //        }
        //    } break;
        //case LLM_ARCH_T5ENCODER:
        //    {
        //        result = llm.build_t5_enc();
        //    } break;
        case LLM_ARCH_JAIS:
            {
                result = llm.build_jais();
            } break;
        case LLM_ARCH_NEMOTRON:
            {
                result = llm.build_nemotron();
            } break;
        case LLM_ARCH_EXAONE:
            {
                result = llm.build_exaone();
            } break;
        //case LLM_ARCH_RWKV6:
        //    {
        //        result = llm.build_rwkv6();
        //    } break;
        //case LLM_ARCH_RWKV6QWEN2:
        //    {
        //        result = llm.build_rwkv6qwen2();
        //    } break;
        case LLM_ARCH_CHAMELEON:
            {
                result = llm.build_chameleon();
            } break;
        case LLM_ARCH_WAVTOKENIZER_DEC:
            {
                result = llm.build_wavtokenizer_dec();
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    // add on pooling layer
    if (lctx.cparams.embeddings) {
        result = llm.append_pooling(result);
    }

    llm.free();

    return result;
}

// decode a batch of tokens by evaluating the transformer
// in case of unsuccessful decoding (error or warning),
// the kv_cache state will be returned to its original state
// (for non-recurrent models) or cleaned (for recurrent models)
//
//   - lctx:      llama context
//   - batch:     batch to evaluate
//
// return 0 on success
// return positive int on warning
// return negative int on error
//
static int llama_decode_impl(
         llama_context & lctx,
           llama_batch   inp_batch) {

    lctx.is_encoding = false;

    if (inp_batch.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // temporary allocate memory for the input batch if needed
    // TODO: this is incorrect for multiple sequences because pos_max() is the maximum across all sequences
    llama_batch_allocr batch_allocr(inp_batch, inp_batch.pos ? -1 : lctx.pos_max() + 1);

    const llama_batch & batch = batch_allocr.batch;

    const uint32_t n_tokens_all = batch.n_tokens;

    const auto & model   = lctx.model;
    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT

    if (batch.token) {
        for (uint32_t i = 0; i < n_tokens_all; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= model.vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return -1;
            }
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (lctx.t_compute_start_us == 0) {
        lctx.t_compute_start_us = ggml_time_us();
    }
    lctx.n_queued_tokens += n_tokens_all;

    const int64_t n_embd  = hparams.n_embd;
    const int64_t n_vocab = vocab.n_tokens();

    uint32_t n_outputs = 0;
    uint32_t n_outputs_prev = 0;

    // this indicates we are doing pooled embedding, so we ignore batch.logits and output all tokens
    const bool embd_pooled = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;

    lctx.embd_seq.clear();

    // count outputs
    if (batch.logits && !embd_pooled) {
        for (uint32_t i = 0; i < n_tokens_all; ++i) {
            n_outputs += batch.logits[i] != 0;
        }
    } else if (lctx.logits_all || embd_pooled) {
        n_outputs = n_tokens_all;
    } else {
        // keep last output only
        n_outputs = 1;
    }

    // reserve output buffer
    if (llama_output_reserve(lctx, n_outputs) < n_outputs) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_outputs);
        return -2;
    };

    const bool logits_all = n_outputs == n_tokens_all;

    //auto & kv_self = lctx.kv_self;
    //llama_kv_slot_restorer kv_slot_restorer(kv_self);

    //lctx.sbatch.from_batch(batch, n_embd,
    //    /* simple_split */ !kv_self.recurrent,
    //    /* logits_all   */ logits_all);

    auto batch_manager = lctx.prepare_batch(batch, logits_all);

    while (lctx.sbatch.n_tokens > 0) {
        llama_ubatch ubatch = batch_manager->next();

        const uint32_t n_tokens = ubatch.n_tokens;

        // count the outputs in this u_batch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs == n_tokens_all) {
                n_outputs_new = n_tokens;
            } else {
                GGML_ASSERT(ubatch.output);
                for (uint32_t i = 0; i < n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            lctx.n_outputs = n_outputs_new;
        }

        if (!batch_manager->prepare()) {
            LLAMA_LOG_ERROR("%s: failed to prepare ubatch\n", __func__);
            batch_manager->restore();
            return -3;
        }

        // reserve a worst case graph if needed
        // TODO: extract to a function
        if (lctx.need_reserve) {
            const auto & cparams = lctx.cparams;
            const auto & model   = lctx.model;

            // build worst-case graph
            uint32_t n_seqs = 1; // TODO: worst-case number of sequences
            uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

            llama_token token = model.vocab.token_bos(); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
            llama_ubatch ubatch = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};

            ggml_cgraph * gf = llama_build_graph(lctx, ubatch, true);

            // initialize scheduler with the worst-case graph
            ggml_backend_sched_reset(lctx.sched.get());
            if (!ggml_backend_sched_reserve(lctx.sched.get(), gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
            }

            lctx.need_reserve = false;
        }

        ggml_backend_sched_reset(lctx.sched.get());
        ggml_backend_sched_set_eval_callback(lctx.sched.get(), lctx.cparams.cb_eval, lctx.cparams.cb_eval_user_data);

        ggml_cgraph * gf = llama_build_graph(lctx, ubatch, false);

        // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

        ggml_backend_sched_alloc_graph(lctx.sched.get(), gf);

        lctx.set_inputs(ubatch);

        // the output is always the last tensor in the graph
        struct ggml_tensor * res  = ggml_graph_node(gf, -1);
        struct ggml_tensor * embd = ggml_graph_node(gf, -2);

        if (lctx.n_outputs == 0) {
            // no output
            res  = nullptr;
            embd = nullptr;
        } else if (cparams.embeddings) {
            res  = nullptr; // do not extract logits for embedding case
            embd = nullptr;
            for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
                if (strcmp(ggml_graph_node(gf, i)->name, "result_embd_pooled") == 0) {
                    embd = ggml_graph_node(gf, i);
                    break;
                }
            }
            GGML_ASSERT(embd != nullptr && "missing embeddings tensor");
        } else {
            embd = nullptr; // do not extract embeddings when not needed
            GGML_ASSERT(strcmp(res->name, "result_output") == 0 && "missing result_output tensor");
        }

        const auto compute_status = lctx.compute_graph(gf, n_tokens > 1);
        if (compute_status != GGML_STATUS_SUCCESS) {
            batch_manager->restore();
            switch (compute_status) {
                case GGML_STATUS_ABORTED:
                    return 2;
                case GGML_STATUS_ALLOC_FAILED:
                    return -2;
                case GGML_STATUS_FAILED:
                default:
                    return -3;
            }
        }

        batch_manager->update();

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        // extract logits
        if (res) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(lctx.sched.get(), res);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(lctx.logits != nullptr);

            float * logits_out = lctx.logits + n_outputs_prev*n_vocab;
            const int32_t n_outputs_new = lctx.n_outputs;

            if (n_outputs_new) {
                GGML_ASSERT( n_outputs_prev + n_outputs_new <= n_outputs);
                GGML_ASSERT((n_outputs_prev + n_outputs_new)*n_vocab <= (int64_t) lctx.logits_size);
                ggml_backend_tensor_get_async(backend_res, res, logits_out, 0, n_outputs_new*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(lctx.sched.get(), embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(lctx.embd != nullptr);
                        float * embd_out = lctx.embd + n_outputs_prev*n_embd;
                        const int32_t n_outputs_new = lctx.n_outputs;

                        if (n_outputs_new) {
                            GGML_ASSERT( n_outputs_prev + n_outputs_new <= n_outputs);
                            GGML_ASSERT((n_outputs_prev + n_outputs_new)*n_embd <= (int64_t) lctx.embd_size);
                            ggml_backend_tensor_get_async(backend_embd, embd, embd_out, 0, n_outputs_new*n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = lctx.embd_seq;

                        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
                            const llama_seq_id seq_id = ubatch.seq_id[s][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd, embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - a single float per sequence
                        auto & embd_seq_out = lctx.embd_seq;

                        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
                            const llama_seq_id seq_id = ubatch.seq_id[s][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(1);
                            ggml_backend_tensor_get_async(backend_embd, embd, embd_seq_out[seq_id].data(), (seq_id)*sizeof(float), sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        n_outputs_prev += lctx.n_outputs;
    }

    // set output mappings
    {
        bool sorted_output = true;

        GGML_ASSERT(lctx.sbatch.out_ids.size() == n_outputs);

        for (size_t i = 0; i < n_outputs; ++i) {
            size_t out_id = lctx.sbatch.out_ids[i];
            lctx.output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        if (sorted_output) {
            lctx.sbatch.out_ids.clear();
        }
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    lctx.n_outputs = n_outputs;

    // wait for the computation to finish (automatically done when obtaining the model output)
    //llama_synchronize(&lctx);

    batch_manager->finalize();

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(lctx.sched.get());

    return 0;
}

// encode a batch of tokens by evaluating the encoder part of the transformer
//
//   - lctx:      llama context
//   - batch:     batch to evaluate
//
// return 0 on success
// return positive int on warning
// return negative int on error
//
static int llama_encode_impl(
         llama_context & lctx,
           llama_batch   inp_batch) {

    lctx.is_encoding = true;

    if (inp_batch.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    // temporary allocate memory for the input batch if needed
    // TODO: this is incorrect for multiple sequences because pos_max() is the maximum across all sequences
    llama_batch_allocr batch_allocr(inp_batch, inp_batch.pos ? -1 : lctx.pos_max() + 1);

    const llama_batch & batch = batch_allocr.batch;
    const uint32_t n_tokens = batch.n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT

    if (batch.token) {
        for (uint32_t i = 0; i < n_tokens; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= model.vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return -1;
            }
        }
    }

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (lctx.t_compute_start_us == 0) {
        lctx.t_compute_start_us = ggml_time_us();
    }

    lctx.n_queued_tokens += n_tokens;

    const int64_t n_embd = hparams.n_embd;

    lctx.sbatch.from_batch(batch, n_embd, /* simple_split */ true, /* logits_all */ true);

    const llama_ubatch ubatch = lctx.sbatch.split_simple(n_tokens);

    // reserve output buffer
    if (llama_output_reserve(lctx, n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        lctx.output_ids[i] = i;
    }

    lctx.inp_embd_enc = NULL;
    lctx.n_outputs = n_tokens;

    //batch_manager->prepare(ubatch);

    // reserve a worst case graph if needed
    // TODO: extract to a function
    if (lctx.need_reserve) {
        // TODO: extract to a function
        const auto & cparams = lctx.cparams;
        const auto & model   = lctx.model;

        // build worst-case graph
        uint32_t n_seqs = 1; // TODO: worst-case number of sequences
        uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        llama_token token = model.vocab.token_bos(); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
        llama_ubatch ubatch = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};

        ggml_cgraph * gf = llama_build_graph(lctx, ubatch, true);

        // initialize scheduler with the worst-case graph
        ggml_backend_sched_reset(lctx.sched.get());
        if (!ggml_backend_sched_reserve(lctx.sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        }

        lctx.need_reserve = false;
    }

    ggml_backend_sched_reset(lctx.sched.get());
    ggml_backend_sched_set_eval_callback(lctx.sched.get(), lctx.cparams.cb_eval, lctx.cparams.cb_eval_user_data);

    ggml_cgraph * gf = llama_build_graph(lctx, ubatch, false);

    ggml_backend_sched_alloc_graph(lctx.sched.get(), gf);

    lctx.set_inputs(ubatch);

    // the output embeddings after the final encoder normalization
    struct ggml_tensor * embd = nullptr;

    // there are two cases here
    if (llama_model_has_decoder(&lctx.model)) {
        // first case is an encoder-decoder T5 model where embeddings are passed to decoder
        embd = ggml_graph_node(gf, -1);
        GGML_ASSERT(strcmp(embd->name, "result_norm") == 0 && "missing result_output tensor");
    } else {
        // second case is an encoder-only T5 model
        if (cparams.embeddings) {
            // only output embeddings if required
            embd = ggml_graph_node(gf, -1);
            if (strcmp(embd->name, "result_embd_pooled") != 0) {
                embd = ggml_graph_node(gf, -2);
            }
            GGML_ASSERT(strcmp(embd->name, "result_embd_pooled") == 0 && "missing embeddings tensor");
        }
    }

    const auto compute_status = lctx.compute_graph(gf, n_tokens > 1);
    switch (compute_status) {
        case GGML_STATUS_SUCCESS:
            break;
        case GGML_STATUS_ABORTED:
            return 2;
        case GGML_STATUS_ALLOC_FAILED:
            return -2;
        case GGML_STATUS_FAILED:
        default:
            return -3;
    }

    // extract embeddings
    if (embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(lctx.sched.get(), embd);
        GGML_ASSERT(backend_embd != nullptr);

        if (llama_model_has_decoder(&lctx.model)) {
            lctx.embd_enc.resize(n_tokens*n_embd);
            float * embd_out = lctx.embd_enc.data();

            ggml_backend_tensor_get_async(backend_embd, embd, embd_out, 0, n_tokens*n_embd*sizeof(float));
            GGML_ASSERT(!ubatch.equal_seqs); // TODO: handle equal splits

            // remember the sequence ids used during the encoding - needed for cross attention later
            lctx.seq_ids_enc.resize(n_tokens);
            for (uint32_t i = 0; i < n_tokens; i++) {
                for (int s = 0; s < ubatch.n_seq_id[i]; s++) {
                    llama_seq_id seq_id = ubatch.seq_id[i][s];
                    lctx.seq_ids_enc[i].insert(seq_id);
                }
            }
        } else {
            GGML_ASSERT(lctx.embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(lctx.embd != nullptr);
                        float * embd_out = lctx.embd;

                        GGML_ASSERT(n_tokens*n_embd <= (int64_t) lctx.embd_size);
                        ggml_backend_tensor_get_async(backend_embd, embd, embd_out, 0, n_tokens*n_embd*sizeof(float));
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings
                        auto & embd_seq_out = lctx.embd_seq;
                        embd_seq_out.clear();

                        GGML_ASSERT(!ubatch.equal_seqs); // TODO: handle equal splits

                        for (uint32_t i = 0; i < n_tokens; i++) {
                            const llama_seq_id seq_id = ubatch.seq_id[i][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd, embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // TODO: this likely should be the same logic as in llama_decoder_internal, but better to
                        //       wait for an encoder model that requires this pooling type in order to test it
                        //       https://github.com/ggerganov/llama.cpp/pull/9510
                        GGML_ABORT("RANK pooling not implemented yet");
                    }
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }
    }

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(lctx.sched.get());

    return 0;
}

//
// interface implementation
//

struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ 1.0f,
        /*.yarn_beta_fast              =*/ 32.0f,
        /*.yarn_beta_slow              =*/ 1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.logits_all                  =*/ false,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.flash_attn                  =*/ false,
        /*.no_perf                     =*/ true,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
    };

    return result;
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf                     =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        numa_init_fn(numa);
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static int llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model, llama_model_params & params) {
    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = 0;
    time_meas tm(model.t_load_us);

    model.t_start_us = tm.t_start_us;

    try {
        llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.kv_overrides);

        ml.print_info();

        model.hparams.vocab_only = params.vocab_only;

        try {
            model.load_arch(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model.load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            model.load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model.load_stats(ml);
        model.print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

        if (!model.load_tensors(ml)) {
            return -2;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

    return 0;
}

static struct llama_model * llama_model_load_from_file_impl(
        const std::string & path_model,
        std::vector<std::string> & splits,
        struct llama_model_params params) {
    ggml_time_init();

    llama_model * model = new llama_model(params);

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    // create list of devices to use with this model
    if (params.devices) {
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
            model->devices.push_back(*dev);
        }
    } else {
        // use all available devices
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // skip CPU backends since they are handled separately
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU:
                    model->devices.push_back(dev);
                    break;
            }
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0 || params.main_gpu >= (int)model->devices.size()) {
            LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %d)\n", __func__, params.main_gpu, (int)model->devices.size());
            llama_model_free(model);
            return nullptr;
        }
        ggml_backend_dev_t main_gpu = model->devices[params.main_gpu];
        model->devices.clear();
        model->devices.push_back(main_gpu);
    }

    for (auto * dev : model->devices) {
        size_t free, total; // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        LLAMA_LOG_INFO("%s: using device %s (%s) - %zu MiB free\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), free/1024/1024);
    }

    const int status = llama_model_load(path_model, splits, *model, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        llama_model_free(model);
        return nullptr;
    }

    return model;
}

// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(path_model, splits, params);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(splits.front(), splits, params);
}

struct llama_context * llama_init_from_model(
                 struct llama_model * model,
        struct llama_context_params   params) {

    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn = false;
    }

    if (params.flash_attn && model->hparams.n_embd_head_k != model->hparams.n_embd_head_v) {
        LLAMA_LOG_WARN("%s: flash_attn requires n_embd_head_k == n_embd_head_v - forcing off\n", __func__);
        params.flash_attn = false;
    }

    if (ggml_is_quantized(params.type_v) && !params.flash_attn) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    llama_context * ctx = nullptr;

    try {
        // TODO: add logic which llama_context implementation to construct
        ctx = new llama_context(*model, params,
                [](llama_context & lctx, const llama_ubatch & ubatch) {
                    return llama_build_graph(lctx, ubatch, true);
                });
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: failed to initialize context: %s\n", __func__, e.what());
        return nullptr;
    }

    return ctx;
}

// deprecated
struct llama_context * llama_new_context_with_model(
                 struct llama_model * model,
        struct llama_context_params   params) {
    return llama_init_from_model(model, params);
}

///

int32_t llama_encode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    const int ret = llama_encode_impl(*ctx, batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    const int ret = llama_decode_impl(*ctx, batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count) {
    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";
    if (snprintf(split_path, maxlen, SPLIT_PATH_FORMAT, path_prefix, split_no + 1, split_count)) {
        return strlen(split_path);
    }
    return 0;
}

int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count) {
    std::string str_split_path(split_path);
    char postfix[32];
    snprintf(postfix, 32, "-%05d-of-%05d.gguf", split_no + 1, split_count);
    std::string str_postfix(postfix);

    // check if split_prefix ends with postfix
    int size_prefix = str_split_path.size() - str_postfix.size();
    if (size_prefix > 0 && str_split_path.find(str_postfix, size_prefix) != std::string::npos) {
        snprintf(split_prefix, std::min((size_t) size_prefix + 1, maxlen), "%s", split_path);
        return size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.


    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

//
// perf
//

struct llama_perf_context_data llama_perf_context(const struct llama_context * ctx) {
    struct llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data.t_start_ms  = 1e-3 * ctx->t_start_us;
    data.t_load_ms   = 1e-3 * ctx->t_load_us;
    data.t_p_eval_ms = 1e-3 * ctx->t_p_eval_us;
    data.t_eval_ms   = 1e-3 * ctx->t_eval_us;
    data.n_p_eval    = std::max(1, ctx->n_p_eval);
    data.n_eval      = std::max(1, ctx->n_eval);

    return data;
}

void llama_perf_context_print(const struct llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
}

void llama_perf_context_reset(struct llama_context * ctx) {
    ctx->t_start_us  = ggml_time_us();
    ctx->t_eval_us   = ctx->n_eval = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}
