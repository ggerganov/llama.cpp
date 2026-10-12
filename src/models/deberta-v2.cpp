#include "models.h"

#include <algorithm>
#include <limits>

// Flat gather indices for the shared c2p/p2c relative attention tables.
class llm_graph_input_deberta_pos : public llm_graph_input_i {
public:
    llm_graph_input_deberta_pos(uint32_t buckets, uint32_t distance, int64_t heads) :
        buckets(buckets), distance(distance), heads(heads) {}

    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n = ubatch->n_tokens;
        std::vector<int32_t> a(n * n * heads);
        std::vector<int32_t> b(a.size());
        const int64_t mid = buckets / 2;
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t q = 0; q < n; ++q) {
                for (int64_t k = 0; k < n; ++k) {
                    const int64_t delta = int64_t(ubatch->pos[q]) - ubatch->pos[k];
                    int64_t bucket = delta;
                    if (std::abs(delta) > mid) {
                        const float log_position = std::ceil(std::log(float(std::abs(delta)) / mid) /
                            std::log(float(distance - 1) / mid) * (mid - 1)) + mid;
                        bucket = int64_t(log_position) * (delta < 0 ? -1 : 1);
                    }
                    bucket = std::clamp<int64_t>(bucket + buckets, 0, 2 * buckets - 1);
                    const size_t at = (h * n + q) * n + k;
                    a[at] = bucket + 2 * buckets * (q + n * h);
                    b[at] = bucket + 2 * buckets * (k + n * h);
                }
            }
        }
        ggml_backend_tensor_set(c2p, a.data(), 0, a.size() * sizeof(int32_t));
        ggml_backend_tensor_set(p2c, b.data(), 0, b.size() * sizeof(int32_t));
    }

    ggml_tensor * c2p = nullptr;
    ggml_tensor * p2c = nullptr;
    const uint32_t buckets;
    const uint32_t distance;
    const int64_t heads;
};

void llama_model_deberta_v2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, n_buckets);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_MAX_DISTANCE, max_distance);
    if (n_buckets < 4 || n_buckets > 16384 || n_buckets % 2 || max_distance <= n_buckets / 2 + 1 ||
        hparams.causal_attn || hparams.n_head() == 0 || hparams.n_head() != hparams.n_head_kv() ||
        hparams.n_embd % hparams.n_head() || !std::isfinite(hparams.f_norm_eps) || hparams.f_norm_eps <= 0) {
        throw std::runtime_error("invalid DeBERTa-v2 configuration");
    }
    hparams.n_embd_out_impl = 1;
}

void llama_model_deberta_v2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias", 0), {n_embd}, 0);
    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD, "weight"), {n_embd, 2 * n_buckets}, 0);
    cls = create_tensor(tn(LLM_TENSOR_CLS, "weight"), {n_embd, 2 * n_embd}, 0);
    cls_b = create_tensor(tn(LLM_TENSOR_CLS, "bias"), {2 * n_embd}, 0);
    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {2 * n_embd, 1}, 0);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"), {1}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd, n_embd, 0);
        if (!layer.wq || !layer.wk || !layer.wv || !layer.wq_b || !layer.wk_b || !layer.wv_b) {
            throw std::runtime_error("DeBERTa needs separate query, key and value weights and biases");
        }
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);
        layer.attn_out_norm = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i), {n_embd}, 0);
        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), {n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, 0);
        layer.layer_out_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias", i), {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deberta_v2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_deberta_v2::graph::graph(const llama_model_deberta_v2 & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t dim = hparams.n_embd_head_k();
    const int64_t n_relative = 2 * model.n_buckets;
    if (n_tokens > std::numeric_limits<int32_t>::max() / (n_relative * n_head)) {
        throw std::runtime_error("DeBERTa relative position index overflow");
    }
    auto positions = std::make_unique<llm_graph_input_deberta_pos>(model.n_buckets, model.max_distance, n_head);
    positions->c2p = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * n_tokens * n_head);
    positions->p2c = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * n_tokens * n_head);
    ggml_set_input(positions->c2p);
    ggml_set_input(positions->p2c);
    ggml_tensor * c2p_indices = positions->c2p;
    ggml_tensor * p2c_indices = positions->p2c;
    res->add_input(std::move(positions));
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    auto * inp_attn = build_attn_inp_no_cache();
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cur = build_norm(cur, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);
    cb(cur, "inp_norm", -1);
    ggml_tensor * relative = model.pos_embd;

    auto split_heads = [&](ggml_tensor * x) {
        return ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, x, dim, n_head, x->ne[1]), 0, 2, 1, 3));
    };
    auto gather = [&](ggml_tensor * x, ggml_tensor * indices) {
        x = ggml_reshape_2d(ctx0, x, 1, ggml_nelements(x));
        return ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, x, indices), n_tokens, n_tokens, n_head);
    };
    const float scale = 1.0f / sqrtf(float(dim * 3));
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * residual = cur;
        auto [q, k, v] = build_qkv(layer, cur, dim, n_head, n_head, il);
        ggml_tensor * pq = split_heads(ggml_add(ctx0, build_lora_mm(layer.wq, relative), layer.wq_b));
        ggml_tensor * pk = split_heads(ggml_add(ctx0, build_lora_mm(layer.wk, relative), layer.wk_b));
        ggml_tensor * c2p = gather(ggml_mul_mat(ctx0, pk, ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3))), c2p_indices);
        ggml_tensor * p2c = gather(ggml_mul_mat(ctx0, pq, ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3))), p2c_indices);
        ggml_tensor * bias = ggml_add(ctx0, c2p, p2c);
        cur = build_attn(inp_attn, layer.wo, layer.wo_b, layer.wo_s, q, k, v, bias, nullptr, nullptr, scale, il);
        cur = build_norm(ggml_add(ctx0, cur, residual), layer.attn_out_norm, layer.attn_out_norm_b, LLM_NORM, il);
        residual = cur;
        cur = ggml_gelu_erf(ctx0, ggml_add(ctx0, build_lora_mm(layer.ffn_up, cur), layer.ffn_up_b));
        cur = ggml_add(ctx0, build_lora_mm(layer.ffn_down, cur), layer.ffn_down_b);
        cur = build_norm(ggml_add(ctx0, cur, residual), layer.layer_out_norm, layer.layer_out_norm_b, LLM_NORM, il);
        cb(cur, "layer_out", il);
    }
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cur = ggml_relu(ctx0, ggml_add(ctx0, build_lora_mm(model.cls, cur), model.cls_b));
    cur = ggml_add(ctx0, build_lora_mm(model.cls_out, cur), model.cls_out_b);
    cb(cur, "decision_scores", -1);
    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}
