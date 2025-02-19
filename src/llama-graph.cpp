#include "llama-graph.h"

#include "llama-impl.h"

ggml_tensor * llama_graph_i::build_inp_s_copy (
        ggml_context * ctx0,
                bool   worst_case) {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(worst_case);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_inp_s_mask(
        ggml_context * ctx0,
                bool   worst_case) {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(worst_case);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_copy_mask_state(
        ggml_context * ctx0,
         ggml_cgraph * gf,
         ggml_tensor * s,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
             int32_t   n_tokens,
             int32_t   n_state,
             int32_t   n_seqs,
                bool   worst_case) {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(gf);
    GGML_UNUSED(s);
    GGML_UNUSED(state_copy);
    GGML_UNUSED(state_mask);
    GGML_UNUSED(n_tokens);
    GGML_UNUSED(n_state);
    GGML_UNUSED(n_seqs);
    GGML_UNUSED(worst_case);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_mamba_layer(
        ggml_context * ctx0,
         ggml_cgraph * gf,
         ggml_tensor * cur,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
  const llama_ubatch & ubatch,
                 int   il,
                bool   worst_case) {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(gf);
    GGML_UNUSED(cur);
    GGML_UNUSED(state_copy);
    GGML_UNUSED(state_mask);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(il);
    GGML_UNUSED(worst_case);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_rwkv_token_shift_load(
        ggml_context * ctx0,
         ggml_cgraph * gf,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
  const llama_ubatch & ubatch,
                 int   il,
                bool   worst_case) {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(gf);
    GGML_UNUSED(state_copy);
    GGML_UNUSED(state_mask);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(il);
    GGML_UNUSED(worst_case);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_rwkv_token_shift_store(
        ggml_context * ctx0,
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il,
                bool   worst_case) {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(token_shift);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(il);
    GGML_UNUSED(worst_case);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}

ggml_tensor * llama_graph_i::build_rwkv6_time_mix(
        ggml_context * ctx0,
         ggml_cgraph * gf,
         ggml_tensor * cur,
         ggml_tensor * x_prev,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
  const llama_ubatch & ubatch,
                 int   il,
                bool   worst_case) {
    GGML_UNUSED(ctx0);
    GGML_UNUSED(gf);
    GGML_UNUSED(cur);
    GGML_UNUSED(x_prev);
    GGML_UNUSED(state_copy);
    GGML_UNUSED(state_mask);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(il);
    GGML_UNUSED(worst_case);

    LLAMA_LOG_ERROR("%s: not implemented\n", __func__);

    return nullptr; // NOLINT
}
