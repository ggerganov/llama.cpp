#pragma once

#include "llama-impl.h"
#include "ggml-cpp.h"

#include "llama-model.h" // TODO: need only hparams

#include <vector>
#include <map>

struct llama_control_vector {
    std::vector<struct ggml_tensor *> tensors; // per layer
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    struct ggml_tensor * tensor_for(int il) const {
        if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
            return nullptr;
        }
        return tensors[il];
    }

    struct ggml_tensor * apply_to(struct ggml_context * ctx, struct ggml_tensor * cur, int  il) const {
        ggml_tensor * layer_dir = tensor_for(il);
        if (layer_dir != nullptr) {
            cur = ggml_add(ctx, cur, layer_dir);
        }
        return cur;
    }
};

static bool llama_control_vector_init(struct llama_control_vector & cvec, const llama_model & model) {
    GGML_ASSERT(cvec.tensors.empty());
    GGML_ASSERT(cvec.ctxs.empty());
    GGML_ASSERT(cvec.bufs.empty());

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            struct ggml_init_params params = {
                /*.mem_size   =*/ model.hparams.n_layer*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }
            ctx_map[buft] = ctx;
            cvec.ctxs.emplace_back(ctx);
            return ctx;
        }
        return it->second;
    };

    // make tensors
    cvec.tensors.reserve(model.hparams.n_layer);
    cvec.tensors.push_back(nullptr); // there's never a tensor for layer 0
    for (size_t il = 1; il < model.hparams.n_layer; il++) {
        ggml_backend_buffer_type_t buft = select_buft(*model.dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
            return false;
        }
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_embd);
        cvec.tensors.push_back(tensor);
    }

    // allocate tensors / buffers and zero
    cvec.bufs.reserve(ctx_map.size());
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        cvec.bufs.emplace_back(buf);
    }

    return true;
}

static int32_t llama_control_vector_apply(struct llama_control_vector & cvec, const llama_model & model, const float * data, size_t len, int32_t n_embd, int32_t il_start, int32_t il_end) {
    if (data == nullptr) {
        // disable the current control vector (but leave allocated for later)
        cvec.layer_start = -1;
        cvec.layer_end   = -1;
        return 0;
    }

    if (n_embd != (int) model.hparams.n_embd) {
        LLAMA_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
        return 1;
    }

    if (cvec.tensors.empty()) {
        if (!llama_control_vector_init(cvec, model)) {
            return 1;
        }
    }

    cvec.layer_start = il_start;
    cvec.layer_end   = il_end;

    for (size_t il = 1; il < model.hparams.n_layer; il++) {
        assert(cvec.tensors[il] != nullptr);

        const size_t off = n_embd * (il - 1); // buffer doesn't have data for layer 0, since it's never present
        if (off + n_embd <= len) {
            ggml_backend_tensor_set(cvec.tensors[il], data + off, 0, n_embd * ggml_element_size(cvec.tensors[il]));
        }
    }

    return 0;
}

