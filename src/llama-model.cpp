#include "llama-model.h"

#include "llama-impl.h"

#include <algorithm>
#include <cassert>

const char * llm_type_name(llm_type type) {
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

static std::string llama_model_ftype_name(llama_ftype ftype) {
    if (ftype & LLAMA_FTYPE_GUESSED) {
        return llama_model_ftype_name((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
    }

    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:         return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:      return "F16";
        case LLAMA_FTYPE_MOSTLY_BF16:     return "BF16";
        case LLAMA_FTYPE_MOSTLY_Q4_0:     return "Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1:     return "Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q5_0:     return "Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1:     return "Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0:     return "Q8_0";
        case LLAMA_FTYPE_MOSTLY_Q2_K:     return "Q2_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:   return "Q2_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:   return "Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:   return "Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:   return "Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:   return "Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:   return "Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:   return "Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:   return "Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:     return "Q6_K";
        case LLAMA_FTYPE_MOSTLY_TQ1_0:    return "TQ1_0 - 1.69 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_TQ2_0:    return "TQ2_0 - 2.06 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:  return "IQ2_XXS - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:   return "IQ2_XS - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_S:    return "IQ2_S - 2.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M:    return "IQ2_M - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:   return "IQ3_XS - 3.3 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:  return "IQ3_XXS - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S:    return "IQ1_S - 1.5625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_M:    return "IQ1_M - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:   return "IQ4_NL - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:   return "IQ4_XS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_S:    return "IQ3_S - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_M:    return "IQ3_S mix - 3.66 bpw";

        default: return "unknown, may not work";
    }
}

std::string llama_model_arch_name (const llama_model & model) {
    return llm_arch_name(model.arch);
}

std::string llama_model_type_name (const llama_model & model) {
    return llm_type_name(model.type);
}

std::string llama_model_ftype_name(const llama_model & model) {
    return llama_model_ftype_name(model.ftype);
}

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

ggml_backend_buffer_type_t llama_model_select_buft(const llama_model & model, int il) {
    return select_buft(
            *model.dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

struct ggml_tensor * llama_model_get_tensor(const struct llama_model & model, const char * name) {
    auto it = std::find_if(model.tensors_by_name.begin(), model.tensors_by_name.end(),
            [name](const std::pair<std::string, struct ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == model.tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}
