#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>
#include <string>

#include "llama.h"

namespace llama_cpp {

struct llama_model_deleter {
    void operator()(llama_model * model) { llama_free_model(model); }
};

struct llama_context_deleter {
    void operator()(llama_context * context) { llama_free(context); }
};

struct llama_sampler_deleter {
    void operator()(llama_sampler * sampler) { llama_sampler_free(sampler); }
};

typedef std::unique_ptr<llama_model, llama_model_deleter>     model;
typedef std::unique_ptr<llama_context, llama_context_deleter> context;
typedef std::unique_ptr<llama_sampler, llama_sampler_deleter> sampler;

inline model load_model_from_file(const std::string & path_model, llama_model_params params) {
    return model(llama_load_model_from_file(path_model.c_str(), params));
}

inline context new_context_with_model(const model & model, llama_context_params params) {
    return context(llama_new_context_with_model(model.get(), params));
}

inline sampler sampler_chain_init(llama_sampler_chain_params params) {
    return sampler(llama_sampler_chain_init(params));
}

std::vector<llama_token> tokenize(
        const llama_cpp::model & model,
             const std::string & raw_text,
                          bool   add_special,
                          bool   parse_special = false);

std::string token_to_piece(
      const llama_cpp::model & model,
                 llama_token   token,
                     int32_t   lstrip,
                        bool   special);

std::string detokenize(
            const llama_cpp::model & model,
    const std::vector<llama_token> & tokens,
                              bool   remove_special,
                              bool   unparse_special);

std::string chat_apply_template(
                   const llama_cpp::model & model,
                        const std::string & tmpl,
    const std::vector<llama_chat_message> & chat,
                                     bool   add_ass);

} // namespace llama_cpp
