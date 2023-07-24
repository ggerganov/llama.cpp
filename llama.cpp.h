#ifndef LLAMA_CPP_H
#define LLAMA_CPP_H

#include "llama.h"

#include <cassert>

static std::string llama_token_to_str(
        const struct llama_context * ctx,
                       llama_token   token) {
    std::string result;
    int length = 8;
    result.resize(length);
    length = llama_token_to_str(ctx, token, (char *)result.data(), result.length());
    if (length < 0) {
        result.resize(-length);
        int check = llama_token_to_str(ctx, token, (char *)result.data(), result.length());
        assert(check == -length);
        GGML_UNUSED(check);
    } else {
        result.resize(length);
    }
    return result;
}

static std::string llama_token_to_str_bpe(
    const struct llama_context * ctx,
                   llama_token   token) {
    std::string result;
    int length = 8;
    result.resize(length);
    length = llama_token_to_str_bpe(ctx, token, (char*)result.data(), result.length());
    if (length < 0) {
        result.resize(-length);
        int check = llama_token_to_str_bpe(ctx, token, (char*)result.data(), result.length());
        assert(check == -length);
        GGML_UNUSED(check);
    } else {
        result.resize(length);
    }
    return result;
}

#endif
