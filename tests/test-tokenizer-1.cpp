#include "llama.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <codecvt>
#include <map>
#include <vector>

std::string detokenize(llama_context * ctx, llama_token * tokens, int count) {
    std::string result;
    for (int i = 0; i < count; ++i) {
        result += llama_token_to_str(ctx, tokens[i]);
        if (i < count - 1) {
            result += "_";
        }
    }
    return result;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <vocab-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    fprintf(stderr, "%s : reading vocab from: '%s'\n", __func__, fname.c_str());

    llama_model * model;
    llama_context * ctx;

    llama_backend_init(false);

    // load the vocab
    {
        auto lparams = llama_context_default_params();

        lparams.vocab_only = true;

        model = llama_load_model_from_file(fname.c_str(), lparams);

        if (model == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            return 1;
        }

        ctx = llama_new_context_with_model(model, lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            llama_free_model(model);
            return 1;
        }
    }

    const int n_vocab = llama_n_vocab(ctx);

    if (n_vocab != 32000) {
        fprintf(stderr, "%s : expected 32000 tokens, got %d\n", __func__, n_vocab);
        llama_free_model(model);
        llama_free(ctx);
        return 2;
    }

    for (int i = 0; i < n_vocab; ++i) {
        const char * forward = llama_token_to_str(ctx, i);
        llama_token tokens[strlen(forward)];
        auto n = llama_tokenize(ctx, forward, tokens, strlen(forward), false);
        if (n == 1) {
            if (i != tokens[0]) {
                const char* backward = llama_token_to_str(ctx, tokens[0]);
                fprintf(stderr, "%s : error: token %d is string %s but tokenize() returns token %d %s\n", __func__, i, forward, tokens[0], backward);
            }
        } else {
            if (i <= 258) {
                fprintf(stderr, "%s : info: token %d is string %s and tokenize() returns tokens %s\n", __func__, i, forward, detokenize(ctx, tokens, n).c_str());
            } else {
                fprintf(stderr, "%s : error: token %d is string %s but tokenize() returns tokens %s\n", __func__, i, forward, detokenize(ctx, tokens, n).c_str());
            }
        }
    }

    std::wstring string_to_convert;
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    for (wchar_t ch = 0x0000; ch < 0xffff; ++ch) {
        std::wstring wstr(1, ch);
        std::string str = converter.to_bytes(wstr);
        llama_token tokens[strlen(str.c_str())];
        auto n = llama_tokenize(ctx, str.c_str(), tokens, str.length(), false);
        if (n == 1) {
            fprintf(stderr, "%s : info: %s tokenized to %d \n", __func__, str.c_str(), tokens[0]);
        }
    }

    llama_free_model(model);
    llama_free(ctx);

    llama_backend_free();

    return 0;
}
