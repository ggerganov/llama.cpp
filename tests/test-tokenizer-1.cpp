#include "llama.h"
#include "common.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <codecvt>
#include <map>
#include <vector>
#include <locale>

static std::string escape_whitespace(const std::string& text) {
    std::string result = "\xe2\x96\x81";
    for (size_t offs = 0; offs < text.length(); ++offs) {
        if (text[offs] == ' ') {
            result += "\xe2\x96\x81";
        } else {
            result += text[offs];
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

    GGML_ASSERT(llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_BPE);

    const int n_vocab = llama_n_vocab(ctx);

    for (int i = 0; i < n_vocab; ++i) {
        std::string forward = llama_token_to_piece(ctx, i);
        std::vector<llama_token> tokens = llama_tokenize(ctx, forward, false);
        if (tokens.size() == 1) {
            if (i != tokens[0]) {
                std::string backward = llama_token_to_piece(ctx, tokens[0]);
                fprintf(stderr, "%s : error: token %d is string %s but bpe returns token %d %s\n",
                    __func__, i, llama_token_to_piece(ctx, i).c_str(), tokens[0], backward.c_str());
                return 2;
            }
        }
    }

#ifdef _WIN32
    std::wstring_convert<typename std::codecvt_utf8<char16_t>, char16_t> u16converter;
    for (char16_t ch = 0x0000; ch < 0xffff; ++ch) {
        std::u16string u16str(1, ch);
        std::string str = u16converter.to_bytes(u16str);
        std::vector<llama_token> tokens = llama_tokenize(ctx, escape_whitespace(str).c_str(), false);
        if (tokens.size() == 1) {
            fprintf(stderr, "%s : info: %s tokenized to %d \n",
                __func__, str.c_str(), tokens[0]);
        }
    }

    std::wstring_convert<typename std::codecvt_utf8<char32_t>, char32_t> u32converter;
    for (char32_t ch = 0x0000; ch < 0x0010ffff; ++ch) {
        std::u32string u32str(1, ch);
        std::string str = u32converter.to_bytes(u32str);
        std::vector<llama_token> tokens = llama_tokenize(ctx, escape_whitespace(str).c_str(), false);
        if (tokens.size() == 1) {
            fprintf(stderr, "%s : info: %s tokenized to %d \n", __func__, str.c_str(), tokens[0]);
        }
    }
#endif

    llama_free_model(model);
    llama_free(ctx);

    llama_backend_free();

    return 0;
}
