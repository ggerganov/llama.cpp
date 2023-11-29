#include "llama.h"
#include "common.h"
#include "unicode.h"
#include "console.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <codecvt>
#include <map>
#include <vector>
#include <locale>

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
        auto mparams = llama_model_default_params();

        mparams.vocab_only = true;

        model = llama_load_model_from_file(fname.c_str(), mparams);

        if (model == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            return 1;
        }

        auto cparams = llama_context_default_params();

        ctx = llama_new_context_with_model(model, cparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load vocab '%s'\n", __func__, fname.c_str());
            llama_free_model(model);
            return 1;
        }
    }

    GGML_ASSERT(llama_vocab_type(model) == LLAMA_VOCAB_TYPE_BPE);

#ifdef _WIN32
    // We need this for unicode console support
    console::init(false, false);
    atexit([]() { console::cleanup(); });
#endif

    const int n_vocab = llama_n_vocab(model);

    for (int i = 0; i < n_vocab; ++i) {
        std::string str = llama_detokenize_bpe(ctx, std::vector<int>(1, i));
        try {
            auto cps = codepoints_from_utf8(str);
            std::vector<llama_token> tokens = llama_tokenize(ctx, str, false);
            std::string check = llama_detokenize_bpe(ctx, tokens);
            if (check != str) {
                fprintf(stderr, "%s : error: token %d detokenizes to '%s'(%zu) but tokenization of this detokenizes to '%s'(%zu)\n",
                    __func__, i, str.c_str(), str.length(), check.c_str(), check.length());
                return 2;
            }
        }
        catch (const std::invalid_argument &) {
            fprintf(stderr, "%s : info: utf8 conversion %d '%s'\n", __func__, i, str.c_str());
        }
    }

    for (uint32_t cp = 0x0000; cp < 0xffff; ++cp) {
        // NOTE: these exceptions seem to be necessary, because the GPT2 tokenizer doesn't want to interfere with some ASCII control characters
        if ((cp < 0x03 || cp > 0x05) && cp != 0x0b && cp != 0x11 && (cp < 0x13 || cp > 0x17) && cp != 0x19 && (cp < 0x1c || cp > 0x1e) && (cp < 0xd800 || cp > 0xdfff)) {
            std::string str = " " + codepoint_to_utf8(cp);
            std::vector<llama_token> tokens = llama_tokenize(ctx, str, false);
            std::string check = llama_detokenize_bpe(ctx, tokens);
            if (str != check) {
                fprintf(stderr, "%s : error: codepoint %x detokenizes to '%s'(%zu) instead of '%s'(%zu)\n",
                    __func__, cp, check.c_str(), check.length(), str.c_str(), str.length());
                return 3;
            }
        }
    }
    // Restrict to assigned unicode planes
    // for (uint32_t cp = 0x10000; cp < 0x0010ffff; ++cp) {
    for (uint32_t cp = 0x10000; cp < 0x00040000; ++cp) {
        std::string str = codepoint_to_utf8(cp);
        std::vector<llama_token> tokens = llama_tokenize(ctx, str, false);
        std::string check = llama_detokenize_bpe(ctx, tokens);
        if (str != check) {
            fprintf(stderr, "%s : error: codepoint %x detokenizes to '%s'(%zu) instead of '%s'(%zu)\n",
                __func__, cp, check.c_str(), check.length(), str.c_str(), str.length());
            return 4;
        }
    }
    for (uint32_t cp = 0x000e0000; cp < 0x0010ffff; ++cp) {
        std::string str = codepoint_to_utf8(cp);
        std::vector<llama_token> tokens = llama_tokenize(ctx, str, false);
        std::string check = llama_detokenize_bpe(ctx, tokens);
        if (str != check) {
            fprintf(stderr, "%s : error: codepoint %x detokenizes to '%s'(%zu) instead of '%s'(%zu)\n",
                __func__, cp, check.c_str(), check.length(), str.c_str(), str.length());
            return 4;
        }
    }
    llama_free_model(model);
    llama_free(ctx);

    llama_backend_free();

    return 0;
}
