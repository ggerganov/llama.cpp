#include "llama.h"
#include "common.h"
#include "unicode.h"
#include "console.h"

#include <cassert>
#include <codecvt>
#include <cstdio>
#include <cstring>
#include <locale>
#include <string>
#include <thread>
#include <vector>

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
            //fprintf(stderr, "%s : info: utf8 conversion %d '%s'\n", __func__, i, str.c_str());
        }
    }

    // unicode
    {
        const int nthread = std::thread::hardware_concurrency();

        std::vector<std::thread> threads(nthread);

        for (int i = 0; i < nthread; ++i) {
            threads[i] = std::thread([i, nthread, ctx]() {
                for (uint32_t cp = i; cp < 0x0010ffff; cp += nthread) {
                    if (!( // NOLINT
                                (cp < 0x03       || cp >  0x05)   && cp != 0x0b && cp != 0x11 &&
                                (cp < 0x13       || cp >  0x17)   && cp != 0x19 &&
                                (cp < 0x1c       || cp >  0x1e)   &&
                                (cp < 0xd800     || cp >  0xdfff) &&
                                (cp < 0x00040000 || cp >= 0x000e0000)
                        )) {
                        continue;
                    }

                    std::string str = codepoint_to_utf8(cp);
                    std::vector<llama_token> tokens = llama_tokenize(ctx, str, false);
                    std::string check = llama_detokenize_bpe(ctx, tokens);
                    if (cp != 9601 && str != check) {
                        fprintf(stderr, "error: codepoint %x detokenizes to '%s'(%zu) instead of '%s'(%zu)\n",
                                cp, check.c_str(), check.length(), str.c_str(), str.length());
                        std::exit(3);
                    }
                }
            });
        }

        for (auto & t : threads) {
            t.join();
        }
    }

    llama_free_model(model);
    llama_free(ctx);

    llama_backend_free();

    return 0;
}
