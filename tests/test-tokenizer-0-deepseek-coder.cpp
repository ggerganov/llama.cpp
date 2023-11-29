#include "llama.h"
#include "common.h"
#include "console.h"

#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include <fstream>

// generate using test-tokenizer-0-falcon.py
static const std::map<std::string, std::vector<llama_token>> & k_tests() {
    static std::map<std::string, std::vector<llama_token>> _k_tests = {
        { ""                      , {    }, },
        { " "                     , {       207, }, },
        { "  "                    , {       243, }, },
        { "   "                   , {       315, }, },
        { "\t"                    , {       184, }, },
        { "\n"                    , {       185, }, },
        { "\t\n"                  , {       184,    185, }, },
        { "Hello world"           , {     17535,   1835, }, },
        { " Hello world"          , {       414,   9489,   1835, }, },
        { "Hello World"           , {     17535,   5414, }, },
        { " Hello World"          , {       414,   9489,   5414, }, },
        { " Hello World!"         , {       414,   9489,   5414,      0, }, },
        { "Hello, world!"         , {     17535,     11,   1835,      0, }, },
        { " Hello, world!"        , {       414,   9489,     11,   1835,      0, }, },
        { " this is 🦙.cpp"        , {       437,    317,  12394,     99,    234,     13,  14789, }, },
        { "w048 7tuijk dsdfhu"    , {        86,     15,     19,     23,    207,     22,     83,   3963,  27659,  26078,   3934,  14072, }, },
        { "нещо на Български"     , {      1593,   6478,    616,   2251,  14994, }, },
        { "កាន់តែពិសេសអាចខលចេញ"   , {       155,    239,    209,    155,    239,    114,    155,    239,    228,    155,    240,    220,    155,    239,    224,    155,    240,    211,    155,    239,    231,    155,    239,    115,    155,    239,    240,    155,    240,    210,    155,    239,    240,    155,    239,     95,    155,    239,    114,    155,    239,    214,    155,    239,    210,    155,    239,    236,    155,    239,    214,    155,    240,    210,    155,    239,    218, }, },
        { "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)", {     10047,    235,    209,    334,   8760,      8,  12394,    233,    114,    350,    222,  10047,    221,    104,    169,    116,    224,    334,   4684,   3909,    992,  24330,    262,  29651,    612,      8,    207,    156,    237,    214,    334,   5950,    992,     78,  12896,    344,    638,    891,   1372,  10736,      8, }, },
        { "Hello"                 , {     17535, }, },
        { " Hello"                , {       414,   9489, }, },
        { "  Hello"               , {       207,    414,   9489, }, },
        { "   Hello"              , {       243,    414,   9489, }, },
        { "    Hello"             , {       315,    414,   9489, }, },
        { "    Hello\n    Hello"  , {       315,    414,   9489,    185,    315,    414,   9489, }, },
        { "\n ="                  , {       185,    405, }, },
        { "' era"                 , {         6,   2895, }, },
        { "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～", {     17535,     11,    320,      6,    435,      0,   1717,    417,    340,  12394,    233,    210,   3015,  19100,    608,   9413,   2668,     16,     18,     16,     19,     16,     20,     16,   1393,    169,    121,    239, }, },

    };

    return _k_tests;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s vocab-file [text-file]\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    std::string fname_text;
    if (argc > 2) {
        fname_text = argv[2];
    }

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

    if (llama_vocab_type(model) != LLAMA_VOCAB_TYPE_DEEPSEEKCODER) {
        fprintf(stderr, "%s : error: vocab type is not DEEPSEEKCODER\n", __func__);
        llama_free_model(model);
        llama_free(ctx);
        return 2;
    }

#ifdef _WIN32
    // We need this for unicode console support
    console::init(false, false);
    atexit([]() { console::cleanup(); });
#endif

    bool success = true;

    for (const auto & test_kv : k_tests()) {
        const std::vector<llama_token> res = llama_tokenize(ctx, test_kv.first, false);

        printf("\n");
        printf("src: '%s'\n", test_kv.first.c_str());
        printf("res: '%s'\n", llama_detokenize_bpe(ctx, res).c_str());
        printf("tok: ");
        for (const auto & tok : res) {
            printf("%d ", tok);
        }
        printf("\n");

        bool correct = res.size() == test_kv.second.size();
        for (int i = 0; i < (int) res.size() && correct; ++i) {
            if (test_kv.second[i] != res[i]) {
                correct = false;
            }
        }

        if (!correct) {
            fprintf(stderr, "%s : failed test:    '%s'\n", __func__, test_kv.first.c_str());
            fprintf(stderr, "%s : detokenized to: '%s' instead of '%s'\n", __func__,
                llama_detokenize_bpe(ctx, res).c_str(),
                llama_detokenize_bpe(ctx, test_kv.second).c_str());
            fprintf(stderr, "%s : expected tokens: ", __func__);
            for (const auto & t : test_kv.second) {
                fprintf(stderr, "%6d, ", t);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "%s : got tokens:      ", __func__);
            for (const auto & t : res) {
                fprintf(stderr, "%6d, ", t);
            }
            fprintf(stderr, "\n");

            success = false;
        }
    }

    if (!fname_text.empty()) {
        fprintf(stderr, "%s : tokenizing: '%s'\n", __func__, fname_text.c_str());

        std::string text;
        {
            std::ifstream ifs(fname_text);
            if (!ifs) {
                fprintf(stderr, "%s : error: could not open file '%s'\n", __func__, fname_text.c_str());
                return 1;
            }
            text = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
        }

        fprintf(stderr, "%s : text size: %zu\n", __func__, text.size());

        const std::vector<llama_token> res = llama_tokenize(ctx, text, false);

        fprintf(stderr, "%s : tokens: %zu\n", __func__, res.size());

        {
            const std::string fname_out = fname_text + ".tokcpp";

            std::ofstream ofs(fname_out);
            if (!ofs) {
                fprintf(stderr, "%s : error: could not open file '%s'\n", __func__, fname_out.c_str());
                return 1;
            }

            for (const auto & tok : res) {
                ofs << tok << " '" << llama_detokenize_bpe(ctx, std::vector<int>{tok}) << "'" << std::endl;
            }
        }

        fprintf(stderr, "%s : tokens written to '%s'\n", __func__, (fname_text + ".tokcpp").c_str());
    }

    llama_free_model(model);
    llama_free(ctx);

    llama_backend_free();

    return success ? 0 : 3;
}
