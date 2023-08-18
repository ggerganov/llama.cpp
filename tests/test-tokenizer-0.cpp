#include "llama.h"
#include "common.h"

#include <cstdio>
#include <string>
#include <map>
#include <vector>

static std::string unescape_whitespace(llama_context* ctx, const std::vector<llama_token>& tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        result += llama_token_to_str(ctx, tokens[i]);
    }
    return result;
}

static const std::map<std::string, std::vector<llama_token>> & k_tests() {
    static std::map<std::string, std::vector<llama_token>> _k_tests = {
        { " ",                      {1,    259, }, },
        { "\t",                     { 1,    29871,   12, }, },
        { "\n",                     { 1,    29871,   13, }, },
        { "\t\n",                   { 1,    29871,   12,     13, }, },
        { "Hello world",            { 1,  15043,   3186, }, },
        { " Hello world",           { 1,  29871,  15043,   3186, }, },
        { "Hello World",            { 1,  15043,   2787, }, },
        { " Hello World",           { 1,  29871,  15043,   2787, }, },
        { " Hello World!",          { 1,  29871,  15043,   2787,  29991, }, },
        { " this is 🦙.cpp",        { 1,  29871,    445,    338,  29871,    243,    162,    169,    156,  29889,   8223, }, },
        { "w048 7tuijk dsdfhu",     { 1,    281,  29900,  29946,  29947,  29871,  29955,   9161,  13535,  18031,   2176,   6905, }, },
        { "нещо на Български",      { 1,   1538,   4851,    665,   1386,  29713,   1305, }, },
        { "កាន់តែពិសេសអាចខលចេញ",   { 1,  29871,  31849,  31324,  31934,    228,    162,    142,    228,    161,
                                     146,    228,    162,    133,    228,    161,    153,    228,    161,    186,
                                     31708,    228,    162,    132,  31708,    228,    161,    165,  31324,    228,
                                     161,    136,    228,    161,    132,    228,    161,    158,    228,    161,
                                     136,    228,    162,    132,    228,    161,    140, }, },
        { "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)",
            { 1,  29871,    243,    162,    157,    131,    313,   8945,  29897,  29871,
                243,    162,    155,    185,  30722,    243,    162,    143,    174,  30598,
                313,  20787,    953,   3848,    275,  16125,    630,  29897,  29871,  31681,
                313,   6194,    953,  29877,   2397,    393,    756,    967,   1914,   5993,  29897, }, },
    };

    return _k_tests;
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

    bool success = true;

    for (const auto & test_kv : k_tests()) {
        std::vector<llama_token> res = llama_tokenize(ctx, test_kv.first, true);
        fprintf(stderr, "%s : '%s' tokenized to '%s'\n",
            __func__, test_kv.first.c_str(), unescape_whitespace(ctx, res).c_str());

        bool correct = res.size() == test_kv.second.size();

        for (int i = 0; i < (int) res.size() && correct; ++i) {
            if (res[i] != test_kv.second[i]) {
                correct = false;
            }
        }

        if (!correct) {
            fprintf(stderr, "%s : failed test:    '%s'\n", __func__, test_kv.first.c_str());
            fprintf(stderr, "%s : detokenized to: '%s'\n", __func__, unescape_whitespace(ctx, test_kv.second).c_str());
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

    llama_free_model(model);
    llama_free(ctx);

    llama_backend_free();

    return success ? 0 : 3;
}
