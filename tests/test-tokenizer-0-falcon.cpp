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
        { ""                      , {  }, },
        { " "                     , {     204, }, },
        { "  "                    , {     258, }, },
        { "   "                   , {     466, }, },
        { "\t"                    , {     192, }, },
        { "\n"                    , {     193, }, },
        { "\t\n"                  , {   19125, }, },
        { "Hello world"           , {    9856,   1079, }, },
        { " Hello world"          , {   23090,   1079, }, },
        { "Hello World"           , {    9856,   2889, }, },
        { " Hello World"          , {   23090,   2889, }, },
        { " Hello World!"         , {   23090,   2889,     12, }, },
        { "Hello, world!"         , {    9856,     23,   1079,     12, }, },
        { " Hello, world!"        , {   23090,     23,   1079,     12, }, },
        { " this is 🦙.cpp"        , {     414,    304,   3346,    111,    231,     25,  29247, }, },
        { "w048 7tuijk dsdfhu"    , {      98,  55866,    204,     34,  16682,   7149,  36190,   6869,  11481, }, },
        { "нещо на Български"     , {     150,    133,   6207,    151,    215,    150,    134,   5052,    133,   6279,   5052,    223,    151,    216,  49679,    123,  53110,  47043,   7795, }, },
        { "កាន់តែពិសេសអាចខលចេញ"   , {   38154,    206,  38154,    126,  38154,    225,    167,    237,    217,  38154,    221,    167,    237,    208,  38154,    228,  38154,    127,  38154,    237,    167,    237,    207,  38154,    237,  38154,    107,  38154,    126,  38154,    211,  38154,    207,  38154,    233,  38154,    211,    167,    237,    207,  38154,    215, }, },
        { "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)", {    2571,    232,    206,    204,     19,  11003,     20,   8196,    126,    283,    219,  48778,    116,  13392,    204,     19,  51831,    732,  63209,   1741,   7955,    522,     20,  22438,    211,    204,     19,   7927,  53360,    325,    504,    701,    946,  10930,     20, }, },
        { "Hello"                 , {    9856, }, },
        { " Hello"                , {   23090, }, },
        { "  Hello"               , {     204,  23090, }, },
        { "   Hello"              , {     258,  23090, }, },
        { "    Hello"             , {     466,  23090, }, },
        { "    Hello\n    Hello"  , {     466,  23090,    742,  23090, }, },
        { "\n ="                  , {    1212,     40, }, },
        { "' era"                 , {      18,   4932, }, },
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

    llama_backend_init();

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

    if (llama_vocab_type(model) != LLAMA_VOCAB_TYPE_BPE) {
        fprintf(stderr, "%s : error: vocab type is not BPE\n", __func__);
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
