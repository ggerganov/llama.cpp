#include "llama.h"
#include "common.h"
#include "console.h"

#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <thread>

using llama_tests = std::map<std::string, std::vector<llama_token>>;

static llama_tests read_tests(const std::string & fname_inp, const std::string & fname_out) {
    llama_tests tests;

    std::ifstream ifs_inp(fname_inp);
    if (!ifs_inp) {
        fprintf(stderr, "%s : error: could not open file '%s'\n", __func__, fname_inp.c_str());
        return tests;
    }

    std::string sraw((std::istreambuf_iterator<char>(ifs_inp)), std::istreambuf_iterator<char>());

    std::ifstream ifs_out(fname_out);
    if (!ifs_out) {
        fprintf(stderr, "%s : error: could not open file '%s'\n", __func__, fname_out.c_str());
        return tests;
    }

    std::vector<std::string> sout;
    for (std::string line; std::getline(ifs_out, line);) {
        sout.push_back(line);
    }

    const std::string sep = "\n__ggml_vocab_test__\n";

    std::vector<std::string> sinp;

    size_t pos = 0;
    while (pos < sraw.size()) {
        const size_t next = sraw.find(sep, pos);
        if (next == std::string::npos) {
            sinp.push_back(sraw.substr(pos));
            break;
        }
        sinp.push_back(sraw.substr(pos, next - pos));
        pos = next + sep.size();
    }

    if (sinp.size() != sout.size()) {
        fprintf(stderr, "%s : error: input and output files have different number of tests\n", __func__);
        return tests;
    }

    for (size_t i = 0; i < sinp.size(); ++i) {
        const std::string & s = sinp[i];
        const std::string & o = string_strip(sout[i]);

        std::vector<llama_token> toks;

        size_t pos = 0;
        while (pos < o.size()) {
            size_t next = o.find(' ', pos);
            if (next == std::string::npos) {
                next = o.size();
            }
            const std::string stok = o.substr(pos, next - pos);
            toks.push_back(std::stoi(stok));
            pos = next + 1;
        }

        tests[s] = toks;
    }

    return tests;
}

int main(int argc, char const *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s vocab-file \n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    const std::string fname_inp = fname + ".inp";
    const std::string fname_out = fname + ".out";

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

#ifdef _WIN32
    // We need this for unicode console support
    console::init(false, false);
    atexit([]() { console::cleanup(); });
#endif

    const int nthread = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(nthread);

    bool success = true;

    const auto k_tests = [&]() -> llama_tests {
        const auto res = read_tests(fname_inp, fname_out);

        if (res.empty()) {
            fprintf(stderr, "%s : error: no tests found\n", __func__);
            exit(1);
        }

        return res;
    }();

    const bool add_special = false;

    for (int i = 0; i < nthread; i++) {
        threads[i] = std::thread([&]() {
            for (const auto & test_kv : k_tests) {
                const std::vector<llama_token> res = llama_tokenize(ctx, test_kv.first, add_special, false);

                bool correct = res.size() == test_kv.second.size();
                for (int i = 0; i < (int) res.size() && correct; ++i) {
                    if (test_kv.second[i] != res[i]) {
                        correct = false;
                    }
                }

                if (!correct) {
                    success = false;
                }
            }
        });
    }

    for (int i = 0; i < nthread; i++) {
        threads[i].join();
    }

    llama_free_model(model);
    llama_free(ctx);

    llama_backend_free();

    printf("\n");
    printf("Tests %s\n", success ? "passed" : "failed");

    return success ? 0 : 3;
}


