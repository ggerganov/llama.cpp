#include "arg.h"
#include "common.h"
#include "ngram-cache.h"
#include "ggml.h"
#include "jarvis.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char ** argv){
    common_params params;

    if (!common_params_parse(argc, argv, params, JARVIS_EXAMPLE_LOOKUP)) {
        return 1;
    }

    // init jarvis.cpp
    jarvis_backend_init();
    jarvis_numa_init(params.numa);

    // load the model
    common_init_result jarvis_init = common_init_from_params(params);

    jarvis_model * model = jarvis_init.model;
    jarvis_context * ctx = jarvis_init.context;
    GGML_ASSERT(model != nullptr);

    // tokenize the prompt
    std::vector<jarvis_token> inp;
    inp = common_tokenize(ctx, params.prompt, true, true);
    fprintf(stderr, "%s: tokenization done\n", __func__);


    common_ngram_cache ngram_cache;
    common_ngram_cache_update(ngram_cache, JARVIS_NGRAM_STATIC, JARVIS_NGRAM_STATIC, inp, inp.size(), true);
    fprintf(stderr, "%s: hashing done, writing file to %s\n", __func__, params.lookup_cache_static.c_str());

    common_ngram_cache_save(ngram_cache, params.lookup_cache_static);

    return 0;
}
