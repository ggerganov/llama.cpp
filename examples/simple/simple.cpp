#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "build-info.h"

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    gpt_params params;

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [PROMPT]\n" , argv[0]);
        return 1 ;
    }

    if (argc >= 2) {
        params.model = argv[1];
    }

    if (argc >= 3) {
        params.prompt = argv[2];
    }

    if (params.prompt.empty()) {
        params.prompt = "Hello my name is";
    }

    // init LLM

    llama_backend_init(params.numa);

    llama_context_params ctx_params = llama_context_default_params();

    llama_model * model = llama_load_model_from_file(params.model.c_str(), ctx_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, params.prompt, true);

    const int max_context_size     = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) tokens_list.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) tokens_list.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    for (auto id : tokens_list) {
        fprintf(stderr, "%s", llama_token_to_str(ctx, id).c_str());
    }

    fflush(stderr);

    // main loop

    // The LLM keeps a contextual cache memory of previous token evaluation.
    // Usually, once this cache is full, it is required to recompute a compressed context based on previous
    // tokens (see "infinite text generation via context swapping" in the main example), but in this minimalist
    // example, we will just stop the loop once this cache is full or once an end of stream is detected.

    const int n_gen = std::min(32, max_context_size);

    while (llama_get_kv_cache_token_count(ctx) < n_gen) {
        // evaluate the transformer

        if (llama_eval(ctx, tokens_list.data(), int(tokens_list.size()), llama_get_kv_cache_token_count(ctx), params.n_threads)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        tokens_list.clear();

        // sample the next token

        llama_token new_token_id = 0;

        auto logits  = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
        }

        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        new_token_id = llama_sample_token_greedy(ctx , &candidates_p);

        // is it an end of stream ?
        if (new_token_id == llama_token_eos(ctx)) {
            fprintf(stderr, " [end of text]\n");
            break;
        }

        // print the new token :
        printf("%s", llama_token_to_str(ctx, new_token_id).c_str());
        fflush(stdout);

        // push this new token for next evaluation
        tokens_list.push_back(new_token_id);
    }

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
