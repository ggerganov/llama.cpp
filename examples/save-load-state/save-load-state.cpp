#include <vector>
#include <cstdio>
#include <chrono>

#include "common.h"
#include "llama.h"
#include "llama.cpp"

using namespace std;

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";
    params.seed = 42;
    params.n_threads = 4;
    params.repeat_last_n = 64;
    params.prompt = "The quick brown fox";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    auto lparams = llama_context_default_params();

    lparams.n_ctx      = params.n_ctx;
    lparams.n_parts    = params.n_parts;
    lparams.seed       = params.seed;
    lparams.f16_kv     = params.memory_f16;
    lparams.use_mmap   = params.use_mmap;
    lparams.use_mlock  = params.use_mlock;

    auto n_past = 0;
    auto last_n_tokens_data = vector<llama_token>(params.repeat_last_n, 0);

    // init
    auto ctx = llama_init_from_file(params.model.c_str(), lparams);
    auto tokens = vector<llama_token>(params.n_ctx);
    auto n_prompt_tokens = llama_tokenize(ctx, params.prompt.c_str(), tokens.data(), tokens.size(), true);

    if (n_prompt_tokens < 1) {
        fprintf(stderr, "%s : failed to tokenize prompt\n", __func__);
        return 1;
    }

    // evaluate prompt

    llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past, params.n_threads);

    last_n_tokens_data.insert(last_n_tokens_data.end(), tokens.data(), tokens.data() + n_prompt_tokens);
    n_past += n_prompt_tokens;

    // Save state (rng, logits, embedding and kv_cache) to file
    FILE *fp_write = fopen("dump_state.bin", "wb");
    auto state_size = llama_get_state_size(ctx);
    auto state_mem = new uint8_t[state_size];
    llama_copy_state_data(ctx, state_mem); // could also copy directly to memory mapped file
    fwrite(state_mem, 1, state_size, fp_write);
    fclose(fp_write);

    // save state (last tokens)
    auto last_n_tokens_data_saved = vector<llama_token>(last_n_tokens_data);
    auto n_past_saved = n_past;

    // first run
    printf("\n%s", params.prompt.c_str());
    for (auto i = 0; i < params.n_predict; i++) {
        auto next_token = llama_sample_top_p_top_k(
            ctx,
            &last_n_tokens_data.back() - params.repeat_last_n,
            params.repeat_last_n,
            40,
            1.0,
            1.0,
            1.1);
        auto next_token_str = llama_token_to_str(ctx, next_token);
        last_n_tokens_data.push_back(next_token);
        printf("%s", next_token_str);
        if (llama_eval(ctx, &next_token, 1, n_past, params.n_threads)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            return 1;
        }
        n_past += 1;
    }
    printf("\n\n");

    // free old model
    llama_free(ctx);

    // load new model

    auto ctx2 = llama_init_from_file(params.model.c_str(), lparams);

    // Load state (rng, logits, embedding and kv_cache) from file
    FILE *fp_read = fopen("dump_state.bin", "rb");
    auto state_size2 = llama_get_state_size(ctx2);
    if (state_size != state_size2) {
        fprintf(stderr, "\n%s : failed to validate state size\n", __func__);
    }
    fread(state_mem, 1, state_size, fp_read);
    llama_set_state_data(ctx2, state_mem);  // could also read directly from memory mapped file
    fclose(fp_read);

    // restore state (last tokens)
    last_n_tokens_data = last_n_tokens_data_saved;
    n_past = n_past_saved;

    // second run
    for (auto i = 0; i < params.n_predict; i++) {
        auto next_token = llama_sample_top_p_top_k(
            ctx2,
            &last_n_tokens_data.back() - params.repeat_last_n,
            params.repeat_last_n,
            40,
            1.0,
            1.0,
            1.1);
        auto next_token_str = llama_token_to_str(ctx2, next_token);
        last_n_tokens_data.push_back(next_token);
        printf("%s", next_token_str);
        if (llama_eval(ctx2, &next_token, 1, n_past, params.n_threads)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            return 1;
        }
        n_past += 1;
    }
    printf("\n\n");
    return 0;
}
