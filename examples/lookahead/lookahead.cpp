#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

struct seq_ngram {
    bool active   = false;

    std::vector<llama_token> tokens;
};

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    const int W = 5; // lookahead window
    const int N = 4; // n-gram size
    const int G = 5; // max verification n-grams

    const bool dump_kv_cache = params.dump_kv_cache;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("lookahead", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model = NULL;
    llama_context * ctx = NULL;

    // load the target model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    // Tokenize the prompt
    const bool add_bos = llama_should_add_bos_token(model);
    LOG("add_bos tgt: %d\n", add_bos);

    std::vector<llama_token> inp;
    std::vector<llama_token> all;

    inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);
    all = inp;

    const int max_context_size     = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    for (auto id : inp) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt
    llama_decode(ctx, llama_batch_get_one( inp.data(), n_input - 1, 0,           0));
    llama_decode(ctx, llama_batch_get_one(&inp.back(),           1, n_input - 1, 0));

    for (int s = 0; s < W + G + 1; ++s) {
        llama_kv_cache_seq_cp(ctx, 0, s, -1, -1);
    }

    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_accept  = 0;

    int n_past = inp.size();

    llama_token id = 0;

    // used to determine end of generation
    bool has_eos = false;

    // seq_id == 0           : the current input token
    // seq_id [1, W]         : tokens from the past N - 1 Jacobi iterations
    // seq_id [W + 1, W + G] : verification n-grams
    llama_batch batch = llama_batch_init(params.n_ctx, 0, W + G + 1);

    // target model sampling context
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    // verification n-grams
    std::vector<seq_ngram> drafts(G);

    // tokens for the past N - 1 Jacobi iterations
    // TODO: how to initialize?
    std::vector<std::vector<llama_token>> tokens_j(N - 1);
    for (int j = 0; j < N - 1; j++) {
        tokens_j[j].resize(W);
        for (int i = 0; i < W; i++) {
            tokens_j[j][i] = all[1 + rand() % (all.size() - 1)];
        }
    }

    std::vector<llama_seq_id> seq_id_look(W + 1);
    for (int i = 0; i < W + 1; i++) {
        seq_id_look[i] = i;
    }

    std::vector<llama_seq_id> seq_id_all(W + G + 1);
    for (int i = 0; i < W + G + 1; i++) {
        seq_id_all[i] = i;
    }

    // debug
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, W + G + 1);

    const auto t_dec_start = ggml_time_us();

    // sample first token
    {
        id = llama_sampling_sample(ctx_sampling, ctx, NULL, 0);

        llama_sampling_accept(ctx_sampling, ctx, id, true);

        {
            const std::string token_str = llama_token_to_piece(ctx, id);

            printf("%s", token_str.c_str());
            fflush(stdout);
        }
    }

    while (true) {
        // debug
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            dump_kv_cache_view_seqs(kvc_view, 40);
        }

        // build the mask from https://lmsys.org/blog/2023-11-21-lookahead-decoding/
        {
            llama_batch_clear(batch);

            llama_batch_add(batch, id, n_past, seq_id_all, true);
            for (int i = 1; i < W; i++) {
                llama_batch_add(batch, tokens_j[0][i], n_past + i, seq_id_look, false);
            }
            for (int j = 1; j < N - 1; j++) {
                for (int i = 0; i < W; i++) {
                    llama_batch_add(batch, tokens_j[j][i], n_past + j + i, { i + 1 }, j == N - 2);
                }
            }

            // TODO: add verification n-grams
        }

        llama_decode(ctx, batch);

        id = llama_sampling_sample(ctx_sampling, ctx, NULL, 0);

        llama_sampling_accept(ctx_sampling, ctx, id, true);

        {
            const std::string token_str = llama_token_to_piece(ctx, id);

            printf("%s", token_str.c_str());
            fflush(stdout);

            if (id == llama_token_eos(model)) {
                has_eos = true;
            }
        }

        ++n_predict;
        ++n_past;

        if (n_predict > params.n_predict || has_eos) {
            break;
        }

        // update Jacobi tokens (or whatever these are called)
        {
            for (int j = 0; j < N - 2; j++) {
                tokens_j[j] = tokens_j[j + 1];
            }

            for (int i = 0; i < W; i++) {
                tokens_j[N - 2][i] = llama_sampling_sample(ctx_sampling, ctx, NULL, W*(N - 2) + i);
            }
        }

        // verification
        // TODO
        {
        }

        llama_kv_cache_seq_rm(ctx, -1, n_past, -1);
    }

    auto t_dec_end = ggml_time_us();

    LOG_TEE("\n\n");

    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_TEE("\n");
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("n_accept  = %d\n", n_accept);

    llama_print_timings(ctx);

    llama_kv_cache_view_free(&kvc_view);
    llama_sampling_free(ctx_sampling);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
