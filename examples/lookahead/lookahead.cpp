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

struct ngram_container {
    ngram_container(int n_vocab, int N, int G) {
        cnt.resize(n_vocab);
        head.resize(n_vocab);
        tokens.resize(n_vocab * (N - 1)*G);
    }

    int n_total = 0;

    std::vector<int> cnt;
    std::vector<int> head;

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
    std::vector<seq_ngram> ngrams(G);

    // tokens for the past N - 1 Jacobi iterations
    std::vector<llama_token> tokens_j_prev(W);
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

    ngram_container ngrams_observed(llama_n_vocab(model), N, G);

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

        // print known n-grams starting with token id
        if (1) {
            if (ngrams_observed.cnt[id] > 0) {
                printf("\n - %d n-grams starting with '%s'\n", ngrams_observed.cnt[id], llama_token_to_piece(ctx, id).c_str());
            }

            for (int i = 0; i < ngrams_observed.cnt[id]; i++) {
                printf("   - ngram %2d: ", i);

                const int idx = id*(N - 1)*G + i*(N - 1);

                for (int j = 0; j < N - 1; j++) {
                    const std::string token_str = llama_token_to_piece(ctx, ngrams_observed.tokens[idx + j]);

                    printf("%s", token_str.c_str());
                }

                printf("\n");
            }
        }

        // update Jacobi tokens (or whatever these are called)
        {
            for (int i = 0; i < W; i++) {
                tokens_j_prev[i] = tokens_j[0][i];
            }

            for (int j = 0; j < N - 2; j++) {
                tokens_j[j] = tokens_j[j + 1];
            }

            for (int i = 0; i < W; i++) {
                tokens_j[N - 2][i] = llama_sampling_sample(ctx_sampling, ctx, NULL, W*(N - 2) + i);
            }
        }

        // update observed ngrams
        {
            // the first token of the n-gram is determined by the index in the container so it is not stored
            std::vector<llama_token> ngram(N - 1);

            // n-gram generation
            for (int f = 0; f < W; ++f) {
                for (int j = 0; j < N - 1; ++j) {
                    ngram[j] = tokens_j[j][f];
                };

                const int ft   = tokens_j_prev[f]; // first token of the n-gram
                const int head = ngrams_observed.head[ft];
                const int idx  = ft*(N - 1)*G + head*(N - 1);

                for (int i = 0; i < N - 1; i++) {
                    ngrams_observed.tokens[idx + i] = ngram[i];
                }

                ngrams_observed.cnt[ft]  = std::min(G, ngrams_observed.cnt[ft] + 1);
                ngrams_observed.head[ft] = (head + 1) % G;

                ngrams_observed.n_total++;
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
