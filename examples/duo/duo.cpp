#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using llama_tokens = std::vector<llama_token>;

struct speculation_context
{
    llama_tokens speculation;
    int32_t      instance_id;
    std::mutex   mtx;
};

speculation_context spec_ctx;

static void split_done_cb(int split)
{
    //fprintf(stderr, "split done: %d\n", split);
    if (split == 1 || split == 2)
    {
        std::lock_guard<std::mutex> guard(spec_ctx.mtx);
        spec_ctx.instance_id = 3 - split;
    }
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    params.cb_split_done = split_done_cb;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    llama_tokens input = llama_tokenize(ctx, params.prompt, true);
    const size_t n_input = input.size();

    // print the prompt token-by-token
    for (auto id : input) {
        fprintf(stdout, "%s", llama_token_to_piece(ctx, id).c_str());
    }
    fflush(stdout);

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < input.size(); i++) {
        llama_batch_add(batch, input[i], i, { 0 }, false);
    }

    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    // we'll use logits from this position to determine next token
    int logit_idx = batch.n_tokens - 1;

    while (n_decode <= params.n_predict) {
        // sample the next token
        {
            auto   n_vocab = llama_n_vocab(model);
            auto * logits  = llama_get_logits_ith(ctx, logit_idx);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // sample the most likely token
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id) || n_decode >= params.n_predict) {
                break;
            }

            fprintf(stdout, "%s", llama_token_to_piece(ctx, new_token_id).c_str());
            fflush(stdout);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            // we still use the 'original' token to sample on next iteration
            logit_idx = batch.n_tokens - 1;

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        // remove the cached entries from mock tokens
        llama_kv_cache_seq_rm(ctx, 0, n_cur, -1);
    }

    LOG_TEE("\n");

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    //llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
