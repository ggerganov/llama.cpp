#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static std::vector<llama_token> heal_last_token(const llama_context * ctx, const std::vector<llama_token> & tokens_list) {
    const llama_token last_token_id = tokens_list.back();
    const llama_model * model = llama_get_model(ctx);
    const int32_t n_vocab = llama_n_vocab(model);

    // Don't roll back e.g. <|endoftext|> (set parse_special=true in llama_tokenize)
    if (llama_token_get_type(model, last_token_id) != LLAMA_TOKEN_TYPE_NORMAL) {
        return {};
    }

    const std::string last_piece = llama_token_to_piece(ctx, last_token_id);
    fprintf(stderr, "token_healing: prefix = '%s'\n", last_piece.c_str());

    fprintf(stderr, "token_healing: candidates:\n");
    fprintf(stderr, " [%6d] '%s'\n", last_token_id, last_piece.c_str());
    std::vector<llama_token> candidates = { last_token_id };
    for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
        if (token_id == last_token_id) {
            continue;
        }
        std::string token_piece = llama_token_to_piece(ctx, token_id);
        if (token_piece.rfind(last_piece, 0) != std::string::npos) {
            candidates.push_back(token_id);
            fprintf(stderr, " [%6d] '%s'\n", token_id, token_piece.c_str());
        }
    }
    if (candidates.size() == 1) {
        // No healing necessary if the last token is the only candidate.
        return {};
    }
    return candidates;
}

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

    // total length of the sequence including the prompt
    const int n_len = 32;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = llama_model_default_params();

    // model_params.n_gpu_layers = 99; // offload all layers to the GPU

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed  = 1234;
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, params.prompt, true);

    // Roll back the last token and constrain tokens to generate in the next step to match the removed last token.
    std::vector<llama_token> token_healing_candidates = heal_last_token(ctx, tokens_list);
    if (!token_healing_candidates.empty()) {
        tokens_list.pop_back();
    }
    if (tokens_list.empty()) {
        // If we remove the first token, llama_decode would crash with an empty sequence, so add bos.
        tokens_list.emplace_back(llama_token_bos(model));
    }

    const int n_ctx    = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    LOG_TEE("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len, n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        LOG_TEE("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    fprintf(stderr, "\n");

    for (auto id : tokens_list) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // main loop

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_len) {
        // sample the next token
        {
            auto   n_vocab = llama_n_vocab(model);
            auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            if (n_decode == 0 && !token_healing_candidates.empty()) {
                for (const llama_token token_id : token_healing_candidates) {
                    candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
                }
            } else {
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
                }
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // sample the most likely token
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_len) {
                LOG_TEE("\n");

                break;
            }

            LOG_TEE("%s", llama_token_to_piece(ctx, new_token_id).c_str());
            fflush(stdout);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    LOG_TEE("\n");

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
