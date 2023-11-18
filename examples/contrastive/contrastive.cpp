#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <limits>

int main(int argc, char ** argv) {
    gpt_params params_expert;
    gpt_params params_amateur;
    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s EXPERT_MODEL_PATH AMATEUR_MODEL_PATH [PROMPT] [alpha] [beta]\n", argv[0]);
        return 1;
    }

    if (argc >= 2) {
        params_expert.model = argv[1];
    }

    if (argc >= 3) {
        params_amateur.model = argv[2];
    }

    if (argc >= 4) {
        params_expert.prompt = argv[3];
        params_amateur.prompt = argv[3];
    }

    float alpha = 0.1;
    float beta = 0.5;

    if (argc >= 5) {
        alpha = std::stof(argv[4]);
    }

    if (argc >= 6) {
        beta = std::stof(argv[5]);
    }

    if (params_expert.prompt.empty()) {
        params_expert.prompt = "Hello my name is";
        params_amateur.prompt = "Hello my name is";
    }

    // total length of the sequence including the prompt
    const int n_len = 32;

    // init LLM

    llama_backend_init(params_expert.numa);

    // initialize the model

    llama_model_params model_params = llama_model_default_params();

    // model_params.n_gpu_layers = 99; // offload all layers to the GPU

    llama_model * model_expert = llama_load_model_from_file(params_expert.model.c_str(), model_params);
    llama_model * model_amateur = llama_load_model_from_file(params_amateur.model.c_str(), model_params);


    if (model_expert == NULL) {
        fprintf(stderr, "%s: error: unable to load expert model\n", __func__);
        return 1;
    }

    if (model_amateur == NULL) {
        fprintf(stderr, "%s: error: unable to load amateur model\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params_expert.n_threads;
    ctx_params.n_threads_batch = params_expert.n_threads_batch == -1 ? params_expert.n_threads : params_expert.n_threads_batch;

    llama_context * ctx_expert = llama_new_context_with_model(model_expert, ctx_params);
    llama_context * ctx_amateur = llama_new_context_with_model(model_amateur, ctx_params);

    if (ctx_expert == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context for expert\n", __func__);
        return 1;
    }

    if (ctx_amateur == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context for amateur\n", __func__);
        return 1;
    }

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx_expert, params_expert.prompt, true);

    const int n_ctx    = llama_n_ctx(ctx_expert);
    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    LOG_TEE("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len, n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        LOG_TEE("%s:        either reduce n_parallel or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    fprintf(stderr, "\n");

    for (auto id : tokens_list) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx_expert, id).c_str());
    }

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx_expert, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    if (llama_decode(ctx_amateur, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // main loop

    int n_cur    = batch.n_tokens;
    int n_decode = 0;
    float log_alpha = std::log(alpha);

    const auto t_main_start = ggml_time_us();
    while (n_cur <= n_len) {
        // sample the next token
        {
            auto   n_vocab = llama_n_vocab(model_expert);
            auto * logits_expert  = llama_get_logits_ith(ctx_expert, batch.n_tokens - 1);
            auto * logits_amateur  = llama_get_logits_ith(ctx_amateur, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            auto largest_expert_logit = *std::max_element(logits_expert, logits_expert + n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                float cd_logit = std::numeric_limits<float>::lowest();
                if (logits_expert[token_id] > log_alpha + largest_expert_logit) {
                    cd_logit = (1+beta)*logits_expert[token_id] - beta*logits_amateur[token_id];
                }
                candidates.emplace_back(llama_token_data{ token_id, cd_logit, 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // sample the most likely token
            const llama_token new_token_id_expert = llama_sample_token_greedy(ctx_expert, &candidates_p);

            // is it an end of stream?
            if (new_token_id_expert == llama_token_eos(model_expert) || n_cur == n_len) {
                LOG_TEE("\n");
                break;
            }

            LOG_TEE("%s", llama_token_to_piece(ctx_expert, new_token_id_expert).c_str());
            fflush(stdout);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id_expert, n_cur, { 0 }, true);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx_expert, batch)) {
            fprintf(stderr, "%s : failed to eval, return code 1\n", __func__);
            return 1;
        }
        if (llama_decode(ctx_amateur, batch)) {
            fprintf(stderr, "%s : failed to eval, return code 1\n", __func__);
            return 1;
        }
    }

    LOG_TEE("\n");

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(ctx_expert);
    llama_print_timings(ctx_amateur);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx_expert);
    llama_free(ctx_amateur);
    llama_free_model(model_expert);
    llama_free_model(model_amateur);

    llama_backend_free();

    return 0;
}

