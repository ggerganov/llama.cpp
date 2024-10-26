#include "arg.h"
#include "common.h"
#include "log.h"
#include "jarvis.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -p \"Hello my name is\" -n 32 -np 4\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;

    params.prompt = "Hello my name is";
    params.n_predict = 32;

    if (!common_params_parse(argc, argv, params, JARVIS_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    common_init();

    // number of parallel batches
    int n_parallel = params.n_parallel;

    // total length of the sequences including the prompt
    int n_predict = params.n_predict;

    // init LLM

    jarvis_backend_init();
    jarvis_numa_init(params.numa);

    // initialize the model

    jarvis_model_params model_params = common_model_params_to_jarvis(params);

    jarvis_model * model = jarvis_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    std::vector<jarvis_token> tokens_list;
    tokens_list = common_tokenize(model, params.prompt, true);

    const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size())*n_parallel;

    // initialize the context

    jarvis_context_params ctx_params = common_context_params_to_jarvis(params);

    ctx_params.n_ctx   = n_kv_req;
    ctx_params.n_batch = std::max(n_predict, n_parallel);

    jarvis_context * ctx = jarvis_new_context_with_model(model, ctx_params);

    auto sparams = jarvis_sampler_chain_default_params();

    jarvis_sampler * smpl = jarvis_sampler_chain_init(sparams);

    jarvis_sampler_chain_add(smpl, jarvis_sampler_init_top_k(params.sparams.top_k));
    jarvis_sampler_chain_add(smpl, jarvis_sampler_init_top_p(params.sparams.top_p, params.sparams.min_keep));
    jarvis_sampler_chain_add(smpl, jarvis_sampler_init_temp (params.sparams.temp));
    jarvis_sampler_chain_add(smpl, jarvis_sampler_init_dist (params.sparams.seed));

    if (ctx == NULL) {
        LOG_ERR("%s: error: failed to create the jarvis_context\n" , __func__);
        return 1;
    }

    const int n_ctx = jarvis_n_ctx(ctx);

    LOG_INF("\n%s: n_predict = %d, n_ctx = %d, n_batch = %u, n_parallel = %d, n_kv_req = %d\n", __func__, n_predict, n_ctx, ctx_params.n_batch, n_parallel, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_ERR("%s: error: n_kv_req (%d) > n_ctx, the required KV cache size is not big enough\n", __func__,  n_kv_req);
        LOG_ERR("%s:        either reduce n_parallel or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    LOG("\n");

    for (auto id : tokens_list) {
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }

    // create a jarvis_batch
    // we use this object to submit token data for decoding
    jarvis_batch batch = jarvis_batch_init(std::max(tokens_list.size(), (size_t) n_parallel), 0, n_parallel);

    std::vector<jarvis_seq_id> seq_ids(n_parallel, 0);
    for (int32_t i = 0; i < n_parallel; ++i) {
        seq_ids[i] = i;
    }

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); ++i) {
        common_batch_add(batch, tokens_list[i], i, seq_ids, false);
    }
    GGML_ASSERT(batch.n_tokens == (int) tokens_list.size());

    if (jarvis_model_has_encoder(model)) {
        if (jarvis_encode(ctx, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        jarvis_token decoder_start_token_id = jarvis_model_decoder_start_token(model);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = jarvis_token_bos(model);
        }

        common_batch_clear(batch);
        common_batch_add(batch, decoder_start_token_id, 0, seq_ids, false);
    }

    // jarvis_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (jarvis_decode(ctx, batch) != 0) {
        LOG_ERR("%s: jarvis_decode() failed\n", __func__);
        return 1;
    }

    //// assign the system KV cache to all parallel sequences
    //// this way, the parallel sequences will "reuse" the prompt tokens without having to copy them
    //for (int32_t i = 1; i < n_parallel; ++i) {
    //    jarvis_kv_cache_seq_cp(ctx, 0, i, -1, -1);
    //}

    if (n_parallel > 1) {
        LOG("\n\n%s: generating %d sequences ...\n", __func__, n_parallel);
    }

    // main loop

    // we will store the parallel decoded sequences in this vector
    std::vector<std::string> streams(n_parallel);

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_predict) {
        // prepare the next batch
        common_batch_clear(batch);

        // sample the next token for each parallel sequence / stream
        for (int32_t i = 0; i < n_parallel; ++i) {
            if (i_batch[i] < 0) {
                // the stream has already finished
                continue;
            }

            const jarvis_token new_token_id = jarvis_sampler_sample(smpl, ctx, i_batch[i]);

            // is it an end of generation? -> mark the stream as finished
            if (jarvis_token_is_eog(model, new_token_id) || n_cur == n_predict) {
                i_batch[i] = -1;
                LOG("\n");
                if (n_parallel > 1) {
                    LOG_INF("%s: stream %d finished at n_cur = %d", __func__, i, n_cur);
                }

                continue;
            }

            // if there is only one stream, we print immediately to stdout
            if (n_parallel == 1) {
                LOG("%s", common_token_to_piece(ctx, new_token_id).c_str());
            }

            streams[i] += common_token_to_piece(ctx, new_token_id);

            i_batch[i] = batch.n_tokens;

            // push this new token for next evaluation
            common_batch_add(batch, new_token_id, n_cur, { i }, true);

            n_decode += 1;
        }

        // all streams are finished
        if (batch.n_tokens == 0) {
            break;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (jarvis_decode(ctx, batch)) {
            LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    if (n_parallel > 1) {
        LOG("\n");

        for (int32_t i = 0; i < n_parallel; ++i) {
            LOG("sequence %d:\n\n%s%s\n\n", i, params.prompt.c_str(), streams[i].c_str());
        }
    }

    const auto t_main_end = ggml_time_us();

    LOG_INF("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    LOG("\n");
    jarvis_perf_sampler_print(smpl);
    jarvis_perf_context_print(ctx);

    fprintf(stderr, "\n");

    jarvis_batch_free(batch);

    jarvis_sampler_free(smpl);
    jarvis_free(ctx);
    jarvis_free_model(model);

    jarvis_backend_free();

    return 0;
}
