#include "llama.h"
#include <cstdio>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s <model.gguf> [prompt]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "Hello my name is";
    int n_predict = 32;

    if (argc < 2) {
        print_usage(argc, argv);
        return 1;
    }
    model_path = argv[1];

    if (argc > 2) {
        prompt = argv[2];
        for (int i = 3; i < argc; i++) {
            prompt += " ";
            prompt += argv[i];
        }
    }

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99; // offload all layers to GPU
    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512; // maximum context size
    ctx_params.no_perf = false;
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    int n_tokens = llama_tokenize(model, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    tokens_list.resize(-n_tokens);
    if (llama_tokenize(model, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    const int n_ctx    = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size());

    fprintf(stderr, "%s: n_predict = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_predict, n_ctx, n_kv_req);


    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        fprintf(stderr, "%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        fprintf(stderr, "%s:        either reduce n_predict or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    fprintf(stderr, "\n");

    for (auto id : tokens_list) {
        char buf[128];
        int n = llama_token_to_piece(model, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size(), 0, 0);

    // evaluate the initial prompt

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // main loop

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_predict) {
        // sample the next token
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        {

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict) {
                fprintf(stderr, "\n");

                break;
            }

            char buf[128];
            int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch
            batch = llama_batch_get_one(&new_token_id, 1, n_cur, 0);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    fprintf(stderr, "\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
