#include "common.h"
#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char ** argv){
    const char * static_input_file = "./wikitext-2-raw/wiki.train.raw";
    std::ifstream file(static_input_file);
    if (!file) {
        fprintf(stderr, "error: failed to open file '%s'\n", static_input_file);
        exit(1);
    }
    std::string static_input;
    std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(static_input));
    if (!static_input.empty() && static_input.back() == '\n') {
        static_input.pop_back();
    }
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    // max/min n-grams size to search for in prompt
    const int ngram_max = 4;
    const int ngram_min = 1;

    // length of the candidate / draft sequence, if match is found
    const int n_draft = params.n_draft;

    const bool dump_kv_cache = params.dump_kv_cache;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("lookup", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model = NULL;
    llama_context * ctx = NULL;

    // load the model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    // tokenize the prompt
    const bool add_bos = llama_should_add_bos_token(model);
    LOG("add_bos tgt: %d\n", add_bos);

    std::vector<llama_token> inp;
    std::vector<llama_token> inp_static;
    inp        = ::llama_tokenize(ctx, params.prompt, add_bos, true);
    inp_static = ::llama_tokenize(ctx, static_input,  add_bos, true);

    std::unordered_map<int64_t, std::unordered_map<llama_token, int>> hashmap = {};
    for (size_t i = 0; i < inp_static.size()-2; ++i) {
        int64_t key_low  = inp_static[i + 0];
        int64_t key_high = inp_static[i + 1];
        key_low  <<=  0;
        key_high <<= 32;
        const int64_t key = key_low | key_high;

        const llama_token value = inp_static[i + 2];

        auto frequency_it = hashmap.find(key);
        std::unordered_map<llama_token, int> frequency;
        if (frequency_it != hashmap.end()) {
            frequency = frequency_it->second;
        }

        auto token_it = frequency.find(value);
        if (token_it != frequency.end()) {
            token_it->second++;
        } else {
            frequency.emplace(std::make_pair(value, 1));
        }

        if (frequency_it == hashmap.end()) {
            hashmap.emplace(std::make_pair(key, frequency));
        }
    }
    printf("\n\n%ld\n\n", hashmap.size());
    std::unordered_map<int64_t, llama_token> hashmap_max;
    for (auto item : hashmap) {
        const int64_t key = item.first;
        const std::unordered_map<llama_token, int> frequency = item.second;
        GGML_ASSERT(!frequency.empty());

        llama_token max_token = -1;
        int max_frequency = 0;
        for (auto item2 : frequency) {
            if (item2.second > max_frequency) {
                max_token = item2.first;
                max_frequency = item2.second;
            }
        }
        GGML_ASSERT(max_token != -1);

        hashmap_max.emplace(std::make_pair(key, max_token));
    }
    printf("\n\n%ld\n\n", hashmap_max.size());

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

    llama_decode(ctx, llama_batch_get_one( inp.data(), n_input - 1, 0,           0));
    llama_decode(ctx, llama_batch_get_one(&inp.back(),           1, n_input - 1, 0));

    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past = inp.size();

    bool has_eos = false;

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    std::vector<llama_token> draft;

    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);

    // debug
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, 1);

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // debug
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            dump_kv_cache_view_seqs(kvc_view, 40);
        }

        // print current draft sequence
        LOG("drafted %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, draft).c_str());

        int i_dft = 0;
        while (true) {
            // sample from the target model
            llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL, i_dft);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            const std::string token_str = llama_token_to_piece(ctx, id);

            if (!params.use_color) {
                printf("%s", token_str.c_str());
            }

            if (id == llama_token_eos(model)) {
                has_eos = true;
            }

            ++n_predict;

            // check if the target token matches the draft
            if (i_dft < (int) draft.size() && id == draft[i_dft]) {
                LOG("the sampled target token matches the %dth drafted token (%d, '%s') - accepted\n", i_dft, id, token_str.c_str());
                ++n_accept;
                ++n_past;
                ++i_dft;
                inp.push_back(id);
                // fprintf(stderr, "pushed: %d\n", id);

                if (params.use_color) {
                    // color accepted draft token
                    printf("\033[34m%s\033[0m", token_str.c_str());
                    fflush(stdout);
                }
                continue;
            }

            if (params.use_color) {
                printf("%s", token_str.c_str());
            }
            fflush(stdout);


            LOG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

            draft.clear();
            draft.push_back(id);
            inp.push_back(id);
            // fprintf(stderr, "pushed: %d\n", id);
            break;
        }

        if ((params.n_predict > 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // KV cache management
        // clean the cache of draft tokens that weren't accepted
        llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

        llama_batch_clear(batch_tgt);
        llama_batch_add(batch_tgt, draft[0], n_past, { 0 }, true);

        // generate n_pred tokens through prompt lookup
        auto prompt_lookup = [&]() -> void {
            for (int i = 0; i < n_draft; ++i) {
                // fprintf(stderr, "lookup: %d %d\n", inp[inp.size() - 2], inp[inp.size() - 1]);
                int64_t key_low  = inp[inp.size() - 2];
                int64_t key_high = inp[inp.size() - 1];
                key_low  <<=  0;
                key_high <<= 32;
                const int64_t key = key_low | key_high;

                auto item_it = hashmap_max.find(key);
                if (item_it == hashmap_max.end()) {
                    break;
                }

                draft.push_back(item_it->second);
                llama_batch_add(batch_tgt, item_it->second, n_past + i + 1, { 0 }, true);
                ++n_drafted;
            }
            return;
        };

        prompt_lookup();

        llama_decode(ctx, batch_tgt);
        ++n_past;

        draft.erase(draft.begin());
    }

    auto t_dec_end = ggml_time_us();

    LOG_TEE("\n\n");

    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_TEE("\n");
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("n_drafted = %d\n", n_drafted);
    LOG_TEE("n_accept  = %d\n", n_accept);
    LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_TEE("\ntarget:\n");
    llama_print_timings(ctx);

    llama_sampling_free(ctx_sampling);
    llama_batch_free(batch_tgt);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
