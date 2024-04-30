#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define TH_VERBOSE  // print token healing candidates

enum class token_healing_type : uint8_t {
    ROLLBACK_LAST,   // roll back last token with a single constrained decoding step
    ROLLBACK_MULTI,  // roll back a fixed amount of tokens, multiple constrained decoding steps
    DYNAMIC_ONCE,    // dynamic roll back, single constrained decoding step
    DYNAMIC_MULTI    // dynamic roll back, multiple constrained decoding steps
};

struct token_healing_context {
    std::string prefix;  // remaining prefix to generate (the input prompt's suffix)

    std::vector<std::string> vocab;  // map token id to token piece
    // TODO consider using a prefix tree
};

static bool startswith(const std::string & str, const std::string & prefix) {
    return str.rfind(prefix, 0) != std::string::npos;
}

static bool token_healing_prefix_exists(const token_healing_context * th_ctx, const std::string & prefix) {
    for (const std::string & token : th_ctx->vocab) {
        if (startswith(token, prefix)) {
            return true;
        }
    }
    return false;
}

static std::vector<llama_token> token_healing_find_prefix(
                                const token_healing_context * th_ctx,
                                const std::string & prefix,
                                const bool include_partial_prefix) {
    // Example: prefix=" world" -> " world", " worldwide", ...
    // If `include_partial_prefix`, include also: " w", " wo", ...
    std::vector<llama_token> candidates;
    const auto & vocab = th_ctx->vocab;
    for (size_t token_id = 0; token_id < vocab.size(); ++token_id) {
        if (startswith(vocab[token_id], prefix)
            || (include_partial_prefix && startswith(prefix, vocab[token_id]))) {
            candidates.push_back((llama_token)token_id);
        }
    }
    return candidates;
}

static token_healing_context * token_healing_init(const llama_context * ctx) {
    auto * th_ctx = new token_healing_context;
    const llama_model * model = llama_get_model(ctx);
    const int32_t n_vocab = llama_n_vocab(model);
    std::vector<std::string> & vocab = th_ctx->vocab;
    vocab.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        vocab.emplace_back(llama_token_to_piece(ctx, token_id, true));
    }
    return th_ctx;
}

static void token_healing_free(token_healing_context * th_ctx) {
    delete th_ctx;
}

static int token_healing_heal(
           const llama_context * ctx,
           std::vector<llama_token> & tokens_list,
           const token_healing_type th_type,
           token_healing_context * th_ctx,
           int n_rollback = 1) {
    if (tokens_list.empty()) {
        return 0;
    }
    const llama_model * model = llama_get_model(ctx);
    const bool is_dynamic = th_type == token_healing_type::DYNAMIC_ONCE || th_type == token_healing_type::DYNAMIC_MULTI;
    const int n_ctx = tokens_list.size();
    const int max_to_remove = is_dynamic ? n_ctx : std::min(n_rollback, n_ctx);
    int n_removed = 0;
    std::string prefix;
    // Roll back tokens a fixed amount or until there does not exist a token that can cover the prompt
    // and stop early if a special token is encountered
    while (n_removed < max_to_remove) {
        const llama_token next_token_id = tokens_list[n_ctx - n_removed - 1];
        if (llama_token_get_type(model, next_token_id) != LLAMA_TOKEN_TYPE_NORMAL) {
            // Don't roll back e.g. <|endoftext|> (if parse_special=true in llama_tokenize)
            break;
        }
        std::string new_prefix = th_ctx->vocab[next_token_id] + prefix;
        if (is_dynamic && !token_healing_prefix_exists(th_ctx, new_prefix)) {
            break;
        }
        n_removed += 1;
        prefix = new_prefix;
    }
    th_ctx->prefix = prefix;

    if (n_removed == 0) {
        return 0;
    }
    // If constrained decoding would give back the original prompt, there is no need to modify the context
    const bool is_multi_decoding = th_type == token_healing_type::DYNAMIC_MULTI || th_type == token_healing_type::ROLLBACK_MULTI;
    const std::vector<llama_token> candidates = token_healing_find_prefix(th_ctx, prefix, is_multi_decoding);
    fprintf(stderr, "token_healing: prefix = '%s' (%d tokens)\n", prefix.c_str(), n_removed);
    if (n_removed == 1 && candidates.size() == 1) {
        fprintf(stderr, "token_healing: nothing to heal\n");
        return 0;
    }
#ifdef TH_VERBOSE
    if (!is_multi_decoding) {
        // Other healing types get printed during decoding
        for (const llama_token token_id : candidates) {
            fprintf(stderr, " [%6d] '%s'\n", token_id, th_ctx->vocab[token_id].c_str());
        }
    }
#endif
    for (int i = 0; i < n_removed; ++i) {
        tokens_list.pop_back();
    }
    if (tokens_list.empty()) {
        // If the first token was removed, llama_decode would crash with an empty sequence, so add bos.
        tokens_list.emplace_back(llama_token_bos(model));
    }
    return n_removed;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [PROMPT] [TOKEN_HEALING 0|1|d1|d|r[N]]\n" , argv[0]);
        return 1;
    }

    if (argc >= 2) {
        params.model = argv[1];
    }

    if (argc >= 3) {
        params.prompt = argv[2];
    }

    bool token_healing_enabled = true;
    auto th_type = token_healing_type::DYNAMIC_MULTI;
    int th_n_rollback = 1;
    if (argc >= 4) {
        std::string value(argv[3]);
        /**/ if (value    == "0" ) { token_healing_enabled = false; }
        else if (value    == "1" ) { th_type = token_healing_type::ROLLBACK_LAST; th_n_rollback = 1; }
        else if (value    == "d1") { th_type = token_healing_type::DYNAMIC_ONCE; }
        else if (value    == "d" ) { th_type = token_healing_type::DYNAMIC_MULTI; }
        else if (value[0] == 'r' ) {
            th_type = token_healing_type::ROLLBACK_MULTI;
            th_n_rollback = std::stoi(value.substr(1));
            if (th_n_rollback <= 0) {
                token_healing_enabled = false;
            }
        } else {
            printf("usage: %s MODEL_PATH [PROMPT] [TOKEN_HEALING 0|1|d1|d|r[N]]\n" , argv[0]);
            return 1;
        }
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

    token_healing_context * th_ctx = nullptr;
    if (token_healing_enabled) {
        th_ctx = token_healing_init(ctx);
        int th_n_tokens_removed = token_healing_heal(ctx, tokens_list, th_type, th_ctx, th_n_rollback);
        if (th_n_tokens_removed == 0) {
            token_healing_enabled = false;
        }
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

            if (token_healing_enabled) {
                // Constrain tokens based on the remaining token healing prefix
                // N.B. We could also set token constraints by setting rejected tokens' logits to -inf
                std::vector<llama_token> th_candidates;
                if (th_type == token_healing_type::ROLLBACK_LAST || th_type == token_healing_type::DYNAMIC_ONCE) {
                    th_candidates = token_healing_find_prefix(th_ctx, th_ctx->prefix, false);
                } else {
                    th_candidates = token_healing_find_prefix(th_ctx, th_ctx->prefix, true);
#ifdef TH_VERBOSE
                    fprintf(stderr, "\ntoken_healing: prefix = '%s'\n", th_ctx->prefix.c_str());
                    for (const llama_token token_id : th_candidates) {
                        fprintf(stderr, " [%6d] '%s'\n", token_id, th_ctx->vocab[token_id].c_str());
                    }
#endif
                }
                for (const llama_token token_id: th_candidates) {
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

            std::string new_token_piece = llama_token_to_piece(ctx, new_token_id);
            LOG_TEE("%s", new_token_piece.c_str());
            fflush(stdout);

            if (token_healing_enabled) {
                if (new_token_piece.size() < th_ctx->prefix.size()) {
                    // Shift prefix constraint (for multi step token healing)
                    th_ctx->prefix = th_ctx->prefix.substr(new_token_piece.size());
                } else {
                    th_ctx->prefix.clear();
                    token_healing_enabled = false;
                }
            }

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

    token_healing_free(th_ctx);
    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
