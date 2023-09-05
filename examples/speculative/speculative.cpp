#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "build-info.h"

#include "common.h"
#include "llama.h"
#include "grammar-parser.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.model_draft.empty()) {
        fprintf(stderr, "%s: error: --model-draft is required\n", __func__);
        return 1;
    }

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("speculative", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    params.perplexity = true; // HACK: enable logits_all = true
    std::tie(model_tgt, ctx_tgt) = llama_init_from_gpt_params(params);

    // load the draft model
    params.model = params.model_draft;
    std::tie(model_dft, ctx_dft) = llama_init_from_gpt_params(params);

    // tokenize the prompt
    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx_tgt, params.prompt, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    fprintf(stderr, "\n\n");

    for (auto id : inp) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx_tgt, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    llama_eval(ctx_tgt,  inp.data(), int(inp.size() - 1), 0, params.n_threads);
    llama_eval(ctx_tgt, &inp.back(),      1, inp.size() - 1, params.n_threads);
    llama_eval(ctx_dft,  inp.data(),     int(inp.size()), 0, params.n_threads);

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    const int n_ctx   = llama_n_ctx(ctx_tgt);
    const int n_vocab = llama_n_vocab(ctx_tgt);
    //GGML_ASSERT(n_vocab == llama_n_vocab(ctx_dft));

    // how many tokens to draft each time
    const int n_draft = params.n_draft;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    std::vector<llama_token> drafted;

    std::vector<llama_token> last_tokens(n_ctx);
    std::fill(last_tokens.begin(), last_tokens.end(), 0);

    for (auto & id : inp) {
        last_tokens.erase(last_tokens.begin());
        last_tokens.push_back(id);
    }

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    // used to determine end of generation
    bool has_eos = false;

    // grammar stuff
    struct llama_grammar * grammar_dft = NULL;
    struct llama_grammar * grammar_tgt = NULL;

    grammar_parser::parse_state parsed_grammar;

    // if requested - load the grammar, error checking is omitted for brevity
    if (!params.grammar.empty()) {
        parsed_grammar = grammar_parser::parse(params.grammar.c_str());
        // will be empty (default) if there are parse errors
        if (parsed_grammar.rules.empty()) {
            return 1;
        }

        std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
        grammar_tgt = llama_grammar_init(grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    }

    const auto t_dec_start = ggml_time_us();

    while (true) {
        LOG("drafted: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_dft, drafted));

        int i_dft = 0;
        while (true) {
            // sample from the target model
            const llama_token id = llama_sample_token(ctx_tgt, NULL, grammar_tgt, params, last_tokens, candidates, i_dft);

            // remember which tokens were sampled - used for repetition penalties during sampling
            last_tokens.erase(last_tokens.begin());
            last_tokens.push_back(id);

            //LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, last_tokens));

            const std::string token_str = llama_token_to_piece(ctx_tgt, id);
            printf("%s", token_str.c_str());
            fflush(stdout);

            if (id == llama_token_eos(ctx_tgt)) {
                has_eos = true;
            }

            ++n_predict;

            // check if the draft matches the target
            if (i_dft < (int) drafted.size() && id == drafted[i_dft]) {
                LOG("the sampled target token matches the %dth drafted token (%d, '%s') - accepted\n", i_dft, id, token_str.c_str());
                ++n_accept;
                ++n_past_tgt;
                ++n_past_dft;
                ++i_dft;

                continue;
            }

            // the drafted token was rejected or we are out of drafted tokens

            if (i_dft < (int) drafted.size()) {
                LOG("the %dth drafted token (%d, '%s') does not match the sampled target token (%d, '%s') - rejected\n",
                        i_dft, drafted[i_dft], llama_token_to_piece(ctx_dft, drafted[i_dft]).c_str(), id, token_str.c_str());
            } else {
                LOG("out of drafted tokens\n");
            }

            llama_eval(ctx_dft, &id, 1, n_past_dft, params.n_threads);
            ++n_past_dft;

            drafted.clear();
            drafted.push_back(id);

            break;
        }

        if (n_predict > params.n_predict || has_eos) {
            break;
        }

        if (grammar_tgt) {
            if (grammar_dft) {
                llama_grammar_free(grammar_dft);
            }
            grammar_dft = llama_grammar_copy(grammar_tgt);

            LOG("copied target grammar to draft grammar\n");
        }

        // sample n_draft tokens from the draft model using greedy decoding
        int n_past_cur = n_past_dft;
        for (int i = 0; i < n_draft; ++i) {
            float * logits = llama_get_logits(ctx_dft);

            candidates.clear();
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            llama_token_data_array cur_p = { candidates.data(), candidates.size(), false };

            if (grammar_dft != NULL) {
                llama_sample_grammar(ctx_dft, &cur_p, grammar_dft);
            }

            // computes softmax and sorts the candidates
            llama_sample_softmax(ctx_dft, &cur_p);

            for (int i = 0; i < 3; ++i) {
                LOG(" - draft candidate %3d: %6d (%8.3f) '%s'\n", i, cur_p.data[i].id, cur_p.data[i].p, llama_token_to_piece(ctx_dft, cur_p.data[i].id).c_str());
            }

            // TODO: better logic?
            if (cur_p.data[0].p < 2*cur_p.data[1].p) {
                LOG("stopping drafting, probability too low: %.3f < 2*%.3f\n", cur_p.data[0].p, cur_p.data[1].p);
                break;
            }

            // drafted token
            const llama_token id = cur_p.data[0].id;

            drafted.push_back(id);
            ++n_drafted;

            // no need to evaluate the last drafted token, since we won't use the result
            if (i == n_draft - 1) {
                break;
            }

            // evaluate the drafted token on the draft model
            llama_eval(ctx_dft, &drafted.back(), 1, n_past_cur, params.n_threads);
            ++n_past_cur;

            if (grammar_dft != NULL) {
                llama_grammar_accept_token(ctx_dft, grammar_dft, id);
            }
        }

        // evaluate the target model on the drafted tokens
        llama_eval(ctx_tgt, drafted.data(), drafted.size(), n_past_tgt, params.n_threads);
        ++n_past_tgt;

        // the first token is always proposed by the traget model before the speculation loop
        drafted.erase(drafted.begin());
    }

    auto t_dec_end = ggml_time_us();

    LOG_TEE("\n\n");

    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));

    // TODO: make sure these numbers are computed correctly
    LOG_TEE("\n");
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("n_drafted = %d\n", n_drafted);
    LOG_TEE("n_accept  = %d\n", n_accept);
    LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_TEE("\ndraft:\n");
    llama_print_timings(ctx_dft);

    LOG_TEE("\ntarget:\n");
    llama_print_timings(ctx_tgt);

    llama_free(ctx_tgt);
    llama_free_model(model_tgt);

    llama_free(ctx_dft);
    llama_free_model(model_dft);

    if (grammar_dft != NULL) {
        llama_grammar_free(grammar_dft);
        llama_grammar_free(grammar_tgt);
    }
    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
