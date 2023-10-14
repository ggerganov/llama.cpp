#include "build-info.h"

#include "common.h"
#include "llama.h"
#include "grammar-parser.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;

    struct llama_grammar * grammar = NULL;

    std::vector<llama_token> last_tokens;
    struct llama_sampling_context ctx_sampling;
};

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.model_draft.empty()) {
        fprintf(stderr, "%s: error: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    int n_seq_dft = 8;

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
    params.logits_all = true;
    std::tie(model_tgt, ctx_tgt) = llama_init_from_gpt_params(params);

    // load the draft model
    params.model = params.model_draft;
    params.n_gpu_layers = params.n_gpu_layers_draft;
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
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1, 0,           0));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1, n_input - 1, 0));
    llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input,     0,           0));

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    const int n_ctx   = llama_n_ctx(ctx_tgt);
    const int n_vocab = llama_n_vocab(model_tgt);
    //GGML_ASSERT(n_vocab == llama_n_vocab(model_dft));

    // how many tokens to draft each time
    int n_draft = params.n_draft;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // grammar stuff
    struct llama_grammar * grammar = NULL;

    grammar_parser::parse_state parsed_grammar;

    // if requested - load the grammar, error checking is omitted for brevity
    if (!params.grammar.empty()) {
        parsed_grammar = grammar_parser::parse(params.grammar.c_str());
        // will be empty (default) if there are parse errors
        if (parsed_grammar.rules.empty()) {
            return 1;
        }

        std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
        grammar = llama_grammar_init(grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    }

    // target model sampling context
    llama_sampling_context ctx_sampling = llama_sampling_context_init(params, grammar);

    // TODO: move to llama_sampling_state
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    std::vector<llama_token> last_tokens;
    last_tokens.resize(n_ctx);
    std::fill(last_tokens.begin(), last_tokens.end(), 0);

    for (auto & id : inp) {
        last_tokens.erase(last_tokens.begin());
        last_tokens.push_back(id);
    }

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    for (int i = 0; i < n_seq_dft; ++i) {
        {
            auto & last_tokens = drafts[i].last_tokens;

            last_tokens.resize(n_ctx);
            std::fill(last_tokens.begin(), last_tokens.end(), 0);

            for (auto & id : inp) {
                last_tokens.erase(last_tokens.begin());
                last_tokens.push_back(id);
            }
        }

        drafts[i].ctx_sampling = llama_sampling_context_init(params, grammar);
    }

    llama_batch batch_dft = llama_batch_init(512, 0, 1);
    llama_batch batch_tgt = llama_batch_init(512, 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    while (true) {
        for (int i = 0; i < n_seq_dft; ++i) {
            if (!drafts[i].active) continue;

            const auto & tokens = drafts[i].tokens;

            LOG("draft %d: %s\n", i, LOG_TOKENS_TOSTR_PRETTY(ctx_dft, tokens));
        }

        int i_dft = 0;
        int i_keep = 0;

        while (true) {
            LOG("sampling target: i_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", i_keep, i_dft, drafts[i_keep].i_batch_tgt[i_dft]);

            // sample from the target model
            llama_token id = llama_sampling_sample(ctx_tgt, NULL, ctx_sampling, last_tokens, candidates, drafts[i_keep].i_batch_tgt[i_dft]);

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

            // check if the target token matches any of the drafts
            {
                bool matches = false;

                for (int i = 0; i < n_seq_dft; ++i) {
                    if (!drafts[i].active) continue;

                    if (i_dft < (int) drafts[i].tokens.size() && id == drafts[i].tokens[i_dft]) {
                        LOG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, i, id, token_str.c_str());

                        i_keep = i;
                        matches = true;
                    } else {
                        drafts[i].active = false;
                    }
                }

                if (matches) {
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;

                    continue;
                }
            }

            LOG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

            // TODO: simplify
            {
                LOG("keeping sequence %d\n", i_keep);

                llama_kv_cache_seq_keep(ctx_dft, i_keep);
                llama_kv_cache_seq_cp  (ctx_dft, i_keep, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_dft, 0);

                llama_kv_cache_seq_rm  (ctx_tgt, i_keep, n_past_tgt, -1);
                llama_kv_cache_seq_keep(ctx_tgt, i_keep);
                llama_kv_cache_seq_cp  (ctx_tgt, i_keep, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_tgt, 0);
            }

            for (int i = 0; i < n_seq_dft; ++i) {
                drafts[i].active = false;
                drafts[i].tokens.clear();
                drafts[i].i_batch_tgt.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(id);
            drafts[0].i_batch_tgt.push_back(0);

            {
                batch_dft.n_tokens = 1;

                batch_dft.token[0]     = id;
                batch_dft.pos[0]       = n_past_dft;
                batch_dft.n_seq_id[0]  = 1;
                batch_dft.seq_id[0][0] = 0;
                batch_dft.logits[0]    = true;
            }

            llama_kv_cache_seq_rm(ctx_dft, 0, n_past_dft, -1);
            llama_decode(ctx_dft, batch_dft);
            ++n_past_dft;

            break;
        }

        if (n_predict > params.n_predict || has_eos) {
            break;
        }

        if (grammar) {
            for (int i = 0; i < n_seq_dft; ++i) {
                auto * grammar_dft = drafts[i].grammar;
                if (grammar_dft) {
                    llama_grammar_free(grammar_dft);
                }

                grammar_dft = llama_grammar_copy(ctx_sampling.grammar);

                LOG("copied target grammar to draft %d grammar\n", i);
            }
        }

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int i = 0; i < n_seq_dft; ++i) {
            drafts[i].active   = false;
            drafts[i].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        batch_tgt.n_tokens     = 1;
        batch_tgt.token[0]     = drafts[0].tokens[0];
        batch_tgt.pos[0]       = n_past_tgt;
        batch_tgt.n_seq_id[0]  = 1;
        batch_tgt.seq_id[0][0] = 0;
        batch_tgt.logits[0]    = true;

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) continue;

                auto & grammar     = drafts[s].grammar;
                auto & i_batch_dft = drafts[s].i_batch_dft;

                float * logits = llama_get_logits_ith(ctx_dft, i_batch_dft);

                // TODO: optimize
                candidates.clear();
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array cur_p = { candidates.data(), candidates.size(), false };

                if (grammar != NULL) {
                    llama_sample_grammar(ctx_dft, &cur_p, grammar);
                }

                // computes softmax and sorts the candidates
                llama_sample_softmax(ctx_dft, &cur_p);

                for (int k = 0; k < 3; ++k) {
                    LOG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p.data[k].id, cur_p.data[k].p, llama_token_to_piece(ctx_dft, cur_p.data[k].id).c_str());
                }

                // TODO: make this configurable
                if (cur_p.data[0].p < 0.1) {
                //if (cur_p.data[0].p < 2*cur_p.data[1].p) {
                    LOG("stopping drafting for seq %3d, probability too low: %.3f < 2*%.3f\n", s, cur_p.data[0].p, cur_p.data[1].p);
                    drafts[s].drafting = false;
                    continue;
                }

                std::vector<int> sa(1, s);

                for (int f = 1; f < 8; ++f) {
                    // TODO: make this configurable
                    if (n_seq_cur < n_seq_dft && cur_p.data[f].p > 0.10) {
                        LOG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        llama_kv_cache_seq_rm(ctx_dft,    n_seq_cur, -1, -1);
                        llama_kv_cache_seq_cp(ctx_dft, s, n_seq_cur, -1, -1);

                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        drafts[n_seq_cur] = drafts[s];
                        drafts[n_seq_cur].skip = true;
                        // TODO: grammar

                        sa.push_back(n_seq_cur);
                        n_seq_cur++;
                    } else {
                        break;
                    }
                }

                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p.data[is].id;

                    int s = sa[is];

                    auto & drafted = drafts[s].tokens;
                    //auto & grammar = drafts[s].grammar;

                    auto & i_batch_dft = drafts[s].i_batch_dft;
                    auto & i_batch_tgt = drafts[s].i_batch_tgt;

                    drafted.push_back(id);

                    // add unique drafted tokens to the target batch
                    batch_tgt.token   [batch_tgt.n_tokens]    = id;
                    batch_tgt.pos     [batch_tgt.n_tokens]    = n_past_tgt + i + 1;
                    batch_tgt.n_seq_id[batch_tgt.n_tokens]    = 1;
                    batch_tgt.seq_id  [batch_tgt.n_tokens][0] = s;
                    batch_tgt.logits  [batch_tgt.n_tokens]    = true;

                    i_batch_tgt.push_back(batch_tgt.n_tokens);

                    batch_tgt.n_tokens++;

                    // no need to evaluate the last drafted token, since we won't use the result
                    if (i == n_draft - 1) {
                        drafts[s].drafting = false;
                        continue;
                    }

                    // add the token to the batch for batched decoding with the draft model
                    batch_dft.token   [batch_dft.n_tokens]    = id;
                    batch_dft.pos     [batch_dft.n_tokens]    = n_past_cur;
                    batch_dft.n_seq_id[batch_dft.n_tokens]    = 1;
                    batch_dft.seq_id  [batch_dft.n_tokens][0] = s;
                    batch_dft.logits  [batch_dft.n_tokens]    = true;

                    i_batch_dft = batch_dft.n_tokens;

                    batch_dft.n_tokens++;
                }
            }

            // no sequence is drafting anymore
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch_dft);
            ++n_past_cur;
            ++n_drafted;

            // update grammar
            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting) continue;

                auto & drafted = drafts[s].tokens;
                auto & grammar = drafts[s].grammar;

                if (grammar != NULL) {
                    llama_grammar_accept_token(ctx_dft, grammar, drafted.back());
                }
            }

            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if (batch_tgt.n_tokens >= n_draft) {
                break;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            llama_kv_cache_seq_keep(ctx_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_kv_cache_seq_cp(ctx_tgt, 0, s, -1, -1);
            }

            //LOG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt));
            llama_decode(ctx_tgt, batch_tgt);
            ++n_past_tgt;
        }

        // the first token is always proposed by the traget model before the speculation loop so we erase it here
        for (int i = 0; i < n_seq_dft; ++i) {
            if (!drafts[i].active) continue;

            drafts[i].tokens.erase(drafts[i].tokens.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG_TEE("\n\n");

    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));

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

    llama_batch_free(batch_dft);

    llama_free(ctx_tgt);
    llama_free_model(model_tgt);

    llama_free(ctx_dft);
    llama_free_model(model_dft);

    if (grammar) {
        llama_grammar_free(grammar);

        for (int i = 0; i < n_seq_dft; ++i) {
            llama_grammar_free(drafts[i].grammar);
        }
    }
    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
