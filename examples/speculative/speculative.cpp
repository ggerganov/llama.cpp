#include "build-info.h"

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define DOFFS 10000

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<float>       tokens_p;

    struct llama_sampling_context * ctx_sampling;
};

static void save_logits(llama_context * ctx, std::vector<float> & v, const int n_vocab, const int count = 1, const int soffs = 0, const int doffs = 0) {
    // printf("SAVE %p: %d, %d, %d\n", (void *)ctx, count, soffs, doffs);
    // printf("<S>");
    GGML_ASSERT(doffs + count <= 30);
    memcpy(
        v.data() + doffs * n_vocab,
        llama_get_logits(ctx) + soffs * n_vocab,
        sizeof(float) * size_t(n_vocab) * count);
}

static void restore_logits(llama_context * ctx, std::vector<float> & v, const int n_vocab, const int count = 1, const int soffs = 0, const int doffs = 0) {
    // printf("<R>");
    // printf("REST %p: %d, %d, %d\n", (void *)ctx, count, soffs, doffs);
    GGML_ASSERT(soffs + count <= 30);
    memcpy(
        llama_get_logits(ctx) + doffs * n_vocab,
        v.data() + soffs * n_vocab,
        sizeof(float) * size_t(n_vocab) * count);
}

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
    const int n_seq_dft = params.n_parallel;

    // TODO: make this configurable
    // const float p_accept = 0.80f;
    // const float p_split  = 0.10f;
    const float p_accept = 0.5f; // 0.80f;
    const float p_split  = p_accept / 8; // 0.10f;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("speculative", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    bool self_speculation   = false;

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
    if (params.model != params.model_draft) {
        params.model = params.model_draft;
        params.n_gpu_layers = params.n_gpu_layers_draft;
        std::tie(model_dft, ctx_dft) = llama_init_from_gpt_params(params);
    } else {
        self_speculation = true;
        model_dft = model_tgt;
        ctx_dft = ctx_tgt;
    }

    const int n_ctx   = llama_n_ctx(ctx_tgt);
    const int n_vocab = llama_n_vocab(model_tgt);

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

    llama_batch batch_dft = llama_batch_init(params.n_ctx, 0, 1);
    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, n_seq_dft);
    std::vector<float> logits_tgt, logits_dft;

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    llama_batch_clear(batch_tgt);
    logits_tgt.resize(n_vocab * 30);
    logits_dft.resize(n_vocab * 30);
    for (int i = 0; i < n_input - 1; i++) {
        llama_batch_add(batch_tgt, inp[i], i, { 0 }, false);
    }
    llama_decode(ctx_tgt, batch_tgt);
    llama_batch_clear(batch_tgt);
    llama_batch_add(batch_tgt, inp.back(), n_input - 1, { 0 }, true);
    llama_decode(ctx_tgt, batch_tgt);
    save_logits(ctx_tgt, logits_tgt, n_vocab);
    if (!self_speculation) {
        llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input,     0,           0 + DOFFS));
    } else {
        // llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input,     0,           0 + DOFFS));
        llama_kv_cache_seq_cp(ctx_dft, 0, 0 + DOFFS, 0, -1);
    }
    // save_logits(ctx_dft, logits_dft, n_vocab, n_input);

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_n_vocab(model_dft));

    // how many tokens to draft each time
    int n_draft = params.n_draft;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    int n_split     = 0;
    int n_bad_split = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    params.sparams.grammar.clear(); // the draft samplers will copy the target sampler's grammar
    params.sparams.temp = std::max(0.01f, params.sparams.temp);

    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].ctx_sampling = llama_sampling_init(params.sparams);
    }

    // std::vector<int32_t> run_layers_dft = {
    //     0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 3, 1, 0, 3, 3, 0, 3, 0, 1, 1,
    //     3, 3, 3, 0, 2, 3, 2, 3, 3, 3, 1, 3, 0, 0, 2, 1, 0, 2, 0, 0,
    //     0, 3, 0, 1, 0, 1, 1, 3, 3, 3, 3, 1, 1, 3, 3, 3, 1, 3, 3, 0,
    //     3, 1, 3, 3, 0, 1, 3, 3, 3, 1, 3, 0, 0, 0, 1, 1, 2, 0, 1, 1, -1, };
    std::vector<int32_t> run_layers_dft = {
        0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 1, 0, 1, 1, 0, 2, 0, 1, 1,
        1, 0, 1, 0, 0, 0, -1, };

    batch_dft.run_layers = run_layers_dft.data();

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    double avg_accepted = 0, avg_rejected = 0;
    float min_accepted = 0, max_rejected = 0;

    while (true) {
        LOG("*** Draft start\n");
        // print current draft sequences
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            const auto & tokens = drafts[s].tokens;

            LOG("draft %d: %s\n", s, LOG_TOKENS_TOSTR_PRETTY(ctx_dft, tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        while (true) {
            LOG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);

            // sample from the target model
            restore_logits(ctx_tgt, logits_tgt, n_vocab, 1, drafts[s_keep].i_batch_tgt[i_dft], drafts[s_keep].i_batch_tgt[i_dft]);
            llama_token id = llama_sampling_sample(ctx_sampling, ctx_tgt, NULL, drafts[s_keep].i_batch_tgt[i_dft]);

            llama_sampling_accept(ctx_sampling, ctx_tgt, id, true);
            save_logits(ctx_tgt, logits_tgt, n_vocab, 1, drafts[s_keep].i_batch_tgt[i_dft], drafts[s_keep].i_batch_tgt[i_dft]);

            //LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, ctx_sampling->prev).c_str());

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

                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].active) {
                        continue;
                    }

                    if (i_dft < (int) drafts[s].tokens.size() && id == drafts[s].tokens[i_dft]) {
                        LOG("the sampled target token matches drafted token %d of sequence %d (%d, '%s') - accepted\n", i_dft, s, id, token_str.c_str());

                        s_keep = s;
                        matches = true;
                        LOG("Derp[%d]: %6d (%5.4f)\n", s, drafts[s].tokens[i_dft], drafts[s].tokens_p[i_dft]);
                        if (min_accepted == 0) min_accepted = drafts[s].tokens_p[i_dft];
                        else min_accepted = std::min(min_accepted, drafts[s].tokens_p[i_dft]);
                        avg_accepted += drafts[s].tokens_p[i_dft] * (avg_accepted == 0 ? 2 : 1);
                        avg_accepted /= 2;
                    } else {
                        if (i_dft < (int) drafts[s].tokens.size() && id != drafts[s].tokens[i_dft]) {
                            if (i_dft == 0 && s > 0) n_bad_split++;
                            max_rejected = std::max(max_rejected, drafts[s].tokens_p[i_dft]);
                            avg_rejected += drafts[s].tokens_p[i_dft] * (avg_rejected == 0 ? 2 : 1);
                            avg_rejected /= 2;
                            LOG("-- Terminate sequence %d+%d: (%d, '%s') != target (%d, '%s') - rejected\n",
                                    s, i_dft, drafts[s].tokens[i_dft],
                                    llama_token_to_piece(ctx_dft, drafts[s].tokens[i_dft]).c_str(),
                                    id, token_str.c_str());
                        }
                        drafts[s].active = false;
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
                LOG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                llama_kv_cache_seq_rm(ctx_dft, s_keep + DOFFS, n_past_dft, -1);
                llama_kv_cache_seq_rm(ctx_tgt, s_keep,         n_past_tgt, -1);
                if (s_keep != 0) {
                    llama_kv_cache_seq_cp(ctx_dft, s_keep + DOFFS, 0 + DOFFS, -1, -1);
                    llama_kv_cache_seq_cp(ctx_tgt, s_keep,         0,         -1, -1);
                }
                for (int s = 1; s < n_seq_dft; ++s) {
                    llama_kv_cache_seq_rm(ctx_dft, s + DOFFS, -1, -1);
                    llama_kv_cache_seq_rm(ctx_tgt, s,         -1, -1);
                }

                /*
                llama_kv_cache_seq_keep(ctx_dft, s_keep);
                llama_kv_cache_seq_cp  (ctx_dft, s_keep, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_dft, 0);

                llama_kv_cache_seq_rm  (ctx_tgt, s_keep, n_past_tgt, -1);
                llama_kv_cache_seq_keep(ctx_tgt, s_keep);
                llama_kv_cache_seq_cp  (ctx_tgt, s_keep, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_tgt, 0);
                */

            }

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].tokens_p.clear();
                drafts[s].i_batch_tgt.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(id);
            drafts[0].tokens_p.push_back(0);
            drafts[0].i_batch_tgt.push_back(0);

            llama_batch_clear(batch_dft);
            llama_batch_add  (batch_dft, id, n_past_dft, { 0 + DOFFS }, true);

            LOG("=== EVAL: DRAFT ACCEPTED ===: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_dft).c_str());

            llama_decode         (ctx_dft, batch_dft);
            save_logits(ctx_dft, logits_dft, n_vocab, batch_dft.n_tokens);

            ++n_past_dft;

            break;
        }

        if (n_predict > params.n_predict || has_eos) {
            break;
        }

        llama_sampling_cp(ctx_sampling, drafts[0].ctx_sampling);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        llama_batch_clear(batch_tgt);
        llama_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        // double avg_accepted = n_accept > 0 ? avg_accepted / double(n_accept) : 0;
        LOG("Average accepted/rejected: %3.5f / %3.5f -- Min accepted/max rejected: %3.5f / %3.5f\n",
                avg_accepted, avg_rejected, min_accepted, max_rejected);

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                restore_logits(ctx_dft, logits_dft, n_vocab, 1, drafts[s].i_batch_dft, drafts[s].i_batch_dft);
                llama_sampling_sample(drafts[s].ctx_sampling, ctx_dft, NULL, drafts[s].i_batch_dft);
                save_logits(ctx_dft, logits_dft, n_vocab, 1, drafts[s].i_batch_dft, drafts[s].i_batch_dft);

                const auto & cur_p = drafts[s].ctx_sampling->cur;

                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p.size()); ++k) {
                    if (cur_p[k].p < 1e-5f) continue;
                    LOG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p[k].id, cur_p[k].p, llama_token_to_piece(ctx_dft, cur_p[k].id).c_str());
                }

                double accept_threshold = avg_rejected == 0 || avg_rejected == 0 || n_drafted < 16
                        ? p_accept
                        : std::max(double(min_accepted * 0.98), avg_accepted * 0.75f);
                // accept_threshold = 0.8;
                if (cur_p[0].p < accept_threshold) {
                    LOG("stopping drafting for seq %3d, probability too low: %.3f < %.3f\n", s, cur_p[0].p, accept_threshold);
                    drafts[s].drafting = false;
                    continue;
                }

                std::vector<int> sa(1, s);

                // attempt to split the branch if the probability is high enough
                for (int f = 1; f < 8; ++f) {
                    // if (n_seq_cur < n_seq_dft && cur_p[f].p > p_split) {
                    // if (n_seq_cur < n_seq_dft && cur_p[f].p > cur_p[0].p / 5) {
                    double split_threshold = avg_accepted == 0 || avg_rejected == 0 || n_drafted < 16
                            ? p_split
                            : ( std::max(double(min_accepted * 0.7), avg_accepted * 0.4)
                                * (n_seq_cur >= 2 ? 0.75 : 1.0) );
                    // split_threshold = 0.1;
                    if (n_seq_cur < n_seq_dft && cur_p[f].p >= split_threshold) {
                        n_split++;
                        LOG(">>>%d<<< splitting seq %3d into %3d on %6d (%8.3f) '%s'\n", f, s, n_seq_cur,
                                cur_p[f].id, cur_p[f].p, llama_token_to_piece(ctx_dft, cur_p[f].id).c_str());

                        llama_kv_cache_seq_rm(ctx_dft,            n_seq_cur + DOFFS, -1, -1);
                        llama_kv_cache_seq_cp(ctx_dft, s + DOFFS, n_seq_cur + DOFFS, -1, -1);

                        // all previous tokens from this branch are now also part of the new branch
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // copy the draft state
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;

                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].tokens_p    = drafts[s].tokens_p;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        llama_sampling_cp(drafts[s].ctx_sampling, drafts[n_seq_cur].ctx_sampling);

                        sa.push_back(n_seq_cur);

                        n_seq_cur++;
                    } else {
                        break;
                    }
                }

                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p[is].id;

                    const int s = sa[is];

                    llama_sampling_accept(drafts[s].ctx_sampling, ctx_dft, id, true);

                    drafts[s].tokens.push_back(id);
                    drafts[s].tokens_p.push_back(cur_p[is].p);

                    // add unique drafted tokens to the target batch
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                    llama_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                    // add the token to the batch for batched decoding with the draft model
                    drafts[s].i_batch_dft = batch_dft.n_tokens;

                    llama_batch_add(batch_dft, id, n_past_cur, { s + DOFFS }, true);

                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            }

            // no sequence is drafting anymore
            if (batch_dft.n_tokens == 0) {
                break;
            }
            // LOG("Draft eval: %d\n", batch_dft.n_tokens);
            // for (int x = 0; x < batch_dft.n_tokens; x++) {
            //     LOG("* %03d: seq %3d, pos %4d, token %6d '%s'", x,
            //         batch_dft.seq_id[x][0], batch_dft.pos[x],
            //         batch_dft.token[x], llama_token_to_piece(ctx_dft, batch_dft.token[x]).c_str());
            // }

            LOG("=== EVAL: DRAFTED ===: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_dft).c_str());

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch_dft);
            save_logits(ctx_dft, logits_dft, n_vocab, batch_dft.n_tokens);
            ++n_past_cur;
            ++n_drafted;

            if (batch_tgt.n_tokens > n_draft) {
                break;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            // llama_kv_cache_seq_keep(ctx_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_kv_cache_seq_rm(ctx_tgt, s,    -1, -1);
            }
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_kv_cache_seq_cp(ctx_tgt, 0, s, -1, -1);
            }

            LOG("=== EVAL: TARGET ===: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            llama_decode(ctx_tgt, batch_tgt);
            save_logits(ctx_tgt, logits_tgt, n_vocab, batch_tgt.n_tokens);
            ++n_past_tgt;
        }

        // the first token is always proposed by the traget model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].tokens_p.erase(drafts[s].tokens_p.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG_TEE("\n\n");

    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_TEE("\n");
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("drafted   = %.3f%%\n", 100.0f * n_drafted / n_predict);
    LOG_TEE("n_drafted = %d\n", n_drafted);
    LOG_TEE("n_accept  = %d\n", n_accept);
    LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_split   = %d\n", n_split);
    LOG_TEE("n_badsplit= %d\n", n_bad_split);

    LOG_TEE("\ndraft:\n");
    llama_print_timings(ctx_dft);

    LOG_TEE("\ntarget:\n");
    llama_print_timings(ctx_tgt);

    llama_sampling_free(ctx_sampling);
    for (int s = 0; s < n_seq_dft; ++s) {
        llama_sampling_free(drafts[s].ctx_sampling);
    }

    llama_batch_free(batch_dft);

    llama_free(ctx_tgt);
    llama_free_model(model_tgt);

    if (!self_speculation) {
        llama_free(ctx_dft);
        llama_free_model(model_dft);
    }

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
