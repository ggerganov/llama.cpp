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

    int split_pos = 0;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<float>       tokens_p;

    struct llama_sampling_context * ctx_sampling;
};

static void save_logits(llama_context * ctx, std::vector<float> & v, const int n_vocab, const int count = 1, const int soffs = 0, const int doffs = 0) {
    // printf("SAVE %p: %d, %d, %d\n", (void *)ctx, count, soffs, doffs);
    // printf("<S>");
    GGML_ASSERT(doffs + count < 64);
    memcpy(
        v.data() + doffs * n_vocab,
        llama_get_logits(ctx) + soffs * n_vocab,
        sizeof(float) * size_t(n_vocab) * count);
}

static void restore_logits(llama_context * ctx, std::vector<float> & v, const int n_vocab, const int count = 1, const int soffs = 0, const int doffs = 0) {
    // printf("<R>");
    // printf("REST %p: %d, %d, %d\n", (void *)ctx, count, soffs, doffs);
    GGML_ASSERT(soffs + count < 64);
    memcpy(
        llama_get_logits(ctx) + doffs * n_vocab,
        v.data() + soffs * n_vocab,
        sizeof(float) * size_t(n_vocab) * count);
}

static llama_token_data_array normalize_candidates(const float * logits, const int n_vocab, std::vector<llama_token_data> & cur) {
    cur.reserve(n_vocab);
    cur.clear();

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), false };
    llama_sample_top_k(NULL, &cur_p, 100, 1);
    llama_sample_softmax(NULL, &cur_p);
    cur.resize(cur_p.size);
    return cur_p;
}

static int32_t find_normalized(const llama_token_data_array & tda, const llama_token id) {
    llama_token_data *item = tda.data;

    for (int32_t i = 0; i < tda.size; i++, item++)
        if (item->id == id) return i;
    return -1;
}

static double running_average(double & cur, double val, double n = 20) {
    if (cur < 1e-5f) {
        cur = val;
        return cur;
    }
    // New average = old average * (n-1)/n + new value /n
    cur = cur * (n - 1) / n + val / n;
    return cur;
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
    const float p_accept = 0.75f; // 0.80f;
    const float p_split  = 0.6f; // p_accept / 8; // 0.10f;

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
    logits_tgt.resize(n_vocab * 64);
    logits_dft.resize(n_vocab * 64);
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
    save_logits(ctx_dft, logits_dft, n_vocab, n_input);

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
    int n_dup_split = 0;
    int n_eff_split = 0;
    int max_streak = 0;

    int64_t t_dft_sample = 0, t_dft_gen = 0, t_dft_accept = 0, t_tgt_predict = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);
    struct llama_sampling_context * ctx_dft_sampling = llama_sampling_init(params.sparams);
    std::vector<llama_token_data> normalized_candidates;
    normalized_candidates.reserve(n_vocab);
    llama_token_data_array normalized_p;

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    params.sparams.grammar.clear(); // the draft samplers will copy the target sampler's grammar
    // params.sparams.temp = std::max(0.01f, params.sparams.temp);

    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].ctx_sampling = llama_sampling_init(params.sparams);
    }

    // 70B (80 layers) skips example
    std::vector<int32_t> run_layers_dft = {
        0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 3, 1, 0, 3, 3, 0, 3, 0, 1, 1,
        3, 3, 3, 0, 2, 3, 2, 3, 3, 3, 1, 3, 0, 0, 2, 1, 0, 2, 0, 0,
        0, 3, 0, 1, 0, 1, 1, 3, 3, 3, 3, 1, 1, 3, 3, 3, 1, 3, 3, 0,
        3, 1, 3, 3, 0, 1, 3, 3, 3, 1, 3, 0, 0, 0, 1, 1, 2, 0, 1, 1, -1, };

    // 3B (26 layers) skips example
    // std::vector<int32_t> run_layers_dft = {
    //        0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 1, 3, 0, 2, 3, 3, 1, 0, 2, 0, 1, 1, 2, 0, 0,
    //     // 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 3, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 2, 0, 1,
    //     -1, };

    // NOTE: Comment this line out to disable skipping.
    batch_dft.run_layers = run_layers_dft.data();

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    double avg_accepted = 0, avg_rejected = 0, tgt_avg_accepted = 0;
    double avg_accept_delta = 0;
    float min_accepted = 0, max_rejected = 0, tgt_min_accepted = 0;

    int64_t t_cur;

    std::vector<std::vector<std::vector<llama_token_data>>> doubt;

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

        float tgt_last_norm = 0, tgt_last_best_norm = 0, tgt_last_orig = 0;

        while (true) {
            LOG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);

            // sample from the target model
            restore_logits(ctx_tgt, logits_tgt, n_vocab, 1, drafts[s_keep].i_batch_tgt[i_dft], drafts[s_keep].i_batch_tgt[i_dft]);
            normalized_p = normalize_candidates(llama_get_logits_ith(ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]), n_vocab, normalized_candidates);
            llama_token id = llama_sampling_sample(ctx_sampling, ctx_tgt, NULL, drafts[s_keep].i_batch_tgt[i_dft]);
            save_logits(ctx_tgt, logits_tgt, n_vocab, 1, drafts[s_keep].i_batch_tgt[i_dft], drafts[s_keep].i_batch_tgt[i_dft]);
            int32_t norm_pos = find_normalized(normalized_p, id);
            int32_t orig_pos = find_normalized({ctx_sampling->cur.data(), ctx_sampling->cur.size(), false}, id);
            if (norm_pos >= 0) {
                tgt_last_norm = normalized_candidates[norm_pos].p;
                tgt_last_best_norm = normalized_candidates[0].p;
                running_average(tgt_avg_accepted, tgt_last_norm);
                tgt_min_accepted = tgt_min_accepted < 1e-4
                        ? tgt_last_norm
                        : std::min(tgt_min_accepted, tgt_last_norm);
            } else {
                tgt_last_norm = tgt_last_best_norm = tgt_avg_accepted;
            }
            if (orig_pos >= 0) {
                tgt_last_orig = ctx_sampling->cur[orig_pos].p;
            }
            LOG("target sampled (%d, '%s') orig_p=%5.4f, norm_p=%5.4f\n",
                    id, llama_token_to_piece(ctx_tgt, id).c_str(),
                    orig_pos >= 0 ? ctx_sampling->cur[orig_pos].p : -1,
                    norm_pos >= 0 ? normalized_candidates[norm_pos].p : -1);

            llama_sampling_accept(ctx_sampling, ctx_tgt, id, true);

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
                    if (!drafts[s].active || i_dft < drafts[s].split_pos) {
                        continue;
                    }

                    if (i_dft < (int) drafts[s].tokens.size() && id == drafts[s].tokens[i_dft]) {
                        LOG("the sampled target token matches drafted token %d of sequence %d (%d, '%s') - accepted\n",
                                i_dft, s, id, token_str.c_str());

                        if (i_dft == 0 && s > 0) {
                            if (matches) n_dup_split++;
                            else n_eff_split++;
                        }
                        s_keep = s;
                        matches = true;
                        LOG("Derp[%d]: %6d (%5.4f)\n", s, drafts[s].tokens[i_dft], drafts[s].tokens_p[i_dft]);
                        if (min_accepted == 0) min_accepted = drafts[s].tokens_p[i_dft];
                        else min_accepted = std::min(min_accepted, drafts[s].tokens_p[i_dft]);
                        running_average(avg_accepted, drafts[s].tokens_p[i_dft]);
                        running_average(avg_accept_delta, tgt_last_norm - drafts[s].tokens_p[i_dft]);
                    } else {
                        if (i_dft < (int) drafts[s].tokens.size() && id != drafts[s].tokens[i_dft]) {
                            if (i_dft == 0 && s > 0) n_bad_split++;
                            max_rejected = std::max(max_rejected, drafts[s].tokens_p[i_dft]);
                            running_average(avg_rejected, drafts[s].tokens_p[i_dft]);
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
                    max_streak = std::max(max_streak, i_dft);
                    continue;
                } else {
                    for (size_t seqnum = 0; seqnum < doubt.size(); seqnum++) {
                        const std::vector<std::vector<llama_token_data>> & sdoubt = doubt[seqnum];
                        if (sdoubt.size() <= i_dft) continue;
                        const std::vector<llama_token_data> & sidoubt = sdoubt[i_dft];
                        for (size_t cidx = 0; cidx < sidoubt.size(); cidx++) {
                            if (sidoubt[cidx].id == id) {
                                LOG("Shoulda picked seq %3zu, pos %4d, candidate %2zu @ p %5.4f: %6d '%s'\n",
                                        seqnum, i_dft, cidx, sidoubt[cidx].p,
                                        id, token_str.c_str());
                                running_average(avg_accepted, sidoubt[cidx].p);
                                if (cidx < 2) {
                                    running_average(avg_accept_delta, tgt_last_norm - sidoubt[cidx].p);
                                    min_accepted = min_accepted < 1e-5f ? sidoubt[cidx].p : std::min(min_accepted, sidoubt[cidx].p);
                                }
                                break;
                            }
                        }
                    }
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
                drafts[s].split_pos = 0;
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
            if (self_speculation) {
                // Copy KV items from non-brain-damaged model... Doesn't seem to help.
                llama_kv_cache_seq_rm(ctx_dft, 0 + DOFFS, 0, n_past_dft - 2);
                llama_kv_cache_seq_cp(ctx_dft, 0, 0 + DOFFS, 0, n_past_dft - 2);
                // llama_kv_cache_seq_rm(ctx_dft, 0 + DOFFS, n_past_dft - 1, -1);
                // llama_kv_cache_seq_cp(ctx_dft, 0, 0 + DOFFS, n_past_dft - 1, -1);
            }

            LOG("=== EVAL: DRAFT ACCEPTED ===: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_dft).c_str());
            t_cur = ggml_time_us();
            llama_decode         (ctx_dft, batch_dft);
            t_dft_accept += ggml_time_us() - t_cur;
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

        avg_rejected = std::max(0.05, std::min(avg_accepted - 0.05, avg_rejected));
        avg_accepted = std::max(0.05, std::max(avg_rejected + 0.05, avg_accepted));
        // double avg_accepted = n_accept > 0 ? avg_accepted / double(n_accept) : 0;
        LOG("STATS: Avg tacc/dacc/drej: %3.5f / %3.5f / %3.5f | Min dacc/min tacc/max drej: %3.5f / %3.5f / %3.5f | delta %3.5f | max streak %d | n_dft/pred/acc: %d / %d / %d\n",
                tgt_avg_accepted, avg_accepted, avg_rejected, min_accepted, tgt_min_accepted, max_rejected, avg_accept_delta, max_streak,
                n_drafted, n_predict, n_accept);
        doubt.clear();
        doubt.resize(n_seq_dft);

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                double accept_threshold, split_threshold;

                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }
                doubt[s].push_back({});

                if (avg_rejected == 0 || avg_rejected == 0 || n_drafted + n_predict < 6) {
                    accept_threshold = std::max(0.6f, tgt_last_norm);
                } else {

                    accept_threshold = (tgt_avg_accepted - avg_accept_delta) * 0.3;
                    accept_threshold *= std::min(0.8, std::max(0.1, double(tgt_last_norm * 1.0)));
                    accept_threshold = std::max(double(min_accepted) * 1.1, accept_threshold);
                    accept_threshold = std::max(std::max(avg_accepted * 0.9, avg_rejected * 1.1), accept_threshold);
                    accept_threshold += 1.0 - (1.2 * n_accept / n_drafted);
                    accept_threshold *= (1.3 - (std::min(n_seq_cur + i, 6) * 0.1));
                    //
                    // accept_threshold = (tgt_avg_accepted - avg_accept_delta) * 0.3;
                    // accept_threshold *= std::min(0.8, std::max(0.1, double(tgt_last_norm * 1.0)));
                    // accept_threshold = std::max(double(min_accepted) * 1.1, accept_threshold);
                    // accept_threshold = std::max(std::max(avg_accepted * 0.9, avg_rejected * 1.1), accept_threshold);
                    // accept_threshold += 1.0 - (1.2 * n_accept / n_drafted);
                    // accept_threshold *= (0.7 + (std::min(n_seq_cur + i, 5) * 0.1));

                }

                std::vector<llama_token_data> cur_p;
                {
                    llama_token d_id;
                    std::vector<llama_token> already_picked;
                    float * logits = NULL;

                    t_cur = ggml_time_us();
                    for (int cidx = 0; cidx < 9; cidx++) {
                        llama_sampling_cp(drafts[s].ctx_sampling, ctx_dft_sampling);
                        restore_logits(ctx_dft, logits_dft, n_vocab, 1, drafts[s].i_batch_dft);
                        logits = llama_get_logits(ctx_dft);
                        normalized_p = normalize_candidates(logits, n_vocab, normalized_candidates);
                        for (size_t x = 0; x < std::min(normalized_p.size, size_t(10)); x++)
                            doubt[s].back().push_back(normalized_p.data[x]);
                        for (const auto & tid : already_picked)
                            logits[tid] = std::numeric_limits<float>::infinity() * -1;
                        d_id = llama_sampling_sample(ctx_dft_sampling, ctx_dft, NULL);
                        already_picked.push_back(d_id);
                        int32_t norm_pos = find_normalized(normalized_p, d_id);
                        if (norm_pos < 0) continue;
                        llama_token_data norm = normalized_candidates[norm_pos];
                        if (norm.p < 0.2) continue;
                        if (ctx_dft_sampling->params.temp <= 0) {
                            llama_token_data_array tda = { ctx_dft_sampling->cur.data(), ctx_dft_sampling->cur.size(), false };
                            llama_sample_top_k(ctx_dft, &tda, 100, 1);
                            llama_sample_softmax(ctx_dft, &tda);
                            ctx_dft_sampling->cur.resize(tda.size);
                        }


                        llama_token_data found;
                        found.id = -1;
                        for (const llama_token_data & td : ctx_dft_sampling->cur) {
                            if (td.id == d_id) {
                                found = td;
                                break;
                            }
                        }
                        GGML_ASSERT(found.id != -1);
                        LOG(" ** draft candidate %3d for seq %3d, pos %3d: %6d (%4.3f, norm %4.3f) '%s'\n",
                            cidx, s, i, found.id, found.p, norm_pos >= 0 ? normalized_candidates[norm_pos].p : -1,
                            llama_token_to_piece(ctx_dft, found.id).c_str());
                        if (found.p < 0.3) continue;
                        if (norm.p < 1e-2f) break;
                        cur_p.push_back(normalized_candidates[norm_pos]);
                    }

                    if (cur_p.size() > 1) {
                        std::sort(cur_p.begin() + 1, cur_p.end(),
                            [](const llama_token_data & a, const llama_token_data & b) {
                                return a.p > b.p;
                            }
                        );
                    }

                }

                t_dft_sample += ggml_time_us() - t_cur;

                if (cur_p.empty()) {
                    LOG("stopping drafting for seq %3d, no viable candidates (%5.3f) \n", s, accept_threshold);
                    drafts[s].drafting = false;
                    continue;
                } else if (cur_p[0].p < accept_threshold && (cur_p[0].p + (cur_p.size() < 2 ? 0 : cur_p[1].p)) < accept_threshold * 1.3) {
                    LOG("stopping drafting for seq %3d, pos %3d - probability too low: %.3f < %.3f\n", s, i, cur_p[0].p, accept_threshold);
                    drafts[s].drafting = false;
                    continue;
                }

                if (cur_p[0].p < accept_threshold) {
                    split_threshold = 0.0;
                } else {
                    split_threshold = cur_p[0].p / 10.0;
                    // split_threshold = std::max(0.01, cur_p[0].p * (n_seq_cur + i > 1 ? 0.15 : 0.2));
                }

                std::vector<int> sa(1, s);



                // LOG("Check splits: %zu\n", cur_p.size());
                // attempt to split the branch if the probability is high enough
                for (int f = 1; f < std::min(8, int(cur_p.size()) - 1); ++f) {
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
                        drafts[n_seq_cur].split_pos = i;
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
                        LOG("Not splitting seq %3d into %3d, choice %2d @ %6d (%8.3f) '%s'\n", s, n_seq_cur, f,
                                cur_p[f].id, cur_p[f].p, llama_token_to_piece(ctx_dft, cur_p[f].id).c_str());
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
            t_cur = ggml_time_us();
            llama_decode(ctx_dft, batch_dft);
            t_dft_gen += ggml_time_us() - t_cur;
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
            t_cur = ggml_time_us();
            llama_decode(ctx_tgt, batch_tgt);
            t_tgt_predict += ggml_time_us() - t_cur;
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
    LOG_TEE("times: target predict: %5.3f, draft gen/accept/sample: %5.3f / %5.3f / %5.3f\n",
            t_tgt_predict / 1e6f, t_dft_gen / 1e6f, t_dft_accept / 1e6f, t_dft_sample / 1e6f);
// int64_t t_dft_sample = 0, t_dft_gen = 0, t_dft_accept = 0, t_tgt_predict = 0;
    LOG_TEE("\n");
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_predict = %d\n", n_predict);
    LOG_TEE("drafted   = %.3f%%\n", 100.0f * n_drafted / n_predict);
    LOG_TEE("n_drafted = %d\n", n_drafted);
    LOG_TEE("n_accept  = %d\n", n_accept);
    LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);
    LOG_TEE("n_draft   = %d\n", n_draft);
    LOG_TEE("n_split   = %d\n", n_split);
    LOG_TEE("n_effsplit= %d\n", n_eff_split);
    LOG_TEE("n_badsplit= %d\n", n_bad_split);
    LOG_TEE("n_dupsplit= %d\n", n_dup_split);
    LOG_TEE("max streak= %d\n", max_streak);

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
