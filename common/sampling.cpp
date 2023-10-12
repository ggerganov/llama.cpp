#include "sampling.h"

llama_sampling_context::~llama_sampling_context() {
    for (auto & it : sequence_contexts) {
        if (it.second.grammar != NULL) {
            llama_grammar_free(it.second.grammar);
            it.second.grammar = NULL;
        }
    }
}

llama_sampling_context llama_sampling_context_init(
        const struct gpt_params & params,
                  llama_grammar * grammar) {
  llama_sampling_context result;

  result.params = params.sampling_params;
  result.grammar = grammar;
  return result;
}

// Note: Creates the context if it doesn't exist, so this always return something.
llama_sampler_sequence_context & llama_sampling_get_sequence_context(
              llama_sampling_context & ctx_sampling,
        const llama_seq_id             seq) {
    const auto it = ctx_sampling.sequence_contexts.find(seq);
    if (it != ctx_sampling.sequence_contexts.end()) {
        return it->second;
    }
    llama_sampler_sequence_context new_ctx = {
        2.0f * ctx_sampling.params.mirostat_tau,
        ctx_sampling.grammar != NULL ? llama_grammar_copy(ctx_sampling.grammar) : NULL,
    };
    return ctx_sampling.sequence_contexts.insert({seq, new_ctx}).first->second;
}

bool llama_sampling_context_reset(
              llama_sampling_context & ctx_sampling,
        const llama_seq_id             seq) {
    const auto it = ctx_sampling.sequence_contexts.find(seq);
    if (it == ctx_sampling.sequence_contexts.end()) return false;
    if (it->second.grammar != NULL) {
        llama_grammar_free(it->second.grammar);
        it->second.grammar = NULL;
    }
    ctx_sampling.sequence_contexts.erase(it);
    return true;
}

llama_token llama_sampling_sample(
                  struct llama_context * ctx,
                  struct llama_context * ctx_guidance,
                  struct llama_sampling_context & ctx_sampling,
        const std::vector<llama_token> & last_tokens,
         std::vector<llama_token_data> & candidates,
        const                      int   idx,
                          llama_seq_id   seq) {
    const int n_ctx   = llama_n_ctx(ctx);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));

    const llama_sampling_params & params = ctx_sampling.params;
    const float   temp            = params.temp;
    const int32_t top_k           = params.top_k <= 0 ? n_vocab : params.top_k;
    const float   top_p           = params.top_p;
    const float   tfs_z           = params.tfs_z;
    const float   typical_p       = params.typical_p;
    const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
    const float   repeat_penalty  = params.repeat_penalty;
    const float   alpha_presence  = params.presence_penalty;
    const float   alpha_frequency = params.frequency_penalty;
    const int     mirostat        = params.mirostat;
    const float   mirostat_tau    = params.mirostat_tau;
    const float   mirostat_eta    = params.mirostat_eta;
    const bool    penalize_nl     = params.penalize_nl;

    llama_token id = 0;

    float * logits = llama_get_logits_ith(ctx, idx);

    // Apply params.logit_bias map
    for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
        logits[it->first] += it->second;
    }

    candidates.clear();
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array cur_p = { candidates.data(), candidates.size(), false };

    if (ctx_guidance) {
        llama_sample_classifier_free_guidance(ctx, &cur_p, ctx_guidance, params.cfg_scale);
    }

    // apply penalties
    if (!last_tokens.empty()) {
        const float nl_logit = logits[llama_token_nl(ctx)];
        const int last_n_repeat = std::min(std::min((int)last_tokens.size(), repeat_last_n), n_ctx);

        llama_sample_repetition_penalty(ctx, &cur_p,
                last_tokens.data() + last_tokens.size() - last_n_repeat,
                last_n_repeat, repeat_penalty);
        llama_sample_frequency_and_presence_penalties(ctx, &cur_p,
                last_tokens.data() + last_tokens.size() - last_n_repeat,
                last_n_repeat, alpha_frequency, alpha_presence);

        if (!penalize_nl) {
            for (size_t idx = 0; idx < cur_p.size; idx++) {
                if (cur_p.data[idx].id == llama_token_nl(ctx)) {
                    cur_p.data[idx].logit = nl_logit;
                    break;
                }
            }
        }
    }

    llama_sampler_sequence_context & ctx_seq = llama_sampling_get_sequence_context(ctx_sampling, seq);

    if (ctx_seq.grammar != NULL) {
        llama_sample_grammar(ctx, &cur_p, ctx_seq.grammar);
    }

    if (temp <= 0) {
        // Greedy sampling
        id = llama_sample_token_greedy(ctx, &cur_p);
    } else {
        if (mirostat == 1) {
            const int mirostat_m = 100;
            llama_sample_temp(ctx, &cur_p, temp);
            id = llama_sample_token_mirostat(ctx, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &ctx_seq.mirostat_mu);
        } else if (mirostat == 2) {
            llama_sample_temp(ctx, &cur_p, temp);
            id = llama_sample_token_mirostat_v2(ctx, &cur_p, mirostat_tau, mirostat_eta, &ctx_seq.mirostat_mu);
        } else {
            // Temperature sampling
            size_t min_keep = std::max(1, params.n_probs);
            llama_sample_top_k      (ctx, &cur_p, top_k, min_keep);
            llama_sample_tail_free  (ctx, &cur_p, tfs_z, min_keep);
            llama_sample_typical    (ctx, &cur_p, typical_p, min_keep);
            llama_sample_top_p      (ctx, &cur_p, top_p, min_keep);
            llama_sample_temp(ctx, &cur_p, temp);

            {
                const int n_top = 10;
                LOG("top %d candidates:\n", n_top);

                for (int i = 0; i < n_top; i++) {
                    const llama_token id = cur_p.data[i].id;
                    (void)id; // To avoid a warning that id is unused when logging is disabled.
                    LOG(" - %5d: '%12s' (%.3f)\n", id, llama_token_to_piece(ctx, id).c_str(), cur_p.data[i].p);
                }
            }

            id = llama_sample_token(ctx, &cur_p);

            LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(ctx, id).c_str());
        }
    }

    if (ctx_seq.grammar != NULL) {
        llama_grammar_accept_token(ctx, ctx_seq.grammar, id);
    }

    return id;
}
