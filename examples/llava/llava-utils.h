#pragma once

// this one and clip lib will be eventually merged to a single lib, let's keep it this way for now

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

inline bool eval_image_embd(llama_context * ctx_llama, float * embd, int N, int n_batch, int * n_past) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < N; i += n_batch) {
        int n_eval = N - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {int32_t(n_eval), nullptr, (embd+i*n_embd), nullptr, nullptr, nullptr, *n_past, 1, 0, };
        if (llama_decode(ctx_llama, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

inline bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

inline bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

inline bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

// TODO: use common/sampling.h
inline llama_token sample_id(llama_context * ctx_llama, gpt_params & params) {
      // out of user input, sample next token
    const float   temp      = params.sampling_params.temp;
    const int32_t top_k     = params.sampling_params.top_k <= 0 ? llama_n_vocab(llama_get_model(ctx_llama)) : params.sampling_params.top_k;
    const float   top_p     = params.sampling_params.top_p;
    const float   tfs_z     = params.sampling_params.tfs_z;
    const float   typical_p = params.sampling_params.typical_p;
      // const int32_t repeat_last_n   = params.sampling_params.repeat_last_n < 0 ? n_ctx : params.sampling_params.repeat_last_n;
      // const float   repeat_penalty  = params.sampling_params.repeat_penalty;
      // const float   alpha_presence  = params.sampling_params.presence_penalty;
      // const float   alpha_frequency = params.sampling_params.frequency_penalty;
    const int     mirostat     = params.sampling_params.mirostat;
    const float   mirostat_tau = params.sampling_params.mirostat_tau;
    const float   mirostat_eta = params.sampling_params.mirostat_eta;
      // const bool    penalize_nl     = params.sampling_params.penalize_nl;

    llama_token id = 0;
    {
        auto logits  = llama_get_logits(ctx_llama);
        auto n_vocab = llama_n_vocab(llama_get_model(ctx_llama));

          // Apply params.logit_bias map
        for (auto it = params.sampling_params.logit_bias.begin(); it != params.sampling_params.logit_bias.end(); it++) {
            logits[it->first] += it->second;
        }

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

          // TODO: Apply penalties
          // float nl_logit = logits[llama_token_nl(ctx)];
          // auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
          // llama_sample_repetition_penalty(ctx, &candidates_p,
          //      last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
          //      last_n_repeat, repeat_penalty);
          // llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
          // last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
          // last_n_repeat, alpha_frequency, alpha_presence);
          // if (!penalize_nl) {
          //     logits[llama_token_nl(ctx)] = nl_logit;
          // }

        if (temp <= 0) {
              // Greedy sampling
            id = llama_sample_token_greedy(ctx_llama, &candidates_p);
        } else {
            if (mirostat == 1) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                const  int mirostat_m    = 100;
                llama_sample_temp(ctx_llama, &candidates_p, temp);
                id = llama_sample_token_mirostat(ctx_llama, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
            } else if (mirostat == 2) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                llama_sample_temp(ctx_llama, &candidates_p, temp);
                id = llama_sample_token_mirostat_v2(ctx_llama, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            } else {
                  // Temperature sampling
                llama_sample_top_k(ctx_llama, &candidates_p, top_k, 1);
                llama_sample_tail_free(ctx_llama, &candidates_p, tfs_z, 1);
                llama_sample_typical(ctx_llama, &candidates_p, typical_p, 1);
                llama_sample_top_p(ctx_llama, &candidates_p, top_p, 1);
                llama_sample_temp(ctx_llama, &candidates_p, temp);
                id = llama_sample_token(ctx_llama, &candidates_p);
            }
        }
    }

    return id;
}

inline const char * sample(struct llama_context * ctx_llama, gpt_params & params, int * n_past) {
    int id = sample_id(ctx_llama, params);
    static std::string ret;
    if (id == llama_token_eos(ctx_llama)) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}
