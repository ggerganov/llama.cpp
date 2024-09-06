#pragma once

#include "llama-grammar.h"

#include <unordered_map>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    std::vector<struct llama_sampler *> samplers;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

using llama_token_cnt = std::unordered_map<llama_token, int>;

// TODO: tmp exposed until test-sampling is fixed
void llama_sampler_penalties_impl(
       llama_token_data_array * cur_p,
        const llama_token_cnt & token_count,
                        float   penalty_repeat,
                        float   penalty_freq,
                        float   penalty_present);

struct llama_sampler * llama_sampler_init_mirostat_impl(
        const struct llama_vocab & vocab,
                        uint32_t   seed,
                           float   tau,
                           float   eta,
                         int32_t   m);

struct llama_sampler * llama_sampler_init_grammar_impl(
        const struct llama_vocab & vocab,
                      const char * grammar_str,
                      const char * grammar_root);

struct llama_sampler * llama_sampler_init_penalties_impl(
        const struct llama_vocab & vocab,
                         int32_t   penalty_last_n,
                           float   penalty_repeat,
                           float   penalty_freq,
                           float   penalty_present,
                            bool   penalize_nl,
                            bool   ignore_eos);

LLAMA_API struct llama_sampler * llama_sampler_init_logit_bias_impl(
        const struct llama_vocab & vocab,
                         int32_t   n_logit_bias,
          const llama_logit_bias * logit_bias);
