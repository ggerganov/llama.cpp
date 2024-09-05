#pragma once

#include "llama-grammar.h"

#include <unordered_map>

struct llama_vocab;
struct llama_grammar;

// samplers

const char *           llama_sampler_name_impl  (const struct llama_sampler & smpl);
void                   llama_sampler_accept_impl(      struct llama_sampler & smpl, llama_token token);
void                   llama_sampler_apply_impl (      struct llama_sampler & smpl, struct llama_token_data_array * cur_p);
void                   llama_sampler_reset_impl (      struct llama_sampler & smpl);
struct llama_sampler * llama_sampler_clone_impl (const struct llama_sampler & smpl);
void                   llama_sampler_free_impl  (      struct llama_sampler * smpl);

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    std::vector<struct llama_sampler *> samplers;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct llama_sampler * llama_sampler_chain_init_impl(      struct llama_sampler_chain_params params);
void                   llama_sampler_chain_add_impl (      struct llama_sampler_chain & chain, struct llama_sampler * smpl);
struct llama_sampler * llama_sampler_chain_get_impl (const struct llama_sampler_chain & chain, int32_t i);
int                    llama_sampler_chain_n_impl   (const struct llama_sampler_chain & chain);

using llama_token_cnt = std::unordered_map<llama_token, int>;

// TODO: tmp exposed until test-sampling is fixed
void llama_sampler_penalties_impl(
       llama_token_data_array * cur_p,
        const llama_token_cnt & token_count,
                        float   penalty_repeat,
                        float   penalty_freq,
                        float   penalty_present);

struct llama_sampler * llama_sampler_init_greedy_impl   ();
struct llama_sampler * llama_sampler_init_dist_impl     (uint32_t seed);
struct llama_sampler * llama_sampler_init_softmax_impl  ();
struct llama_sampler * llama_sampler_init_top_k_impl    (int32_t k);
struct llama_sampler * llama_sampler_init_top_p_impl    (float   p, size_t min_keep);
struct llama_sampler * llama_sampler_init_min_p_impl    (float   p, size_t min_keep);
struct llama_sampler * llama_sampler_init_tail_free_impl(float   z, size_t min_keep);
struct llama_sampler * llama_sampler_init_typical_impl  (float   p, size_t min_keep);
struct llama_sampler * llama_sampler_init_temp_impl     (float   t);
struct llama_sampler * llama_sampler_init_temp_ext_impl (float   t, float  delta, float exponent);

struct llama_sampler * llama_sampler_init_mirostat_impl(
        const struct llama_vocab & vocab,
                           float   tau,
                           float   eta,
                         int32_t   m);

struct llama_sampler * llama_sampler_init_mirostat_v2_impl(
                           float   tau,
                           float   eta);

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
