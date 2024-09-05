#pragma once

#include "llama-grammar.h"

#include <random>
#include <unordered_map>

struct llama_vocab;
struct llama_grammar;

using llama_token_cnt = std::unordered_map<llama_token, int>;

// TODO: tmp exposed until test-sampling is fixed
void llama_constraint_penalties_impl(
       llama_token_data_array * cur_p,
        const llama_token_cnt & token_count,
                        float   penalty_repeat,
                        float   penalty_freq,
                        float   penalty_present);

// constraints

struct llama_constraint * llama_constraint_init_softmax_impl    ();
struct llama_constraint * llama_constraint_init_top_k_impl      (int32_t k);
struct llama_constraint * llama_constraint_init_top_p_impl      (float   p, size_t min_keep);
struct llama_constraint * llama_constraint_init_min_p_impl      (float   p, size_t min_keep);
struct llama_constraint * llama_constraint_init_tail_free_impl  (float   z, size_t min_keep);
struct llama_constraint * llama_constraint_init_typical_impl    (float   p, size_t min_keep);
struct llama_constraint * llama_constraint_init_temp_impl       (float   t);
struct llama_constraint * llama_constraint_init_temp_ext_impl   (float   t, float  delta, float exponent);

struct llama_constraint * llama_constraint_init_mirostat_impl(
        const struct llama_vocab & vocab,
                           float   tau,
                           float   eta,
                         int32_t   m);

struct llama_constraint * llama_constraint_init_mirostat_v2_impl(
                           float   tau,
                           float   eta);

struct llama_constraint * llama_constraint_init_grammar_impl(
        const struct llama_vocab & vocab,
                      const char * grammar_str,
                      const char * grammar_root);

struct llama_constraint * llama_constraint_init_penalties_impl(
        const struct llama_vocab & vocab,
                         int32_t   penalty_last_n,
                           float   penalty_repeat,
                           float   penalty_freq,
                           float   penalty_present,
                            bool   penalize_nl,
                            bool   ignore_eos);

    LLAMA_API struct llama_constraint * llama_constraint_init_logit_bias_impl(
        const struct llama_vocab & vocab,
                         int32_t   n_logit_bias,
          const llama_logit_bias * logit_bias);

struct llama_constraint * llama_constraint_clone_impl(const struct llama_constraint & cnstr);

void llama_constraint_free_impl(struct llama_constraint * cnstr);

const char * llama_constraint_name_impl  (const struct llama_constraint & cnstr);
void         llama_constraint_accept_impl(      struct llama_constraint & cnstr, llama_token token);
void         llama_constraint_apply_impl (      struct llama_constraint & cnstr, struct llama_token_data_array * cur_p);
void         llama_constraint_reset_impl (      struct llama_constraint & cnstr);

// samplers

struct llama_sampler {
    llama_sampler_params params;

    const struct llama_vocab * vocab;

    // state

    std::mt19937 rng;

    ring_buffer<llama_token> prev;

    std::vector<llama_constraint *> constraints;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct llama_sampler * llama_sampler_init_impl  (const struct llama_vocab   & vocab, struct llama_sampler_params params);
void                   llama_sampler_free_impl  (      struct llama_sampler * smpl);
struct llama_sampler * llama_sampler_clone_impl (const struct llama_sampler & smpl);
void                   llama_sampler_reset_impl (      struct llama_sampler & smpl);
void                   llama_sampler_accept_impl(      struct llama_sampler & smpl, llama_token token);
void                   llama_sampler_apply_impl (      struct llama_sampler & smpl, struct llama_token_data_array * cur_p);

void                      llama_sampler_constraint_add_impl(      struct llama_sampler & smpl, struct llama_constraint * cnstr);
int                       llama_sampler_n_constraints_impl (const struct llama_sampler & smpl);
struct llama_constraint * llama_sampler_constraint_get_impl(const struct llama_sampler & smpl, int ith);

llama_token llama_sampler_sample_impl(struct llama_token_data_array * cur_p, std::mt19937 & rng, enum llama_sampler_type type);

llama_token llama_sampler_prev_impl  (const struct llama_sampler & smpl, int ith);
int         llama_sampler_n_prev_impl(const struct llama_sampler & smpl);
