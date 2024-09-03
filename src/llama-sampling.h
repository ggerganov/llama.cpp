#pragma once

#include "llama-grammar.h"

#include <random>
#include <unordered_map>

struct llama_vocab;
struct llama_grammar;

using llama_token_cnt = std::unordered_map<llama_token, int>;

// TODO: remove before merge
struct llama_sampling {
    llama_sampling(const struct llama_vocab & vocab);
    ~llama_sampling();

    llama_sampling_params params;

    std::string grammar_str;
    std::string grammar_root;

    std::vector<llama_logit_bias> logit_bias; // logit biases to apply

    // state

    std::mt19937 rng;

    const struct llama_vocab & vocab;

    std::vector<llama_constraint_type> samplers;

    ring_buffer<llama_token> prev;

    struct llama_grammar * grammar = nullptr;

    // mirostat sampler state
    float mirostat_mu;

    mutable int64_t t_sample_us  = 0;
    mutable int64_t t_grammar_us = 0;
    mutable int64_t t_accept_us  = 0;

    mutable int32_t n_sample  = 0;
    mutable int32_t n_grammar = 0;
    mutable int32_t n_accept  = 0;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;
};

//
// internal API
//

struct llama_sampling * llama_sampling_init_impl(const struct llama_vocab & vocab, struct llama_sampling_params params);

void llama_sampling_free_impl(struct llama_sampling * sampling);

struct llama_sampling * llama_sampling_cp_impl(const struct llama_sampling & smpl);

void llama_sampling_reset_impl(struct llama_sampling & smpl);

// TODO: move the API below as member functions of llama_sampling
void llama_sampling_set_rng_seed_impl  (struct llama_sampling & smpl, uint32_t seed);
void llama_sampling_set_grammar_impl   (struct llama_sampling & smpl, const char * grammar_str, const char * grammar_root);
void llama_sampling_set_logit_bias_impl(struct llama_sampling & smpl, int32_t n_logit_bias, const llama_logit_bias * logit_bias);

void llama_sampling_softmax_impl  (struct llama_token_data_array * candidates);
void llama_sampling_top_k_impl    (struct llama_token_data_array * candidates, int32_t k, size_t min_keep);
void llama_sampling_top_p_impl    (struct llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_min_p_impl    (struct llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_tail_free_impl(struct llama_token_data_array * candidates, float z, size_t min_keep);
void llama_sampling_typical_impl  (struct llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_entropy_impl  (struct llama_token_data_array * candidates, float min_temp, float max_temp, float exponent_val);
void llama_sampling_temp_impl     (struct llama_token_data_array * candidates, float temp);
void llama_sampling_grammar_impl  (struct llama_token_data_array * candidates, const struct llama_grammar & grammar);

void llama_sampling_penalties_impl(
       llama_token_data_array * candidates,
        const llama_token_cnt & token_count,
                        float   penalty_repeat,
                        float   penalty_freq,
                        float   penalty_present);

/// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
/// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
/// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
llama_token llama_sampling_sample_mirostat_impl   (struct llama_token_data_array * candidates, std::mt19937 & rng, float tau, float eta, int32_t m, int32_t n_vocab, float & mu);

/// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
/// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
llama_token llama_sampling_sample_mirostat_v2_impl(struct llama_token_data_array * candidates, std::mt19937 & rng, float tau, float eta, float & mu);

llama_token llama_sampling_sample_greedy_impl(struct llama_token_data_array * candidates);
llama_token llama_sampling_sample_dist_impl  (struct llama_token_data_array * candidates, std::mt19937 & rng);

void llama_sampling_accept_impl(struct llama_sampling & smpl, llama_token token, bool apply_grammar);

llama_token llama_sampling_prev_impl  (const struct llama_sampling & smpl, int ith);
int         llama_sampling_n_prev_impl(const struct llama_sampling & smpl);


//
// sampling v2
//

// constraints

struct llama_constraint * llama_constraint_init_top_k_impl    (int32_t k, size_t min_keep);
struct llama_constraint * llama_constraint_init_top_p_impl    (float   p, size_t min_keep);
struct llama_constraint * llama_constraint_init_min_p_impl    (float   p, size_t min_keep);
struct llama_constraint * llama_constraint_init_tail_free_impl(float   z, size_t min_keep);
struct llama_constraint * llama_constraint_init_typical_impl  (float   p, size_t min_keep);
struct llama_constraint * llama_constraint_init_temp_impl     (float   t);
struct llama_constraint * llama_constraint_init_temp_ext_impl (float   t, float  delta, float exponent);

struct llama_constraint * llama_constraint_init_grammar_impl  (
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

void llama_constraint_free_impl(struct llama_constraint * cnstr);

void llama_constraint_accept_impl(struct llama_constraint & cnstr, llama_token token);
void llama_constraint_apply_impl (struct llama_constraint & cnstr, struct llama_token_data_array * candidates);
void llama_constraint_reset_impl (struct llama_constraint & cnstr);

// samplers

struct llama_sampler {
    llama_sampler_params params;

    const struct llama_vocab * vocab;

    // state

    std::mt19937 rng;

    float mirostat_mu;

    ring_buffer<llama_token> prev;

    std::vector<llama_constraint *> constraints;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // timing

    mutable int64_t t_sample_us = 0;

    mutable int32_t n_sample = 0;
};

struct llama_sampler * llama_sampler_init_impl (const struct llama_vocab   & vocab, struct llama_sampler_params params);
void                   llama_sampler_free_impl (      struct llama_sampler * smpl);
struct llama_sampler * llama_sampler_cp_impl   (const struct llama_sampler & smpl);
void                   llama_sampler_reset_impl(      struct llama_sampler & smpl);

void llama_sampler_add_constraint_impl(struct llama_sampler & smpl, struct llama_constraint * cnstr);

void llama_sampler_accept_impl(struct llama_sampler & smpl, llama_token token);
void llama_sampler_apply_impl (struct llama_sampler & smpl, struct llama_token_data_array * candidates);

llama_token llama_sampler_prev_impl  (const struct llama_sampler & smpl, int ith);
int         llama_sampler_n_prev_impl(const struct llama_sampler & smpl);
