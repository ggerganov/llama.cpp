#pragma once

#include "llama-grammar.h"

struct llama_vocab;
struct llama_grammar;

struct llama_sampling {
    llama_sampling(const struct llama_vocab & vocab);
    ~llama_sampling();

    llama_sampling_params params;

    std::string grammar_str;
    std::string grammar_root;

    std::string cfg_prompt;
    float       cfg_scale = 1.0f;

    std::vector<llama_logit_bias> logit_bias; // logit biases to apply

    // state

    std::mt19937 rng;

    const struct llama_vocab & vocab;

    struct llama_grammar * grammar = nullptr;

    mutable int64_t t_total_us = 0;

    mutable int32_t n_sample = 0;
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
void llama_sampling_set_cfg_impl       (struct llama_sampling & smpl, const char * cfg_prompt, float cfg_scale);
void llama_sampling_set_logit_bias_impl(struct llama_sampling & smpl, int32_t n_logit_bias, const llama_logit_bias * logit_bias);

void llama_sampling_softmax_impl  (struct llama_sampling & smpl, llama_token_data_array * candidates);
void llama_sampling_top_k_impl    (struct llama_sampling & smpl, llama_token_data_array * candidates, int32_t k, size_t min_keep);
void llama_sampling_top_p_impl    (struct llama_sampling & smpl, llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_min_p_impl    (struct llama_sampling & smpl, llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_tail_free_impl(struct llama_sampling & smpl, llama_token_data_array * candidates, float z, size_t min_keep);
void llama_sampling_typical_impl  (struct llama_sampling & smpl, llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_entropy_impl  (struct llama_sampling & smpl, llama_token_data_array * candidates, float min_temp, float max_temp, float exponent_val);
void llama_sampling_temp_impl     (struct llama_sampling & smpl, llama_token_data_array * candidates, float temp);
void llama_sampling_grammar_impl  (struct llama_sampling & smpl, llama_token_data_array * candidates);

void llama_sampling_repetition_penalties_impl(
        struct llama_sampling & smpl,
       llama_token_data_array * candidates,
            const llama_token * last_tokens,
                       size_t   penalty_last_n,
                        float   penalty_repeat,
                        float   penalty_freq,
                        float   penalty_present);

void llama_sampling_apply_guidance_impl(
        struct llama_sampling & smpl,
                        float * logits,
                        float * logits_guidance,
                        float   scale);

llama_token llama_sampling_sample_mirostat_impl   (struct llama_sampling & smpl, llama_token_data_array * candidates, float tau, float eta, int32_t m, float * mu);
llama_token llama_sampling_sample_mirostat_v2_impl(struct llama_sampling & smpl, llama_token_data_array * candidates, float tau, float eta, float * mu);
llama_token llama_sampling_sample_greedy_impl     (struct llama_sampling & smpl, llama_token_data_array * candidates);
llama_token llama_sampling_sample_with_rng_impl   (struct llama_sampling & smpl, llama_token_data_array * candidates, std::mt19937 & rng);
llama_token llama_sampling_sample_impl            (struct llama_sampling & smpl, llama_token_data_array * candidates);

void llama_sampling_accept_impl(struct llama_sampling & smpl, llama_token token);
