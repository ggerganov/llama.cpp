#pragma once

#include "llama-impl.h"

struct llama_sampling {
    llama_sampling(int32_t n_vocab) : n_vocab(n_vocab) {}

    const int32_t n_vocab;

    std::mt19937 rng;

    mutable int64_t t_total_us = 0;

    mutable int32_t n_sample = 0;
};

//
// internal API
//

struct llama_sampling * llama_sampling_init_impl(int32_t n_vocab);

void llama_sampling_free_impl(struct llama_sampling * sampling);

void llama_sampling_set_rng_seed_impl(struct llama_sampling & smpl, uint32_t seed);

void llama_sampling_softmax_impl  (struct llama_sampling & smpl, llama_token_data_array * candidates);
void llama_sampling_top_k_impl    (struct llama_sampling & smpl, llama_token_data_array * candidates, int32_t k, size_t min_keep);
void llama_sampling_top_p_impl    (struct llama_sampling & smpl, llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_min_p_impl    (struct llama_sampling & smpl, llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_tail_free_impl(struct llama_sampling & smpl, llama_token_data_array * candidates, float z, size_t min_keep);
void llama_sampling_typical_impl  (struct llama_sampling & smpl, llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sampling_entropy_impl  (struct llama_sampling & smpl, llama_token_data_array * candidates, float min_temp, float max_temp, float exponent_val);
void llama_sampling_temp_impl     (struct llama_sampling & smpl, llama_token_data_array * candidates, float temp);

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

