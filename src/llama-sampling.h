#pragma once

#include "llama-impl.h"

struct llama_sampling {
    llama_sampling(int32_t n_vocab) : n_vocab(n_vocab) {}

    std::mt19937 rng;

    int64_t t_sample_us = 0;

    int32_t n_sample = 0;
    int32_t n_vocab = 0;

    void reset_timings() {
        t_sample_us = 0;
        n_sample = 0;
    }
};

struct llama_sampling * llama_get_sampling(struct llama_context * ctx);
