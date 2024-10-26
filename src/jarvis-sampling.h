#pragma once

// TODO: rename jarvis-sampling.h/.cpp to jarvis-sampler.h/.cpp ?

#include "jarvis-grammar.h"

struct jarvis_vocab;
struct jarvis_grammar;

// sampler chain

struct jarvis_sampler_chain {
    jarvis_sampler_chain_params params;

    std::vector<struct jarvis_sampler *> samplers;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct jarvis_sampler * jarvis_sampler_init_grammar_impl(
        const struct jarvis_vocab & vocab,
                      const char * grammar_str,
                      const char * grammar_root);

struct jarvis_sampler * jarvis_sampler_init_infill_impl(
        const struct jarvis_vocab & vocab);

struct jarvis_sampler * jarvis_sampler_init_dry_impl(
        const struct jarvis_vocab &  vocab,
                         int32_t    context_size,
                           float    dry_multiplier,
                           float    dry_base,
                         int32_t    dry_allowed_length,
                         int32_t    dry_penalty_last_n,
                      const char ** seq_breakers,
                          size_t    num_breakers);

struct jarvis_sampler * jarvis_sampler_init_dry_testing(
                         int32_t   context_size,
                           float   dry_multiplier,
                           float   dry_base,
                         int32_t   dry_allowed_length,
                         int32_t   dry_penalty_last_n,
  const std::vector<std::vector<jarvis_token>>& seq_breakers);
