#pragma once

#include "llama.h"

#include <string>
#include <vector>

// sampling parameters
struct gpt_sampling_params {
    uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampling

    int32_t n_prev            = 64;    // number of previous tokens to remember
    int32_t n_probs           = 0;     // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t min_keep          = 0;     // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t top_k             = 40;    // <= 0 to use vocab size
    float   top_p             = 0.95f; // 1.0 = disabled
    float   min_p             = 0.05f; // 0.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   typ_p             = 1.00f; // typical_p, 1.0 = disabled
    float   temp              = 0.80f; // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float   dynatemp_range    = 0.00f; // 0.0 = disabled
    float   dynatemp_exponent = 1.00f; // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t penalty_last_n    = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   penalty_repeat    = 1.00f; // 1.0 = disabled
    float   penalty_freq      = 0.00f; // 0.0 = disabled
    float   penalty_present   = 0.00f; // 0.0 = disabled
    int32_t mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 5.00f; // target entropy
    float   mirostat_eta      = 0.10f; // learning rate
    bool    penalize_nl       = false; // consider newlines as a repeatable token
    bool    ignore_eos        = false;

    std::vector<enum llama_constraint_type> samplers = {
        LLAMA_CONSTRAINT_TYPE_TOP_K,
        LLAMA_CONSTRAINT_TYPE_TFS_Z,
        LLAMA_CONSTRAINT_TYPE_TYPICAL_P,
        LLAMA_CONSTRAINT_TYPE_TOP_P,
        LLAMA_CONSTRAINT_TYPE_MIN_P,
        LLAMA_CONSTRAINT_TYPE_TEMPERATURE
    };

    std::string grammar; // optional BNF-like grammar to constrain sampling

    std::vector<llama_logit_bias> logit_bias; // logit biases to apply

    // print the parameters into a string
    std::string print_all() const;

    // print the samplers into a string
    std::string print_samplers() const;
};

// TODO: implement
struct gpt_sampler {
    gpt_sampling_params params;

    struct llama_constraint * grmr = nullptr;

    struct llama_sampler * smpl = nullptr;
};

// overload of llama_sampling_init using gpt_sampling_params
struct llama_sampling * llama_sampling_init(const struct llama_model * model, const struct gpt_sampling_params & params);

void llama_sampling_cp(llama_sampling * src, llama_sampling *& dst);

// common sampling implementation:
//
// - set logits
// - apply the configured sampling constraints
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
llama_token llama_sampling_sample(
        struct llama_sampling * smpl,
         struct llama_context * ctx,
                          int   idx);

// helpers

// get a string representation of the last accepted tokens
std::string llama_sampling_prev_str(llama_sampling * smpl, llama_context * ctx, int n);

char        llama_sampling_type_to_chr(enum llama_constraint_type sampler_type);
std::string llama_sampling_type_to_str(enum llama_constraint_type sampler_type);

std::vector<enum llama_constraint_type> llama_sampling_types_from_names(const std::vector<std::string> & names, bool allow_alt_names);
std::vector<enum llama_constraint_type> llama_sampling_types_from_chars(const std::string & chars);
