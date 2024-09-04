#pragma once

#include "llama.h"

#include <string>
#include <vector>

enum gpt_constraint_type {
    GPT_CONSTRAINT_TYPE_NONE        = 0,
    GPT_CONSTRAINT_TYPE_TOP_K       = 1,
    GPT_CONSTRAINT_TYPE_TOP_P       = 2,
    GPT_CONSTRAINT_TYPE_MIN_P       = 3,
    GPT_CONSTRAINT_TYPE_TFS_Z       = 4,
    GPT_CONSTRAINT_TYPE_TYPICAL_P   = 5,
    GPT_CONSTRAINT_TYPE_TEMPERATURE = 6,
};

// sampling parameters
struct gpt_sampler_params {
    uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler

    int32_t n_prev            = 64;    // number of previous tokens to remember
    int32_t n_probs           = 0;     // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t min_keep          = 0;     // 0 = disabled, otherwise constraints should return at least min_keep tokens
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

    std::vector<enum gpt_constraint_type> constraints = {
        GPT_CONSTRAINT_TYPE_TOP_K,
        GPT_CONSTRAINT_TYPE_TFS_Z,
        GPT_CONSTRAINT_TYPE_TYPICAL_P,
        GPT_CONSTRAINT_TYPE_TOP_P,
        GPT_CONSTRAINT_TYPE_MIN_P,
        GPT_CONSTRAINT_TYPE_TEMPERATURE
    };

    std::string grammar; // optional BNF-like grammar to constrain sampling

    std::vector<llama_logit_bias> logit_bias; // logit biases to apply

    // print the parameters into a string
    std::string print_all() const;

    // print the constraints into a string
    std::string print_constraints() const;
};

// gpt_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the paramerters
//
struct gpt_sampler;

struct gpt_sampler * gpt_sampler_init(const struct llama_model * model, const struct gpt_sampler_params & params);

void gpt_sampler_free(struct gpt_sampler * gsmpl);

struct gpt_sampler * gpt_sampler_cp(gpt_sampler * gsmpl);

void gpt_sampler_accept(struct gpt_sampler * gsmpl, llama_token token, bool apply_grammar);
void gpt_sampler_reset (struct gpt_sampler * gsmpl);

void gpt_sampler_set_logits(struct gpt_sampler * gsmpl, const float * logits);

llama_token_data_array * gpt_sampler_get_candidates(struct gpt_sampler * gsmpl);

llama_token gpt_sampler_last(const struct gpt_sampler * gsmpl);

void gpt_print_timings(struct llama_context * ctx, struct gpt_sampler * gsmpl);

// common sampling implementation:
//
// - set logits
// - apply the configured sampling constraints
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
llama_token gpt_sampler_sample(struct gpt_sampler * gsmpl, struct llama_context * ctx, int idx);

void gpt_sampler_apply_grammar(struct gpt_sampler * gsmpl, llama_token_data_array * cur_p);

llama_token gpt_sampler_sample_dist  (struct gpt_sampler * gsmpl, llama_token_data_array * cur_p);
llama_token gpt_sampler_sample_greedy(struct gpt_sampler * gsmpl, llama_token_data_array * cur_p, bool probs);

// helpers

// get a string representation of the last accepted tokens
std::string gpt_sampler_prev_str(gpt_sampler * gsmpl, llama_context * ctx, int n);

char        gpt_constraint_type_to_chr(enum gpt_constraint_type cnstr);
std::string gpt_constraint_type_to_str(enum gpt_constraint_type cnstr);

std::vector<enum gpt_constraint_type> gpt_constraint_types_from_names(const std::vector<std::string> & names, bool allow_alt_names);
std::vector<enum gpt_constraint_type> gpt_constraint_types_from_chars(const std::string & chars);
