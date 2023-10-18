#pragma once

#include "llama.h"

#include "grammar-parser.h"

#include <string>
#include <vector>
#include <unordered_map>

// sampling parameters
typedef struct llama_sampling_params {
    int32_t top_k             = 40;    // <= 0 to use vocab size
    float   top_p             = 0.95f; // 1.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   typical_p         = 1.00f; // 1.0 = disabled
    float   temp              = 0.80f; // 1.0 = disabled
    float   repeat_penalty    = 1.10f; // 1.0 = disabled
    int32_t repeat_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   frequency_penalty = 0.00f; // 0.0 = disabled
    float   presence_penalty  = 0.00f; // 0.0 = disabled
    int32_t mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 5.00f; // target entropy
    float   mirostat_eta      = 0.10f; // learning rate

    bool    penalize_nl       = true;  // consider newlines as a repeatable token

    int32_t n_probs           = 0;     // if greater than 0, output the probabilities of top n_probs tokens.

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    std::string cfg_negative_prompt;   // string to help guidance
    float       cfg_scale     = 1.f;   // How strong is guidance

    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens

} llama_sampling_params;

// general sampler context
// TODO: move to llama.h
struct llama_sampling_context {
    // parameters that will be used for sampling
    llama_sampling_params params;

    // mirostat sampler state
    float mirostat_mu;

    llama_grammar * grammar;

    // internal
    grammar_parser::parse_state parsed_grammar;

    // TODO: replace with ring-buffer
    std::vector<llama_token>      prev;
    std::vector<llama_token_data> cur;
};

#include "common.h"

// Create a new sampling context instance.
struct llama_sampling_context * llama_sampling_init(const struct gpt_params & params);

void llama_sampling_free(struct llama_sampling_context * ctx);

// Reset the sampler context
// - clear prev tokens
// - reset grammar
void llama_sampling_reset(llama_sampling_context * ctx);

// Copy the sampler context
void llama_sampling_cp(llama_sampling_context * src, llama_sampling_context * dst);

// this is a common sampling function used across the examples for convenience
// it can serve as a starting point for implementing your own sampling function
// Note: When using multiple sequences, it is the caller's responsibility to call
//       llama_sampling_reset when a sequence ends
//
// required:
//  - ctx_main:     context to use for sampling
//  - ctx_sampling: sampling-specific context
//
// optional:
//  - ctx_cfg:      context to use for classifier-free guidance
//  - idx:          sample from llama_get_logits_ith(ctx, idx)
//
// returns:
//  - token:      sampled token
//  - candidates: vector of candidate tokens
//
llama_token llama_sampling_sample(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        struct llama_context * ctx_cfg,
        int idx = 0);

void llama_sampling_accept(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id);
