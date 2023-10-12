#pragma once

#include "llama.h"

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
typedef struct llama_sampling_context {
    // parameters that will be used for sampling
    llama_sampling_params params;

    // mirostat sampler state
    float mirostat_mu;

    llama_grammar * grammar;
} llama_sampling_context;

#include "common.h"

// Create a new sampling context instance.
llama_sampling_context llama_sampling_context_init(
        const struct gpt_params & params,
                  llama_grammar * grammar = NULL);

// Reset the sampler context for the supplied sequence id (defaults to 0).
// This is necessary to reuse a sequence id or free memory used by sequences
// that are no longer required.
bool llama_sampling_context_reset(
              llama_sampling_context & ctx_sampling,
        const llama_seq_id             seq = 0);

// this is a common sampling function used across the examples for convenience
// it can serve as a starting point for implementing your own sampling function
// Note: When using multiple sequences, it is the caller's responsibility to call
//       llama_sampling_context_reset when a sequence ends
//
// required:
//  - ctx:          context to use for sampling
//  - ctx_sampling: sampling-specific context
//
// optional:
//  - ctx_guidance:  context to use for classifier-free guidance, ignore if NULL
//  - last_tokens:   needed for repetition penalty, ignore if empty
//  - idx:           sample from llama_get_logits_ith(ctx, idx)
//  - seq:           sequence id to associate sampler state with
//
// returns:
//  - token:      sampled token
//  - candidates: vector of candidate tokens
//
llama_token llama_sampling_sample(
                  struct llama_context * ctx,
                  struct llama_context * ctx_guidance,
                  struct llama_sampling_context & ctx_sampling,
        const std::vector<llama_token> & last_tokens,
         std::vector<llama_token_data> & candidates,
        const                      int   idx = 0);
