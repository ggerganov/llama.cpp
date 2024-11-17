#pragma once

#include "llama.h"

#include <vector>

struct common_speculative;

struct common_speculative_params {
    int n_draft = 16;
    int n_min   = 5;  // do not add drafts smaller than this, TODO: leave this to user?

    struct llama_model * model_dft = nullptr;

    struct llama_context * ctx_dft = nullptr;
};

struct common_speculative * common_speculative_init(struct common_speculative_params params);

void common_speculative_free(struct common_speculative * spec);

// TODO: remove
void common_speculative_set_prompt(struct common_speculative * spec, llama_token * tokens, int32_t n_tokens);

// sample up to n_draft tokens and add them to the batch using the draft model
//
// TODO: change to:
//
//    void common_speculative_add_draft(
//            struct common_speculative * spec,
//            struct llama_batch & batch_tgt,
//            llama_token * tokens,
//            int32_t n_tokens);
//
//       and update the internal logic to compute only the new tokens
//
void common_speculative_add_draft(
        struct common_speculative * spec,
        struct llama_batch & batch_tgt,
        llama_token id_last,
        int n_past);

std::vector<llama_token> common_speculative_sample(
        struct common_speculative * spec,
        struct common_sampler * smpl,
        struct llama_context * ctx_tgt);
