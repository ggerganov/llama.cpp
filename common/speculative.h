#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

struct common_speculative_params {
    int n_draft = 16;
    int n_min   = 5;  // do not add drafts smaller than this, TODO: leave this to user?
    int n_reuse = 256;

    float p_min = 0.9f;
};

struct common_speculative * common_speculative_init(
        struct common_speculative_params params,
        struct llama_context * ctx_dft);

void common_speculative_free(struct common_speculative * spec);

bool common_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft);

// sample up to n_draft tokens and add them to the batch using the draft model
//
void common_speculative_add_draft(
        struct common_speculative * spec,
        struct llama_batch & batch_tgt,
        const llama_tokens & prompt,
        llama_token id_last,
        llama_token n_past_tgt);
