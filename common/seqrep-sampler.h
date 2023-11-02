#pragma once

#include <stddef.h>

#include <vector>
#include <regex>

#include "llama.h"

enum llama_sampler_seqrep_flags {
    // Tolerance charges can't be used consecutively.
    LLAMA_SEQREP_TOLERANCE_NO_CONSECUTIVE  = (1 << 0),

    // Tolerance charges can't be used before the first actual match.
    LLAMA_SEQREP_TOLERANCE_NO_FIRST        = (1 << 1),

    // When applying the length penalty, use the length of the longest observed
    // sequence matching the token rather than the total length of
    // sequences matching the token. In other words, if we find a sequence
    // of length 3 and a sequence of length 4 continued by token 69 then
    // with this flag on we penalize based on length 4, with it off we
    // penalize based on length 7 (3 + 4).
    LLAMA_SEQREP_PENALIZE_LENGTH_MAX_SEEN  = (1 << 2),

    // Apply an absolute penalty rather than dividing the logit by the penalty.
    LLAMA_SEQREP_ABSOLUTE_PENALTY          = (1 << 3),

    // Rewind to cut off the head of sequences rather than the end.
    // Ignored when min_length < 2.
    // Since it wouldn't make sense to rewind and then let sampling pick
    // the same token again, penalty values and mid_word_scale have no
    // effect.
    LLAMA_SEQREP_REWIND_MODE               = (1 << 4),

    // When rewinding, skip past whitespace and punctuation. For example,
    // if the matched sequence was "<NL>'hello" then we will rewind to the
    // token starting with 'h' and ban it.
    LLAMA_SEQREP_REWIND_SKIP_WS_PUNCT      = (1 << 5),

    // Rewind to the shortest matching sequence of at least min_length rather than the longest.
    LLAMA_SEQREP_REWIND_USE_SHORTEST_MATCH = (1 << 6),

    // Rewinding requires a word boundary. Only has an effect when rewind_seek_word_boundary isn't 0.
    LLAMA_SEQREP_REWIND_REQUIRE_WBOUND     = (1 << 7),

    // Persisted bans are only applied if at a word bound.
    LLAMA_SEQREP_REWIND_PERSIST_REQUIRE_WBOUND = (1 << 8),
};

typedef struct llama_sampler_seqrep_params {
    // The minimum length of a matching sequence of tokens. When this is < 2 then
    // the sampler works in single token mode and tolerance is ignored.
    size_t min_length;

    // Maximum length for a matching sequence of tokens.
    size_t max_length;

    // Starting offset for matching against the end of the sequence. This can be used
    // to only match against sequences in the initial prompt, for example. Matching
    // starts at the offset and moves toward the beginning of the list.
    // Use 0 for penultimate token when min_length > 1 otherwise 0 for last token.
    size_t start_offset;

    // Window of last tokens to consider, starting from the end. < 0 means
    // the whole list.
    int    last_n;

    // Flags based on llama_sampler_seqrep_flags enum values ORed together.
    int    flags;

    // Tolerance for non-matching tokens in a sequence.
    float  tolerance;

    // Flat penalty applied to the token that can continue a repeated sequence.
    float  presence_penalty;

    // Scaling penalty applied to the token that can continue a repeated sequence.
    // The penalty is multiplied by the total length of sequences that are continued by this token unless
    // the PENALIZE_LENGTH_MAX_SEEN is set.
    float  length_penalty;

    // Scale for penalizing tokens from repeated sequences that aren't at/form a word boundary.
    float  mid_word_scale;

    // Tolerance credit per real match. I.E. .5 means +1 tolerance per 2 matched tokens.
    float  tolerance_match_credit;

    // Caps tolerance at the specified value. Only meaningful when tolerance_match_credit > 0
    float  tolerance_cap;

    // Ensure the sequence is at least the specified length in rewind mode after
    // whitespace skipping and other modifications.
    size_t rewind_min_length;

    // When rewinding, try to find a word boundary within the specified distance, starting with tokens earlier than the rewind point.
    size_t rewind_seek_word_boundary;

    // A position is limited to the specified number of rewinds. When the limit is exceeded, future rewinds cannot target it or earlier tokens.
    size_t rewind_max_visits;

    // Tokens banned by rewind remain banned for an additional number of positions equal to the value. i.e. setting this to 1 would mean the token is banned for 2 positions.
    size_t rewind_persist_bans;

    // Number of tokens from the sequence to ban when rewinding.
    size_t rewind_ban_length;

    std::vector<std::wregex> include_re;
    std::vector<std::wregex> exclude_re;
} llama_sampler_seqrep_params;

enum seqrep_check_word_flags {
    SEQREP_CW_START_IS_WBOUND  = 1 << 0,
    SEQREP_CW_END_IS_WBOUND    = 1 << 1,
    SEQREP_CW_ALL_WS_PUNCT     = 1 << 2,
    SEQREP_CW_START_IS_INVALID = 1 << 3, // Start of token is invalid/incomplete UTF8
    SEQREP_CW_END_IS_INVALID   = 1 << 4  // End of token is invalid/incomplete UTF8
};


struct seqrep_logit_info {
    const int n_vocab;
    std::vector<llama_token_data> token_data;

    seqrep_logit_info(llama_context * ctx, const size_t k, const int32_t ith);

    const std::vector<llama_token_data> & get_token_data(void);

    llama_token_data get_token_id(const llama_token token_id) const;

    void rebuild(llama_context *ctx, const size_t k, const int32_t ith);

    void populate_logits(float * logits);

    // Yoinked from beam search code.
    // Return top k token_data by logit.
    std::vector<llama_token_data> top_k(const float * const logits, const size_t k);

    seqrep_logit_info(const int n_vocab, const std::vector<llama_token_data> & token_data = {})
        : n_vocab(n_vocab)
        , token_data(token_data)
        {}
};

struct seqrep_rewind_slot {
  size_t count;
  std::vector<llama_token> tokens;
  struct llama_sampling_context * ctx_sampling = nullptr;
};

struct seqrep_rewind_state {
    const size_t n_vocab;
    const size_t n_ctx;
    const size_t k;

    std::vector<seqrep_logit_info>  logit_slots;
    std::vector<seqrep_rewind_slot> rewind_slots;

    seqrep_rewind_state(
        const size_t n_vocab,
        const size_t n_ctx,
        const size_t k = 2000);

    struct seqrep_rewind_slot & get_rewind_slot(const size_t idx);

    void set_logits_slot(llama_context * ctx, const size_t idx, const int32_t ith = 0);

    void populate_logits(llama_context * ctx, const size_t idx, const int32_t ith = 0);

};

// Sequence repetition penalty with semi-fuzzy matching. Note: Handles the last_n window itself.
size_t llama_sample_seqrep_penalty(
    struct llama_context * ctx,
    llama_token_data_array * candidates,
    const std::vector<llama_token> & last_tokens,
    const llama_sampler_seqrep_params * params);

int llama_seqrep_check_word(
    const struct llama_context * ctx,
    const llama_token token,
    std::vector<char> & buf);

size_t llama_seqrep_handle_rewind(
        struct llama_context * ctx,
        struct seqrep_rewind_state & rewind_state,
        const std::vector<llama_token> & generated_tokens,
        const size_t n_generated,
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_sampler_seqrep_params> & params_list,
        size_t * high_water_mark,
        const int32_t ith = 0);

void seqrep_sampler_help();
void seqrep_sampler_params_init(llama_sampler_seqrep_params * params);
void seqrep_sampler_params_dump(const llama_sampler_seqrep_params * params);
bool seqrep_sampler_params_parse(char * s, llama_sampler_seqrep_params * params);
struct llama_sampler_seqrep_params llama_seqrep_merge_params(
    const std::vector<llama_sampler_seqrep_params> & params_list,
    const int and_flags,
    const int not_flags);
