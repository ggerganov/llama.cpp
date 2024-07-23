#pragma once

#include "llama.h"

#include <random>
#include <string>
#include <vector>

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

LLAMA_ATTRIBUTE_FORMAT(2, 3)
void llama_log_internal        (ggml_log_level level, const char * format, ...);
void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
);

struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_partial_utf8   partial_utf8;
};

using llama_grammar_rule  = std::vector<      llama_grammar_element>;
using llama_grammar_stack = std::vector<const llama_grammar_element *>;

using llama_grammar_rules      = std::vector<llama_grammar_rule>;
using llama_grammar_stacks     = std::vector<llama_grammar_stack>;
using llama_grammar_candidates = std::vector<llama_grammar_candidate>;

const llama_grammar_rules  & llama_grammar_get_rules (const struct llama_grammar * grammar);
      llama_grammar_stacks & llama_grammar_get_stacks(      struct llama_grammar * grammar);

void llama_grammar_accept(
        const llama_grammar_rules  & rules,
        const llama_grammar_stacks & stacks,
        const uint32_t chr,
              llama_grammar_stacks & new_stacks);

std::vector<llama_grammar_candidate> llama_grammar_reject_candidates_for_stack(
        const llama_grammar_rules & rules,
        const llama_grammar_stack & stack,
        const llama_grammar_candidates & candidates);

std::pair<std::vector<uint32_t>, llama_partial_utf8> decode_utf8(
        const std::string & src,
        llama_partial_utf8 partial_start);
