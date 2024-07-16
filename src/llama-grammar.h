#pragma once

#include "llama-impl.h"

struct llama_vocab;

struct llama_grammar {
    const llama_grammar_rules  rules;
          llama_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8 partial_utf8;
};

struct llama_grammar * llama_get_grammar(struct llama_context * ctx);
