#pragma once

#include "jarvis-impl.h"

#include <map>

struct jarvis_vocab;

// grammar element type
enum jarvis_gretype {
    // end of rule definition
    JARVIS_GRETYPE_END            = 0,

    // start of alternate definition for rule
    JARVIS_GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    JARVIS_GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    JARVIS_GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    JARVIS_GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding JARVIS_GRETYPE_CHAR or JARVIS_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    JARVIS_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding JARVIS_GRETYPE_CHAR or
    // JARVIS_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    JARVIS_GRETYPE_CHAR_ALT       = 6,

    // any character (.)
    JARVIS_GRETYPE_CHAR_ANY       = 7,
};

typedef struct jarvis_grammar_element {
    enum jarvis_gretype type;
    uint32_t           value; // Unicode code point or rule ID
} jarvis_grammar_element;

struct jarvis_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct jarvis_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    jarvis_partial_utf8   partial_utf8;
};

using jarvis_grammar_rule  = std::vector<      jarvis_grammar_element>;
using jarvis_grammar_stack = std::vector<const jarvis_grammar_element *>;

using jarvis_grammar_rules      = std::vector<jarvis_grammar_rule>;
using jarvis_grammar_stacks     = std::vector<jarvis_grammar_stack>;
using jarvis_grammar_candidates = std::vector<jarvis_grammar_candidate>;

const jarvis_grammar_rules  & jarvis_grammar_get_rules (const struct jarvis_grammar * grammar);
      jarvis_grammar_stacks & jarvis_grammar_get_stacks(      struct jarvis_grammar * grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `jarvis_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void jarvis_grammar_accept(
        const jarvis_grammar_rules  & rules,
        const jarvis_grammar_stacks & stacks,
                          uint32_t   chr,
              jarvis_grammar_stacks & stacks_new);

std::vector<jarvis_grammar_candidate> jarvis_grammar_reject_candidates_for_stack(
        const jarvis_grammar_rules      & rules,
        const jarvis_grammar_stack      & stack,
        const jarvis_grammar_candidates & candidates);

struct jarvis_grammar_parser {
    std::map<std::string, uint32_t> symbol_ids;

    jarvis_grammar_rules rules;

    jarvis_grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const jarvis_grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested);

    const char * parse_sequence(
            const char         * src,
            const std::string  & rule_name,
            jarvis_grammar_rule & rule,
            bool               is_nested);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct jarvis_grammar {
    // note: allow null vocab for testing (not great)
    const jarvis_vocab * vocab;

    const jarvis_grammar_rules  rules;  // TODO: shared ptr
          jarvis_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    jarvis_partial_utf8 partial_utf8;
};

//
// internal API
//

// note: needed for tests (not great)
struct jarvis_grammar * jarvis_grammar_init_impl(
        const struct jarvis_vocab * vocab,
        const jarvis_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

struct jarvis_grammar * jarvis_grammar_init_impl(const struct jarvis_vocab * vocab, const char * grammar_str, const char * grammar_root);

void jarvis_grammar_free_impl(struct jarvis_grammar * grammar);

struct jarvis_grammar * jarvis_grammar_clone_impl(const struct jarvis_grammar & grammar);

// TODO: move the API below as member functions of jarvis_grammar
void jarvis_grammar_apply_impl(
        const struct jarvis_grammar & grammar,
            jarvis_token_data_array * cur_p);

void jarvis_grammar_accept_impl(
              struct jarvis_grammar & grammar,
                       jarvis_token   token);
