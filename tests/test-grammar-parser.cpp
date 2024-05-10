#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"
#include "grammar-parser.h"

#include <cassert>

int main()
{
    grammar_parser::parse_state parsed_grammar;

    const char *grammar_bytes = R"""(root  ::= (expr "=" term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= [0-9]+)""";

    parsed_grammar = grammar_parser::parse(grammar_bytes);

    std::vector<std::pair<std::string, uint32_t>> expected = {
        {"expr", 2},
        {"expr_5", 5},
        {"expr_6", 6},
        {"root", 0},
        {"root_1", 1},
        {"root_4", 4},
        {"term", 3},
        {"term_7", 7},
    };

    uint32_t index = 0;
    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it)
    {
        std::string key = it->first;
        uint32_t value = it->second;
        std::pair<std::string, uint32_t> expected_pair = expected[index];

        // pretty print error message before asserting
        if (expected_pair.first != key || expected_pair.second != value)
        {
            fprintf(stderr, "expected_pair: %s, %u\n", expected_pair.first.c_str(), expected_pair.second);
            fprintf(stderr, "actual_pair: %s, %u\n", key.c_str(), value);
            fprintf(stderr, "expected_pair != actual_pair\n");
        }

        assert(expected_pair.first == key && expected_pair.second == value);

        index++;
    }
    std::vector<llama_grammar_element> expected_rules = {
        {LLAMA_GRETYPE_RULE_REF, 4},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 2},
        {LLAMA_GRETYPE_CHAR, 61},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_CHAR, 10},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_RULE_REF, 6},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 7},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 1},
        {LLAMA_GRETYPE_RULE_REF, 4},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_RULE_REF, 1},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 45},
        {LLAMA_GRETYPE_CHAR_ALT, 43},
        {LLAMA_GRETYPE_CHAR_ALT, 42},
        {LLAMA_GRETYPE_CHAR_ALT, 47},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 5},
        {LLAMA_GRETYPE_RULE_REF, 6},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 48},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
        {LLAMA_GRETYPE_RULE_REF, 7},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR, 48},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
        {LLAMA_GRETYPE_END, 0},
    };

    index = 0;
    for (auto rule : parsed_grammar.rules)
    {
        // compare rule to expected rule
        for (uint32_t i = 0; i < rule.size(); i++)
        {
            llama_grammar_element element = rule[i];
            llama_grammar_element expected_element = expected_rules[index];

            // pretty print error message before asserting
            if (expected_element.type != element.type || expected_element.value != element.value)
            {
                fprintf(stderr, "index: %u\n", index);
                fprintf(stderr, "expected_element: %d, %u\n", expected_element.type, expected_element.value);
                fprintf(stderr, "actual_element: %d, %u\n", element.type, element.value);
                fprintf(stderr, "expected_element != actual_element\n");
            }

            assert(expected_element.type == element.type && expected_element.value == element.value);
            index++;
        }
    }

    const char *longer_grammar_bytes = R"""(
    root  ::= (expr "=" ws term "\n")+
    expr  ::= term ([-+*/] term)*
    term  ::= ident | num | "(" ws expr ")" ws
    ident ::= [a-z] [a-z0-9_]* ws
    num   ::= [0-9]+ ws
    ws    ::= [ \t\n]*
    )""";

    parsed_grammar = grammar_parser::parse(longer_grammar_bytes);

    expected = {
        {"expr", 2},
        {"expr_6", 6},
        {"expr_7", 7},
        {"ident", 8},
        {"ident_10", 10},
        {"num", 9},
        {"num_11", 11},
        {"root", 0},
        {"root_1", 1},
        {"root_5", 5},
        {"term", 4},
        {"ws", 3},
        {"ws_12", 12},
    };

    index = 0;
    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it)
    {
        std::string key = it->first;
        uint32_t value = it->second;
        std::pair<std::string, uint32_t> expected_pair = expected[index];

        // pretty print error message before asserting
        if (expected_pair.first != key || expected_pair.second != value)
        {
            fprintf(stderr, "expected_pair: %s, %u\n", expected_pair.first.c_str(), expected_pair.second);
            fprintf(stderr, "actual_pair: %s, %u\n", key.c_str(), value);
            fprintf(stderr, "expected_pair != actual_pair\n");
        }

        assert(expected_pair.first == key && expected_pair.second == value);

        index++;
    }
    expected_rules = {
        {LLAMA_GRETYPE_RULE_REF, 5},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 2},
        {LLAMA_GRETYPE_CHAR, 61},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_RULE_REF, 4},
        {LLAMA_GRETYPE_CHAR, 10},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 4},
        {LLAMA_GRETYPE_RULE_REF, 7},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 12},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 8},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_RULE_REF, 9},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR, 40},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_RULE_REF, 2},
        {LLAMA_GRETYPE_CHAR, 41},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 1},
        {LLAMA_GRETYPE_RULE_REF, 5},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_RULE_REF, 1},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 45},
        {LLAMA_GRETYPE_CHAR_ALT, 43},
        {LLAMA_GRETYPE_CHAR_ALT, 42},
        {LLAMA_GRETYPE_CHAR_ALT, 47},
        {LLAMA_GRETYPE_RULE_REF, 4},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 6},
        {LLAMA_GRETYPE_RULE_REF, 7},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 97},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
        {LLAMA_GRETYPE_RULE_REF, 10},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 11},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 97},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
        {LLAMA_GRETYPE_CHAR_ALT, 48},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
        {LLAMA_GRETYPE_CHAR_ALT, 95},
        {LLAMA_GRETYPE_RULE_REF, 10},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 48},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
        {LLAMA_GRETYPE_RULE_REF, 11},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR, 48},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 32},
        {LLAMA_GRETYPE_CHAR_ALT, 9},
        {LLAMA_GRETYPE_CHAR_ALT, 10},
        {LLAMA_GRETYPE_RULE_REF, 12},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    };

    index = 0;
    for (auto rule : parsed_grammar.rules)
    {
        // compare rule to expected rule
        for (uint32_t i = 0; i < rule.size(); i++)
        {
            llama_grammar_element element = rule[i];
            llama_grammar_element expected_element = expected_rules[index];

            // pretty print error message before asserting
            if (expected_element.type != element.type || expected_element.value != element.value)
            {
                fprintf(stderr, "index: %u\n", index);
                fprintf(stderr, "expected_element: %d, %u\n", expected_element.type, expected_element.value);
                fprintf(stderr, "actual_element: %d, %u\n", element.type, element.value);
                fprintf(stderr, "expected_element != actual_element\n");
            }

            assert(expected_element.type == element.type && expected_element.value == element.value);
            index++;
        }
    }

    return 0;
}
