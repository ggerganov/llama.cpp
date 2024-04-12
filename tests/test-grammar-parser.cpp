#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"
#include "grammar-parser.h"

#include <cassert>

static const char * type_str(llama_gretype type) {
    switch (type) {
        case LLAMA_GRETYPE_CHAR: return "LLAMA_GRETYPE_CHAR";
        case LLAMA_GRETYPE_CHAR_NOT: return "LLAMA_GRETYPE_CHAR_NOT";
        case LLAMA_GRETYPE_CHAR_ALT: return "LLAMA_GRETYPE_CHAR_ALT";
        case LLAMA_GRETYPE_CHAR_RNG_UPPER: return "LLAMA_GRETYPE_CHAR_RNG_UPPER";
        case LLAMA_GRETYPE_RULE_REF: return "LLAMA_GRETYPE_RULE_REF";
        case LLAMA_GRETYPE_ALT: return "LLAMA_GRETYPE_ALT";
        case LLAMA_GRETYPE_END: return "LLAMA_GRETYPE_END";
        default: return "?";
    }
}

static void verify_parsing(const char *grammar_bytes, const std::vector<std::pair<std::string, uint32_t>> expected, const std::vector<llama_grammar_element> &expected_rules) {
    uint32_t index = 0;
    grammar_parser::parse_state parsed_grammar = grammar_parser::parse(grammar_bytes);

    auto print_all = [&]() {
        fprintf(stderr, "Code to update expectation:\n");
        fprintf(stderr, "    verify_parsing(R\"\"\"(%s)\"\"\", {\n", grammar_bytes);
        for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it) {
            fprintf(stderr, "        {\"%s\", %u},\n", it->first.c_str(), it->second);
        }
        fprintf(stderr, "    }, {\n");
        for (auto rule : parsed_grammar.rules) {
            for (uint32_t i = 0; i < rule.size(); i++) {
                fprintf(stderr, "        {%s, %u},\n", type_str(rule[i].type), rule[i].value);
            }
        }
        fprintf(stderr, "    });\n");
    };
    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it)
    {
        std::string key = it->first;
        uint32_t value = it->second;
        std::pair<std::string, uint32_t> expected_pair = expected[index];

        // pretty print error message before asserting
        if (expected_pair.first != key || expected_pair.second != value)
        {
            fprintf(stderr, "index: %u\n", index);
            fprintf(stderr, "expected_pair: %s, %u\n", expected_pair.first.c_str(), expected_pair.second);
            fprintf(stderr, "actual_pair: %s, %u\n", key.c_str(), value);
            fprintf(stderr, "expected_pair != actual_pair\n");
            print_all();
        }

        assert(expected_pair.first == key && expected_pair.second == value);

        index++;
    }

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
                fprintf(stderr, "expected_element: %s, %u\n", type_str(expected_element.type), expected_element.value);
                fprintf(stderr, "actual_element: %s, %u\n", type_str(element.type), element.value);
                fprintf(stderr, "expected_element != actual_element\n");
                fprintf(stderr, "all elements:\n");
                print_all();
            }

            assert(expected_element.type == element.type && expected_element.value == element.value);
            index++;
        }
    }
}

int main()
{
    verify_parsing(R"""(
        root  ::= "a"
    )""", {
        {"root", 0},
    }, {
        {LLAMA_GRETYPE_CHAR, 97},
        {LLAMA_GRETYPE_END, 0},

    });

    verify_parsing(R"""(
        root  ::= "a" | [bdx-z] | [^1-3]
    )""", {
        {"root", 0},
    }, {
        {LLAMA_GRETYPE_CHAR, 97},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR, 98},
        {LLAMA_GRETYPE_CHAR_ALT, 100},
        {LLAMA_GRETYPE_CHAR_ALT, 120},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR_NOT, 49},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 51},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"+
    )""", {
        {"root", 0},
        {"root_1", 1},
        {"root_2", 2},
        {"root_star_3", 3},
    }, {
        {LLAMA_GRETYPE_RULE_REF, 2},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 97},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 1},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 1},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= (expr "=" term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= [0-9]+
    )""", {
        {"expr", 2},
        {"expr_6", 6},
        {"expr_7", 7},
        {"expr_star_8", 8},
        {"root", 0},
        {"root_1", 1},
        {"root_4", 4},
        {"root_star_5", 5},
        {"term", 3},
        {"term_10", 10},
        {"term_9", 9},
        {"term_star_11", 11},
    }, {
        {LLAMA_GRETYPE_RULE_REF, 4},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 2},
        {LLAMA_GRETYPE_CHAR, 61},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_CHAR, 10},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_RULE_REF, 7},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 10},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 1},
        {LLAMA_GRETYPE_RULE_REF, 5},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 1},
        {LLAMA_GRETYPE_RULE_REF, 5},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 45},
        {LLAMA_GRETYPE_CHAR_ALT, 43},
        {LLAMA_GRETYPE_CHAR_ALT, 42},
        {LLAMA_GRETYPE_CHAR_ALT, 47},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 8},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 6},
        {LLAMA_GRETYPE_RULE_REF, 8},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 48},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 9},
        {LLAMA_GRETYPE_RULE_REF, 11},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 9},
        {LLAMA_GRETYPE_RULE_REF, 11},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= (expr "=" ws term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= ident | num | "(" ws expr ")" ws
        ident ::= [a-z] [a-z0-9_]* ws
        num   ::= [0-9]+ ws
        ws    ::= [ \t\n]*
    )""", {
        {"expr", 2},
        {"expr_7", 7},
        {"expr_8", 8},
        {"expr_star_9", 9},
        {"ident", 10},
        {"ident_12", 12},
        {"ident_13", 13},
        {"ident_star_14", 14},
        {"num", 11},
        {"num_15", 15},
        {"num_16", 16},
        {"num_star_17", 17},
        {"root", 0},
        {"root_1", 1},
        {"root_5", 5},
        {"root_star_6", 6},
        {"term", 4},
        {"ws", 3},
        {"ws_18", 18},
        {"ws_19", 19},
        {"ws_star_20", 20},
    }, {
        {LLAMA_GRETYPE_RULE_REF, 5},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 2},
        {LLAMA_GRETYPE_CHAR, 61},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_RULE_REF, 4},
        {LLAMA_GRETYPE_CHAR, 10},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 4},
        {LLAMA_GRETYPE_RULE_REF, 8},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 19},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 10},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_RULE_REF, 11},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR, 40},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_RULE_REF, 2},
        {LLAMA_GRETYPE_CHAR, 41},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 1},
        {LLAMA_GRETYPE_RULE_REF, 6},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 1},
        {LLAMA_GRETYPE_RULE_REF, 6},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 45},
        {LLAMA_GRETYPE_CHAR_ALT, 43},
        {LLAMA_GRETYPE_CHAR_ALT, 42},
        {LLAMA_GRETYPE_CHAR_ALT, 47},
        {LLAMA_GRETYPE_RULE_REF, 4},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 9},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 7},
        {LLAMA_GRETYPE_RULE_REF, 9},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 97},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
        {LLAMA_GRETYPE_RULE_REF, 13},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 16},
        {LLAMA_GRETYPE_RULE_REF, 3},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 97},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
        {LLAMA_GRETYPE_CHAR_ALT, 48},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
        {LLAMA_GRETYPE_CHAR_ALT, 95},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 14},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 12},
        {LLAMA_GRETYPE_RULE_REF, 14},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 48},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 15},
        {LLAMA_GRETYPE_RULE_REF, 17},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 15},
        {LLAMA_GRETYPE_RULE_REF, 17},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_CHAR, 32},
        {LLAMA_GRETYPE_CHAR_ALT, 9},
        {LLAMA_GRETYPE_CHAR_ALT, 10},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 20},
        {LLAMA_GRETYPE_END, 0},
        {LLAMA_GRETYPE_RULE_REF, 18},
        {LLAMA_GRETYPE_RULE_REF, 20},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    return 0;
}
