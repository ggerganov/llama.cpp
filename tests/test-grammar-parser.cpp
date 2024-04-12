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
        fprintf(stderr, "    verify_parsing(R\"\"\"(%s)\"\"\", {\n", grammar_bytes);
        for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it) {
            fprintf(stderr, "        {\"%s\", %u},\n", it->first.c_str(), it->second);
        }
        fprintf(stderr, "    }, {\n");
        for (size_t i_rule = 0; i_rule < parsed_grammar.rules.size(); i_rule++) {
            fprintf(stderr, "        // %s (index %zu)\n", expected[i_rule].first.c_str(), i_rule);
            auto & rule = parsed_grammar.rules[i_rule];
            for (uint32_t i = 0; i < rule.size(); i++) {
                std::string rule_str;
                fprintf(stderr, "        {%s, ", type_str(rule[i].type));
                if (rule[i].type == LLAMA_GRETYPE_CHAR || rule[i].type == LLAMA_GRETYPE_CHAR_ALT ||
                    rule[i].type == LLAMA_GRETYPE_CHAR_NOT || rule[i].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                    char c = rule[i].value;
                    if (c == '\n') {
                        fprintf(stderr, "'\\n'");
                    } else if (c == '\t') {
                        fprintf(stderr, "'\\t'");
                    } else if (c == '\r') {
                        fprintf(stderr, "'\\r'");
                    } else if (c == '\0') {
                        fprintf(stderr, "'\\0'");
                    } else {
                        fprintf(stderr, "'%c'", c);
                    }
                } else if (rule[i].type == LLAMA_GRETYPE_RULE_REF) {
                    fprintf(stderr, "/* %s */ %u", expected[rule[i].value].first.c_str(), rule[i].value);
                } else {
                    fprintf(stderr, "%u", rule[i].value);
                }
                fprintf(stderr, "},\n");
            }
        }
        fprintf(stderr, "    });\n");
    };

    if (getenv("TEST_GRAMMAR_PARSER_PRINT_ALL")) {
        print_all();
        fprintf(stderr, "\n");
        return;
    }

    fprintf(stderr, "Testing grammar:%s\n", grammar_bytes);

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
            fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
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
                fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
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
        // root (index 0)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a" | [bdx-z] | [^1-3]
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR, 'b'},
        {LLAMA_GRETYPE_CHAR_ALT, 'd'},
        {LLAMA_GRETYPE_CHAR_ALT, 'x'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR_NOT, '1'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '3'},
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
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_star_3 */ 3},
        {LLAMA_GRETYPE_END, 0},
        // root_star_3 (index 3)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_star_3 */ 3},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"?
    )""", {
        {"root", 0},
        {"root_1", 1},
        {"root_1_3", 3},
        {"root_2", 2},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_1_3 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // root_1_3 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 3},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 3)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"*
    )""", {
        {"root", 0},
        {"root_1", 1},
        {"root_2", 2},
        {"root_star_3", 3},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* root_star_3 */ 3},
        {LLAMA_GRETYPE_END, 0},
        // root_star_3 (index 3)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_star_3 */ 3},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2}
    )""", {
        {"root", 0},
        {"root_1", 1},
        {"root_2", 2},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2,}
    )""", {
        {"root", 0},
        {"root_1", 1},
        {"root_2", 2},
        {"root_star_3", 3},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_star_3 */ 3},
        {LLAMA_GRETYPE_END, 0},
        // root_star_3 (index 3)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_star_3 */ 3},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{ 4}
    )""", {
        {"root", 0},
        {"root_1", 1},
        {"root_2", 2},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2,4}
    )""", {
        {"root", 0},
        {"root_1", 1},
        {"root_1_3", 3},
        {"root_2", 2},
        {"root_2_4", 4},
    }, {
        // root (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root_1_3 */ 2},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 1)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_END, 0},
        // root_1_3 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_2_4 */ 4},
        {LLAMA_GRETYPE_END, 0},
        // root_2 (index 3)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // root_2_4 (index 4)
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_2 */ 3},
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
        // expr (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* root */ 4},
        {LLAMA_GRETYPE_END, 0},
        // expr_6 (index 1)
        {LLAMA_GRETYPE_RULE_REF, /* expr_7 */ 2},
        {LLAMA_GRETYPE_CHAR, '='},
        {LLAMA_GRETYPE_RULE_REF, /* expr_star_8 */ 3},
        {LLAMA_GRETYPE_CHAR, '\n'},
        {LLAMA_GRETYPE_END, 0},
        // expr_7 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* expr_star_8 */ 3},
        {LLAMA_GRETYPE_RULE_REF, /* root_star_5 */ 7},
        {LLAMA_GRETYPE_END, 0},
        // expr_star_8 (index 3)
        {LLAMA_GRETYPE_RULE_REF, /* term_9 */ 10},
        {LLAMA_GRETYPE_END, 0},
        // root (index 4)
        {LLAMA_GRETYPE_RULE_REF, /* expr_6 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 5},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 5)
        {LLAMA_GRETYPE_RULE_REF, /* expr_6 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 5},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // root_4 (index 6)
        {LLAMA_GRETYPE_CHAR, '-'},
        {LLAMA_GRETYPE_CHAR_ALT, '+'},
        {LLAMA_GRETYPE_CHAR_ALT, '*'},
        {LLAMA_GRETYPE_CHAR_ALT, '/'},
        {LLAMA_GRETYPE_RULE_REF, /* expr_star_8 */ 3},
        {LLAMA_GRETYPE_END, 0},
        // root_star_5 (index 7)
        {LLAMA_GRETYPE_RULE_REF, /* term */ 8},
        {LLAMA_GRETYPE_END, 0},
        // term (index 8)
        {LLAMA_GRETYPE_RULE_REF, /* root_4 */ 6},
        {LLAMA_GRETYPE_RULE_REF, /* term */ 8},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // term_10 (index 9)
        {LLAMA_GRETYPE_CHAR, '0'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
        {LLAMA_GRETYPE_END, 0},
        // term_9 (index 10)
        {LLAMA_GRETYPE_RULE_REF, /* term_10 */ 9},
        {LLAMA_GRETYPE_RULE_REF, /* term_star_11 */ 11},
        {LLAMA_GRETYPE_END, 0},
        // term_star_11 (index 11)
        {LLAMA_GRETYPE_RULE_REF, /* term_10 */ 9},
        {LLAMA_GRETYPE_RULE_REF, /* term_star_11 */ 11},
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
        // expr (index 0)
        {LLAMA_GRETYPE_RULE_REF, /* ident_12 */ 5},
        {LLAMA_GRETYPE_END, 0},
        // expr_7 (index 1)
        {LLAMA_GRETYPE_RULE_REF, /* expr_8 */ 2},
        {LLAMA_GRETYPE_CHAR, '='},
        {LLAMA_GRETYPE_RULE_REF, /* expr_star_9 */ 3},
        {LLAMA_GRETYPE_RULE_REF, /* ident */ 4},
        {LLAMA_GRETYPE_CHAR, '\n'},
        {LLAMA_GRETYPE_END, 0},
        // expr_8 (index 2)
        {LLAMA_GRETYPE_RULE_REF, /* ident */ 4},
        {LLAMA_GRETYPE_RULE_REF, /* num */ 8},
        {LLAMA_GRETYPE_END, 0},
        // expr_star_9 (index 3)
        {LLAMA_GRETYPE_RULE_REF, /* ws_19 */ 19},
        {LLAMA_GRETYPE_END, 0},
        // ident (index 4)
        {LLAMA_GRETYPE_RULE_REF, /* num_16 */ 10},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_RULE_REF, /* num_star_17 */ 11},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_CHAR, '('},
        {LLAMA_GRETYPE_RULE_REF, /* expr_star_9 */ 3},
        {LLAMA_GRETYPE_RULE_REF, /* expr_8 */ 2},
        {LLAMA_GRETYPE_CHAR, ')'},
        {LLAMA_GRETYPE_RULE_REF, /* expr_star_9 */ 3},
        {LLAMA_GRETYPE_END, 0},
        // ident_12 (index 5)
        {LLAMA_GRETYPE_RULE_REF, /* expr_7 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* ident_13 */ 6},
        {LLAMA_GRETYPE_END, 0},
        // ident_13 (index 6)
        {LLAMA_GRETYPE_RULE_REF, /* expr_7 */ 1},
        {LLAMA_GRETYPE_RULE_REF, /* ident_13 */ 6},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // ident_star_14 (index 7)
        {LLAMA_GRETYPE_CHAR, '-'},
        {LLAMA_GRETYPE_CHAR_ALT, '+'},
        {LLAMA_GRETYPE_CHAR_ALT, '*'},
        {LLAMA_GRETYPE_CHAR_ALT, '/'},
        {LLAMA_GRETYPE_RULE_REF, /* ident */ 4},
        {LLAMA_GRETYPE_END, 0},
        // num (index 8)
        {LLAMA_GRETYPE_RULE_REF, /* num_15 */ 9},
        {LLAMA_GRETYPE_END, 0},
        // num_15 (index 9)
        {LLAMA_GRETYPE_RULE_REF, /* ident_star_14 */ 7},
        {LLAMA_GRETYPE_RULE_REF, /* num_15 */ 9},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // num_16 (index 10)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {LLAMA_GRETYPE_RULE_REF, /* root_1 */ 13},
        {LLAMA_GRETYPE_RULE_REF, /* expr_star_9 */ 3},
        {LLAMA_GRETYPE_END, 0},
        // num_star_17 (index 11)
        {LLAMA_GRETYPE_RULE_REF, /* term */ 16},
        {LLAMA_GRETYPE_RULE_REF, /* expr_star_9 */ 3},
        {LLAMA_GRETYPE_END, 0},
        // root (index 12)
        {LLAMA_GRETYPE_CHAR, 'a'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {LLAMA_GRETYPE_CHAR_ALT, '0'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
        {LLAMA_GRETYPE_CHAR_ALT, '_'},
        {LLAMA_GRETYPE_END, 0},
        // root_1 (index 13)
        {LLAMA_GRETYPE_RULE_REF, /* root_5 */ 14},
        {LLAMA_GRETYPE_END, 0},
        // root_5 (index 14)
        {LLAMA_GRETYPE_RULE_REF, /* root */ 12},
        {LLAMA_GRETYPE_RULE_REF, /* root_5 */ 14},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // root_star_6 (index 15)
        {LLAMA_GRETYPE_CHAR, '0'},
        {LLAMA_GRETYPE_CHAR_RNG_UPPER, '9'},
        {LLAMA_GRETYPE_END, 0},
        // term (index 16)
        {LLAMA_GRETYPE_RULE_REF, /* root_star_6 */ 15},
        {LLAMA_GRETYPE_RULE_REF, /* ws */ 17},
        {LLAMA_GRETYPE_END, 0},
        // ws (index 17)
        {LLAMA_GRETYPE_RULE_REF, /* root_star_6 */ 15},
        {LLAMA_GRETYPE_RULE_REF, /* ws */ 17},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
        // ws_18 (index 18)
        {LLAMA_GRETYPE_CHAR, ' '},
        {LLAMA_GRETYPE_CHAR_ALT, '\t'},
        {LLAMA_GRETYPE_CHAR_ALT, '\n'},
        {LLAMA_GRETYPE_END, 0},
        // ws_19 (index 19)
        {LLAMA_GRETYPE_RULE_REF, /* ws_star_20 */ 20},
        {LLAMA_GRETYPE_END, 0},
        // ws_star_20 (index 20)
        {LLAMA_GRETYPE_RULE_REF, /* ws_18 */ 18},
        {LLAMA_GRETYPE_RULE_REF, /* ws_star_20 */ 20},
        {LLAMA_GRETYPE_ALT, 0},
        {LLAMA_GRETYPE_END, 0},
    });

    return 0;
}
