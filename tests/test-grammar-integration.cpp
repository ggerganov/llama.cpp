#ifdef NDEBUG
#undef NDEBUG
#endif

#define LLAMA_API_INTERNAL

#include "ggml.h"
#include "llama.h"
#include "grammar-parser.h"
#include "unicode.h"
#include <cassert>
#include <string>

static void test_simple_grammar() {
    // Test case for a simple grammar
    const std::string grammar_str = R"""(root ::= expr
expr ::= term ("+" term)*
term ::= number
number ::= [0-9]+)""";

    grammar_parser::parse_state parsed_grammar = grammar_parser::parse(grammar_str.c_str());

    // Ensure we parsed correctly
    assert(!parsed_grammar.rules.empty());

    // Ensure we have a root node
    assert(!(parsed_grammar.symbol_ids.find("root") == parsed_grammar.symbol_ids.end()));

    std::vector<const llama_grammar_element*> grammar_rules(parsed_grammar.c_rules());
    llama_grammar* grammar = llama_grammar_init(
        grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));

    std::string input = "123+456";

    auto decoded = decode_utf8(input, {});

    const auto & code_points = decoded.first;

    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        auto prev_stacks = grammar->stacks;
        llama_grammar_accept(grammar->rules, prev_stacks, *it, grammar->stacks);
        assert(!grammar->stacks.empty());
    }

    bool completed_grammar = false;

    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            completed_grammar = true;
            break;
        }
    }

    assert(completed_grammar);

    // Clean up allocated memory
    llama_grammar_free(grammar);
}

static void test_complex_grammar() {
    // Test case for a more complex grammar, with both failure strings and success strings
    const std::string grammar_str = R"""(root ::= expression
expression ::= term ws (("+"|"-") ws term)*
term ::= factor ws (("*"|"/") ws factor)*
factor ::= number | variable | "(" expression ")" | function-call
number ::= [0-9]+
variable ::= [a-zA-Z_][a-zA-Z0-9_]*
function-call ::= variable ws "(" (expression ("," ws expression)*)? ")"
ws ::= [ \t\n\r]?)""";

    grammar_parser::parse_state parsed_grammar = grammar_parser::parse(grammar_str.c_str());

    // Ensure we parsed correctly
    assert(!parsed_grammar.rules.empty());

    // Ensure we have a root node
    assert(!(parsed_grammar.symbol_ids.find("root") == parsed_grammar.symbol_ids.end()));

    std::vector<const llama_grammar_element*> grammar_rules(parsed_grammar.c_rules());
    llama_grammar* grammar = llama_grammar_init(
        grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));

    // Save the original grammar stacks so that we can reset after every new string we want to test
    auto original_stacks = grammar->stacks;

    // Test a few strings
    std::vector<std::string> test_strings_pass = {
        "42",
        "1*2*3*4*5",
        "x",
        "x+10",
        "x1+y2",
        "(a+b)*(c-d)",
        "func()",
        "func(x,y+2)",
        "a*(b+c)-d/e",
        "f(g(x),h(y,z))",
        "x + 10",
        "x1 + y2",
        "(a + b) * (c - d)",
        "func()",
        "func(x, y + 2)",
        "a * (b + c) - d / e",
        "f(g(x), h(y, z))",
        "123+456",
        "123*456*789-123/456+789*123",
        "123+456*789-123/456+789*123-456/789+123*456-789/123+456*789-123/456+789*123-456"
    };

    std::vector<std::string> test_strings_fail = {
        "+",
        "/ 3x",
        "x + + y",
        "a * / b",
        "func(,)",
        "func(x y)",
        "(a + b",
        "x + y)",
        "a + b * (c - d",
        "42 +",
        "x +",
        "x + 10 +",
        "(a + b) * (c - d",
        "func(",
        "func(x, y + 2",
        "a * (b + c) - d /",
        "f(g(x), h(y, z)",
        "123+456*789-123/456+789*123-456/789+123*456-789/123+456*789-123/456+789*123-456/",
    };

    // Passing strings
    for (const auto & test_string : test_strings_pass) {
        auto decoded = decode_utf8(test_string, {});

        const auto & code_points = decoded.first;

        int pos = 0;
        for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
            ++pos;
            auto prev_stacks = grammar->stacks;
            llama_grammar_accept(grammar->rules, prev_stacks, *it, grammar->stacks);

            // Expect that each code point will not cause the grammar to fail
            if (grammar->stacks.empty()) {
                fprintf(stdout, "Error at position %d\n", pos);
                fprintf(stderr, "Unexpected character '%s'\n", unicode_cpt_to_utf8(*it).c_str());
                fprintf(stderr, "Input string is %s:\n", test_string.c_str());
            }
            assert(!grammar->stacks.empty());
        }

        bool completed_grammar = false;

        for (const auto & stack : grammar->stacks) {
            if (stack.empty()) {
                completed_grammar = true;
                break;
            }
        }

        assert(completed_grammar);

        // Reset the grammar stacks
        grammar->stacks = original_stacks;
    }

    // Failing strings
    for (const auto & test_string : test_strings_fail) {
        auto decoded = decode_utf8(test_string, {});

        const auto & code_points = decoded.first;
        bool parse_failed = false;

        for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
            auto prev_stacks = grammar->stacks;
            llama_grammar_accept(grammar->rules, prev_stacks, *it, grammar->stacks);
            if (grammar->stacks.empty()) {
                parse_failed = true;
                break;
            }
            assert(!grammar->stacks.empty());
        }

        bool completed_grammar = false;

        for (const auto & stack : grammar->stacks) {
            if (stack.empty()) {
                completed_grammar = true;
                break;
            }
        }

        // Ensure that the grammar is not completed, or that each string failed to match as-expected
        assert((!completed_grammar) || parse_failed);

        // Reset the grammar stacks
        grammar->stacks = original_stacks;
    }

    // Clean up allocated memory
    llama_grammar_free(grammar);
}

static void test_failure_missing_root() {
    // Test case for a grammar that is missing a root rule
    const std::string grammar_str = R"""(rot ::= expr
expr ::= term ("+" term)*
term ::= number
number ::= [0-9]+)""";

    grammar_parser::parse_state parsed_grammar = grammar_parser::parse(grammar_str.c_str());

    // Ensure we parsed correctly
    assert(!parsed_grammar.rules.empty());

    // Ensure we do NOT have a root node
    assert(parsed_grammar.symbol_ids.find("root") == parsed_grammar.symbol_ids.end());
}

static void test_failure_missing_reference() {
    // Test case for a grammar that is missing a referenced rule
    const std::string grammar_str = R"""(root ::= expr
expr ::= term ("+" term)*
term ::= numero
number ::= [0-9]+)""";

    fprintf(stderr, "Expected error:  ");

    grammar_parser::parse_state parsed_grammar = grammar_parser::parse(grammar_str.c_str());

    // Ensure we did NOT parsed correctly
    assert(parsed_grammar.rules.empty());

    fprintf(stderr, "End of expected error. Test successful.\n");
}

int main() {
    test_simple_grammar();
    test_complex_grammar();
    test_failure_missing_root();
    test_failure_missing_reference();
    return 0;
}
