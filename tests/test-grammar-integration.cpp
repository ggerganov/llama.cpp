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
#include <vector>

static llama_grammar* build_grammar(const std::string & grammar_str) {
    auto parsed_grammar = grammar_parser::parse(grammar_str.c_str());

    // Ensure we parsed correctly
    assert(!parsed_grammar.rules.empty());

    // Ensure we have a root node
    assert(!(parsed_grammar.symbol_ids.find("root") == parsed_grammar.symbol_ids.end()));

    std::vector<const llama_grammar_element*> grammar_rules(parsed_grammar.c_rules());
    llama_grammar* grammar = llama_grammar_init(
        grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));

    return grammar;
}

static bool test_build_grammar_fails(const std::string & grammar_str) {
    fprintf(stderr, "⚫ Testing failure for grammar: %s\n", grammar_str.c_str());
    bool grammar_fails = false;
    try {
        build_grammar(grammar_str);
        fprintf(stderr, "  ❌ Expected build failure, but succeeded\n");
    } catch (const std::exception & err) {
        grammar_fails = true;
        fprintf(stdout, "  ✅︎\n");
    }
    return grammar_fails;
}

static bool match_string(const std::string & input, llama_grammar* grammar) {
    auto decoded = decode_utf8(input, {});

    const auto & code_points = decoded.first;

    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        auto prev_stacks = grammar->stacks;
        llama_grammar_accept(grammar->rules, prev_stacks, *it, grammar->stacks);
        if (grammar->stacks.empty()) {
            // no stacks means that the grammar failed to match at this point
            return false;
        }
    }

    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            // An empty stack means that the grammar has been completed
            return true;
        }
    }

    return false;
}

static void test_grammar(const std::string & test_desc, const std::string & grammar_str, const std::vector<std::string> & passing_strings, const std::vector<std::string> & failing_strings) {
    fprintf(stderr, "⚫ Testing %s. Grammar: %s\n", test_desc.c_str(), grammar_str.c_str());
    fflush(stderr);

    auto grammar = build_grammar(grammar_str);

    // Save the original grammar stacks so that we can reset after every new string we want to test
    auto original_stacks = grammar->stacks;

    fprintf(stderr, "  🔵 Valid strings:\n");

    // Passing strings
    for (const auto & test_string : passing_strings) {
        fprintf(stderr, "    \"%s\" ", test_string.c_str());
        fflush(stderr);

        bool matched = match_string(test_string, grammar);

        if (!matched) {
            fprintf(stderr, "❌ (failed to match)\n");
        } else {
            fprintf(stdout, "✅︎\n");
        }

        assert(matched);

        // Reset the grammar stacks
        grammar->stacks = original_stacks;
    }

    fprintf(stderr, "  🟠 Invalid strings:\n");

    // Failing strings
    for (const auto & test_string : failing_strings) {
        fprintf(stderr, "    \"%s\" ", test_string.c_str());
        fflush(stderr);

        bool matched = match_string(test_string, grammar);

        if (matched) {
            fprintf(stderr, "❌ (incorrectly matched)\n");
        } else {
            fprintf(stdout, "✅︎\n");
        }
        assert(!matched);

        // Reset the grammar stacks
        grammar->stacks = original_stacks;
    }

    // Clean up allocated memory
    llama_grammar_free(grammar);
}

static void test_simple_grammar() {
    // Test case for a simple grammar
    test_grammar(
        "simple grammar",
        R"""(
            root ::= expr
            expr ::= term ("+" term)*
            term ::= number
            number ::= [0-9]+)""",
        // Passing strings
        {
            "42",
            "1+2+3+4+5",
            "123+456",
        },
        // Failing strings
        {
            "+",
            "/ 3",
            "1+2+3+4+5+",
            "12a45",
        }
    );
}

static void test_complex_grammar() {
    // Test case for a more complex grammar, with both failure strings and success strings
    test_grammar(
        "medium complexity grammar",
        // Grammar
        R"""(
            root ::= expression
            expression ::= term ws (("+"|"-") ws term)*
            term ::= factor ws (("*"|"/") ws factor)*
            factor ::= number | variable | "(" expression ")" | function-call
            number ::= [0-9]+
            variable ::= [a-zA-Z_][a-zA-Z0-9_]*
            function-call ::= variable ws "(" (expression ("," ws expression)*)? ")"
            ws ::= [ \t\n\r]?)""",
        // Passing strings
        {
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
        },
        // Failing strings
        {
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
        }
    );
}

static void test_quantifiers() {
    // A collection of tests to exercise * + and ? quantifiers

    test_grammar(
        "* quantifier",
        // Grammar
        R"""(root ::= "a"*)""",
        // Passing strings
        {
            "",
            "a",
            "aaaaa",
            "aaaaaaaaaaaaaaaaaa",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        },
        // Failing strings
        {
            "b",
            "ab",
            "aab",
            "ba",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
        }
    );
    test_grammar(
        "+ quantifier",
        // Grammar
        R"""(root ::= "a"+)""",
        // Passing strings
        {
            "a",
            "aaaaa",
            "aaaaaaaaaaaaaaaaaa",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        },
        // Failing strings
        {
            "",
            "b",
            "ab",
            "aab",
            "ba",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
        }
    );
    test_grammar(
        "? quantifier",
        // Grammar
        R"""(root ::= "a"?)""",
        // Passing strings
        {
            "",
            "a"
        },
        // Failing strings
        {
            "b",
            "ab",
            "aa",
            "ba",
        }
    );
    test_grammar(
        "mixed quantifiers",
        // Grammar
        R"""(
            root ::= cons+ vowel* cons? (vowel cons)*
            vowel ::= [aeiouy]
            cons ::= [bcdfghjklmnpqrstvwxyz]
            )""",
        // Passing strings
        {
            "yes",
            "no",
            "noyes",
            "crwth",
            "four",
            "bryyyy",
        },
        // Failing strings
        {
            "yess",
            "yesno",
            "forty",
            "catyyy",
        }
    );
}

static void test_failure_missing_root() {
    fprintf(stderr, "⚫ Testing missing root node:\n");
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
    fprintf(stderr, "  ✅︎ Passed\n");
}

static void test_failure_missing_reference() {
    fprintf(stderr, "⚫ Testing missing reference node:\n");

    // Test case for a grammar that is missing a referenced rule
    const std::string grammar_str =
R"""(root ::= expr
expr ::= term ("+" term)*
term ::= numero
number ::= [0-9]+)""";

    fprintf(stderr, "    Expected error:  ");

    grammar_parser::parse_state parsed_grammar = grammar_parser::parse(grammar_str.c_str());

    // Ensure we did NOT parsed correctly
    assert(parsed_grammar.rules.empty());

    fprintf(stderr, "    End of expected error.\n");
    fprintf(stderr, "  ✅︎ Passed\n");
}

static void test_failure_left_recursion() {
    fprintf(stderr, "⚫ Testing left recursion detection:\n");

    // Test simple left recursion detection
    const std::string simple_str = R"""(root ::= "a" | root "a")""";
    assert(test_build_grammar_fails(simple_str));

    // Test more complicated left recursion detection
    const std::string medium_str = R"""(
root ::= asdf
asdf ::= "a" | asdf "a"
)""";
    assert(test_build_grammar_fails(medium_str));

    // Test even more complicated left recursion detection
    const std::string hard_str = R"""(
root ::= asdf
asdf ::= "a" | foo "b"
foo ::= "c" | asdf "d" | "e")""";
    assert(test_build_grammar_fails(hard_str));

    // Test yet even more complicated left recursion detection
    const std::string hardest_str = R"""(
root ::= asdf
asdf ::= "a" | foo "b"
foo ::= "c" | empty asdf "d" | "e"
empty ::= "blah" | )""";
    assert(test_build_grammar_fails(hardest_str));

    fprintf(stderr, "  ✅︎ Passed\n");
}

int main() {
    fprintf(stdout, "Running grammar integration tests...\n");
    test_simple_grammar();
    test_complex_grammar();
    test_quantifiers();
    test_failure_missing_root();
    test_failure_missing_reference();
    test_failure_left_recursion();
    fprintf(stdout, "All tests passed.\n");
    return 0;
}
