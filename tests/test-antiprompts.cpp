#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"
#include "common.h"

#include <cassert>

template <typename T>
void assert_equal(const T & actual, const T & expected) {
    if (expected == actual) return;
    printf("Expected: %s, Actual: %s\n", ((std::string)expected).c_str(), ((std::string)actual).c_str());
    assert(expected == actual);
}

//  cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CURL=1 && cmake --build build -j -t test-jinja -t test-antiprompts  && ./build/bin/test-antiprompts
int main()
{
    auto tokenizer = [&](const std::string & text) {
        std::vector<llama_token> tokens;
        for (size_t i = 0; i < text.length(); ++i) {
            tokens.push_back(text[i]);
        }
        return tokens;
    };
    const std::vector<std::string> stop_words { };
    const std::vector<std::string> grammar_trigger_words { };

    printf("Testing antiprompts\n");

    llama_antiprompts antiprompts;
    antiprompts.build(tokenizer, {"abc", "bcd"}, {"bca", "x"});

    assert_equal(antiprompts.findSingleTokenMatch('x'), {
        /* .pos = */ 0,
        /* .pattern = */ "x",
        /* .is_partial = */ false,
        /* .matchLength = */ 1,
        /* .is_grammar_trigger = */ true,
    });
    assert_equal(antiprompts.findSingleTokenMatch('a'), {
        /* .pos = */ std::string::npos,
        /* .pattern = */ "",
        /* .is_partial = */ false,
        /* .matchLength = */ 0,
        /* .is_grammar_trigger = */ false,
    });
    assert_equal(antiprompts.findFirstMatch(" ab", 0), {
        /* .pos = */ 1,
        /* .pattern = */ "",
        /* .is_partial = */ true,
        /* .matchLength = */ 2,
        /* .is_grammar_trigger = */ false,
    });
    assert_equal(antiprompts.findFirstMatch(" abc", 0), {
        /* .pos = */ 1,
        /* .pattern = */ "abc",
        /* .is_partial = */ false,
        /* .matchLength = */ 3,
        /* .is_grammar_trigger = */ false,
    });
    assert_equal(antiprompts.findFirstMatch(" ab c", 0), {
        /* .pos = */ std::string::npos,
        /* .pattern = */ "",
        /* .is_partial = */ false,
        /* .matchLength = */ 0,
        /* .is_grammar_trigger = */ false,
    });
    assert_equal(antiprompts.findFirstMatch(" abc abc", 0), {
        /* .pos = */ 1,
        /* .pattern = */ "abc",
        /* .is_partial = */ false,
        /* .matchLength = */ 3,
        /* .is_grammar_trigger = */ false,
    });
    assert_equal(antiprompts.findFirstMatch(" ab abc", 0), {
        /* .pos = */ 4,
        /* .pattern = */ "abc",
        /* .is_partial = */ false,
        /* .matchLength = */ 3,
        /* .is_grammar_trigger = */ false,
    });
    assert_equal(antiprompts.findFirstMatch(" bc", 0), {
        /* .pos = */ 1,
        /* .pattern = */ "",
        /* .is_partial = */ true,
        /* .matchLength = */ 2,
        /* .is_grammar_trigger = */ false,
    });
    assert_equal(antiprompts.findFirstMatch(" bcd", 0), {
        /* .pos = */ 1,
        /* .pattern = */ "bcd",
        /* .is_partial = */ false,
        /* .matchLength = */ 3,
        /* .is_grammar_trigger = */ false,
    });
    assert_equal(antiprompts.findFirstMatch(" bca", 0), {
        /* .pos = */ 1,
        /* .pattern = */ "bca",
        /* .is_partial = */ false,
        /* .matchLength = */ 3,
        /* .is_grammar_trigger = */ true,
    });
    printf("OK\n");
    // llama_antiprompts::MatchResult{0, "a", .is_partial = false, . 1, false});

    return 0;
}
