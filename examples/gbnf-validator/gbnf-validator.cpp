#define LLAMA_API_INTERNAL

#include "grammar-parser.h"
#include "ggml.h"
#include "llama.h"
#include "unicode.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

static bool llama_sample_grammar_string(struct llama_grammar * grammar, const std::string & input_str, size_t & error_pos, std::string & error_msg) {
    auto decoded = decode_utf8(input_str, {});
    const auto & code_points = decoded.first;

    size_t pos = 0;
    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        auto prev_stacks = grammar->stacks;
        grammar->stacks = llama_grammar_accept(grammar->rules, grammar->stacks, *it);
        if (grammar->stacks.empty()) {
            error_pos = pos;
            error_msg = "Unexpected character '" + unicode_cpt_to_utf8(*it) + "'";
            grammar->stacks = prev_stacks;
            return false;
        }
        ++pos;
    }

    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            return true;
        }
    }

    error_pos = pos;
    error_msg = "Unexpected end of input";
    return false;
}

static void print_error_message(const std::string & input_str, size_t error_pos, const std::string & error_msg) {
    std::cout << "Input string is invalid according to the grammar." << std::endl;
    std::cout << "Error: " << error_msg << " at position " << std::to_string(error_pos) << std::endl;
    std::cout << std::endl;
    std::cout << "Input string:" << std::endl;
    std::cout << input_str.substr(0, error_pos);
    if (error_pos < input_str.size()) {
        std::cout << "\033[1;31m" << input_str[error_pos];
        if (error_pos+1 < input_str.size()) {
            std::cout << "\033[0;31m" << input_str.substr(error_pos+1);
        }
        std::cout << "\033[0m" << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <grammar_file> <input_file>" << std::endl;
        return 1;
    }

    const std::string grammar_file = argv[1];
    const std::string input_file = argv[2];

    // Read the GBNF grammar file
    std::ifstream grammar_stream(grammar_file);
    if (!grammar_stream.is_open()) {
        std::cerr << "Failed to open grammar file: " << grammar_file << std::endl;
        return 1;
    }

    std::string grammar_str((std::istreambuf_iterator<char>(grammar_stream)), std::istreambuf_iterator<char>());
    grammar_stream.close();

    // Parse the GBNF grammar
    auto parsed_grammar = grammar_parser::parse(grammar_str.c_str());

    // will be empty (default) if there are parse errors
    if (parsed_grammar.rules.empty()) {
        fprintf(stderr, "%s: failed to parse grammar\n", __func__);
        return 1;
    }

    // Ensure that there is a "root" node.
    if (parsed_grammar.symbol_ids.find("root") == parsed_grammar.symbol_ids.end()) {
        fprintf(stderr, "%s: grammar does not contain a 'root' symbol\n", __func__);
        return 1;
    }

    std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());

    // Create the LLAMA grammar
    auto grammar = llama_grammar_init(
            grammar_rules.data(),
            grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));

    // Read the input file
    std::ifstream input_stream(input_file);
    if (!input_stream.is_open()) {
        std::cerr << "Failed to open input file: " << input_file << std::endl;
        return 1;
    }

    std::string input_str((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
    input_stream.close();

    // Validate the input string against the grammar
    size_t error_pos;
    std::string error_msg;
    bool is_valid = llama_sample_grammar_string(grammar, input_str, error_pos, error_msg);

    if (is_valid) {
        std::cout << "Input string is valid according to the grammar." << std::endl;
    } else {
        print_error_message(input_str, error_pos, error_msg);
    }

    // Clean up
    llama_grammar_free(grammar);

    return 0;
}