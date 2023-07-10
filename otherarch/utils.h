// Various helper functions and utilities

#pragma once

#include <string>
#include <map>
#include <vector>
#include <random>
#include <thread>
#include "common.h"

//
// CLI argument parsing
//


//
// Vocab utils
//

struct gpt_vocab {
    using id    = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    std::vector<std::string> special_tokens;

    void add_special_token(const std::string & token);
};

void utreplace(std::string & str, const std::string & needle, const std::string & replacement);

// poor-man's JSON parsing
std::map<std::string, int32_t> json_parse(const std::string & fname);

std::string convert_to_utf8(const std::wstring & input);

std::wstring convert_to_wstring(const std::string & input);

void gpt_split_words(std::string str, std::vector<std::string>& words);

// split text into tokens
//
// ref: https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L53
//
// Regex (Python):
// r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
//
// Regex (C++):
// R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"
//
std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab & vocab, const std::string & text);



bool should_transpose_layer(std::string name);

void kcpp_graph_compute_helper(ggml_cgraph * graph, int n_threads);