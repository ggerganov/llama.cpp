/*
    Copyright 2024 Google LLC

    Use of this source code is governed by an MIT-style
    license that can be found in the LICENSE file or at
    https://opensource.org/licenses/MIT.
*/
// SPDX-License-Identifier: MIT
#pragma once

#include "common.h"
#include <json.hpp>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct common_chat_params {
    json messages;
    json tools;
    json tool_choice;
    json json_schema;
    bool parallel_tool_calls;
    bool stream;
    std::string grammar;
    bool add_generation_prompt = true;
};

class common_chat_parser {
public:
    virtual ~common_chat_parser() = default;

    virtual std::optional<common_chat_msg> parse_partial(const std::string & input) = 0;
    virtual common_chat_msg parse_final(const std::string & input) = 0;
    virtual std::unique_ptr<common_chat_parser> clone() const = 0;
};

struct common_chat_data {
    json prompt;
    std::string grammar;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string> additional_stops;
    std::unique_ptr<class common_chat_parser> parser;
};

struct common_chat_data common_chat_init(const common_chat_template & tmpl, const struct common_chat_params & params);
