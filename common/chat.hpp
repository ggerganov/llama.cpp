// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include "common.h"
#include <json.hpp>
#include <optional>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct common_chat_inputs {
    json messages;
    json tools;
    json tool_choice;
    json json_schema;
    bool parallel_tool_calls;
    bool stream;
    std::string grammar;
    bool add_generation_prompt = true;
};

typedef std::function<common_chat_msg(const std::string & input)> common_chat_parser;

struct common_chat_params {
    json prompt;
    std::string grammar;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string> additional_stops;// std::unique_ptr<class common_chat_parser> parser;
    common_chat_parser parser;
    std::string format; // For debugging and testing.
    bool grammar_lazy = false;
};

struct common_chat_params common_chat_params_init(const common_chat_template & tmpl, const struct common_chat_inputs & params);
