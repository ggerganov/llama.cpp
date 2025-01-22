#pragma once

#include "ggml.h"
#include "common.h"
#include "chat-template.hpp"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

enum common_tool_call_style {
    UnknownToolCallStyle,
    None,
    Generic,
    Llama31,
    Llama32,
    FunctionaryV3Llama3,
    FunctionaryV3Llama31,
    Hermes2Pro,
    CommandRPlus,
    MistralNemo,
    FirefunctionV2,
};

struct common_tool_call {
    std::string name;
    std::string arguments;
    std::string id;
};

struct common_tool_calls {
    std::string content;
    std::vector<common_tool_call> tool_calls;
};

struct common_tool_call_handler {
    std::string prompt;
    std::string grammar;
    std::vector<std::string> grammar_triggers;
    std::vector<std::string> additional_stops;
};

std::string common_tool_call_style_name(common_tool_call_style style);

common_tool_call_style common_tool_call_style_detect(const common_chat_template & chat_template);

common_tool_calls parse_tool_calls(common_tool_call_style style, const nlohmann::ordered_json & tools, const std::string& input);

common_tool_call_handler common_tool_call_handler_init(
    common_tool_call_style style,
    const common_chat_template & tmpl,
    bool allow_content,
    const nlohmann::ordered_json & parallel_tool_calls,
    const nlohmann::ordered_json & messages,
    const nlohmann::ordered_json & tools,
    const nlohmann::ordered_json & json_schema = {});
