#pragma once

#include "ggml.h"
#include "common.h"
#include "chat-template.hpp"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

enum common_tool_call_style {
    COMMON_TOOL_CALL_STYLE_UNKNOWN,
    COMMON_TOOL_CALL_STYLE_NONE,
    COMMON_TOOL_CALL_STYLE_GENERIC,
    COMMON_TOOL_CALL_STYLE_LLAMA_3_1,
    COMMON_TOOL_CALL_STYLE_LLAMA_3_2,
    COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3,
    COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3_1,
    COMMON_TOOL_CALL_STYLE_HERMES_2_PRO,
    COMMON_TOOL_CALL_STYLE_COMMAND_R_PLUS,
    COMMON_TOOL_CALL_STYLE_MISTRAL_NEMO,
    COMMON_TOOL_CALL_STYLE_FIRE_FUNCTION_V2,
};

struct common_tool_call_handler {
    std::string prompt;
    std::string grammar;
    std::vector<std::string> grammar_triggers;
    std::vector<std::string> additional_stops;
};

std::string common_tool_call_style_name(common_tool_call_style style);

common_tool_call_style common_tool_call_style_detect(const common_chat_template & chat_template);

common_chat_msg parse_tool_calls(common_tool_call_style style, const nlohmann::ordered_json & tools, const std::string& input);

common_tool_call_handler common_tool_call_handler_init(
    common_tool_call_style style,
    const common_chat_template & tmpl,
    bool allow_content,
    const nlohmann::ordered_json & parallel_tool_calls,
    const nlohmann::ordered_json & messages,
    const nlohmann::ordered_json & tools,
    const nlohmann::ordered_json & json_schema = {});
