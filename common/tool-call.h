#pragma once

#include "ggml.h"
#include "common.h"
#include "chat-template.hpp"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

enum llama_tool_call_style {
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
};

struct llama_tool_call {
    std::string name;
    std::string arguments;
    std::string id;
};

struct llama_tool_calls {
    std::string content;
    std::vector<llama_tool_call> tool_calls;
};

struct llama_tool_call_handler {
    std::string prompt;
    std::string grammar;
    std::vector<std::string> grammar_trigger_words;
    std::vector<std::string> additional_stop_words;
};

std::string llama_tool_call_style_name(llama_tool_call_style style);

llama_tool_call_style llama_tool_call_style_detect(const minja::chat_template & chat_template);

llama_tool_calls parse_tool_calls(llama_tool_call_style style, const nlohmann::ordered_json & tools, const std::string& input);

llama_tool_call_handler llama_tool_call_handler_init(
    llama_tool_call_style style,
    const minja::chat_template & tmpl,
    bool allow_content,
    const nlohmann::ordered_json & parallel_tool_calls,
    const nlohmann::ordered_json & messages,
    const nlohmann::ordered_json & tools,
    const nlohmann::ordered_json & json_schema = {});
