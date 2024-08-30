#pragma once

#include "llama.h"
#include "common.h"
#include "utils.hpp"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

#include <string>
#include <vector>
#include <sstream>

using json = nlohmann::ordered_json;

enum llama_tool_format {
    LLAMA_TOOL_FORMAT_NOT_SUPPORTED,
    LLAMA_TOOL_FORMAT_HERMES_3,
};

enum llama_response_state {
    LLAMA_RESPONSE_STATE_UNKNOWN,
    LLAMA_RESPONSE_STATE_TEXT,
    LLAMA_RESPONSE_STATE_TOOL_CALL,
};

// get the tool call format for the loaded model
// this function does linear search, so do not call it repeatedly
inline enum llama_tool_format get_tool_format(const struct llama_context * ctx) {
    auto model = llama_get_model(ctx);
    auto has_token = [&](std::string piece) {
        for (int i = 0; i < llama_n_vocab(model); i++) {
            const std::string token_str = llama_token_to_piece(ctx, i, true);
            if (token_str == piece) {
                return true;
            }
        }
        return false;
    };
    if (has_token("<|im_start|>") && has_token("<tool_call>")) {
        return LLAMA_TOOL_FORMAT_HERMES_3;
    }
    return LLAMA_TOOL_FORMAT_NOT_SUPPORTED;
}

inline std::string format_chat_with_tool(enum llama_tool_format format, const std::vector<json> & messages, json tools) {
    if (!tools.is_array()) {
        throw std::runtime_error("tools must be an array");
    }
    std::stringstream ss;
    auto chat = parse_chat_messages(messages);
    if (format == LLAMA_TOOL_FORMAT_HERMES_3) {
        ss << "<|im_start|>system\n\n";
        ss << "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools>\n\n";
        for (auto tool : tools) {
            ss << tool.dump(1, '\t') << "\n\n";
        }
        ss << "</tools> Use the following pydantic model json schema for each tool call you will make: {\"properties\": {\"arguments\": {\"title\": \"Arguments\", \"type\": \"object\"}, \"name\": {\"title\": \"Name\", \"type\": \"string\"}}, \"required\": [\"arguments\", \"name\"], \"title\": \"FunctionCall\", \"type\": \"object\"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n";
        ss << "<tool_call>\n";
        ss << "{\"arguments\": <args-dict>, \"name\": <function-name>}\n";
        ss << "</tool_call><|im_end|>\n";
        for (auto & message : chat) {
            std::string role(message.role);
            if (role == "system") {
                continue; // for optimal performance, we skip user-defined system message
            }
            ss << "<|im_start|>" << role << "\n\n";
            if (role == "tool") {
                ss << "<tool_response>\n" << string_strip(message.content) << "\n</tool_response>\n";
            } else {
                ss << string_strip(message.content) << "<|im_end|>\n";
            }
        }
        ss << "<|im_start|>assistant\n\n";
    } else {
        throw std::runtime_error("tool_call is not supported by this model");
    }
    LOG_VERBOSE("format_chat_with_tool", {{"text", ss.str()}});
    return ss.str();
}

// check if the response is text or tool_call
// if it is tool_call, we may have to disable streaming, because we must parse the whole JSON response
inline enum llama_response_state check_response_state(enum llama_tool_format format, const std::string & generated_text) {
    if (format == LLAMA_TOOL_FORMAT_NOT_SUPPORTED) {
        return LLAMA_RESPONSE_STATE_TEXT;
    } else if (format == LLAMA_TOOL_FORMAT_HERMES_3 && generated_text.rfind("<tool_call>", 0) == 0) {
        return LLAMA_RESPONSE_STATE_TOOL_CALL;
    }
    return LLAMA_RESPONSE_STATE_TEXT;
}

// convert model's response to OAI format
inline json parse_tool_response(enum llama_tool_format format, const std::string & generated_text) {
    if (format == LLAMA_TOOL_FORMAT_NOT_SUPPORTED) {
        return json{};
    } else if (format == LLAMA_TOOL_FORMAT_HERMES_3) {
        std::string tmp(generated_text);
        string_replace_all(tmp, "<tool_call>", "");
        string_replace_all(tmp, "</tool_call>", "");
        json tool = json::parse(tmp);
        std::vector<json> tool_calls = {json{
            {"id",       tool.at("name")},
            {"type",     "function"},
            {"function", {
                {"name",      tool.at("name")},
                {"arguments", tool.at("arguments").dump()}, // OAI requires this to be JSON-stringified
            }},
        }};
        return tool_calls;
    }
    return generated_text;
}
