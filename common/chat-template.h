#pragma once

#include <json.hpp>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

enum llama_tool_call_style {
    Unknown,
    Llama31,
    FunctionaryV3Llama3,
    FunctionaryV3Llama31,
    Hermes2Pro,
};

class llama_chat_template {
  public:

  private:
    llama_tool_call_style _tool_call_style = Unknown;
    bool _supports_tools = true;
    // Meta-Llama-3.1-8B-Instruct's template expects arguments to be an object.
    // Most other templates (and OpenAI's API) expect the arguments object to be stringified.
    bool _requires_object_arguments = false;
    bool _supports_system_role = true;
    std::string _chat_template;
    std::string _bos_token;
    std::string _eos_token;
  public:
    llama_chat_template(const std::string & chat_template, const std::string & bos_token, const std::string & eos_token)
        : _chat_template(chat_template), _bos_token(bos_token), _eos_token(eos_token) {

        _supports_tools = chat_template.find("tools") != std::string::npos;
        _requires_object_arguments = chat_template.find("tool_call.arguments | items") != std::string::npos;
        _supports_system_role = chat_template.find("System role not supported") == std::string::npos;

        if (chat_template.find("<tool_call>") != std::string::npos) {
            _tool_call_style = Hermes2Pro;
        } else if (chat_template.find(">>>all") != std::string::npos) {
            _tool_call_style = FunctionaryV3Llama3;
        } else if (chat_template.find("<|start_header_id|>") != std::string::npos) {
            if (chat_template.find("<function=") != std::string::npos) {
                _tool_call_style = FunctionaryV3Llama31;
            } else if (chat_template.find("<|python_tag|>") != std::string::npos) {
                _tool_call_style = Llama31;
            }
        }
    }

    static llama_chat_template from_model(
        const struct llama_model * model,
        const std::string & chat_template_override);

    llama_tool_call_style tool_call_style() const { return _tool_call_style; }

    const std::string & chat_template() const { return _chat_template; }
    bool supports_tools() const { return _supports_tools; }

    std::string apply(
        const nlohmann::ordered_json & messages,
        const nlohmann::ordered_json & tools,
        bool add_generation_prompt) const;
};
