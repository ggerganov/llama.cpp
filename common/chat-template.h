#pragma once

#include "minja.hpp"
#include <json.hpp>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;


enum llama_tool_call_style {
    UnknownToolCallStyle,
    Llama31,
    Llama32,
    FunctionaryV3Llama3,
    FunctionaryV3Llama31,
    Hermes2Pro,
    CommandRPlus,
};

class llama_chat_template {
  public:

  private:
    llama_tool_call_style _tool_call_style = UnknownToolCallStyle;
    bool _supports_tools = true;
    // Meta-Llama-3.1-8B-Instruct's template expects arguments to be an object.
    // Most other templates (and OpenAI's API) expect the arguments object to be stringified.
    bool _requires_object_arguments = false;
    bool _supports_system_role = true;
    std::string _chat_template;
    std::string _bos_token;
    std::string _eos_token;
    std::unique_ptr<minja::TemplateNode> _template_root;

  public:
    llama_chat_template(const std::string & chat_template, const std::string & bos_token, const std::string & eos_token);

    static llama_chat_template from_model(
        const struct llama_model * model,
        const char * chat_template_override = nullptr);

    llama_tool_call_style tool_call_style() const { return _tool_call_style; }

    const std::string & chat_template() const { return _chat_template; }
    bool supports_tools() const { return _supports_tools; }

    std::string apply(
        const nlohmann::ordered_json & messages,
        const nlohmann::ordered_json & tools,
        bool add_generation_prompt,
        const nlohmann::ordered_json & extra_context = nlohmann::ordered_json()) const;
};
