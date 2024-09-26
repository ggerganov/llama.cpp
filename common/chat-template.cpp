#include "chat-template.h"
#include "minja.hpp"
#include "llama.h"

using json = nlohmann::ordered_json;

static std::string _llama_token_to_piece(const struct llama_model * model, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(model, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(model, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

static std::string llama_model_meta_val_str(const struct llama_model * model, const char * key) {
    int32_t tlen = llama_model_meta_val_str(model, key, nullptr, 0);
    if (tlen > 0) {
        std::vector<char> curr_tmpl_buf(tlen + 1, 0);
        if (llama_model_meta_val_str(model, key, curr_tmpl_buf.data(), curr_tmpl_buf.size()) == tlen) {
            return std::string(curr_tmpl_buf.data(), tlen);
        }
    }
    return "";
}

llama_chat_template llama_chat_template::from_model(
    const struct llama_model * model,
    const std::string & chat_template_override)
{
    // TODO: handle "chatml"?
    auto chat_template = chat_template_override.empty()
        ? llama_model_meta_val_str(model, "tokenizer.chat_template")
        : chat_template_override;
    auto bos_token = _llama_token_to_piece(model, llama_token_bos(model), true);
    auto eos_token = _llama_token_to_piece(model, llama_token_eos(model), true);
    return llama_chat_template(chat_template, bos_token, eos_token);
}

std::string llama_chat_template::apply(
    const json & messages,
    const json & tools,
    bool add_generation_prompt) const
{
    auto actual_messages = messages;

    // First, "fix" messages so they have a chance to be rendered correctly by the template

    if (_requires_object_arguments || !_supports_system_role) {
        std::string pending_system;
        auto flush_sys = [&]() {
            if (!pending_system.empty()) {
                actual_messages.push_back({
                    {"role", "user"},
                    {"content", pending_system},
                });
                pending_system.clear();
            }
        };
        for (auto & message : actual_messages) {
            if (!message.contains("role") || !message.contains("content")) {
                throw std::runtime_error("message must have 'role' and 'content' fields: " + message.dump());
            }
            std::string role = message.at("role");
            std::string content = message.at("content");

            if (!_supports_system_role) {
                if (role == "system") {
                    if (!pending_system.empty()) pending_system += "\n";
                    pending_system += content;
                    continue;
                } else {
                    if (role == "user") {
                        if (!pending_system.empty()) {
                            message["content"] = pending_system + (content.empty() ? "" : "\n" + content);
                            pending_system.clear();
                        }
                    } else {
                        flush_sys();
                    }
                }
            }
            if (_requires_object_arguments && message.contains("tool_calls")) {
                for (auto & tool_call : message.at("tool_calls")) {
                    std::string arguments = tool_call.at("arguments");
                    tool_call["arguments"] = json::parse(arguments);
                }
            }
        }
        flush_sys();
    }

    auto context = minja::Context::make(json({
        {"messages", actual_messages},
        {"add_generation_prompt", add_generation_prompt},
        {"bos_token", _bos_token},
        {"eos_token", _eos_token},
    }));

    if (!tools.is_null() && !tools.empty()) {
        auto tools_val = minja::Value(tools);
        context->set("tools", tools_val);
    }

    auto tmpl_root = minja::Parser::parse(_chat_template, {
        /* .trim_blocks = */ true,
        /* .lstrip_blocks = */ true,
        /* .keep_trailing_newline = */ false,
    });
    return tmpl_root->render(context);
}
