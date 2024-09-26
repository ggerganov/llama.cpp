#include "tool-call.h"
#include "json-schema-to-grammar.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;

// https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3.llama3.txt
static bool needs_functionary_v3_tool_call(const std::string & chat_template) {
    return chat_template.find("<|start_header_id|>") != std::string::npos
        && chat_template.find(">>>all") != std::string::npos;
}

// https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3-llama3.1.txt
static bool needs_functionary_v3_llama_3_1_tool_call(const std::string & chat_template) {
    return chat_template.find("<|start_header_id|>") != std::string::npos
        && chat_template.find("<function=") != std::string::npos;
}

static bool needs_llama_3_1_tool_call(const std::string & chat_template) {
    return chat_template.find("<|start_header_id|>") != std::string::npos
        && chat_template.find("<|python_tag|>") != std::string::npos;
}

static bool needs_hermes_pro_tool_call(const std::string & chat_template) {
    return chat_template.find("<tool_call>") != std::string::npos;
}

static bool parse_json(std::string::const_iterator & it, const std::string::const_iterator & end, json & out) {
    // // https://json.nlohmann.me/features/parsing/sax_interface/
    struct json_error_locator : public nlohmann::json_sax<json> {
        std::size_t position;
        bool found_error;

        json_error_locator() : position(0), found_error(false) {}

        bool parse_error(std::size_t position, const std::string & last_token, const json::exception & ex) override {
            // LOG_WARNING("JSON error (Expected)", {{"position", position}, {"last_token", last_token}, {"error", ex.what()}});
            this->position = position - 1;
            this->found_error = true;
            return false;
        }
        bool null() override { return true; }
        bool boolean(bool) override { return true; }
        bool number_integer(number_integer_t) override { return true; }
        bool number_unsigned(number_unsigned_t) override { return true; }
        bool number_float(number_float_t, const string_t &) override { return true; }
        bool string(string_t &) override { return true; }
        bool binary(binary_t &) override { return true; }
        bool start_object(std::size_t) override { return true; }
        bool key(string_t &) override { return true; }
        bool end_object() override { return true; }
        bool start_array(std::size_t) override { return true; }
        bool end_array() override { return true; }
    };
    json_error_locator err_loc;
    json::sax_parse(it, end, &err_loc);

    std::string::const_iterator temptative_end;
    if (err_loc.found_error) {
        temptative_end = it + err_loc.position;
    } else {
        temptative_end = end;
    }
    std::string json_sub {it, temptative_end};
    // LOG_WARNING("Parsing json", {{"json_sub", json_sub}});
    try {
        out = json::parse(json_sub);
        it = temptative_end;
        return true;
    } catch (const std::exception & e) {
        // LOG_WARNING("Failed to parse tool call", {{"json_sub", json_sub}, {"error", e.what()}});
        return false;
    }
}

static llama_tool_calls parse_hermes_tool_calls(const std::string& input) {
    try {
        std::regex start_pattern(R"([\n\s]*<tool_call>)");
        std::regex middle_pattern(R"([\n\s]*</tool_call>[\n\s]*<tool_call>)");
        std::regex end_pattern(R"([\n\s]*</tool_call>[\n\s]*$)");

        auto end = input.end();
        std::sregex_iterator rend;
        std::sregex_iterator rit(input.begin(), end, start_pattern);
        if (rit == rend) {
            return {input, {}};
        }

        llama_tool_calls result;
        result.content = rit->prefix();

        auto it = rit->suffix().first;
        while (it != end) {
            json call;
            if (!parse_json(it, end, call)) {
                throw std::runtime_error("Failed to parse json tool call");
            }
            result.tool_calls.push_back({
                call["name"],
                call["arguments"].dump(),
            });
            rit = {it, end, middle_pattern};
            if (rit != rend) {
                it = rit->suffix().first;
            } else {
                rit = {it, end, end_pattern};
                if (rit == rend) {
                    throw std::runtime_error("Malformed input, missing </tool_call>");
                }
                break;
            }
        }
        return result;
    } catch (const std::exception & e) {
        return {input, {}};
    }
}

static llama_tool_calls parse_llama_3_1_tool_calls(const json & tools, const std::string& input) {
    static std::regex python_tag_regex(R"(<\|python_tag\|>([\s\S\n]*)$)");
    std::smatch match;
    if (std::regex_search(input, match, python_tag_regex)) {
        return {
            match.prefix().str(), {
                {"ipython", (json {{"code", match[1].str()}}).dump()},
            }
        };
    }
    try {
        auto call = json::parse(input);
        // Only treat JSON as a tool call if it has a name attribute that matches any of the tools specified in the request.
        // There doesn't seem to be any better way to detect a tool call.
        if (call.contains("name") && call["name"].is_string()) {
            std::string name = call["name"];
            for (const auto & tool : tools) {
                if (tool.at("function").at("name") == name) {
                    return {
                        "",
                        {
                            {name, call["parameters"].dump()},
                        }
                    };
                }
            }
        }
    } catch (const std::exception & e) {
        // Do nothing
    }
    return {input, {}};
}

static llama_tool_calls parse_functionary_tool_calls(const std::string& input, const std::regex & function_regex, const std::regex & close_regex) {
    std::smatch match;

    llama_tool_calls result;
    auto end = input.end();
    auto it = input.begin();

    while (it != end) {
        std::sregex_iterator rend;
        std::sregex_iterator rit(it, end, function_regex);
        if (rit == rend) {
            result.content += std::string(it, end);
            break;
        }

        result.content += std::string(it, rit->prefix().second);
        it = rit->suffix().first;

        auto name = rit->str(1);

        json arguments;
        if (!parse_json(it, end, arguments)) {
            throw std::runtime_error("Failed to parse json tool call arguments");
        }
        if (!std::regex_search(it, end, match, close_regex)) {
            throw std::runtime_error("Malformed input, missing closing pattern");
        }
        it = match.suffix().first;
        result.tool_calls.push_back({name, arguments.dump()});
    }
    return result;
}

static llama_tool_calls parse_functionary_v3_llama_3_1_tool_calls(const std::string& input) {
    static std::regex function_regex(R"(<function=(\w+)>)");
    static std::regex close_regex(R"(</function>)");
    return parse_functionary_tool_calls(input, function_regex, close_regex);
}

static llama_tool_calls parse_functionary_v3_tool_calls(const std::string& input) {
    static std::regex function_regex(R"(>>>(\w+)\n)");
    static std::regex close_regex(R"($|\n(?=>>>))");
    return parse_functionary_tool_calls(input, function_regex, close_regex);
}

llama_tool_calls parse_tool_calls(const json & tools, const std::string & chat_template, const std::string& input) {
    if (needs_hermes_pro_tool_call(chat_template)) {
        return parse_hermes_tool_calls(input);
    } else if (needs_llama_3_1_tool_call(chat_template)) {
        return parse_llama_3_1_tool_calls(tools, input);
    } else if (needs_functionary_v3_tool_call(chat_template)) {
        return parse_functionary_v3_tool_calls(input);
    } else if (needs_functionary_v3_llama_3_1_tool_call(chat_template)) {
        return parse_functionary_v3_llama_3_1_tool_calls(input);
    } else {
        throw std::runtime_error("Unsupported chat template for tool calls");
    }
}

llama_tool_call_handler llama_tool_call_handler_init(
    const std::string & chat_template,
    bool allow_content,
    bool parallel_tool_calls,
    const nlohmann::ordered_json & tools)
{
    llama_tool_call_handler handler;

    if (needs_functionary_v3_tool_call(chat_template)) {
        // MeetKaiFunctionary_3_2
        // >>>all\nlet's call functions>>>fn1\n{"arg1": 1...}\n>>>fn2\n{"arg1": 1...}...
        // Using ">>>f1\n", ">>>f2\n"... as trigger words for the grammar
        handler.grammar = build_grammar([&](const llama_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            for (size_t i = 0, n = tools.size(); i < n; i++) {
                auto & tool = tools[i];
                const auto & function = tool["function"];
                std::string name = function["name"];
                auto parameters = function["parameters"];
                auto tool_rule = builder.add_rule(name + "-call", "\">>>" + name + "\\n\" " + builder.add_schema(name + "-args", parameters));
                tool_rules.push_back(tool_rule);
                if (allow_content) {
                    handler.grammar_trigger_words.push_back(">>>" + name + "\n");
                }
            }
            auto tool_call = builder.add_rule("tool_call", join(tool_rules.begin(), tool_rules.end(), " | ")) + " space";
            builder.add_rule("root", parallel_tool_calls ? "(" + tool_call + ")+" : tool_call);
        });
        // handler.parser = parse_functionary_3_2_tool_calls;
    } else if (needs_functionary_v3_llama_3_1_tool_call(chat_template)) {
        // https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3-llama3.1.txt
        handler.grammar = build_grammar([&](const llama_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            for (size_t i = 0, n = tools.size(); i < n; i++) {
                auto & tool = tools[i];
                const auto & function = tool["function"];
                std::string name = function["name"];
                auto parameters = function["parameters"];
                auto tool_rule = builder.add_rule(name + "-call", "\"<function=" + name + ">\" " + builder.add_schema(name + "-args", parameters) + " \"</function>\"");
                tool_rules.push_back(tool_rule);
            }
            auto tool_call = builder.add_rule("tool_call", join(tool_rules.begin(), tool_rules.end(), " | ")) + " space";
            builder.add_rule("root", parallel_tool_calls ? "(" + tool_call + ")+" : tool_call);
            if (allow_content) {
                handler.grammar_trigger_words.push_back("<function=");
            }
        });
        // handler.parser = parse_functionary_3_2_tool_calls;
    } else if (needs_hermes_pro_tool_call(chat_template)) {
        // NousResearchHermesPro_2
        // (content)?(<tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>)*
        handler.grammar = build_grammar([&](const llama_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            for (const auto & tool : tools) {
                const auto & function = tool["function"];
                std::string name = function["name"];
                auto parameters = function["parameters"];
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_schema(name + "-call", {
                    {"type", "object"},
                    {"properties", json {
                        {"name", json {{"const", name}}},
                        {"arguments", parameters},
                    }},
                    {"required", json::array({"name", "arguments"})},
                }));
            }

            auto tool_call = "\"<tool_call>\" " + builder.add_rule("tool_call", join(tool_rules.begin(), tool_rules.end(), " | ")) + " \"</tool_call>\" space";
            builder.add_rule("root", parallel_tool_calls ? "(" + tool_call + ")+" : tool_call);
            if (allow_content) {
                handler.grammar_trigger_words.push_back("<tool_call>");
            }
        });
    } else if (needs_llama_3_1_tool_call(chat_template)) {
        handler.grammar = build_grammar([&](const llama_grammar_builder & builder) {
            static std::vector<std::string> builtin_tools {"wolfram_alpha", "brave_search"};
            std::vector<std::string> tool_rules;

            for (const auto & tool : tools) {
                const auto & function = tool["function"];
                std::string name = function["name"];
                auto parameters = function["parameters"];
                builder.resolve_refs(parameters);
                if (name == "ipython" || std::find(builtin_tools.begin(), builtin_tools.end(), name) != builtin_tools.end()) {
                    tool_rules.push_back(builder.add_rule("ipython-call", "\"<|python_tag|>\" .*"));
                    if (allow_content) {
                        handler.grammar_trigger_words.push_back("<|python_tag|>");
                    }
                } else {
                    //"<|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"" + name + "\", " +
                    tool_rules.push_back(
                        builder.add_rule(
                            name + "-call",
                            "\"\\n{\\\"name\\\": \\\"" + name + "\\\", \\\"parameters\\\": \" " +
                                builder.add_schema(name + "-args", parameters) +
                            " \"}\""));
                    if (allow_content) {
                        handler.grammar_trigger_words.push_back("\n{\"" + name + "\"");
                    }
                }
            }

            builder.add_rule("root", join(tool_rules.begin(), tool_rules.end(), " | "));
        });
        handler.additional_stop_words.push_back("<|eom_id|>");
    } else {
        // TODO: generic thoughtful schema.
        throw std::runtime_error("Unsupported tool call style!");
    }
    return handler;
}
