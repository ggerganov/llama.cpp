#include "chat-handler.hpp"
#include "chat-template.hpp"
#include "json-schema-to-grammar.h"
#include "minja.hpp"

const common_grammar_options grammar_options {
    /* .dotall = */ false,
    /* .compact_spaces = */ false,
    // /* .compact_spaces = */ true,
};

static bool parse_json(std::string::const_iterator & it, const std::string::const_iterator & end, json & out) {
    // // https://json.nlohmann.me/features/parsing/sax_interface/
    struct json_error_locator : public nlohmann::json_sax<json> {
        std::size_t position;
        bool found_error;

        json_error_locator() : position(0), found_error(false) {}

        bool parse_error(std::size_t position, const std::string &, const json::exception &) override {
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
    try {
        out = json::parse(json_sub);
        it = temptative_end;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

/**
 * Takes a prefix regex that must have 1 group to capture the function name, a closing suffix, and expects json parameters in between.
 * Aggregates the prefix, suffix and in-between text into the content.
 */
static common_chat_msg parse_json_tool_calls(const json & tools, const std::string& input, const std::regex & function_regex, const std::regex & close_regex, bool check_names) {
    std::smatch match;

    common_chat_msg result;
    result.role = "assistant";
    auto end = input.end();
    auto it = input.begin();

    std::vector<std::string> tool_names;
    if (check_names) {
        for (const auto & tool : tools) {
            if (!tool.contains("type")) {
                continue;
            }
            std::string type = tool.at("type");
            if (type == "function") {
                tool_names.push_back(tool["function"]["name"]);
            } else if (type == "code_interpreter") {
                tool_names.push_back("python");
            }
        }
    }

    while (it != end) {
        std::sregex_iterator rend;
        std::sregex_iterator rit(it, end, function_regex);
        if (rit == rend) {
            fprintf(stderr, "No more tool calls found\n");
            result.content += std::string(it, end);
            break;
        }
        auto name = rit->str(1);
        if (check_names && std::find(tool_names.begin(), tool_names.end(), name) == tool_names.end()) {
            fprintf(stderr, "Skipping unknown tool name: %s (known tools: %s)\n", name.c_str(), string_join(tool_names, ", ").c_str());
            result.content += std::string(it, rit->suffix().first);
            break;
        }

        result.content += std::string(it, rit->prefix().second);
        it = rit->suffix().first;


        json arguments;
        if (!parse_json(it, end, arguments)) {
            throw std::runtime_error("Failed to parse json tool call arguments");
        }
        if (!std::regex_search(it, end, match, close_regex)) {
            throw std::runtime_error("Malformed input, missing closing pattern");
        }
        it = match.suffix().first;
        result.tool_calls.push_back({name, arguments.is_string() ? arguments.get<std::string>() : arguments.dump(), /* id= */ ""});
    }
    return result;
}

static common_chat_msg parse_prefixed_json_tool_call_array(const std::string& input, const std::string & prefix, size_t rstrip_prefix = 0) {
    auto content_end = input.find(prefix);
    size_t tc_start = std::string::npos;

    common_chat_msg result;
    result.role = "assistant";
    const auto process_tool_calls = [&](const json & tool_calls) {
        for (const auto & tool_call : tool_calls) {
            const auto & arguments = tool_call["arguments"];
            result.tool_calls.push_back({
                tool_call["name"],
                arguments.is_string() ? arguments.get<std::string>() : arguments.dump(),
                tool_call.contains("id") ? tool_call["id"] : "",
            });
        }
    };
    if (content_end == std::string::npos) {
        result.content = input;
    } else {
        tc_start = content_end + prefix.size() - rstrip_prefix;
        result.content = input.substr(0, content_end);
        auto tool_calls = json::parse(input.substr(tc_start));
        process_tool_calls(tool_calls);
    }
    return result;
}

static nlohmann::ordered_json add_system(const nlohmann::ordered_json & messages, const std::string & system_prompt) {
    json messages_with_system = messages;

    if (messages_with_system.size() > 0 && messages_with_system[0].at("role") == "system") {
        std::string existing_system = messages_with_system.at(0).at("content");
        messages_with_system[0] = json {
            {"role", "system"},
            {"content", existing_system + "\n" + system_prompt},
        };
    } else {
        messages_with_system.insert(messages_with_system.begin(), json {
            {"role", "system"},
            {"content", system_prompt},
        });
    }
    return messages_with_system;
}

class text_chat_parser : public common_chat_parser {
public:
    std::optional<common_chat_msg> parse_partial(const std::string & input) override {
        return parse_final(input);
    }

    common_chat_msg parse_final(const std::string & input) override {
        return {
            /* .role = */ "assistant",
            /* .content = */ input,
            /* .tool_calls = */ {},
        };
    }

    std::unique_ptr<common_chat_parser> clone() const override {
        return std::make_unique<text_chat_parser>();
    }
};

class monolithic_chat_parser : public common_chat_parser {

    std::string input_buffer_;
    std::function<common_chat_msg(const std::string & input)> parse_final_;

public:
    monolithic_chat_parser(const std::function<common_chat_msg(const std::string & input)> & parse_final) : parse_final_(parse_final) {}

    std::optional<common_chat_msg> parse_partial(const std::string & input) override {
        input_buffer_ += input;
        return std::nullopt;
    }

    common_chat_msg parse_final(const std::string & input) override {
        input_buffer_ += input;
        auto out = parse_final_(input_buffer_);
        input_buffer_.clear();
        return out;
    }

    std::unique_ptr<common_chat_parser> clone() const override {
        return std::make_unique<monolithic_chat_parser>(parse_final_);
    }
};

const auto python_tool = json::parse(R"({
  "type": "function",
  "function": {
    "name": "python",
    "description": "an ipython interpreter",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {
          "type": "string",
          "description": "Python code to execute."
        }
      },
      "required": ["code"]
    }
  }
})");

static void foreach_normalized_tool(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type")) {
            continue;
        }
        if (tool["type"] == "code_interpreter") {
            fn(python_tool);
        } else {
            fn(tool);
        }
    }
}

static common_chat_data common_chat_init_generic_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;

    auto tool_call_schemas = json::array();
    foreach_normalized_tool(params.tools, [&](const json & tool) {
        const auto & function = tool["function"];
        auto tool_schema = json {
            {"type", "object"},
            {"properties", {
                {"name", {
                    {"type", "string"},
                    {"const", function["name"]},
                }},
                {"arguments", function["parameters"]},
            }},
            {"required", json::array({"name", "arguments"})},
        };
        if (function.contains("description")) {
            tool_schema["description"] = function["description"];
        }
        if (params.parallel_tool_calls) {
            tool_schema["properties"]["id"] = {
                {"type", "string"},
                {"minLength", 4},
            };
            tool_schema["required"].push_back("id");
        }
        tool_call_schemas.emplace_back(tool_schema);
    });
    const auto tool_call =
        params.parallel_tool_calls
            ? json {
                {"type", "object"},
                {"properties", {
                    {"tool_calls", {
                        {"type", "array"},
                        {"items", tool_call_schemas.size() == 1 ? tool_call_schemas[0] : json {
                            {"anyOf", tool_call_schemas},
                        }},
                        {"minItems", 1},
                    }},
                }},
                {"required", json::array({"tool_calls"})},
            }
            : json {
                {"type", "object"},
                {"properties", {
                    {"tool_call", tool_call_schemas.size() == 1 ? tool_call_schemas[0] : json {
                        {"anyOf", tool_call_schemas},
                    }},
                }},
                {"required", json::array({"tool_call"})},
            };
    const auto schema =
        params.tool_choice != "required"
            ? json {
                {"anyOf", json::array({
                    tool_call,
                    {
                        {"type", "object"},
                        {"properties", {
                            {"response", params.json_schema.is_null()
                                ? json {{"type", "string"}}
                                : params.json_schema
                            },
                        }},
                        {"required", json::array({"response"})},
                    },
                })}
            }
            : tool_call;

    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        builder.add_schema("root", schema);
    }, grammar_options);

    // TODO: add schema to system prompt.
    auto tweaked_messages = add_system(
        params.messages,
        "Respond in JSON format, either with a request to call tools or with a response to the user's request. Here is the schema for all responses:\n\n```json\n" + schema.dump(2) + "\n```");

    data.prompt = tmpl.apply(tweaked_messages, params.tools.empty() ? json() : params.tools, /* add_generation_prompt= */ true);
    data.parser = std::make_unique<monolithic_chat_parser>([&](const std::string & input) {
        json data = json::parse(input);
        common_chat_msg result;
        result.role = "assistant";
        if (data.contains("tool_calls")) {
            for (const auto & tool_call : data["tool_calls"]) {
                result.tool_calls.push_back({
                    tool_call["name"],
                    tool_call["arguments"].dump(),
                    tool_call.contains("id") ? tool_call["id"] : "",
                });
            }
        } else if (data.contains("tool_call")) {
            result.tool_calls.push_back({
                data["tool_call"]["name"],
                data["tool_call"]["arguments"].dump(),
                /* id= */ "",
            });
        } else if (data.contains("response")) {
            const auto & response = data["response"];
            result.content = response.is_string() ? response.get<std::string>() : response.dump(2);
        }
        return result;
    });
    return data;
}

static common_chat_data common_chat_init_mistral_nemo_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        auto schemas = json::array();
        foreach_normalized_tool(params.tools, [&](const json & tool) {
            const auto & function = tool["function"];
            schemas.push_back({
                {"type", "object"},
                {"properties", {
                    // Important note: the model is probably trained to take a JSON stringified arguments value.
                    // It's hard to constrain that for now (while reusing the JSON schema conversion), so we're just expecting a plain object.
                    {"name", {
                        {"type", "string"},
                        {"const", function["name"]},
                    }},
                    {"arguments", function["parameters"]},
                    {"id", {
                        {"type", "string"},
                        // Nemo's template expects a 9-character alphanumeric ID.
                        {"pattern", "^[a-zA-Z0-9]{9}$"},
                    }},
                }},
                {"required", json::array({"name", "arguments", "id"})},
            });
        });
        auto schema = json {
            {"type", "array"},
            {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
            {"minItems", 1},
        };
        if (!params.parallel_tool_calls) {
            schema["maxItems"] = 1;
        }
        builder.add_rule("root", "\"[TOOL_CALLS]\" " + builder.add_schema("tool_calls", schema));
    }, grammar_options);
    if (params.tool_choice != "required") {
        data.grammar_triggers.push_back({"[TOOL_CALLS]", /* .at_start = */ true});
    }
    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, /* add_generation_prompt= */ true);
    data.parser = std::make_unique<monolithic_chat_parser>([](const std::string & input) -> common_chat_msg {
            return parse_prefixed_json_tool_call_array(input, "[TOOL_CALLS]");
        });
    return data;
}

static common_chat_data common_chat_init_llama_3_tool_calls(const common_chat_template & tmpl, const struct common_chat_params & params, bool uses_python_tag, bool eagerly_match_any_json) {
    auto builtin_tools = json {"wolfram_alpha", "brave_search"};
    common_chat_data data;

    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        std::vector<std::string> tool_rules;

        auto has_python = false;

        for (const auto & tool : params.tools) {
            if (!tool.contains("type")) {
                continue;
            }

            if (tool["type"] == "code_interpreter") {
                builtin_tools.push_back("code_interpreter");
                has_python = true;
            } else if (tool["type"] == "function" && tool.contains("function")) {
                const auto & function = tool["function"];
                std::string name = function["name"];
                auto parameters = function["parameters"];
                builder.resolve_refs(parameters);
                if (uses_python_tag && (name == "python" || name == "ipython" || builtin_tools.contains(name))) {
                    has_python = true;
                } else {
                    //"<|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"" + name + "\", " +
                    tool_rules.push_back(
                        builder.add_rule(
                            name + "-call",
                            "\"\\n\"? \"{\" ( \"\\\"type\\\": \\\"function\\\", \" | space ) \"\\\"name\\\": \\\"" + name + "\\\", \\\"parameters\\\": \" " +
                                builder.add_schema(name + "-args", parameters) +
                            " \"}\""));
                    if (params.tool_choice != "required" && !eagerly_match_any_json) {
                        data.grammar_triggers.push_back({"{\"name\": \"" + name + "\"", /* .at_start = */ false});
                        // Accommodate most common tool call variations from Llama-3.1-8B and Llama-3.2-3B.
                        // Note that c++11's regex doesn't support partial matches, otherwise it would make
                        // sense to add support for trigger regexes to the antiprompt mechanism.
                        data.grammar_triggers.push_back({"{\n\t\"name\": \"" + name + "\"", /* .at_start = */ false});
                        data.grammar_triggers.push_back({"{\n  \"name\": \"" + name + "\"", /* .at_start = */ false});
                        data.grammar_triggers.push_back({"{\n    \"name\": \"" + name + "\"", /* .at_start = */ false});
                        data.grammar_triggers.push_back({"{\"type\": \"function\", \"name\": \"" + name + "\"", /* .at_start = */ false});
                    }
                }
            }
        }

        if (has_python) {
            tool_rules.push_back(builder.add_rule("ipython-call", "\"<|python_tag|>\" .*"));
            if (params.tool_choice != "required") {
                data.grammar_triggers.push_back({"<|python_tag|>", /* .at_start = */ false});
            }
        }

        if (params.tool_choice != "required" && eagerly_match_any_json) {
            data.grammar_triggers.push_back({"{\"", /* .at_start = */ true});
            data.grammar_triggers.push_back({"{\n\t\"", /* .at_start = */ true});
            data.grammar_triggers.push_back({"{\n  \"", /* .at_start = */ true});
            data.grammar_triggers.push_back({"{\n    \"", /* .at_start = */ true});
        }

        builder.add_rule("root", string_join(tool_rules, " | "));
    }, grammar_options);
    data.additional_stops.push_back("<|eom_id|>");
    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, /* add_generation_prompt= */ true, {
        {"builtin_tools", builtin_tools},
    });
    data.parser = std::make_unique<monolithic_chat_parser>([params, uses_python_tag](const std::string & input) -> common_chat_msg {
        if (uses_python_tag) {
            static std::regex python_tag_regex(R"(<\|python_tag\|>([\s\S\n]*)$)");
            std::smatch match;
            if (std::regex_search(input, match, python_tag_regex)) {
                return {
                    /* .role = */ "assistant",
                    /* .content = */ match.prefix().str(),
                    /* .tool_calls = */ {
                        {
                            /* .name = */ "python",
                            /* .arguments = */ match[1].str(),
                            /* .id = */ "",
                        },
                    }
                };
            }
        }
        static std::regex function_regex("\\{(?:\"type\": \"function\", |[\\s\\n\\r]*)\"name\": \"([^\"]+)\", \"parameters\": ");
        static std::regex close_regex("\\}");
        return parse_json_tool_calls(params.tools, input, function_regex, close_regex, /* check_names= */ true);
    });
    return data;
}

static common_chat_data common_chat_init_firefunction_v2_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        auto schemas = json::array();
        foreach_normalized_tool(params.tools, [&](const json & tool) {
            const auto & function = tool["function"];
            schemas.push_back({
                {"type", "object"},
                {"properties", {
                    {"name", {
                        {"type", "string"},
                        {"const", function["name"]},
                    }},
                    {"arguments", function["parameters"]},
                }},
                {"required", json::array({"name", "arguments", "id"})},
            });
        });
        auto schema = json {
            {"type", "array"},
            {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
            {"minItems", 1},
        };
        if (!params.parallel_tool_calls) {
            schema["maxItems"] = 1;
        }
        builder.add_rule("root", "\" functools\"? " + builder.add_schema("tool_calls", schema));
    }, grammar_options);
    if (params.tool_choice != "required") {
        data.grammar_triggers.push_back({" functools[", /* .at_start = */ false});
    }
    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, /* add_generation_prompt= */ true);
    data.parser = std::make_unique<monolithic_chat_parser>([](const std::string & input) -> common_chat_msg {
        return parse_prefixed_json_tool_call_array(input, " functools[", /* rstrip_prefix= */ 1);
    });
    return data;
}

static common_chat_data common_chat_init_functionary_v3_llama_3_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    // >>>all\nlet's call functions>>>fn1\n{"arg1": 1...}\n>>>fn2\n{"arg1": 1...}...
    // Using ">>>f1\n", ">>>f2\n"... as trigger words for the grammar
    common_chat_data data;

    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        std::vector<std::string> first_tool_rules;
        std::vector<std::string> subsequent_tool_rules;
        auto has_python = false;
        for (const auto & tool : params.tools) {
            if (!tool.contains("type")) {
                continue;
            }
            if (tool["type"] == "code_interpreter") {
                has_python = true;
            } else if (tool["type"] == "function" && tool.contains("function")) {
                const auto & function = tool["function"];
                std::string name = function["name"];
                auto parameters = function["parameters"];
                auto args_rule = builder.add_schema(name + "-args", parameters);
                first_tool_rules.push_back(builder.add_rule(name + "-call", "\"" + name + "\\n\" " + args_rule));
                subsequent_tool_rules.push_back(builder.add_rule(name + "-call2", "\"\\n>>>" + name + "\\n\" " + args_rule));
                if (params.tool_choice != "required") {
                    data.grammar_triggers.push_back({name + "\n", /* .at_start = */ true});
                    data.grammar_triggers.push_back({"\n>>>" + name + "\n", /* .at_start = */ false});
                }
            }
        }
        auto first_rule = builder.add_rule("first_tool_call", string_join(first_tool_rules, " | ")) + " space";
        // Note: if there's a python rule, it needs to come last.
        auto python_rule = builder.add_rule("python-call", "\"python\\n\" .*");
        if (has_python && params.tool_choice != "required") {
            data.grammar_triggers.push_back({"python\n", /* .at_start = */ true});
            data.grammar_triggers.push_back({"\n>>>python\n", /* .at_start = */ false});
        }
        if (params.parallel_tool_calls) {
            auto subsequent_rule = builder.add_rule("subsequent_tool_call", string_join(subsequent_tool_rules, " | ")) + " space";
            builder.add_rule("root", python_rule + " | " + first_rule + " (" + subsequent_rule + ")*" + (has_python ? " ( \">>>\\n\" " + python_rule + " )?" : ""));
        } else {
            builder.add_rule("root", first_rule + (has_python ? " | " + python_rule : ""));
        }
    }, grammar_options);

    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, /* add_generation_prompt= */ true);
    data.parser = std::make_unique<monolithic_chat_parser>([params](const std::string & input) -> common_chat_msg {
        static std::regex function_regex(R"((?:>>>)?(\w+)\n)");
        static std::regex close_regex(R"($|(?=>>>))");
        return parse_json_tool_calls(params.tools, input, function_regex, close_regex, /* check_names= */ true);
    });
    return data;
}

static common_chat_data common_chat_init_functionary_v3_llama_3_1_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    // ./tests/chat/templates/meetkai-functionary-medium-v3.1.jinja
    // https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3-llama3.1.txt
    // TODO: handle tool {type: code_interpreter} as python
    common_chat_data data;
    json tools = params.tools.is_null() ? params.tools : json::array();

    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        std::vector<std::string> tool_rules;
        auto has_python = false;
        for (const auto & tool : params.tools) {
            if (!tool.contains("type")) {
                continue;
            }
            if (tool["type"] == "code_interpreter") {
                has_python = true;
            } else if (tool["type"] == "function" && tool.contains("function")) {
                const auto & function = tool["function"];
                std::string name = function["name"];
                if (name == "python" || name == "ipython") {
                    has_python = true;
                } else {
                    auto parameters = function["parameters"];
                    tool_rules.push_back(builder.add_rule(name + "-call", "\"<function=" + name + ">\" " + builder.add_schema(name + "-args", parameters) + " \"</function>\" space"));
                }
            }
        }
        if (has_python) {
            tool_rules.push_back(builder.add_rule("python-call", "\"<|python_tag|>\" .*"));
            if (params.tool_choice != "required") {
                data.grammar_triggers.push_back({"<|python_tag|>", /* .at_start = */ false});
            }
        }
        auto tool_call = builder.add_rule("tool_call", string_join(tool_rules, " | ")) + " space";
        builder.add_rule("root", params.parallel_tool_calls ? "(" + tool_call + ")+" : tool_call);
        if (params.tool_choice != "required") {
            data.grammar_triggers.push_back({"<function=", /* .at_start = */ false});
        }
    }, grammar_options);

    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, /* add_generation_prompt= */ true);
    data.parser = std::make_unique<monolithic_chat_parser>([params](const std::string & input) -> common_chat_msg {
        // This version of Functionary still supports the llama 3.1 tool call format for the python tool.
        static std::regex python_tag_regex(R"(<\|python_tag\|>([\s\S\n]*)$)");
        std::smatch match;
        if (std::regex_search(input, match, python_tag_regex)) {
            return {
                /* .role = */ "assistant",
                /* .content = */ match.prefix().str(),
                /* .tool_calls = */ {
                    {
                        /* .name = */ "python",
                        /* .arguments = */ match[1].str(),
                        /* .id = */ "",
                    },
                }
            };
        }
        static std::regex function_regex(R"(<function=(\w+)>)");
        static std::regex close_regex(R"(</function>)");
        return parse_json_tool_calls(params.tools, input, function_regex, close_regex, /* check_names= */ false);
    });
    return data;
}

static common_chat_data common_chat_init_hermes_2_pro_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;
    // (content)?(<tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>)*
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        std::vector<std::string> tool_rules;
        foreach_normalized_tool(params.tools, [&](const json & tool) {
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
        });
        auto tool_call = "\"<tool_call>\" space " + builder.add_rule("tool_call", string_join(tool_rules, " | ")) + " \"</tool_call>\" space";
        builder.add_rule("root", params.parallel_tool_calls ? "(" + tool_call + ")+" : tool_call);
        if (params.tool_choice != "required") {
            data.grammar_triggers.push_back({"<tool_call>", /* .at_start = */ false});
        }
    }, grammar_options);

    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, /* add_generation_prompt= */ true);
    data.parser = std::make_unique<monolithic_chat_parser>([&](const std::string & input) -> common_chat_msg {
        try {
            std::regex start_pattern(R"([\n\s]*<tool_call>)");
            std::regex middle_pattern(R"([\n\s]*</tool_call>[\n\s]*<tool_call>)");
            std::regex end_pattern(R"([\n\s]*</tool_call>[\n\s]*$)");

            auto end = input.end();
            std::sregex_iterator rend;
            std::sregex_iterator rit(input.begin(), end, start_pattern);
            if (rit == rend) {
                return {"assistant", input, {}};
            }

            common_chat_msg result;
            result.role = "assistant";
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
                    /* id= */ "",
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
            return {"assistant", input, {}};
        }
    });
    return data;
}

static common_chat_data common_chat_init_without_tools(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;
    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, /* add_generation_prompt= */ true);
    data.parser = std::make_unique<text_chat_parser>();
    if (!params.json_schema.is_null()) {
        if (!params.grammar.empty()) {
            throw std::runtime_error("Either \"json_schema\" or \"grammar\" can be specified, but not both");
        }
        data.grammar = json_schema_to_grammar(params.json_schema);
    } else {
        data.grammar = params.grammar.empty();
    }
    return data;
}

common_chat_data common_chat_init(const common_chat_template & tmpl, const struct common_chat_params & params) {
    if (params.tools.is_null()) {
        return common_chat_init_without_tools(tmpl, params);
    }

    if (!params.grammar.empty()) {
        throw std::runtime_error("Cannot specify grammar with tools");
    }

    const auto & src = tmpl.source();
    if (src.find("<tool_call>") != std::string::npos) {
        return common_chat_init_hermes_2_pro_tool_call(tmpl, params);
    }
    if (src.find(">>>all") != std::string::npos) {
        return common_chat_init_functionary_v3_llama_3_tool_call(tmpl, params);
    }
    if (src.find("<|start_header_id|>") != std::string::npos
        && src.find("<function=") != std::string::npos) {
        return common_chat_init_functionary_v3_llama_3_1_tool_call(tmpl, params);
    }
    if (src.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos) {
        auto uses_python_tag = src.find("<|python_tag|>") != std::string::npos;

        // Technically we should only trigger on `"\n{\"name\": \"" + name + "\""` for each tool name,
        // but Llama-3.2-3B (and 1B) struggles to output valid tool calls so we're "guiding" it strongly as soon
        // as it seems to be outputting some JSON.
        // TODO: make this conditional on a very small model (e.g. 1B / 3B).
        auto eagerly_match_any_json = false; // style == common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_2;

        return common_chat_init_llama_3_tool_calls(tmpl, params, uses_python_tag, eagerly_match_any_json);
    }
    // if (src.find("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>") != std::string::npos) {
    //     TODO: Command-R-Plus
    // }
    if (src.find("[TOOL_CALLS]") != std::string::npos) {
        return common_chat_init_mistral_nemo_tool_call(tmpl, params);
    }
    if (src.find(" functools[") != std::string::npos) {
        return common_chat_init_firefunction_v2_tool_call(tmpl, params);
    }
    return common_chat_init_generic_tool_call(tmpl, params);
}
