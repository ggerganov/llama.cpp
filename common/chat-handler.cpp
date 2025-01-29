#include "chat-handler.hpp"
#include "chat-template.hpp"
#include "json-schema-to-grammar.h"
#include "log.h"
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
static common_chat_msg parse_json_tool_calls(const json & tools, const std::string& input, const std::optional<std::regex> & trigger_opt, const std::regex & function_regex, const std::regex & close_regex, bool check_names, bool allow_raw_python = false) {
    std::smatch match;

    common_chat_msg result;
    result.role = "assistant";

    std::vector<std::string> tool_names;
    if (check_names) {
        for (const auto & tool : tools) {
            if (!tool.contains("type") || tool["type"] != "function" || !tool.contains("function")) {
                continue;
            }
            tool_names.push_back(tool["function"]["name"]);
        }
    }

    auto end = input.end();
    auto it = input.begin();

    if (trigger_opt) {
        if (!std::regex_search(it, end, match, *trigger_opt)) {
            result.content = input;
            return result;
        }
        result.content = match.prefix().str();
        it = match.suffix().first;
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
            it = rit->suffix().first;
            continue;
        }

        result.content += std::string(it, rit->prefix().second);
        it = rit->suffix().first;


        json arguments;
        if (!parse_json(it, end, arguments)) {
            if (allow_raw_python && name == "python" && std::regex_match("", close_regex)) {
                std::string src(it, end);
                result.tool_calls.push_back({name, src, /* id= */ ""});
                break;
            }
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

static void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool["type"] != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(tool);
    }
}

static common_chat_msg no_op_text_parser(const std::string & input) {
    return {
        /* .role = */ "assistant",
        /* .content = */ input,
        /* .tool_calls = */ {},
    };
}

static common_chat_data common_chat_init_generic_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;

    auto tool_call_schemas = json::array();
    foreach_function(params.tools, [&](const json & tool) {
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

    data.grammar_lazy = false;
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        builder.add_schema("root", schema);
    }, grammar_options);

    auto tweaked_messages = common_chat_template::add_system(
        params.messages,
        "Respond in JSON format, either with a request to call tools or with a response to the user's request. Here is the schema for all responses:\n\n```json\n" + schema.dump(2) + "\n```");

    data.prompt = tmpl.apply(tweaked_messages, params.tools.empty() ? json() : params.tools, params.add_generation_prompt);
    data.format = "generic tool calls";
    data.parser = [&](const std::string & input) {
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
    };
    return data;
}

static common_chat_data common_chat_init_mistral_nemo_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;
    data.grammar_lazy = params.tool_choice != "required";
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        auto schemas = json::array();
        foreach_function(params.tools, [&](const json & tool) {
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
    data.grammar_triggers.push_back({"[TOOL_CALLS]", /* .at_start = */ true});
    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, params.add_generation_prompt);
    data.format = "mistral nemo tool calls";
    data.parser = [](const std::string & input) {
        return parse_prefixed_json_tool_call_array(input, "[TOOL_CALLS]");
    };
    return data;
}

static void expect_tool_parameters(const std::string & name, const json & parameters, const std::vector<std::string> & expected_properties) {
    if (!parameters.is_object() || !parameters.contains("type") || parameters["type"] != "object" || !parameters.contains("properties") || !parameters.contains("required")) {
        throw std::runtime_error("Parameters of tool " + name + " must be an object w/ required properties");
    }
    const auto & parameters_properties = parameters.at("properties");
    const auto & parameters_required = parameters.at("required");
    for (const auto & prop : expected_properties) {
        if (!parameters_properties.contains(prop)) {
            throw std::runtime_error("Parameters of tool " + name + " is missing property: " + prop);
        }
        if (std::find(parameters_required.begin(), parameters_required.end(), json(prop)) == parameters_required.end()) {
            throw std::runtime_error("Parameters of tool " + name + " must have property marked as required: " + prop);
        }
    }
    if (parameters_properties.size() != expected_properties.size()) {
        throw std::runtime_error("Parameters of tool " + name + " must only have these properties:" + string_join(expected_properties, ", "));
    }
}

static common_chat_data common_chat_init_llama_3_1_python_tag_tool_calls(const common_chat_template & tmpl, const struct common_chat_params & params) {
    auto builtin_tools = json::array();
    common_chat_data data;
    data.grammar_lazy = params.tool_choice != "required";
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        std::vector<std::string> tool_rules;

        auto handle_builtin_tool = [&](const std::string & name, const json & parameters) {
            if (name == "wolfram_alpha") {
                // https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/remote/tool_runtime/wolfram_alpha/wolfram_alpha.py
                expect_tool_parameters(name, parameters, {"query"});
            } else if (name == "web_search" || name == "brave_search") {
                // https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/remote/tool_runtime/brave_search/brave_search.py
                expect_tool_parameters(name, parameters, {"query"});
            } else if (name == "python" || name == "code_interpreter") {
                // https://github.com/meta-llama/llama-stack/blob/main/llama_stack/providers/inline/tool_runtime/code_interpreter/code_interpreter.py
                expect_tool_parameters(name, parameters, {"code"});
            } else {
                return false;
            }

            std::vector<std::string> kvs;
            for (const auto & [key, value] : parameters.at("properties").items()) {
                kvs.push_back("\"" + key + "=\" " + builder.add_schema(name + "-args-" + key, value));
            }

            tool_rules.push_back(
                builder.add_rule(
                    name + "-call",
                    "\"<|python_tag|>" + name + ".call(\" " + string_join(kvs, " \", \" ") + " \")\""));
            builtin_tools.push_back(name);

            return true;
        };

        foreach_function(params.tools, [&](const json & tool) {
            const auto & function = tool["function"];
            std::string name = function["name"];
            auto parameters = function["parameters"];

            // https://github.com/meta-llama/llama-stack/tree/main/llama_stack/providers/remote/tool_runtime
            if (handle_builtin_tool(name, parameters)) {
                return;
            }
            builder.resolve_refs(parameters);
            tool_rules.push_back(
                builder.add_rule(
                    name + "-call",
                    "\"{\\\"name\\\": \\\"" + name + "\\\", \\\"parameters\\\": \" " +
                        builder.add_schema(name + "-args", parameters) +
                    " \"}\""));
            data.grammar_triggers.push_back({"{\"name\": \"" + name + "\"", /* .at_start = */ true});
        });
        if (!builtin_tools.empty()) {
            data.grammar_triggers.push_back({"<|python_tag|>", /* .at_start = */ false});
        }
        builder.add_rule("root", string_join(tool_rules, " | "));
    }, grammar_options);
    data.additional_stops.push_back("<|eom_id|>");
    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, params.add_generation_prompt, {
        {"tools_in_user_message", false},
        {"builtin_tools", builtin_tools.empty() ? json() : builtin_tools},
    });
    data.format = "llama 3.1 tool calls";
    data.parser = [params](const std::string & input) -> common_chat_msg {
        static std::regex function_regex("\\{\"name\": \"([^\"]+)\", \"parameters\": ");
        static std::regex close_regex("\\}");
        static std::regex builtin_call_regex("<\\|python_tag\\|>([^.(]+)\\.call\\((.*)\\)");

        std::smatch match;
        if (std::regex_match(input, match, builtin_call_regex)) {
            auto name = match[1].str();
            auto raw_args = match[2].str();

            // TODO: if/when builtin tools start accepting more than 1 argument, use parse_json for real parsing.
            auto it_eq = raw_args.find('=');
            auto arg_name = raw_args.substr(0, it_eq);
            auto arg_value_str = raw_args.substr(it_eq + 1);
            auto arg_value = json::parse(arg_value_str);

            return {
                /* .role = */ "assistant",
                /* .content = */ match.prefix().str(),
                /* .tool_calls = */ {
                    {
                        /* .name = */ match[1],
                        /* .arguments = */ (json {
                            {arg_name, arg_value},
                        }).dump(),
                        /* .id = */ "",
                    },
                },
            };
        }
        return parse_json_tool_calls(params.tools, input, std::nullopt, function_regex, close_regex, /* check_names= */ true);
    };
    return data;
}

static common_chat_data common_chat_init_llama_3_2_tool_calls(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;

    data.grammar_lazy = params.tool_choice != "required";
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        std::vector<std::string> tool_rules;

        // auto add_tool = [&](const json & tool) {
        foreach_function(params.tools, [&](const json & tool) {
            const auto & function = tool["function"];
            std::string name = function["name"];
            auto parameters = function["parameters"];
            builder.resolve_refs(parameters);
            tool_rules.push_back(
                builder.add_rule(
                    name + "-call",
                    "\"{\" "
                    // " ( \"\\\"type\\\": \\\"function\\\", \" | space ) "
                    "\"\\\"name\\\": \\\"" + name + "\\\", \\\"parameters\\\": \" " +
                        builder.add_schema(name + "-args", parameters) +
                    " \"}\""));
            data.grammar_triggers.push_back({"{\"name\": \"" + name + "\"", /* .at_start = */ true});
        });

        builder.add_rule("root", string_join(tool_rules, " | "));
    }, grammar_options);
    data.additional_stops.push_back("<|eom_id|>");
    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, params.add_generation_prompt, {});
    data.format = "llama 3.2 tool calls";
    data.parser = [params](const std::string & input) {
        static std::regex function_regex("\\{[\\s\\n\\r]*(?:\"type\"[\\s\\n\\r]*:[\\s\\n\\r]*\"function\"[\\s\\n\\r]*,[\\s\\n\\r]*|[\\s\\n\\r]*)\"name\"[\\s\\n\\r]*:[\\s\\n\\r]*\"([^\"]+)\"[\\s\\n\\r]*,[\\s\\n\\r]*\"parameters\": ");
        static std::regex close_regex("\\}");
        auto res = parse_json_tool_calls(params.tools, input, std::nullopt, function_regex, close_regex, /* check_names= */ true);
        return res;
    };
    return data;
}

static common_chat_data common_chat_init_deepseek_r1_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;
    data.grammar_lazy = params.tool_choice != "required";
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        std::vector<std::string> tool_rules;
        foreach_function(params.tools, [&](const json & tool) {
            const auto & function = tool["function"];
            std::string name = function["name"];
            auto parameters = function["parameters"];
            auto args_rule = builder.add_schema(name + "-args", parameters);
            tool_rules.push_back(builder.add_rule(name + "-call",
                "\"<｜tool▁call▁begin｜>function<｜tool▁sep｜>" + name + "\\n```json\\n\" " + args_rule + " \"```<｜tool▁call▁end｜>\""));
        });
        data.grammar_triggers.push_back({"<｜tool▁calls▁begin｜>", /* .at_start = */ false});
        builder.add_rule("root", "\"<｜tool▁calls▁begin｜>\" (" + string_join(tool_rules, " | ") + ")" + (params.parallel_tool_calls ? "*" : "") + " space");
    }, grammar_options);
    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, params.add_generation_prompt);
    data.format = "deepseek r1 tool calls";
    data.parser = [params](const std::string & input) {
        static std::regex trigger_regex("<｜tool▁calls▁begin｜>");
        static std::regex function_regex("<｜tool▁call▁begin｜>function<｜tool▁sep｜>([^\n]+)\n```json\n");
        static std::regex close_regex("```<｜tool▁call▁end｜>");
        return parse_json_tool_calls(params.tools, input, trigger_regex, function_regex, close_regex, /* check_names= */ true);
    };
    return data;
}

static common_chat_data common_chat_init_firefunction_v2_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    fprintf(stderr, "%s\n", __func__);
    common_chat_data data;
    if (!params.tools.is_null() && !params.tools.empty()) {
        data.grammar_lazy = params.tool_choice != "required";
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(params.tools, [&](const json & tool) {
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
        data.grammar_triggers.push_back({" functools[", /* .at_start = */ false});
        data.parser = [](const std::string & input) {
            return parse_prefixed_json_tool_call_array(input, " functools[", /* rstrip_prefix= */ 1);
        };
        data.format = "firefunction v2 tool calls";
    } else {
        data.parser = no_op_text_parser;
        data.format = "firefunction v2 text-only";
    }
    data.prompt = tmpl.apply(params.messages, /* tools= */ nullptr, params.add_generation_prompt, {
        {"datetime", "Jan 29 2025 13:00:00 GMT"},
        {"functions", json(params.tools.empty() ? "" : params.tools.dump(2))},
    }, /* adjust_inputs= */ false);
    return data;
}

static common_chat_data common_chat_init_functionary_v3_2_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    // >>>all\nlet's call functions>>>fn1\n{"arg1": 1...}\n>>>fn2\n{"arg1": 1...}...
    // Using ">>>f1\n", ">>>f2\n"... as trigger words for the grammar
    common_chat_data data;

    data.grammar_lazy = params.tool_choice != "required";
    if (!params.tools.is_null() && !params.tools.empty()) {
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> first_tool_rules;
            std::vector<std::string> subsequent_tool_rules;
            foreach_function(params.tools, [&](const json & tool) {
                const auto & function = tool["function"];
                std::string name = function["name"];
                auto parameters = function["parameters"];
                auto args_rule = builder.add_schema(name + "-args", parameters);
                first_tool_rules.push_back(builder.add_rule(name + "-call", "\"" + name + "\\n\" " + args_rule));
                subsequent_tool_rules.push_back(builder.add_rule(name + "-call2", "\">>>" + name + "\\n\" " + args_rule));
                data.grammar_triggers.push_back({name, /* .at_start = */ true});
                data.grammar_triggers.push_back({">>>" + name, /* .at_start = */ false});
            });
            auto first_rule = first_tool_rules.empty() ? "" : builder.add_rule("first_tool_call", string_join(first_tool_rules, " | ")) + " space";
            if (params.parallel_tool_calls) {
                auto subsequent_rule = builder.add_rule("subsequent_tool_call", string_join(subsequent_tool_rules, " | ")) + " space";
                builder.add_rule("root", first_rule + " (" + subsequent_rule + ")*");
            } else {
                builder.add_rule("root", first_rule);
            }

        }, grammar_options);
        data.format = "functionary v3.2 tool calls";
    } else {
        data.format = "functionary v3.2 content-only";
    }

    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, params.add_generation_prompt);
    data.parser = [params](const std::string & input) {
        static std::regex function_regex(R"((?:>>>)?(\w+)\n)");
        static std::regex close_regex(R"($|(?=>>>))");

        auto res = parse_json_tool_calls(params.tools, input, std::nullopt, function_regex, close_regex, /* check_names= */ true, /* allow_raw_python= */ true);
        if (res.content.find("all\n") == 0) {
            res.content = res.content.substr(4);
        }
        return res;
    };
    return data;
}

static common_chat_data common_chat_init_functionary_v3_1_llama_3_1_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    // ./tests/chat/templates/meetkai-functionary-medium-v3.1.jinja
    // https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3-llama3.1.txt
    common_chat_data data;
    json tools = params.tools.is_null() ? params.tools : json::array();
    std::string python_code_argument_name;
    auto has_raw_python = false;

    data.grammar_lazy = params.tool_choice != "required";
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        std::vector<std::string> tool_rules;
        foreach_function(params.tools, [&](const json & tool) {
            const auto & function = tool["function"];
            const auto & parameters = function["parameters"];
            std::string name = function["name"];
            if (name == "python" || name == "ipython") {
                if (!parameters.contains("type")) {
                    throw std::runtime_error("Missing type in python tool");
                }
                has_raw_python = true;
                auto type = parameters.at("type");
                if (type == "object") {
                    auto properties = parameters.at("properties");
                    for (auto it = properties.begin(); it != properties.end(); ++it) {
                        if (it.value().at("type") == "string") {
                            if (!python_code_argument_name.empty()) {
                                throw std::runtime_error("Multiple string arguments found in python tool");
                            }
                            python_code_argument_name = it.key();
                        }
                    }
                    if (python_code_argument_name.empty()) {
                        throw std::runtime_error("No string argument found in python tool");
                    }
                } else if (type != "string") {
                    throw std::runtime_error("Invalid type in python tool: " + type.dump());
                }
            }
            tool_rules.push_back(builder.add_rule(name + "-call", "\"<function=" + name + ">\" " + builder.add_schema(name + "-args", parameters) + " \"</function>\" space"));
        });
        if (has_raw_python) {
            tool_rules.push_back(builder.add_rule("python-call", "\"<|python_tag|>\" .*"));
            data.grammar_triggers.push_back({"<|python_tag|>", /* .at_start = */ false});
        }
        auto tool_call = builder.add_rule("tool_call", string_join(tool_rules, " | ")) + " space";
        builder.add_rule("root", params.parallel_tool_calls ? "(" + tool_call + ")+" : tool_call);
        data.grammar_triggers.push_back({"<function=", /* .at_start = */ false});
    }, grammar_options);

    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, params.add_generation_prompt);
    data.format = "functionary v3.1 llama 3.1 tool calls";
    data.parser = [params, has_raw_python, python_code_argument_name](const std::string & input) -> common_chat_msg {
        // This version of Functionary still supports the llama 3.1 tool call format for the python tool.
        static std::regex python_tag_regex(R"(<\|python_tag\|>([\s\S\n]*)$)");
        std::smatch match;
        if (std::regex_search(input, match, python_tag_regex)) {
            auto code = match[1].str();
            return {
                /* .role = */ "assistant",
                /* .content = */ match.prefix().str(),
                /* .tool_calls = */ {
                    {
                        /* .name = */ "python",
                        /* .arguments = */ python_code_argument_name.empty() ? code : (json {{python_code_argument_name, code}}).dump(),
                        /* .id = */ "",
                    },
                }
            };
        }
        static std::regex function_regex(R"(<function=(\w+)>)");
        static std::regex close_regex(R"(</function>)");
        return parse_json_tool_calls(params.tools, input, std::nullopt, function_regex, close_regex, /* check_names= */ false, has_raw_python);
    };
    return data;
}

static common_chat_data common_chat_init_hermes_2_pro_tool_call(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;
    // (content)?(<tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>)*
    data.grammar_lazy = params.tool_choice != "required";
    data.grammar = build_grammar([&](const common_grammar_builder & builder) {
        std::vector<std::string> tool_rules;
        foreach_function(params.tools, [&](const json & tool) {
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
        data.grammar_triggers.push_back({"<tool_call>", /* .at_start = */ false});
    }, grammar_options);

    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, params.add_generation_prompt);
    data.format = "hermes 2 pro tool calls";
    data.parser = [&](const std::string & input) -> common_chat_msg {
        try {
            std::regex start_pattern(R"([\n\s]*<tool_call>)");
            std::regex middle_pattern(R"([\n\s]*</tool_call>[\n\s]*<tool_call>)");
            std::regex end_pattern(R"([\n\s]*</tool_call>[\n\s]*$)");

            auto end = input.end();
            std::sregex_iterator rend;
            std::sregex_iterator rit(input.begin(), end, start_pattern);
            if (rit == rend) {
                return {
                    /* .role = */ "assistant",
                    /* .content = */ input,
                    /* .tool_calls = */ {},
                };
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
                const auto & arguments = call["arguments"];
                result.tool_calls.push_back({
                    call["name"],
                    arguments.dump(),
                    // arguments.is_string() ? arguments.get<std::string>() : arguments.dump(),
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
            return {
                /* .role = */ "assistant",
                /* .content = */ input,
                /* .tool_calls = */ {},
            };
        }
    };
    return data;
}

static common_chat_data common_chat_init_without_tools(const common_chat_template & tmpl, const struct common_chat_params & params) {
    common_chat_data data;
    data.prompt = tmpl.apply(params.messages, params.tools.empty() ? json() : params.tools, params.add_generation_prompt);
    data.format = "content-only";
    data.parser = no_op_text_parser;
    data.grammar_lazy = false;
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
    auto has_tools = params.tools.is_null() || params.tool_choice == "none";
    if (has_tools && !params.grammar.empty()) {
        throw std::runtime_error("Cannot specify grammar with tools");
    }

    const auto & src = tmpl.source();
    if (src.find(">>>all") != std::string::npos) {
        // Functionary prepends "all\n" to plain content outputs, so we use the parser no matter when
        return common_chat_init_functionary_v3_2_tool_call(tmpl, params);
    }
    if (src.find(" functools[") != std::string::npos) {
        // Firefunction v2 requires datetime and functions in the context
        return common_chat_init_firefunction_v2_tool_call(tmpl, params);
    }

    if (has_tools) {
        return common_chat_init_without_tools(tmpl, params);
    }

    if (src.find("<tool_call>") != std::string::npos) {
        return common_chat_init_hermes_2_pro_tool_call(tmpl, params);
    }
    if (src.find("<|start_header_id|>") != std::string::npos
        && src.find("<function=") != std::string::npos) {
        return common_chat_init_functionary_v3_1_llama_3_1_tool_call(tmpl, params);
    }
    if (src.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos) {
        auto uses_python_tag = src.find("<|python_tag|>") != std::string::npos;

        if (uses_python_tag) {
            return common_chat_init_llama_3_1_python_tag_tool_calls(tmpl, params);
        } else {
            return common_chat_init_llama_3_2_tool_calls(tmpl, params);
        }
    }
    if (src.find("<｜tool▁calls▁begin｜>") != std::string::npos) {
        return common_chat_init_deepseek_r1_tool_call(tmpl, params);
    }
    if (src.find("[TOOL_CALLS]") != std::string::npos) {
        return common_chat_init_mistral_nemo_tool_call(tmpl, params);
    }
    return common_chat_init_generic_tool_call(tmpl, params);
}

