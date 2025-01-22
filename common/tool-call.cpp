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

static json normalize_tools(const json & tools) {
    static const auto python_tool = json::parse(R"({
        "type": "function",
        "function": {
            "name": "python",
            "description": "Runs code in an Python interpreter and returns the result of the execution after 60 seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to run in the Python interpreter."
                    }
                },
                "required": ["code"]
            }
        }
    })");

    auto results = json::array();
    for (const auto & tool : tools) {
        if (!tool.contains("type")) {
            continue;
        }
        if (tool["type"] == "code_interpreter") {
            results.push_back(python_tool);
        } else if (tool["type"] == "function") {
            results.push_back(tool);
        } else {
            continue;
        }
    }
    return results;
}

std::string common_tool_call_style_name(common_tool_call_style style) {
    switch (style) {
        case COMMON_TOOL_CALL_STYLE_NONE:
            return "None";
        case COMMON_TOOL_CALL_STYLE_GENERIC:
            return "Generic";
        case COMMON_TOOL_CALL_STYLE_LLAMA_3_1:
            return "Llama-3.1";
        case COMMON_TOOL_CALL_STYLE_LLAMA_3_2:
            return "Llama-3.2";
        case COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3:
            return "FunctionaryV3Llama3";
        case COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3_1:
            return "FunctionaryV3Llama3.1";
        case COMMON_TOOL_CALL_STYLE_HERMES_2_PRO:
            return "Hermes2Pro";
        case COMMON_TOOL_CALL_STYLE_COMMAND_R_PLUS:
            return "CommandRPlus";
        case COMMON_TOOL_CALL_STYLE_MISTRAL_NEMO:
            return "MistralNemo";
        case COMMON_TOOL_CALL_STYLE_FIRE_FUNCTION_V2:
            return "FirefunctionV2";
        default:
            return "Unknown";
    }
}

common_tool_call_style common_tool_call_style_detect(const common_chat_template & chat_template) {
    const auto & src = chat_template.source();

    if (src.find("<tool_call>") != std::string::npos) {
        return COMMON_TOOL_CALL_STYLE_HERMES_2_PRO;
    } else if (src.find(">>>all") != std::string::npos) {
        return COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3;
    } else if (src.find("<|start_header_id|>") != std::string::npos
        && src.find("<function=") != std::string::npos) {
        return COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3_1;
    } else if (src.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos) {
        if (src.find("<|python_tag|>") != std::string::npos) {
            return COMMON_TOOL_CALL_STYLE_LLAMA_3_1;
        } else {
            return COMMON_TOOL_CALL_STYLE_LLAMA_3_2;
        }
    } else if (src.find("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>") != std::string::npos) {
        return COMMON_TOOL_CALL_STYLE_COMMAND_R_PLUS;
    } else if (src.find("[TOOL_CALLS]") != std::string::npos) {
        return COMMON_TOOL_CALL_STYLE_MISTRAL_NEMO;
    } else if (src.find(" functools[") != std::string::npos) {
        return COMMON_TOOL_CALL_STYLE_FIRE_FUNCTION_V2;
    } else {
        return COMMON_TOOL_CALL_STYLE_GENERIC;
    }
}

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

    std::unordered_set<std::string> tool_names;
    if (check_names) {
        for (const auto & tool : tools) {
            if (!tool.contains("type")) {
                continue;
            }
            std::string type = tool.at("type");
            if (type == "function") {
                tool_names.insert(tool["function"]["name"]);
            } else if (type == "code_interpreter") {
                tool_names.insert("python");
            }
        }
    }

    while (it != end) {
        std::sregex_iterator rend;
        std::sregex_iterator rit(it, end, function_regex);
        if (rit == rend) {
            result.content += std::string(it, end);
            break;
        }
        auto name = rit->str(1);
        if (check_names && tool_names.find(name) == tool_names.end()) {
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
        result.tool_calls.push_back({name, arguments.dump(), /* id= */ ""});
    }
    return result;
}

static common_chat_msg parse_hermes_tool_calls(const std::string& input) {
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
}

static common_chat_msg parse_llama_3_tool_calls(const json & tools, const std::string& input, bool allow_python_tag) {
    if (allow_python_tag) {
        static std::regex python_tag_regex(R"(<\|python_tag\|>([\s\S\n]*)$)");
        std::smatch match;
        if (std::regex_search(input, match, python_tag_regex)) {
            return {
                /* .role = */ "assistant",
                /* .content = */ match.prefix().str(),
                /* .tool_calls = */ {
                    {
                        /* .name = */ "python",
                        /* .arguments = */ (json {{"code", match[1].str()}}).dump(),
                        /* .id = */ "",
                    },
                }
            };
        }
    }
    static std::regex function_regex("\\{(?:\"type\": \"function\", |[\\s\\n\\r]*)\"name\": \"([^\"]+)\", \"parameters\": ");
    static std::regex close_regex("\\}");
    return parse_json_tool_calls(tools, input, function_regex, close_regex, /* check_names= */ true);
}

static common_chat_msg parse_functionary_v3_llama_3_1_tool_calls(const json & tools, const std::string& input) {
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
                    /* .arguments = */ (json {{"code", match[1].str()}}).dump(),
                    /* .id = */ "",
                },
            }
        };
    }
    static std::regex function_regex(R"(<function=(\w+)>)");
    static std::regex close_regex(R"(</function>)");
    return parse_json_tool_calls(tools, input, function_regex, close_regex, /* check_names= */ false);
}

static common_chat_msg parse_functionary_v3_tool_calls(const json & tools, const std::string& input) {
    static std::regex function_regex(R"((?:>>>)?(\w+)\n)");
    static std::regex close_regex(R"($|(?=>>>))");
    return parse_json_tool_calls(tools, input, function_regex, close_regex, /* check_names= */ true);
}

static common_chat_msg parse_generic_tool_calls(const std::string& input) {
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

static common_chat_msg parse_mistral_nemo_tool_calls(const std::string& input) {
    return parse_prefixed_json_tool_call_array(input, "[TOOL_CALLS]");
}

static common_chat_msg parse_firefunction_v2_tool_calls(const std::string& input) {
    return parse_prefixed_json_tool_call_array(input, " functools[", /* rstrip_prefix= */ 1);
}

common_chat_msg parse_tool_calls(common_tool_call_style style, const json & tools, const std::string& input) {
    fprintf(stderr, "# parse_tool_calls(%s):\n\n%s\n\n", common_tool_call_style_name(style).c_str(), input.c_str());
    switch (style) {
        case COMMON_TOOL_CALL_STYLE_NONE:
            return {"assistant", input, {}};
        case COMMON_TOOL_CALL_STYLE_GENERIC:
            return parse_generic_tool_calls(input);
        case COMMON_TOOL_CALL_STYLE_LLAMA_3_1:
            return parse_llama_3_tool_calls(tools, input, /* parse_llama_3_tool_calls= */ true);
        case COMMON_TOOL_CALL_STYLE_LLAMA_3_2:
            return parse_llama_3_tool_calls(tools, input, /* parse_llama_3_tool_calls= */ false);
        case COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3:
            return parse_functionary_v3_tool_calls(tools, input);
        case COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3_1:
            return parse_functionary_v3_llama_3_1_tool_calls(tools, input);
        case COMMON_TOOL_CALL_STYLE_HERMES_2_PRO:
            return parse_hermes_tool_calls(input);
        case COMMON_TOOL_CALL_STYLE_MISTRAL_NEMO:
            return parse_mistral_nemo_tool_calls(input);
        case COMMON_TOOL_CALL_STYLE_FIRE_FUNCTION_V2:
            return parse_firefunction_v2_tool_calls(input);
        default:
            throw std::runtime_error("Unsupported tool call style");
    }
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

common_tool_call_handler common_tool_call_handler_init(
    common_tool_call_style style,
    const common_chat_template & tmpl,
    bool allow_content,
    const nlohmann::ordered_json & parallel_tool_calls,
    const nlohmann::ordered_json & messages,
    const nlohmann::ordered_json & tools,
    const nlohmann::ordered_json & json_schema)
{
    common_grammar_options grammar_options {
        /* .dotall = */ false,
        /* .compact_spaces = */ true,
    };
    common_tool_call_handler handler;
    auto parallel = parallel_tool_calls.is_null() ? tmpl.supports_parallel_tool_calls() : parallel_tool_calls.get<bool>();

    switch (style) {
        case COMMON_TOOL_CALL_STYLE_NONE:
            handler.prompt = tmpl.apply(messages, tools, /* add_generation_prompt= */ true);
            break;
        case COMMON_TOOL_CALL_STYLE_GENERIC: {
            auto actual_tools = normalize_tools(tools);
            auto tool_call_schemas = json::array();
            for (const auto & tool : actual_tools) {
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
                if (parallel) {
                    tool_schema["properties"]["id"] = {
                        {"type", "string"},
                        {"minLength", 4},
                    };
                    tool_schema["required"].push_back("id");
                }
                tool_call_schemas.emplace_back(tool_schema);
            }
            const auto tool_call =
                parallel
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
                allow_content
                    ? json {
                        {"anyOf", json::array({
                            tool_call,
                            {
                                {"type", "object"},
                                {"properties", {
                                    {"response", json_schema.is_null()
                                        ? json {{"type", "string"}}
                                        : json_schema
                                    },
                                }},
                                {"required", json::array({"response"})},
                            },
                        })}
                    }
                    : tool_call;
            handler.grammar = build_grammar([&](const common_grammar_builder & builder) {
                builder.add_schema("root", schema);
            }, grammar_options);
            // TODO: add schema to system prompt.
            auto tweaked_messages = add_system(
                messages,
                "Respond in JSON format, either with a request to call tools or with a response to the user's request. Here is the schema for all responses:\n\n```json\n" + schema.dump(2) + "\n```");
            handler.prompt = tmpl.apply(tweaked_messages, actual_tools.empty() ? json() : actual_tools, /* add_generation_prompt= */ true);
            break;
        }
        case COMMON_TOOL_CALL_STYLE_MISTRAL_NEMO: {
            auto actual_tools = normalize_tools(tools);
            handler.grammar = build_grammar([&](const common_grammar_builder & builder) {
                auto schemas = json::array();
                for (const auto & tool : actual_tools) {
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
                }
                auto schema = json {
                    {"type", "array"},
                    {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
                    {"minItems", 1},
                };
                if (!parallel) {
                    schema["maxItems"] = 1;
                }
                builder.add_rule("root", "\"[TOOL_CALLS]\" " + builder.add_schema("tool_calls", schema));
            }, grammar_options);
            if (allow_content) {
                handler.grammar_triggers.push_back("[TOOL_CALLS]");
            }
            handler.prompt = tmpl.apply(messages, actual_tools.empty() ? json() : actual_tools, /* add_generation_prompt= */ true);
            break;
        }
        case COMMON_TOOL_CALL_STYLE_FIRE_FUNCTION_V2: {
            auto actual_tools = normalize_tools(tools);
            handler.grammar = build_grammar([&](const common_grammar_builder & builder) {
                auto schemas = json::array();
                for (const auto & tool : actual_tools) {
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
                }
                auto schema = json {
                    {"type", "array"},
                    {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
                    {"minItems", 1},
                };
                if (!parallel) {
                    schema["maxItems"] = 1;
                }
                builder.add_rule("root", "\" functools\"? " + builder.add_schema("tool_calls", schema));
            }, grammar_options);
            if (allow_content) {
                handler.grammar_triggers.push_back(" functools[");
            }
            handler.prompt = tmpl.apply(messages, actual_tools.empty() ? json() : actual_tools, /* add_generation_prompt= */ true);
            break;
        }
        case COMMON_TOOL_CALL_STYLE_LLAMA_3_1:
        case COMMON_TOOL_CALL_STYLE_LLAMA_3_2: {
            auto builtin_tools = json {"wolfram_alpha", "brave_search"};
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    continue;
                }
                if (tool["type"] == "code_interpreter") {
                    builtin_tools.push_back("code_interpreter");
                    break;
                }
            }
            auto actual_tools = normalize_tools(tools);

            auto uses_python_tag = style == common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_1;

            // Technically we should only trigger on `"\n{\"name\": \"" + name + "\""` for each tool name,
            // but Llama-3.2-3B (and 1B) struggles to output valid tool calls so we're "guiding" it strongly as soon
            // as it seems to be outputting some JSON.
            // TODO: make this conditional on a very small model (e.g. 1B / 3B).
            auto eagerly_match_any_json = style == common_tool_call_style::COMMON_TOOL_CALL_STYLE_LLAMA_3_2;

            handler.grammar = build_grammar([&](const common_grammar_builder & builder) {
                std::vector<std::string> tool_rules;

                for (const auto & tool : actual_tools) {
                    const auto & function = tool["function"];
                    std::string name = function["name"];
                    auto parameters = function["parameters"];
                    builder.resolve_refs(parameters);
                    if (uses_python_tag && (name == "ipython" || builtin_tools.contains(name))) {
                        tool_rules.push_back(builder.add_rule("ipython-call", "\"<|python_tag|>\" .*"));
                        if (allow_content) {
                            handler.grammar_triggers.push_back("<|python_tag|>");
                        }
                    } else {
                        //"<|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"" + name + "\", " +
                        tool_rules.push_back(
                            builder.add_rule(
                                name + "-call",
                                "\"\\n\"? \"{\" ( \"\\\"type\\\": \\\"function\\\", \" | space ) \"\\\"name\\\": \\\"" + name + "\\\", \\\"parameters\\\": \" " +
                                    builder.add_schema(name + "-args", parameters) +
                                " \"}\""));
                        if (allow_content && !eagerly_match_any_json) {
                            handler.grammar_triggers.push_back("{\"name\": \"" + name + "\"");
                            // Accommodate most common tool call variations from Llama-3.1-8B and Llama-3.2-3B.
                            // Note that c++11's regex doesn't support partial matches, otherwise it would make
                            // sense to add support for trigger regexes to the antiprompt mechanism.
                            handler.grammar_triggers.push_back("{\n\t\"name\": \"" + name + "\"");
                            handler.grammar_triggers.push_back("{\n  \"name\": \"" + name + "\"");
                            handler.grammar_triggers.push_back("{\n    \"name\": \"" + name + "\"");
                            handler.grammar_triggers.push_back("{\"type\": \"function\", \"name\": \"" + name + "\"");
                        }
                    }
                }

                if (allow_content && eagerly_match_any_json) {
                    handler.grammar_triggers.push_back("{\"");
                    handler.grammar_triggers.push_back("{\n\t\"");
                    handler.grammar_triggers.push_back("{\n  \"");
                    handler.grammar_triggers.push_back("{\n    \"");
                }

                builder.add_rule("root", string_join(tool_rules, " | "));
            }, grammar_options);
            handler.additional_stops.push_back("<|eom_id|>");
            handler.prompt = tmpl.apply(messages, actual_tools.empty() ? json() : actual_tools, /* add_generation_prompt= */ true, {
                {"builtin_tools", builtin_tools},
            });
            break;
        }
        case COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3: {
            // >>>all\nlet's call functions>>>fn1\n{"arg1": 1...}\n>>>fn2\n{"arg1": 1...}...
            // Using ">>>f1\n", ">>>f2\n"... as trigger words for the grammar
            auto actual_tools = normalize_tools(tools);
            handler.grammar = build_grammar([&](const common_grammar_builder & builder) {
                std::vector<std::string> first_tool_rules;
                std::vector<std::string> subsequent_tool_rules;
                for (const auto & tool : actual_tools) {
                    const auto & function = tool["function"];
                    std::string name = function["name"];
                    auto parameters = function["parameters"];
                    auto args_rule = builder.add_schema(name + "-args", parameters);
                    first_tool_rules.push_back(builder.add_rule(name + "-call", "\"" + name + "\\n\" " + args_rule));
                    subsequent_tool_rules.push_back(builder.add_rule(name + "-call2", "\"\\n>>>" + name + "\\n\" " + args_rule));
                    if (allow_content) {
                        handler.grammar_triggers.push_back(name + "\n");
                        handler.grammar_triggers.push_back("\n>>>" + name + "\n");
                    }
                }
                auto first_rule = builder.add_rule("first_tool_call", string_join(first_tool_rules, " | ")) + " space";
                if (parallel) {
                    auto subsequent_rule = builder.add_rule("subsequent_tool_call", string_join(subsequent_tool_rules, " | ")) + " space";
                    builder.add_rule("root", first_rule + " (" + subsequent_rule + ")*");
                } else {
                    builder.add_rule("root", first_rule);
                }
            }, grammar_options);
            handler.prompt = tmpl.apply(messages, actual_tools.empty() ? json() : actual_tools, /* add_generation_prompt= */ true);
            // handler.parser = parse_functionary_3_2_tool_calls;
            break;
        }
        case COMMON_TOOL_CALL_STYLE_FUNCTIONARY_V3_LLAMA_3_1: {
            // ./tests/chat/templates/meetkai-functionary-medium-v3.1.jinja
            // https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3-llama3.1.txt
            // TODO: handle tool {type: code_interpreter} as python
            auto actual_tools = normalize_tools(tools);
            handler.grammar = build_grammar([&](const common_grammar_builder & builder) {
                std::vector<std::string> tool_rules;
                for (const auto & tool : actual_tools) {
                    const auto & function = tool["function"];
                    std::string name = function["name"];
                    auto parameters = function["parameters"];
                    if (name == "python" || name == "ipython") {
                        tool_rules.push_back(builder.add_rule("python-call", "\"<|python_tag|>\" .*"));
                        if (allow_content) {
                            handler.grammar_triggers.push_back("<|python_tag|>");
                        }
                    } else {
                        tool_rules.push_back(builder.add_rule(name + "-call", "\"<function=" + name + ">\" " + builder.add_schema(name + "-args", parameters) + " \"</function>\" space"));
                    }
                }
                auto tool_call = builder.add_rule("tool_call", string_join(tool_rules, " | ")) + " space";
                builder.add_rule("root", parallel ? "(" + tool_call + ")+" : tool_call);
                if (allow_content) {
                    handler.grammar_triggers.push_back("<function=");
                }
            }, grammar_options);
            handler.prompt = tmpl.apply(messages, actual_tools.empty() ? json() : actual_tools, /* add_generation_prompt= */ true);
            // handler.parser = parse_functionary_3_2_tool_calls;
            break;
        }
        case COMMON_TOOL_CALL_STYLE_HERMES_2_PRO: {
            // NousResearchHermesPro_2
            // (content)?(<tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>)*
            auto actual_tools = normalize_tools(tools);
            handler.grammar = build_grammar([&](const common_grammar_builder & builder) {
                std::vector<std::string> tool_rules;
                for (const auto & tool : actual_tools) {
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

                auto tool_call = "\"<tool_call>\" space " + builder.add_rule("tool_call", string_join(tool_rules, " | ")) + " \"</tool_call>\" space";
                builder.add_rule("root", parallel ? "(" + tool_call + ")+" : tool_call);
                if (allow_content) {
                    handler.grammar_triggers.push_back("<tool_call>");
                }
            }, grammar_options);
            handler.prompt = tmpl.apply(messages, actual_tools.empty() ? json() : actual_tools, /* add_generation_prompt= */ true);
            break;
        }
        default:
            throw std::runtime_error("Unsupported tool call style");
    }
    return handler;
}
