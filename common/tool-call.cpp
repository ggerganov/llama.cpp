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

std::string llama_tool_call_style_name(llama_tool_call_style style) {
    switch (style) {
        case llama_tool_call_style::None:
            return "None";
        case llama_tool_call_style::Generic:
            return "Generic";
        case llama_tool_call_style::Llama31:
            return "Llama-3.1";
        case llama_tool_call_style::Llama32:
            return "Llama-3.2";
        case llama_tool_call_style::FunctionaryV3Llama3:
            return "FunctionaryV3Llama3";
        case llama_tool_call_style::FunctionaryV3Llama31:
            return "FunctionaryV3Llama3.1";
        case llama_tool_call_style::Hermes2Pro:
            return "Hermes2Pro";
        case llama_tool_call_style::CommandRPlus:
            return "CommandRPlus";
        case llama_tool_call_style::MistralNemo:
            return "MistralNemo";
        default:
            return "Unknown";
    }
}

llama_tool_call_style llama_tool_call_style_detect(const minja::chat_template & chat_template) {
    const auto & src = chat_template.source();

    if (src.find("<tool_call>") != std::string::npos) {
        return Hermes2Pro;
    } else if (src.find(">>>all") != std::string::npos) {
        return FunctionaryV3Llama3;
    } else if (src.find("<|start_header_id|>") != std::string::npos
        && src.find("<function=") != std::string::npos) {
        return FunctionaryV3Llama31;
    } else if (src.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos) {
        if (src.find("<|python_tag|>") != std::string::npos) {
            return Llama31;
        } else {
            return Llama32;
        }
    } else if (src.find("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>") != std::string::npos) {
        return CommandRPlus;
    } else if (src.find("[TOOL_CALLS]") != std::string::npos) {
        return MistralNemo;
    } else {
        return Generic;
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
static llama_tool_calls parse_json_tool_calls(const json & tools, const std::string& input, const std::regex & function_regex, const std::regex & close_regex, bool check_names) {
    std::smatch match;

    llama_tool_calls result;
    auto end = input.end();
    auto it = input.begin();

    std::unordered_set<std::string> tool_names;
    if (check_names) {
        for (const auto & tool : tools) {
            if (tool.contains("type") && tool["type"] == "function") {
                tool_names.insert(tool["function"]["name"]);
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
        return {input, {}};
    }
}

static llama_tool_calls parse_llama_3_tool_calls(const json & tools, const std::string& input, bool allow_python_tag) {
    if (allow_python_tag) {
        static std::regex python_tag_regex(R"(<\|python_tag\|>([\s\S\n]*)$)");
        std::smatch match;
        if (std::regex_search(input, match, python_tag_regex)) {
            return {
                match.prefix().str(), {
                    {"ipython", (json {{"code", match[1].str()}}).dump()},
                }
            };
        }
    }
    static std::regex function_regex("\\{(?:\"type\": \"function\", |[\\s\\n\\r]*)\"name\": \"([^\"]+)\", \"parameters\": ");
    static std::regex close_regex("\\}");
    return parse_json_tool_calls(tools, input, function_regex, close_regex, /* check_names= */ true);
}

static llama_tool_calls parse_functionary_v3_llama_3_1_tool_calls(const json & tools, const std::string& input) {
    // This version of Functionary still supports the llama 3.1 tool call format for the python tool.
    static std::regex python_tag_regex(R"(<\|python_tag\|>([\s\S\n]*)$)");
    std::smatch match;
    if (std::regex_search(input, match, python_tag_regex)) {
        return {
            match.prefix().str(), {
                {"ipython", (json {{"code", match[1].str()}}).dump()},
            }
        };
    }
    static std::regex function_regex(R"(<function=(\w+)>)");
    static std::regex close_regex(R"(</function>)");
    return parse_json_tool_calls(tools, input, function_regex, close_regex, /* check_names= */ false);
}

static llama_tool_calls parse_functionary_v3_tool_calls(const json & tools, const std::string& input) {
    static std::regex function_regex(R"((?:>>>)?(\w+)\n)");
    static std::regex close_regex(R"($|(?=>>>))");
    return parse_json_tool_calls(tools, input, function_regex, close_regex, /* check_names= */ true);
}

static llama_tool_calls parse_generic_tool_calls(const std::string& input) {
    json data = json::parse(input);
    llama_tool_calls result;
    if (data.contains("tool_calls")) {
        for (const auto & tool_call : data["tool_calls"]) {
            result.tool_calls.push_back({
                tool_call["name"],
                tool_call["arguments"].dump(),
                /* id= */ "",
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

static llama_tool_calls parse_mistral_nemo_tool_calls(const std::string& input) {
    auto content_end = input.find("[TOOL_CALLS]");
    size_t tc_start = std::string::npos;
    if (content_end != std::string::npos) {
        tc_start = content_end + 12;
    } else {
        // Somehow not getting [TOOL_CALLS] in the output. Oh well, just do without it.
        content_end = input.find("[{\"");
        if (content_end == std::string::npos || content_end > 0) {
            return {input, {}};
        }
        tc_start = content_end;
    }
    llama_tool_calls result;
    result.content = input.substr(0, content_end);
    auto tool_calls = json::parse(input.substr(tc_start));
    for (const auto & tool_call : tool_calls) {
        const auto & arguments = tool_call["arguments"];
        result.tool_calls.push_back({
            tool_call["name"],
            arguments.is_string() ? arguments.get<std::string>() : arguments.dump(),
            tool_call["id"],
        });
    }
    return result;
}

llama_tool_calls parse_tool_calls(llama_tool_call_style style, const json & tools, const std::string& input) {
    // fprintf(stderr, "# parse_tool_calls:\n\n%s\n\n", input.c_str());
    switch (style) {
        case llama_tool_call_style::None:
            return {input, {}};
        case llama_tool_call_style::Generic:
            return parse_generic_tool_calls(input);
        case llama_tool_call_style::Llama31:
            return parse_llama_3_tool_calls(tools, input, /* parse_llama_3_tool_calls= */ true);
        case llama_tool_call_style::Llama32:
            return parse_llama_3_tool_calls(tools, input, /* parse_llama_3_tool_calls= */ false);
        case llama_tool_call_style::FunctionaryV3Llama3:
            return parse_functionary_v3_tool_calls(tools, input);
        case llama_tool_call_style::FunctionaryV3Llama31:
            return parse_functionary_v3_llama_3_1_tool_calls(tools, input);
        case llama_tool_call_style::Hermes2Pro:
            return parse_hermes_tool_calls(input);
        case llama_tool_call_style::MistralNemo:
            return parse_mistral_nemo_tool_calls(input);   
        default:
            throw std::runtime_error("Unsupported tool call style");
    }
}

static nlohmann::ordered_json add_system(const nlohmann::ordered_json & messages, const std::string & system_prompt) {
    json messages_with_system = messages;

    if (messages_with_system.size() > 0 && messages_with_system[0].at("role") == "system") {
        messages_with_system.at(0).at("content") += ("\n" + system_prompt);
    } else {
        messages_with_system.insert(messages_with_system.begin(), json {
            {"role", "system"},
            {"content", system_prompt},
        });
    }
    return messages_with_system;
}

llama_tool_call_handler llama_tool_call_handler_init(
    llama_tool_call_style style,
    const minja::chat_template & tmpl,
    bool allow_content,
    const nlohmann::ordered_json & parallel_tool_calls,
    const nlohmann::ordered_json & messages,
    const nlohmann::ordered_json & tools,
    const nlohmann::ordered_json & json_schema)
{
    llama_tool_call_handler handler;
    auto parallel = parallel_tool_calls.is_null() ? tmpl.supports_parallel_tool_calls() : parallel_tool_calls.get<bool>();

    switch (style) {
        case llama_tool_call_style::None:
            handler.prompt = tmpl.apply(messages, tools, /* add_generation_prompt= */ true);
            break;
        case llama_tool_call_style::Generic: {
            auto tool_call_schemas = json::array();
            for (const auto & tool : tools) {
                if (tool["type"] != "function") {
                    continue;
                }
                const auto & function = tool["function"];
                std::string name = function["name"];
                auto parameters = function["parameters"];
                tool_call_schemas.emplace_back(json {
                    {"type", "object"},
                    {"properties", {
                        {"name", {
                            {"type", "string"},
                            {"const", name},
                        }},
                        {"arguments", parameters},
                    }},
                    {"required", json::array({"name", "arguments"})},
                });
            }
            const auto tool_call =
                parallel
                    ? json {
                        {"type", "object"},
                        {"properties", {
                            {"tool_calls", {
                                {"type", "array"},
                                {"items", json {{"anyOf", tool_call_schemas}}}
                            }},
                        }},
                        {"required", json::array({"tool_calls"})},
                    }
                    : json {
                        {"type", "object"},
                        {"properties", {
                            {"tool_call", json {{"anyOf", tool_call_schemas}}},
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
                            },
                        })}
                    }
                    : tool_call;
            handler.grammar = build_grammar([&](const llama_grammar_builder & builder) {
                builder.add_schema("", schema);
            });
            // TODO: add schema to system prompt.
            auto tweaked_messages = add_system(
                messages, 
                "Respond in JSON format, either with a request to call tools or with a response to the user's request. Here is the schema for all responses:\n\n```json\n" + schema.dump(2) + "\n```");
            handler.prompt = tmpl.apply(tweaked_messages, tools, /* add_generation_prompt= */ true);
            break;
        }
        case llama_tool_call_style::MistralNemo: {
            handler.grammar = build_grammar([&](const llama_grammar_builder & builder) {
                auto schemas = json::array();
                for (const auto & tool : tools) {
                    if (tool["type"] != "function") {
                        continue;
                    }
                    const auto & function = tool["function"];
                    std::string name = function["name"];
                    auto parameters = function["parameters"];
                    auto schema = json {
                        {"type", "object"},
                        {"properties", {
                            // Important note: the model is probably trained to take a JSON stringified arguments value.
                            // It's hard to constrain that for now (while reusing the JSON schema conversion), so we're just expecting a plain object.
                            {"arguments", parameters},
                            {"name", {
                                {"type", "string"},
                                {"const", name},
                            }},
                            {"id", {
                                {"type", "string"},
                                // Nemo's template expects a 9-character alphanumeric ID.
                                {"pattern", "^[a-zA-Z0-9]{9}$"},
                            }},
                        }},
                        {"required", json::array({"arguments", "id", "name"})},
                    };
                    schemas.push_back(schema);
                }
                auto schema = json {
                    {"type", "array"},
                    {"items", json {{"anyOf", schemas}}},
                    {"minItems", 1},
                };
                if (!parallel) {
                    schema["maxItems"] = 1;
                }
                builder.add_schema("", schema);
            });
            if (allow_content) {
                handler.grammar_trigger_words.push_back("[TOOL_CALLS]");
                handler.grammar_trigger_words.push_back("[{\"");
            }
            auto tweaked_messages = add_system(messages, "Prefix any tool calls with [TOOL_CALLS]");
            handler.prompt = tmpl.apply(tweaked_messages, tools, /* add_generation_prompt= */ true);
            break;
        }
        case llama_tool_call_style::Llama31:
        case llama_tool_call_style::Llama32: {
            static auto builtin_tools = json {"wolfram_alpha", "brave_search"};

            auto uses_python_tag = style == llama_tool_call_style::Llama31;

            // Technically we should only trigger on `"\n{\"name\": \"" + name + "\""` for each tool name,
            // but Llama-3.2-3B (and 1B) struggles to output valid tool calls so we're "guiding" it strongly as soon
            // as it seems to be outputting some JSON.
            // TODO: make this conditional on a very small model (e.g. 1B / 3B).
            auto eagerly_match_any_json = style == llama_tool_call_style::Llama32;

            handler.grammar = build_grammar([&](const llama_grammar_builder & builder) {
                std::vector<std::string> tool_rules;

                for (const auto & tool : tools) {
                    const auto & function = tool["function"];
                    std::string name = function["name"];
                    auto parameters = function["parameters"];
                    builder.resolve_refs(parameters);
                    if (uses_python_tag && (name == "ipython" || builtin_tools.contains(name))) {
                        tool_rules.push_back(builder.add_rule("ipython-call", "\"<|python_tag|>\" .*"));
                        if (allow_content) {
                            handler.grammar_trigger_words.push_back("<|python_tag|>");
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
                            handler.grammar_trigger_words.push_back("{\"name\": \"" + name + "\"");
                            // Accommodate most common tool call variations from Llama-3.1-8B and Llama-3.2-3B.
                            // Note that c++11's regex doesn't support partial matches, otherwise it would make
                            // sense to add support for trigger regexes to the antiprompt mechanism.
                            handler.grammar_trigger_words.push_back("{\n\t\"name\": \"" + name + "\"");
                            handler.grammar_trigger_words.push_back("{\n  \"name\": \"" + name + "\"");
                            handler.grammar_trigger_words.push_back("{\n    \"name\": \"" + name + "\"");
                            handler.grammar_trigger_words.push_back("{\"type\": \"function\", \"name\": \"" + name + "\"");
                        }
                    }
                }

                if (allow_content && eagerly_match_any_json) {
                    handler.grammar_trigger_words.push_back("{\"");
                    handler.grammar_trigger_words.push_back("{\n\t\"");
                    handler.grammar_trigger_words.push_back("{\n  \"");
                    handler.grammar_trigger_words.push_back("{\n    \"");
                }

                builder.add_rule("root", join(tool_rules.begin(), tool_rules.end(), " | "));
            });
            handler.additional_stop_words.push_back("<|eom_id|>");
            handler.prompt = tmpl.apply(messages, tools, /* add_generation_prompt= */ true, {
                {"builtin_tools", builtin_tools},
            });
            break;
        }
        case llama_tool_call_style::FunctionaryV3Llama3: {
            // >>>all\nlet's call functions>>>fn1\n{"arg1": 1...}\n>>>fn2\n{"arg1": 1...}...
            // Using ">>>f1\n", ">>>f2\n"... as trigger words for the grammar
            handler.grammar = build_grammar([&](const llama_grammar_builder & builder) {
                std::vector<std::string> first_tool_rules;
                std::vector<std::string> subsequent_tool_rules;
                for (size_t i = 0, n = tools.size(); i < n; i++) {
                    auto & tool = tools[i];
                    const auto & function = tool["function"];
                    std::string name = function["name"];
                    auto parameters = function["parameters"];
                    auto args_rule = builder.add_schema(name + "-args", parameters);
                    first_tool_rules.push_back(builder.add_rule(name + "-call", "\"" + name + "\\n\" " + args_rule));
                    subsequent_tool_rules.push_back(builder.add_rule(name + "-call2", "\">>>" + name + "\\n\" " + args_rule));
                    if (allow_content) {
                        handler.grammar_trigger_words.push_back(name + "\n");
                        handler.grammar_trigger_words.push_back(">>>" + name + "\n");
                    }
                }
                auto first_rule = builder.add_rule("first_tool_call", join(first_tool_rules.begin(), first_tool_rules.end(), " | ")) + " space";
                if (parallel) {
                    auto subsequent_rule = builder.add_rule("subsequent_tool_call", join(subsequent_tool_rules.begin(), subsequent_tool_rules.end(), " | ")) + " space";
                    builder.add_rule("root", first_rule + " (" + subsequent_rule + ")*");
                } else {
                    builder.add_rule("root", first_rule);
                }
            });
            handler.prompt = tmpl.apply(messages, tools, /* add_generation_prompt= */ true);
            // handler.parser = parse_functionary_3_2_tool_calls;
            break;
        }
        case llama_tool_call_style::FunctionaryV3Llama31: {
            // ./tests/chat/templates/meetkai-functionary-medium-v3.1.jinja
            // https://github.com/MeetKai/functionary/blob/main/tests/prompt_test_v3-llama3.1.txt
            // TODO: handle tool {type: code_interpreter} as python
            handler.grammar = build_grammar([&](const llama_grammar_builder & builder) {
                std::vector<std::string> tool_rules;
                for (size_t i = 0, n = tools.size(); i < n; i++) {
                    auto & tool = tools[i];
                    const auto & function = tool["function"];
                    std::string name = function["name"];
                    auto parameters = function["parameters"];
                    if (name == "python") {
                        tool_rules.push_back(builder.add_rule("python-call", "\"<|python_tag|>\" .*"));
                        if (allow_content) {
                            handler.grammar_trigger_words.push_back("<|python_tag|>");
                        }
                    } else {
                        tool_rules.push_back(builder.add_rule(name + "-call", "\"<function=" + name + ">\" " + builder.add_schema(name + "-args", parameters) + " \"</function>\" space"));
                    }
                }
                auto tool_call = builder.add_rule("tool_call", join(tool_rules.begin(), tool_rules.end(), " | ")) + " space";
                builder.add_rule("root", parallel ? "(" + tool_call + ")+" : tool_call);
                if (allow_content) {
                    handler.grammar_trigger_words.push_back("<function=");
                }
            });
            handler.prompt = tmpl.apply(messages, tools, /* add_generation_prompt= */ true);
            // handler.parser = parse_functionary_3_2_tool_calls;
            break;
        }
        case llama_tool_call_style::Hermes2Pro: {
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

                auto tool_call = "\"<tool_call>\" space " + builder.add_rule("tool_call", join(tool_rules.begin(), tool_rules.end(), " | ")) + " \"</tool_call>\" space";
                builder.add_rule("root", parallel ? "(" + tool_call + ")+" : tool_call);
                if (allow_content) {
                    handler.grammar_trigger_words.push_back("<tool_call>");
                }
            });
            handler.prompt = tmpl.apply(messages, tools, /* add_generation_prompt= */ true);
            break;
        }
        default:
            throw std::runtime_error("Unsupported tool call style");
    }
    return handler;
}
