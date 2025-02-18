//  Tests chat handling, including grammar generation and parsing for tool calling, for various templates.
//
//  Also acts as a CLI to generate a Markdown summary of the formats of Jinja templates,
//  e.g. given Minja (http://github.com/google/minja) checked out in parent dir:
//
//    cmake -B build && cmake --build build --parallel && ./build/bin/test-chat ../minja/build/tests/*.jinja 2>/dev/null
//
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <string>

#include "chat-template.hpp"
#include "chat.hpp"
#include "llama-grammar.h"
#include "unicode.h"

using json = nlohmann::ordered_json;

static common_chat_msg msg_from_json(const json & message) {
    common_chat_msg ret;
    ret.role = "assistant";
    if (message.contains("content") && !message.at("content").is_null()) {
        ret.content = message.at("content");
    }
    if (message.contains("tool_plan")) {
        ret.reasoning_content = message.at("tool_plan");
    }
    if (message.contains("reasoning_content")) {
        ret.reasoning_content = message.at("reasoning_content");
    }
    auto has_tool_calls = message.contains("tool_calls");
    if (has_tool_calls) {
        for (const auto & tc : message.at("tool_calls")) {
            const auto & arguments = tc.at("function").at("arguments");
            ret.tool_calls.push_back({
                tc.at("function").at("name").get<std::string>(),
                arguments.is_string() ? arguments.get<std::string>() : arguments.dump(),
                tc.contains("id") ? tc.at("id").get<std::string>() : "",
            });
        }
    }
    return ret;
}

template <class T> static void assert_equals(const T & expected, const T & actual) {
    if (expected != actual) {
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

static std::string read_file(const std::string & path) {
    std::cerr << "# Reading: " << path << std::endl << std::flush;
    std::ifstream fs(path, std::ios_base::binary);
    if (!fs.is_open()) {
        fs = std::ifstream("../" + path, std::ios_base::binary);
        if (!fs.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }
    }
    fs.seekg(0, std::ios_base::end);
    auto size = fs.tellg();
    fs.seekg(0);
    std::string out;
    out.resize(static_cast<size_t>(size));
    fs.read(&out[0], static_cast<std::streamsize>(size));
    return out;
}

static std::unique_ptr<llama_grammar> build_grammar(const std::string & grammar_str) {
    return std::unique_ptr<llama_grammar>(
        llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0));
}

// TODO: extract to common helper (copied from test-grammar-integration.cpp)
static bool match_string(const std::string & input, llama_grammar * grammar) {
    const auto cpts = unicode_cpts_from_utf8(input);

    auto & stacks_cur = llama_grammar_get_stacks(grammar);

    for (const auto & cpt : cpts) {
        llama_grammar_accept(grammar, cpt);

        if (stacks_cur.empty()) {
            // no stacks means that the grammar failed to match at this point
            return false;
        }
    }

    for (const auto & stack : stacks_cur) {
        if (stack.empty()) {
            // An empty stack means that the grammar has been completed
            return true;
        }
    }

    return false;
}

// Dumps `{"a": 1}` as `"{\"a\": 1}"`, unlike nlohmann::json::dump which would dump it as `"{\"a\":1}"`.
static std::string dump(const json & j) {
    return minja::Value(j).dump(-1, /* to_json= */ true);
}

static void assert_msg_equals(const common_chat_msg & expected, const common_chat_msg & actual) {
    assert_equals(expected.role, actual.role);
    assert_equals(expected.content, actual.content);
    assert_equals(expected.reasoning_content, actual.reasoning_content);
    assert_equals(expected.tool_calls.size(), actual.tool_calls.size());
    for (size_t i = 0; i < expected.tool_calls.size(); i++) {
        const auto & expected_tool_call = expected.tool_calls[i];
        const auto & actual_tool_call   = actual.tool_calls[i];
        assert_equals(expected_tool_call.name, actual_tool_call.name);
        assert_equals(dump(json::parse(expected_tool_call.arguments)), dump(json::parse(actual_tool_call.arguments)));
        assert_equals(expected_tool_call.id, actual_tool_call.id);
    }
}

const auto special_function_tool = json::parse(R"({
  "type": "function",
  "function": {
    "name": "special_function",
    "description": "I'm special",
    "parameters": {
      "type": "object",
      "properties": {
        "arg1": {
          "type": "integer",
          "description": "The arg."
        }
      },
      "required": ["arg1"]
    }
  }
})");
const auto python_tool           = json::parse(R"({
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
const auto code_interpreter_tool = json::parse(R"({
  "type": "function",
  "function": {
    "name": "code_interpreter",
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
const json tools                 = { special_function_tool, python_tool };
const json llama_3_1_tools       = { special_function_tool, code_interpreter_tool };

struct delta_data {
    std::string        delta;
    common_chat_params params;
};

static delta_data init_delta(const common_chat_template & tmpl, const std::vector<std::string> & end_tokens,
                             const json & user_message, const json & delta_message, const json & tools,
                             const json & tool_choice,
                             bool think = false) {
    common_chat_inputs inputs;
    inputs.parallel_tool_calls = true;
    inputs.messages            = json::array();
    inputs.messages.push_back(user_message);
    inputs.tools       = tools;
    inputs.tool_choice = tool_choice;
    inputs.extract_reasoning = think;
    auto params_prefix = common_chat_params_init(tmpl, inputs);

    inputs.messages.push_back(delta_message);
    inputs.add_generation_prompt = false;
    auto params_full             = common_chat_params_init(tmpl, inputs);

    std::string prefix = params_prefix.prompt;
    std::string full   = params_full.prompt;

    if (full == prefix) {
        throw std::runtime_error("Full message is the same as the prefix");
    }

    size_t common_prefix_length = 0;
    for (size_t i = 0; i < prefix.size() && i < full.size(); ++i) {
        if (prefix[i] != full[i]) {
            break;
        }
        if (prefix[i] == '<') {
            // DeepSeek R1's template (as of 20250209) adds a trailing <think> if add_generation_prompt,
            // but it removes thinking tags for past messages.
            // The prefix and full strings diverge at <think> vs. <｜tool▁calls▁begin｜>, we avoid consuming the leading <.
            continue;
        }
        common_prefix_length = i + 1;
    }
    auto delta = full.substr(common_prefix_length);

    // Strip end tokens
    for (const auto & end_token : end_tokens) {
        // rfind to find the last occurrence
        auto pos = delta.rfind(end_token);
        if (pos != std::string::npos) {
            delta = delta.substr(0, pos);
            break;
        }
    }
    return { delta, params_full };
}

/*
  Applies the template to 1 user message w/ add_generation_prompt=true, then w/ the test message w/ add_generation_prompt=false,
  gets the diff, removes any end tokens and parses the result w/ the grammar, checking that
  the parsed message is the same as the test_message
*/
static void test_template(const common_chat_template & tmpl, const std::vector<std::string> & end_tokens,
                          const json & test_message, const json & tools = {}, const std::string & expected_delta = "",
                          bool expect_grammar_triggered = true,
                          bool test_grammar_if_triggered = true,
                          bool think = false) {
    common_chat_msg expected_msg = msg_from_json(test_message);

    auto user_message = json{
        { "role",    "user"          },
        { "content", "Hello, world!" }
    };

    for (const auto & tool_choice : json({ "auto", "required" })) {
        auto data = init_delta(tmpl, end_tokens, user_message, test_message, tools, tool_choice, think);
        if (!expected_delta.empty()) {
            assert_equals(expected_delta, data.delta);
        }

        if (expect_grammar_triggered) {
            const auto msg = common_chat_parse(data.delta, data.params.format);
            assert_msg_equals(expected_msg, msg);
        }

        if (!expected_msg.tool_calls.empty()) {
            GGML_ASSERT(!data.params.grammar.empty());
        }
        if (!data.params.grammar.empty()) {
            auto grammar = build_grammar(data.params.grammar);
            if (!grammar) {
                throw std::runtime_error("Failed to build grammar");
            }
            auto earliest_trigger_pos = std::string::npos;
            auto constrained = data.delta;
            for (const auto & trigger : data.params.grammar_triggers) {
                auto pos = constrained.find(trigger.word);
                if (pos == std::string::npos) {
                    continue;
                }
                if (pos > 0 && trigger.at_start) {
                    fprintf(stderr, "Trigger %s not at start of message, skipping:\n\n%s\n\n", trigger.word.c_str(), constrained.c_str());
                    continue;
                }
                if (earliest_trigger_pos == std::string::npos || pos < earliest_trigger_pos) {
                    earliest_trigger_pos = pos;
                }
            }
            auto grammar_triggered = false;
            if (earliest_trigger_pos != std::string::npos) {
                constrained = constrained.substr(earliest_trigger_pos);
                grammar_triggered = true;
            }
            if (data.params.grammar_lazy) {
                assert_equals(expect_grammar_triggered, grammar_triggered);
            }

            if (grammar_triggered && test_grammar_if_triggered && !match_string(constrained, grammar.get())) {
                throw std::runtime_error("Failed to match delta against grammar:\n\n" + data.delta +
                                            "\n\nGrammar: " + data.params.grammar);
            }
        }
    }
}

static void test_template_output_parsers() {
    json message_user {
        { "role",    "user"     },
        { "content", "Hey there!" },
    };
    json message_assist {
        { "role",    "assistant"     },
        { "content", "Hello, world!\nWhat's up?" },
    };
    json message_assist_thoughts_unparsed_think {
        { "role",    "assistant"     },
        { "content", "<think>I'm thinking</think>Hello, world!\nWhat's up?" },
    };
    json message_assist_thoughts_unparsed_r7b {
        { "role",    "assistant"     },
        { "content", "<|START_THINKING|>I'm thinking<|END_THINKING|>Hello, world!\nWhat's up?" },
    };
    json message_assist_thoughts {
        { "role",    "assistant"     },
        { "content", "Hello, world!\nWhat's up?" },
        { "reasoning_content", "I'm thinking" },
    };
    json tool_calls = json::array({{
        { "type", "function" },
        { "function", { { "name", "special_function" }, { "arguments", "{\"arg1\": 1}" } } },
    }});

    json message_assist_call {
        { "role",       "assistant"},
        { "content",    {}},
        { "tool_calls", {
            {
                { "type", "function" },
                { "function", {
                    { "name", "special_function" },
                    { "arguments", "{\"arg1\": 1}" },
                }},
            },
        }},
    };
    json message_assist_call_thoughts = {
        { "role",       "assistant"                },
        { "content",    nullptr                    },
        { "reasoning_content",   "I'm\nthinking"              },
        { "tool_calls",  {
            {
                { "type", "function" },
                { "function", {
                    { "name", "special_function" },
                    { "arguments", "{\"arg1\": 1}" },
                }},
            },
        }},
    };
    json message_assist_call_thoughts_unparsed = {
        { "role",       "assistant"                },
        { "content",    "<think>I'm\nthinking</think>" },
        { "tool_calls",  {
            {
                { "type", "function" },
                { "function", {
                    { "name", "special_function" },
                    { "arguments", "{\"arg1\": 1}" },
                }},
            },
        }},
    };
    json message_assist_call_id {
        { "role",       "assistant"},
        { "content",    {}},
        { "tool_calls", {
            {
                { "type", "function" },
                { "function", {
                    { "name", "special_function" },
                    { "arguments", "{\"arg1\": 1}" },
                }},
                {"id", "123456789"},
            },
        }},
        { "role",       "assistant"                },
        { "content",    {}                         },
        { "tool_calls", tool_calls                  }
    };
    json message_assist_call_idx {
        { "role",       "assistant"},
        { "content",    {}},
        { "tool_calls", {
            {
                { "type", "function" },
                { "function", {
                    { "name", "special_function" },
                    { "arguments", "{\"arg1\": 1}" },
                }},
                // Index of the tool call in the tool_calls array
                {"id", "0"},
            },
        }},
        { "role",       "assistant"                },
        { "content",    {}                         },
        { "tool_calls", tool_calls                  }
    };
    json message_assist_call_tool_plan_idx = message_assist_call_idx;
    message_assist_call_tool_plan_idx["tool_plan"] = "I'm thinking";

    auto python_message_assist_call = json{
        { "role",       "assistant"                },
        { "content",    {}                         },
        { "tool_calls", json{ {
                            { "type", "function" },
                            { "function",
                              {
                                  { "name", "python" },
                                  { "arguments",
                                    {
                                        { "code", "print('hey')" },
                                    } },
                              } },
                        } } }
    };
    auto code_interpreter_message_assist_call = json{
        { "role",       "assistant"                },
        { "content",    {}                         },
        { "tool_calls", json{ {
                            { "type", "function" },
                            { "function",
                              {
                                  { "name", "code_interpreter" },
                                  { "arguments",
                                    {
                                        { "code", "print('hey')" },
                                    } },
                              } },
                        } } }
    };

    common_chat_inputs inputs_no_tools;
    inputs_no_tools.messages                = json::array({message_user});
    inputs_no_tools.extract_reasoning       = false;

    common_chat_inputs inputs_no_tools_think;
    inputs_no_tools_think.messages          = json::array({message_user});
    inputs_no_tools_think.extract_reasoning = true;

    common_chat_inputs inputs_tools;
    inputs_tools.messages                   = json::array({message_user});
    inputs_tools.tools                      = json::array({special_function_tool});
    inputs_tools.extract_reasoning          = false;

    common_chat_inputs inputs_tools_think;
    inputs_tools_think.messages             = json::array({message_user});
    inputs_tools_think.tools                = json::array({special_function_tool});
    inputs_tools_think.extract_reasoning    = true;

    common_chat_inputs inputs_tools_builtin;
    inputs_tools_builtin.messages           = json::array({message_user});
    inputs_tools_builtin.tools              = json::array({python_tool});
    inputs_tools_builtin.extract_reasoning  = false;

    {
        // Not supported yet
        const common_chat_template tmpl(read_file("models/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja"), "<s>", "</s>");
        assert_equals(COMMON_CHAT_FORMAT_GENERIC, common_chat_params_init(tmpl, inputs_tools).format);
    }
    {
        const common_chat_template tmpl(read_file("models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja"), "<s>", "</s>");
        std::vector<std::string>   end_tokens{ "<|END_OF_TURN_TOKEN|>" };

        assert_equals(COMMON_CHAT_FORMAT_COMMAND_R7B,                   common_chat_params_init(tmpl, inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_COMMAND_R7B,                   common_chat_params_init(tmpl, inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_COMMAND_R7B_EXTRACT_REASONING, common_chat_params_init(tmpl, inputs_tools_think).format);

        assert_msg_equals(msg_from_json(message_assist),
            common_chat_parse(
                "Hello, world!\nWhat's up?",
                COMMON_CHAT_FORMAT_COMMAND_R7B));
        assert_msg_equals(msg_from_json(message_assist),
            common_chat_parse(
                "Hello, world!\nWhat's up?<|END_RESPONSE|>",
                COMMON_CHAT_FORMAT_COMMAND_R7B));
        assert_msg_equals(msg_from_json(message_assist),
            common_chat_parse(
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                COMMON_CHAT_FORMAT_COMMAND_R7B));
        assert_msg_equals(msg_from_json(message_assist_thoughts_unparsed_r7b),
            common_chat_parse(
                "<|START_THINKING|>I'm thinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                COMMON_CHAT_FORMAT_COMMAND_R7B));
        assert_msg_equals(msg_from_json(message_assist_thoughts_unparsed_r7b),
            common_chat_parse(
                "<|START_THINKING|>I'm thinking<|END_THINKING|>"
                "Hello, world!\nWhat's up?<|END_RESPONSE|>",
                COMMON_CHAT_FORMAT_COMMAND_R7B));

        assert_msg_equals(msg_from_json(message_assist_thoughts),
            common_chat_parse(
                "<|START_THINKING|>I'm thinking<|END_THINKING|>"
                "<|START_RESPONSE|>Hello, world!\nWhat's up?<|END_RESPONSE|>",
                COMMON_CHAT_FORMAT_COMMAND_R7B_EXTRACT_REASONING));

        test_template(tmpl, end_tokens, message_assist_call_idx, tools,
                      "<|START_THINKING|><|END_THINKING|>"
                      "<|START_ACTION|>[\n"
                      "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
                      "]<|END_ACTION|>");
        test_template(tmpl, end_tokens, message_assist_call_tool_plan_idx, tools,
                      "<|START_THINKING|>I'm thinking<|END_THINKING|>"
                      "<|START_ACTION|>[\n"
                      "    {\"tool_call_id\": \"0\", \"tool_name\": \"special_function\", \"parameters\": {\"arg1\": 1}}\n"
                      "]<|END_ACTION|>",
                      /* expect_grammar_triggered= */ true,
                      /* test_grammar_if_triggered= */ true,
                      /* think= */ true);
        test_template(tmpl, end_tokens, message_assist, tools,
                      "<|START_RESPONSE|>Hello, world!\n"
                      "What's up?<|END_RESPONSE|>",
                      /* expect_grammar_triggered= */ false);
    }
    {
        const common_chat_template tmpl(read_file("models/templates/google-gemma-2-2b-it.jinja"), "<s>", "</s>");
        std::vector<std::string>   end_tokens{ "<end_of_turn>" };

        assert_equals(COMMON_CHAT_FORMAT_CONTENT_ONLY, common_chat_params_init(tmpl, inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_GENERIC, common_chat_params_init(tmpl, inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_GENERIC,
                      common_chat_params_init(
                          common_chat_template(read_file("models/templates/microsoft-Phi-3.5-mini-instruct.jinja"),
                                               "<s>", "</s>"),
                          inputs_tools)
                          .format);

        // Generic tool calls doesn't generate / parse content-only messages symmetrically.

        assert_msg_equals(msg_from_json(message_assist),
                          common_chat_parse("{\n"
                                            "  \"response\": \"Hello, world!\\nWhat's up?\"\n"
                                            "}",
                                            common_chat_params_init(tmpl, inputs_tools).format));
        test_template(tmpl, end_tokens, message_assist_call_id, tools,
                      "{\n"
                      "  \"tool_calls\": [\n"
                      "    {\n"
                      "      \"name\": \"special_function\",\n"
                      "      \"arguments\": {\n"
                      "        \"arg1\": 1\n"
                      "      },\n"
                      "      \"id\": \"123456789\"\n"
                      "    }\n"
                      "  ]\n"
                      "}");
    }
    {
        const common_chat_template tmpl(read_file("models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja"), "<s>",
                                        "</s>");
        std::vector<std::string>   end_tokens{ "</s>" };

        assert_equals(COMMON_CHAT_FORMAT_MISTRAL_NEMO, common_chat_params_init(tmpl, inputs_tools).format);

        test_template(tmpl, end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_template(
            tmpl, end_tokens, message_assist_call_id, tools,
            "[TOOL_CALLS][{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}, \"id\": \"123456789\"}]");
    }
    {
        const common_chat_template tmpl(
            read_file("models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja"), "<s>", "</s>");
        std::vector<std::string> end_tokens{ "<|im_end|>" };

        assert_equals(COMMON_CHAT_FORMAT_HERMES_2_PRO, common_chat_params_init(tmpl, inputs_tools).format);
        assert_equals(
            COMMON_CHAT_FORMAT_HERMES_2_PRO,
            common_chat_params_init(
                common_chat_template(read_file("models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja"),
                                     "<s>", "</s>"),
                inputs_tools)
                .format);
        assert_equals(
            COMMON_CHAT_FORMAT_HERMES_2_PRO,
            common_chat_params_init(
                common_chat_template(read_file("models/templates/Qwen-Qwen2.5-7B-Instruct.jinja"), "<s>", "</s>"),
                inputs_tools)
                .format);

        test_template(tmpl, end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_template(tmpl, end_tokens, message_assist_call, tools,
                      "<tool_call>\n"
                      "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
                      "</tool_call>");
        test_template(tmpl, end_tokens, python_message_assist_call, tools,
                      "<tool_call>\n"
                      "{\"name\": \"python\", \"arguments\": {\"code\": \"print('hey')\"}}\n"
                      "</tool_call>");
    }
    {
        const common_chat_template tmpl(read_file("models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja"), "<s>",
                                        "</s>");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X, common_chat_params_init(tmpl, inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
                      common_chat_params_init(tmpl, inputs_tools_builtin).format);
        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS,
                      common_chat_params_init(
                          common_chat_template(read_file("models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja"),
                                               "<s>", "</s>"),
                          inputs_tools_builtin)
                          .format);

        // test_template(tmpl, end_tokens, message_assist, tools, R"(?)", /* expect_grammar_triggered= */ false);
        test_template(tmpl, end_tokens, code_interpreter_message_assist_call, llama_3_1_tools,
                      "<|python_tag|>code_interpreter.call(code=\"print('hey')\")");
        test_template(tmpl, end_tokens, python_message_assist_call, tools,
                      "<|python_tag|>python.call(code=\"print('hey')\")");
        test_template(tmpl, end_tokens, message_assist_call, tools,
                      "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}");
    }
    {
        const common_chat_template tmpl(read_file("models/templates/meta-llama-Llama-3.2-3B-Instruct.jinja"), "<s>",
                                        "</s>");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_LLAMA_3_X, common_chat_params_init(tmpl, inputs_tools).format);

        test_template(tmpl, end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_template(tmpl, end_tokens, message_assist_call, tools,
                      "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}");
    }
    {
        const common_chat_template tmpl(read_file("models/templates/meetkai-functionary-medium-v3.1.jinja"), "<s>",
                                        "</s>");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1,
                      common_chat_params_init(tmpl, inputs_tools).format);

        test_template(tmpl, end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_template(tmpl, end_tokens, message_assist_call, tools,
                      "<function=special_function>{\"arg1\": 1}</function>");
    }
    {
        const common_chat_template tmpl(read_file("models/templates/meetkai-functionary-medium-v3.2.jinja"), "<s>",
                                        "</s>");
        std::vector<std::string>   end_tokens{ "<|eom_id|>", "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2, common_chat_params_init(tmpl, inputs_no_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2, common_chat_params_init(tmpl, inputs_tools).format);

        test_template(tmpl, end_tokens, message_assist, {},
                      "all\n"
                      "Hello, world!\n"
                      "What's up?",
                      /* expect_grammar_triggered= */ false);
        test_template(tmpl, end_tokens, message_assist_call, tools,
                      "special_function\n"
                      "{\"arg1\": 1}");
    }
    {
        const common_chat_template tmpl(read_file("models/templates/fireworks-ai-llama-3-firefunction-v2.jinja"), "<s>",
                                        "</s>");
        std::vector<std::string>   end_tokens{ "<|eot_id|>" };

        assert_equals(COMMON_CHAT_FORMAT_FIREFUNCTION_V2, common_chat_params_init(tmpl, inputs_tools).format);

        test_template(tmpl, end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_template(tmpl, end_tokens, message_assist_call, tools,
                      " functools[{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}]");
    }
    {
        // Original DeepSeek R1 template. Leaves <｜tool▁calls▁begin｜> and others unclosed. Our logic fixes the prompt.
        const common_chat_template tmpl(read_file("models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja"),
                                        "<s>", "</s>");
        std::vector<std::string>   end_tokens{ "<｜end▁of▁sentence｜>" };

        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1,                   common_chat_params_init(tmpl, inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1_EXTRACT_REASONING, common_chat_params_init(tmpl, inputs_tools_think).format);

        test_template(tmpl, end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_template(tmpl, end_tokens, message_assist_thoughts, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        assert_msg_equals(msg_from_json(message_assist_thoughts_unparsed_think),
            common_chat_parse("<think>I'm thinking</think>Hello, world!\nWhat's up?",
            COMMON_CHAT_FORMAT_DEEPSEEK_R1));
        assert_msg_equals(msg_from_json(message_assist_thoughts),
            common_chat_parse("<think>I'm thinking</think>Hello, world!\nWhat's up?",
            COMMON_CHAT_FORMAT_DEEPSEEK_R1_EXTRACT_REASONING));
        assert_msg_equals(msg_from_json(message_assist_thoughts),
            // Latest template update (ast of 20250209) adds a trailing <think>\n if add_generation_prompt is true.
            common_chat_parse("I'm thinking</think>Hello, world!\nWhat's up?",
            COMMON_CHAT_FORMAT_DEEPSEEK_R1_EXTRACT_REASONING));
        // test_template(tmpl, end_tokens, message_assist_call, tools,
        //               "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
        //               "```json\n"
        //               "{\"arg1\": 1}\n"
        //               // Look what's not here: <｜tool▁calls▁end｜> (also missing the <｜end▁of▁sentence｜>, but that is removed lazily by the test's delta logic)
        //               "```<｜tool▁call▁end｜>",
        //               /* expect_grammar_triggered= */ true,
        //               /* test_grammar_if_triggered= */ false);
    }
    {
        // Replacement DeepSeek R1 template. Makes the Distill Qwen 7B/32B models happy to call tools and all.
        const common_chat_template tmpl(read_file("models/templates/llama-cpp-deepseek-r1.jinja"),
                                        "<s>", "</s>");
        std::vector<std::string>   end_tokens{ "<｜end▁of▁sentence｜>" };

        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1,                   common_chat_params_init(tmpl, inputs_tools).format);
        assert_equals(COMMON_CHAT_FORMAT_DEEPSEEK_R1_EXTRACT_REASONING, common_chat_params_init(tmpl, inputs_tools_think).format);

        test_template(tmpl, end_tokens, message_assist, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        test_template(tmpl, end_tokens, message_assist_thoughts, tools, "Hello, world!\nWhat's up?", /* expect_grammar_triggered= */ false);
        assert_msg_equals(msg_from_json(message_assist_thoughts_unparsed_think),
            common_chat_parse("<think>I'm thinking</think>Hello, world!\nWhat's up?",
            COMMON_CHAT_FORMAT_DEEPSEEK_R1));
        assert_msg_equals(msg_from_json(message_assist_thoughts),
            common_chat_parse("<think>I'm thinking</think>Hello, world!\nWhat's up?",
            COMMON_CHAT_FORMAT_DEEPSEEK_R1_EXTRACT_REASONING));

        assert_msg_equals(msg_from_json(message_assist_call_thoughts_unparsed),
            common_chat_parse(
                "<think>I'm\nthinking</think>\n\n"
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                COMMON_CHAT_FORMAT_DEEPSEEK_R1));
        assert_msg_equals(msg_from_json(message_assist_call_thoughts),
            common_chat_parse(
                "<think>I'm\nthinking</think>\n\n"
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
                COMMON_CHAT_FORMAT_DEEPSEEK_R1_EXTRACT_REASONING));
        test_template(tmpl, end_tokens, message_assist_call, tools,
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>special_function\n"
                "```json\n"
                "{\"arg1\": 1}\n"
                "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>");
    }
}

int main(int argc, char ** argv) {
#ifndef _WIN32
    if (argc > 1) {
        common_chat_inputs inputs;
        inputs.messages = {
            { { "role", "user" }, { "content", "Hey" } }
        };
        inputs.tools = json::array({ special_function_tool });

        std::cout << "| Template | Format |\n";
        std::cout << "|----------|--------|\n";

        for (int i = 1; i < argc; i++) {
            try {
                std::string path = argv[i];
                if (path.rfind(".jinja") != path.size() - 6) {
                    std::cerr << "Skipping non-jinja file: " << path << std::endl;
                    continue;
                }
                common_chat_template tmpl(read_file(path), "", "");
                auto parts  = string_split(path, "/");
                auto name   = parts[parts.size() - 1];
                auto format = common_chat_format_name(common_chat_params_init(tmpl, inputs).format);
                std::cout << "| " << name << " | " << format << " |\n";
            } catch (const std::exception & e) {
                std::cerr << "Failed to process " << argv[i] << ": " << e.what() << std::endl;
            }
        }
    } else
#endif
    {
        test_template_output_parsers();
        std::cout << "\n[chat] All tests passed!" << std::endl;
    }
    return 0;
}
