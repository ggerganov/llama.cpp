#include "tool-call.h"
#include "llama-grammar.h"
#include "unicode.h"

#include <fstream>
#include <iostream>
#include <string>
#include <json.hpp>

using json = nlohmann::ordered_json;

template <class T>
static void assert_equals(const T & expected, const T & actual) {
    if (expected != actual) {
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

static std::string read_file(const std::string &path) {
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
    return std::unique_ptr<llama_grammar>(llama_grammar_init_impl(nullptr, grammar_str.c_str(), "root"));
}

// TODO: extract to common helper (copied from test-grammar-integration.cpp)
static bool match_string(const std::string & input, llama_grammar * grammar) {
    const auto cpts = unicode_cpts_from_utf8(input);

    const llama_grammar_rules  & rules      = llama_grammar_get_rules (grammar);
          llama_grammar_stacks & stacks_cur = llama_grammar_get_stacks(grammar);

    for (const auto & cpt : cpts) {
        const llama_grammar_stacks stacks_prev = llama_grammar_get_stacks(grammar); // copy

        llama_grammar_accept(rules, stacks_prev, cpt, stacks_cur);

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

static void test_parse_tool_call(llama_tool_call_style style, const json & tools, const std::string & input, const std::string & expected_content, const json & expected_tool_calls) {
    std::cout << "# Testing: " << input << std::endl << std::flush;
    auto result = parse_tool_calls(style, tools, input);
    assert_equals(expected_content, result.content);
    auto tool_calls = json::array();
    for (const auto & tc : result.tool_calls) {
      auto tool_call = json {
        {"type", "function"},
        {"function", {
          {"arguments", dump(json::parse(tc.arguments))},
          {"name", tc.name},
        }},
      };
      if (!tc.id.empty()) {
        tool_call["id"] = tc.id;
      }
      tool_calls.push_back(tool_call);
    }
    // Reparse / dump w/ non-ordered JSON variant.
    auto expected = nlohmann::json::parse(expected_tool_calls.dump()).dump();
    auto actual = nlohmann::json::parse(tool_calls.dump()).dump();
    assert_equals(expected, actual);
}

const json tools = json::parse(R"([
  {
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
  },
  {
    "type": "function",
    "function": {
      "name": "ipython",
      "description": "a python interpreter",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {
            "type": "string",
            "description": "The code."
          }
        },
        "required": ["code"]
      }
    }
  }
])");

static void test_parsing() {
    json request = {
      {"tools", tools}
    };

    const auto fooBarCall = json {
      {"type", "function"},
      {"function", {
        {"name", "foo"},
        {"arguments", dump({
          {"bar", 1}
        })},
      }}
    };

    test_parse_tool_call(llama_tool_call_style::Generic, tools,
      "{\"tool_call\": {\"name\": \"foo\", \"arguments\": {\"bar\": 1}}}",
      "",
      json::array({fooBarCall}));
    test_parse_tool_call(llama_tool_call_style::Generic, tools,
      "{\"tool_calls\": [{\"name\": \"foo\", \"arguments\": {\"bar\": 1}}]}",
      "",
      json::array({fooBarCall}));

    test_parse_tool_call(llama_tool_call_style::Hermes2Pro, tools,
      "<tool_call>{\"name\": \"foo\", \"arguments\": {\"bar\": 1}}</tool_call>",
      "",
      json::array({fooBarCall}));

    test_parse_tool_call(llama_tool_call_style::FunctionaryV3Llama3, tools,
      ">>>ipython\n{\"code\": \"print('Hello, world!')\"}",
      "",
      json {{
        {"type", "function"},
        {"function", {
          {"name", "ipython"},
          {"arguments", dump({
            {"code", "print('Hello, world!')"}
          })}
        }}
      }});
    test_parse_tool_call(llama_tool_call_style::FunctionaryV3Llama3, tools,
      ">>>special_function\n{\"arg1\": 1}\n ",
      "",
      json {{
        {"type", "function"},
        {"function", {
          {"name", "special_function"},
          {"arguments", dump({
            {"arg1", 1}
          })}
        }}
      }});

    test_parse_tool_call(llama_tool_call_style::FunctionaryV3Llama31, tools,
      "Hell<function=foo>{\"arg1\": 1}</function>o, world<function=bar>{\"arg2\": 2}</function>!",
      "Hello, world!",
      json {
        {
          {"type", "function"},
          {"function", {
            {"name", "foo"},
            {"arguments", dump({
              {"arg1", 1}
            })}
          }}
        },
        {
          {"type", "function"},
          {"function", {
            {"name", "bar"},
            {"arguments", dump({
              {"arg2", 2}
            })}
          }}
        },
      });
    test_parse_tool_call(llama_tool_call_style::FunctionaryV3Llama31, tools,
      "<function=test>{ } </function> ",
      " ",
      json {{
        {"type", "function"},
        {"function", {
          {"name", "test"},
          {"arguments", "{}"}
        }}
      }});

    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "<|python_tag|>this could be anything",
      "",
      json {{
        {"type", "function"},
        {"function", {
          {"name", "ipython"},
          {"arguments", dump({
            {"code", "this could be anything"}
          })}
        }}
      }});
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "I'm thinking<|python_tag|>",
      "I'm thinking",
      json {{
        {"type", "function"},
        {"function", {
          {"name", "ipython"},
          {"arguments", dump({{"code", ""}})}
        }}
      }});
    auto special_function_call = json {
        {"type", "function"},
        {"function", {
          {"arguments", dump({{"arg1", 1}})},
          {"name", "special_function"},
        }},
    };
    auto special_function_call_with_id = json::parse(special_function_call.dump());
    special_function_call_with_id["id"] = "123456789";
    
    auto no_function_call = json::array();

    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
      "",
      json::array({special_function_call}));
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "{\n  \"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
      "",
      json::array({special_function_call}));
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "{\n\t\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
      "",
      json::array({special_function_call}));
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "{\n    \"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
      "",
      json::array({special_function_call}));
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "{\"type\": \"function\", \"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
      "",
      json::array({special_function_call}));

    // No match: function unknown
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "{\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
      "{\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
      no_function_call);
    // No match: bad indentation
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "{\n\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
      "{\n\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
      no_function_call);
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "{\n \"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
      "{\n \"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
      no_function_call);

    test_parse_tool_call(llama_tool_call_style::MistralNemo, tools,
      "Bleh[TOOL_CALLS][{\"arguments\": {\"arg1\": 1}, \"name\": \"special_function\", \"id\": \"123456789\"}]",
      "Bleh",
      json::array({special_function_call_with_id}));
    test_parse_tool_call(llama_tool_call_style::MistralNemo, tools,
      "[{\"arguments\": {\"arg1\": 1}, \"name\": \"special_function\", \"id\": \"123456789\"}]",
      "",
      json::array({special_function_call_with_id}));
}

static void test_tool_call_style(const std::string & template_file, llama_tool_call_style expected) {
    const minja::chat_template tmpl(read_file(template_file), "<s>", "</s>");
    auto tool_call_style = llama_tool_call_style_detect(tmpl);
    std::cout << "# Testing tool call style of: " << template_file << std::endl << std::flush;
    assert_equals(expected, tool_call_style);
}

static void test_tool_call_style_detection() {
    test_tool_call_style("tests/chat/templates/meetkai-functionary-medium-v3.1.jinja", FunctionaryV3Llama31);
    test_tool_call_style("tests/chat/templates/meetkai-functionary-medium-v3.2.jinja", FunctionaryV3Llama3);
    test_tool_call_style("tests/chat/templates/meta-llama-Meta-Llama-3.1-8B-Instruct.jinja", Llama31);
    test_tool_call_style("tests/chat/templates/meta-llama-Llama-3.2-3B-Instruct.jinja", Llama32);
    test_tool_call_style("tests/chat/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja", CommandRPlus);
    test_tool_call_style("tests/chat/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja", MistralNemo);
    test_tool_call_style("tests/chat/templates/google-gemma-7b-it.jinja", Generic);
}

static std::string get_message_prompt_delta(const minja::chat_template & tmpl, const std::vector<std::string> & end_tokens, const json & user_message, const json & delta_message, const json & tools) {
  auto prefix = tmpl.apply(json::array({user_message}), tools, /* add_generation_prompt= */ true, json::object());
  auto full = tmpl.apply(json::array({user_message, delta_message}), tools, /* add_generation_prompt= */ false, json::object());

  // Check full starts with prefix
  if (full.find(prefix) != 0) {
    throw std::runtime_error("Full message does not start with prefix");
  }

  auto delta = full.substr(prefix.size());

  // Strip end tokens
  for (const auto & end_token : end_tokens) {
    // rfind to find the last occurrence
    auto pos = delta.rfind(end_token);
    if (pos != std::string::npos) {
      delta = delta.substr(0, pos);
      break;
    }
  }
  return delta;
}

static void test_template(const std::string & template_file, const char * bos_token, const char * eos_token, const std::vector<std::string> & end_tokens, const json & tool_calling_message, const json & tools, bool skip_grammar_test = false) {
  std::cout << "# Testing template: " << template_file << std::endl << std::flush;
  const minja::chat_template tmpl(read_file(template_file), bos_token, eos_token);
  auto tool_call_style = llama_tool_call_style_detect(tmpl);
  auto & tool_calls = tool_calling_message.at("tool_calls");

  // Format the message: apply the template to 1 user message w/ add_generation_prompt=true, then w/ the extra message w/ add_generation_prompt=false,
  // get the diff and try and parse it w/ the grammar.
  auto user_message = json {
      {"role", "user"},
      {"content", "Hello, world!"}
  };

  auto handler = llama_tool_call_handler_init(tool_call_style, tmpl, /* allow_content= */ true, /* parallel_tool_calls= */ true, {user_message, tool_calling_message}, tools);
  auto grammar = build_grammar(handler.grammar);
  if (!grammar) {
    throw std::runtime_error("Failed to build grammar");
  }

  if (!skip_grammar_test) {
    auto full_delta = get_message_prompt_delta(tmpl, end_tokens, user_message, tool_calling_message, tools);
    std::cout << "Full delta:\n```\n" << full_delta << "\n```" << std::endl;
    test_parse_tool_call(tool_call_style, tools, full_delta, "", tool_calls);

    auto content_less_delta = get_message_prompt_delta(tmpl, end_tokens, user_message, {
      {"role", "assistant"},
      {"content", ""},
      {"tool_calls", tool_calls}
    }, tools);
    if (!match_string(content_less_delta, grammar.get())) {
      throw std::runtime_error("Failed to match content-less delta against grammar:\n\nContent-less delta: " + content_less_delta + "\n\nGrammar: " + handler.grammar);
    }
  }
}

static void test_grammars() {
  auto tool_call_message = json {
    {"role", "assistant"},
    {"content", ""},
    {"tool_calls", json {{
      {"type", "function"},
      {"function", {
        {"name", "special_function"},
        {"arguments", "{\"arg1\": 1}"}
      }},
    }}}
  };
  auto tool_call_message_with_id = json::parse(tool_call_message.dump());
  tool_call_message_with_id["tool_calls"][0]["id"] = "123456789";

  test_template("tests/chat/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja", "<s>", "</s>", { "</s>" }, tool_call_message_with_id, tools,
    /* skip_grammar_test= */ true);
  test_template("tests/chat/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja", "<s>", "</s>", { "<|im_end|>" }, tool_call_message, tools);
  test_template("tests/chat/templates/meta-llama-Meta-Llama-3.1-8B-Instruct.jinja", "<s>", "</s>", { "<|eom_id|>", "<|eot_id|>" }, tool_call_message, tools);
  test_template("tests/chat/templates/meta-llama-Llama-3.2-3B-Instruct.jinja", "<s>", "</s>", { "<|eom_id|>", "<|eot_id|>" }, tool_call_message, tools);
  test_template("tests/chat/templates/meetkai-functionary-medium-v3.1.jinja", "<s>", "</s>", { "<|eom_id|>", "<|eot_id|>" }, tool_call_message, tools);
  test_template("tests/chat/templates/meetkai-functionary-medium-v3.2.jinja", "<s>", "</s>", { "<|eom_id|>", "<|eot_id|>" }, tool_call_message, tools);
}

int main() {
    test_tool_call_style_detection();
    test_parsing();
    test_grammars();

    std::cout << "[tool-call] All tests passed!" << std::endl;
    return 0;
}
