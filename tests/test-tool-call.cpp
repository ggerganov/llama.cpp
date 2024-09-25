#include "tool-call.h"

#include <fstream>
#include <iostream>
#include <string>
#include <json.hpp>

using json = nlohmann::ordered_json;

static void assert_equals(const std::string & expected, const std::string & actual) {
    if (expected != actual) {
        std::cerr << "Expected: " << expected << std::endl;
        std::cerr << "Actual: " << actual << std::endl;
        std::cerr << std::flush;
        throw std::runtime_error("Test failed");
    }
}

/*
    cmake -B build -DLLAMA_CURL=1 -DCMAKE_BUILD_TYPE=Release && cmake --build build -t test-tool-call -j && ./build/bin/test-tool-call
*/

static void test_parse_tool_call(const json & tools, const std::string & chat_template, const std::string & input, const std::string & expected_content, const json & expected_tool_calls) {
    std::cout << "# Testing: " << input << std::endl << std::flush;
    auto result = parse_tool_calls(tools, chat_template, input);
    assert_equals(expected_content, result.content);
    auto tool_calls = json::array();
    for (const auto & tc : result.tool_calls) {
        tool_calls.push_back({
          {"function", {
            {"name", tc.name},
            {"arguments", tc.arguments},
          }}
        });
    }
    assert_equals(expected_tool_calls.dump(), tool_calls.dump());
}
int main() {
    json tools = json::parse(R"([
      {
        "type": "function",
        "function": {
          "name": "special_function",
          "description": "I'm special",
          "parameters": {
            "type": "object",
            "properties": {
              "arg1": {
                "type": "string",
                "description": "The arg."
              }
            },
            "required": ["arg1"]
          }
        }
      }
    ])");
    json request = {
      {"tools", tools}
    };
    
    std::string hermes_2_pro_like_tmpl = "Hermes 2 Pro template should have <tool_call> inside it";
    test_parse_tool_call(tools, hermes_2_pro_like_tmpl,
      "<tool_call>{\"name\": \"foo\", \"arguments\": {\"bar\": 1}}</tool_call>",
      "",
      json {{
        {"function", {
          {"name", "foo"},
          {"arguments", (json {
            {"bar", 1}
          }).dump()}
        }}
      }});
   
    std::string functionary_v3_like_tmpl = "Functionary 3.2 template should have <|start_header_id|> and then some >>>all inside it";
    test_parse_tool_call(tools, functionary_v3_like_tmpl,
      ">>>ipython\nprint('Hello, world!')",
      "",
      json {{
        {"function", {
          {"name", "ipython"},
          {"arguments", (json {
            {"code", "print('Hello, world!')"}
          }).dump()}
        }}
      }});
   
    std::string functionary_v3_llama_3_1_like_tmpl = "Functionary 3.2 template for llama 3.1 should have <|start_header_id|> and then some <function=foo>{...}</function> inside it";
    test_parse_tool_call(tools, functionary_v3_llama_3_1_like_tmpl,
      "Hell<function=foo>{\"arg1\": 1}</function>o, world<function=bar>{\"arg2\": 2}</function>!",
      "Hello, world!",
      json {
        {
          {"function", {
            {"name", "foo"},
            {"arguments", (json {
              {"arg1", 1}
            }).dump()}
          }}
        },
        {
          {"function", {
            {"name", "bar"},
            {"arguments", (json {
              {"arg2", 2}
            }).dump()}
          }}
        },
      });
   
    std::string llama_3_1_like_tmpl = "Llama 3.1 template should have <|start_header_id|> and <|python_tag|> inside it";
    test_parse_tool_call(tools, llama_3_1_like_tmpl,
      "<|python_tag|>this could be anything",
      "",
      json {{
        {"function", {
          {"name", "ipython"},
          {"arguments", (json {
            {"code", "this could be anything"}
          }).dump()}
        }}
      }});
    test_parse_tool_call(tools, llama_3_1_like_tmpl,
      "I'm thinking<|python_tag|>",
      "I'm thinking",
      json {{
        {"function", {
          {"name", "ipython"},
          {"arguments", (json {{"code", ""}}).dump()}
        }}
      }});
    test_parse_tool_call(tools, llama_3_1_like_tmpl,
      "{\"name\": \"special_function\", \"parameters\": {\"arg1\": 1}}",
      "",
      json {{
        {"function", {
          {"name", "special_function"},
          {"arguments", (json {
            {"arg1", 1}
          }).dump()}
        }}
      }});
    test_parse_tool_call(tools, llama_3_1_like_tmpl,
      "{\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
      "{\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}", json::array());

    return 0;
}