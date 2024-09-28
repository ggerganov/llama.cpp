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

static std::string read_file(const std::string &path) {
  std::ifstream fs(path, std::ios_base::binary);
  if (!fs.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  fs.seekg(0, std::ios_base::end);
  auto size = fs.tellg();
  fs.seekg(0);
  std::string out;
  out.resize(static_cast<size_t>(size));
  fs.read(&out[0], static_cast<std::streamsize>(size));
  return out;
}

/*
    cmake -B build -DLLAMA_CURL=1 -DCMAKE_BUILD_TYPE=Release && cmake --build build -t test-tool-call -j && ./build/bin/test-tool-call
*/

static void test_parse_tool_call(llama_tool_call_style style, const json & tools, const std::string & input, const std::string & expected_content, const json & expected_tool_calls) {
    std::cout << "# Testing: " << input << std::endl << std::flush;
    auto result = parse_tool_calls(style, tools, input);
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
    json request = {
      {"tools", tools}
    };

    test_parse_tool_call(llama_tool_call_style::Hermes2Pro, tools,
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

    test_parse_tool_call(llama_tool_call_style::FunctionaryV3Llama3, tools,
      ">>>ipython\n{\"code\": \"print('Hello, world!')\"}",
      "",
      json {{
        {"function", {
          {"name", "ipython"},
          {"arguments", (json {
            {"code", "print('Hello, world!')"}
          }).dump()}
        }}
      }});
    test_parse_tool_call(llama_tool_call_style::FunctionaryV3Llama3, tools,
      ">>>special_function\n{\"arg1\": 1}\n ",
      "",
      json {{
        {"function", {
          {"name", "special_function"},
          {"arguments", (json {
            {"arg1", 1}
          }).dump()}
        }}
      }});

    test_parse_tool_call(llama_tool_call_style::FunctionaryV3Llama31, tools,
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
    test_parse_tool_call(llama_tool_call_style::FunctionaryV3Llama31, tools,
      "<function=test>{ } </function> ",
      " ",
      json {{
        {"function", {
          {"name", "test"},
          {"arguments", "{}"}
        }}
      }});

    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
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
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "I'm thinking<|python_tag|>",
      "I'm thinking",
      json {{
        {"function", {
          {"name", "ipython"},
          {"arguments", (json {{"code", ""}}).dump()}
        }}
      }});
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
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
    test_parse_tool_call(llama_tool_call_style::Llama31, tools,
      "{\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}",
      "{\"name\": \"unknown_function\", \"arguments\": {\"arg1\": 1}}", json::array());

    std::cout << "[tool-call] All tests passed!" << std::endl;
    return 0;
}
