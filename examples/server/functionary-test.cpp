#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include "json.hpp"
#include "functionary.hpp"

using json = nlohmann::json;

/**
 * A simple test program that allow testing functionary.hpp without using server.
 * TODO: how to add this test to CI?
 * 
 * Compile command: clear && g++ functionary-test.cpp -o functionary-test && ./functionary-test
 */

std::string test_oai_input_json = R"(
{
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_n_day_weather_forecast",
                "description": "Get an N-day weather forecast",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location."
                        },
                        "num_days": {
                            "type": "integer",
                            "description": "The number of days to forecast"
                        }
                    },
                    "required": ["location", "format", "num_days"]
                }
            }
        }
    ],
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "What is the weather like in Boston?"
        },
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {
                    "type": "function",
                    "id":"get_car_price",
                    "function": {
                        "arguments": "{\"car_name\": \"Song\"}",
                        "name": "get_car_price"
                    }
                },
                {
                    "type": "function",
                    "id":"get_car_price",
                    "function": {
                        "arguments": "{\"car_name\": \"Tang\"}",
                        "name": "get_car_price"
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "get_car_price",
            "name": "get_car_price",
            "content": "{\"price\": {\"price\": \"$25000\"}}"
        },
        {
            "role": "tool",
            "tool_call_id": "get_car_price",
            "name": "get_car_price",
            "content": "{\"price\": {\"price\": \"$20000\"}}"
        }
    ]
}
)";


std::string test_response = R"(<|from|>assistant
<|recipient|>all
<|content|>I will get the price of 2 cars and compare
<|from|>assistant
<|recipient|>get_car_price
<|content|>{"car_name": "Song"}
<|from|>assistant
<|recipient|>get_car_price
<|content|>{"car_name": "Tang"}<|stop|>)";

int main() {
    auto test_oai_input = json::parse(test_oai_input_json);
    auto prompt = llama_functionary::convert_oai_to_prompt(test_oai_input);
    std::cout << "\n" << prompt << "\n";

    std::cout << "\n" << llama_functionary::convert_response_to_oai_choices(test_response) << "\n";

    return 0;
}
