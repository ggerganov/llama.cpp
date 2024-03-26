#!/bin/bash
set -euo pipefail

SERVER_PID=""
function cleanup() {
  if [ -n "$SERVER_PID" ]; then
    echo "# Killing server"
    kill $SERVER_PID
    wait $SERVER_PID
  fi
}
trap cleanup EXIT

echo "# Starting the server"
python -m examples.openai --model ~/AI/Models/Hermes-2-Pro-Mistral-7B.Q8_0.gguf &
SERVER_PID=$!

sleep 5

echo "# Send a message to the chat API"

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "tools": [{
          "type": "function",
          "function": {
              "name": "get_current_weather",
              "description": "Get the current weather",
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
                      }
                  },
                  "required": ["location", "format"]
              }
          }
      }, {
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
      }],
    "messages": [
      {"role": "system", "content": "Do not make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
      {"role": "user", "content": "what is the weather going to be like in San Francisco and Glasgow over the next 4 days. Give the temperatyre in celsius for both locations."}
    ]
  }'

