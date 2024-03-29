#!/bin/bash
set -euo pipefail

SERVER_PID=""
function cleanup() {
  if [ -n "$SERVER_PID" ]; then
    echo "# Killing server" >&2
    kill $SERVER_PID
    wait $SERVER_PID
  fi
}
trap cleanup EXIT

echo "# Starting the server" >&2

args=(
    # --cpp_server_endpoint "http://localhost:8081"
    
    # --model ~/AI/Models/functionary-medium-v2.2.q4_0.gguf
    
    --model ~/AI/Models/mixtral-8x7b-instruct-v0.1.Q8_0.gguf
    # --model ~/AI/Models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf

    # --model ~/AI/Models/Hermes-2-Pro-Mistral-7B.Q8_0.gguf
    # --model ~/AI/Models/Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf
)
python -m examples.openai "${args[@]}" &
SERVER_PID=$!

sleep 5

echo "# Send a message to the chat API" >&2

python -m examples.openai.reactor
exit

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
      {"role": "user", "content": "I live in the UK. what is the weather going to be like in San Francisco and Glasgow over the next 4 days."}
    ]
  }' | \
  jq .

#   {"role": "system", "content": "Do not make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
