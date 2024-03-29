# examples.openai: OpenAI-compatibility layer on top of server.cpp

New Python OpenAI API compatibility server, which calls into / spawns the C++ server under the hood:

```bash
python -m examples.openai.server --model model.gguf
```

## Prerequisites

Note: To get conda, just install Miniforge (it's OSS): https://github.com/conda-forge/miniforge

```bash
conda create -n agent python=3.11
conda activate agent
pip install -r examples/openai/requirements.txt
```

## Features

The new [examples/openai/server.py](./server.py):

- Supports grammar-constrained tool calling for **all** models (incl. Mixtral 7x8B)

    - Optimised support for Functionary & Nous Hermes, easy to extend to other tool-calling schemes

    - Generic support w/ JSON schema that guides the model towards tool usage (at the cost of extra tokens):

        ```ts
          {
            // original_thought: string,
            thought_about_next_step_only: string,
            next_step: {tool_calls: {name: string, arguments: any}} | {result: T}
          }
          // Where T is the output JSON schema, or 'any'
        ```

        - Option to publicise schemas to models as TypeScript signatures (as for Functionary) or JSON schema.

        - Supports models that require user/assistant alternance (like Mixtral Instruct) by merging system messages into user messages.

- Spawns the C++ [llama.cpp server](../server) under the hood (unless passed `--endpoint`), but only uses its non-chat endpoint

  (depending on the prompting strategy, we weave the tool & output schema along with the chat template into the raw model grammar constraints)

- Uses the actual Jinja2 templates stored in the GGUF models

- Will eventually also spawn `whisper.cpp` and another server subprocess for the embeddings endpoint

Rationale: the C++ server lacks some OpenAI compatibility features (and can't realistically keep up with prompt templates w/o bringing in too many dependencies), this new layer could allow focusing the C++ server on serving efficiency and delegate OAI compliance to a layer easier to maintain.

## Test

If you want to see tools in action, look at the [agent example](../agent). Otherwise:

Start the server in Terminal 1:

```bash
python -m examples.openai --model  ~/AI/Models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
```

Query it in Terminal 2 (or use it from any framework that makes use of tools: note tool calls are guaranteed to comply to the schema, so retries are likely not necessary!):

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
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
      {"role": "user", "content": "what is the weather going to be like in San Francisco and Glasgow over the next 4 days"}
    ]
  }'
```

<details>
<summary>Show output</summary>

```json
{
  "id": "chatcmpl-3095057176",
  "object": "chat.completion",
  "created": 1711726921,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "name": null,
        "tool_call_id": null,
        "content": "In order to provide the required information, I need to call the get_n_day_weather_forecast function twice, once for San Francisco and once for Glasgow.",
        "tool_calls": [
          {
            "id": "call_970977",
            "type": "function",
            "function": {
              "name": "get_n_day_weather_forecast",
              "arguments": {
                "location": "San Francisco, CA",
                "format": "celsius",
                "num_days": 4
              }
            }
          }
        ]
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 546,
    "completion_tokens": 118,
    "total_tokens": 664
  },
  "system_fingerprint": "...",
  "error": null
}
```

</details>

## TODO

- Embedding endpoint w/ distinct server subprocess

- Evaluate options for session caching

    - Pass session id & store / read from file?

    - Support parent session ids for trees of thought?

    - Support precaching long prompts from CLI / read session files?

- Follow-ups

    - Remove OAI support from server

    - Remove non-Python json-schema-to-grammar versions

    - Reach out to frameworks to advertise new option.
