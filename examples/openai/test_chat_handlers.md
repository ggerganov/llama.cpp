
Messages:

```js
[
  {
    "role": "user",
    "name": null,
    "tool_call_id": null,
    "content": "Add two numbers for the purpose of this test.",
    "tool_calls": null
  },
  {
    "role": "assistant",
    "name": null,
    "tool_call_id": null,
    "content": null,
    "tool_calls": [
      {
        "id": "call_531873",
        "type": "function",
        "function": {
          "name": "superSecretTool",
          "arguments": {
            "a": 2535,
            "b": 32222000403
          }
        }
      }
    ]
  },
  {
    "role": "tool",
    "name": "superSecretTool",
    "tool_call_id": "call_531873",
    "content": "32222002938",
    "tool_calls": null
  },
  {
    "role": "assistant",
    "name": null,
    "tool_call_id": null,
    "content": "The sum of 2535 and 32222000403 is 42.",
    "tool_calls": null
  }
]
```


# mistral_instruct_v0_1


Template:

```js
{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}
```


## mistral_instruct_v0_1 / TOOLS_THOUGHTFUL_STEPS


### mistral_instruct_v0_1 / TOOLS_THOUGHTFUL_STEPS / with tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST]{
  "thought_about_next_step_only": "",
  "next_step": {
    "tool_calls": [
      {
        "id": "call_531873",
        "type": "function",
        "function": {
          "name": "superSecretTool",
          "arguments": {
            "a": 2535,
            "b": 32222000403
          }
        }
      }
    ]
  }
}</s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST]{
  "thought_about_next_step_only": "",
  "next_step": {
    "result": "The sum of 2535 and 32222000403 is 42."
  }
}</s>
```


Prompt:

```json
You are a function calling AI model.
Here are the tools available:
{
  "type": "function",
  "function": {
    "name": "superSecretTool",
    "description": "Adds two numbers",
    "parameters": {
      "properties": {
        "a": {
          "type": "integer"
        },
        "b": {
          "type": "integer"
        }
      },
      "required": [
        "a",
        "b"
      ]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "say",
    "description": "Says something out loud (TTS)",
    "parameters": {
      "properties": {
        "text": {
          "description": "The text to say out loud",
          "type": "string"
        }
      },
      "required": [
        "text"
      ]
    }
  }
}
Please respond in JSON format with the following schema: {
  "type": "object",
  "properties": {
    "thought_about_next_step_only": {
      "title": "Thought about next step",
      "type": "string"
    },
    "next_step": {
      "title": "Next Step: either a result or one or more tool calls to achieve the original goal",
      "oneOf": [
        {
          "properties": {
            "tool_calls": {
              "prefixItems": [
                {
                  "properties": {
                    "name": {
                      "title": "Name of the tool to call",
                      "type": "string"
                    },
                    "arguments": {
                      "title": "Arguments to pass to the tool",
                      "type": "object"
                    }
                  },
                  "required": [
                    "name",
                    "arguments"
                  ]
                }
              ]
            }
          },
          "required": [
            "tool_calls"
          ]
        },
        {
          "title": "Result (achieving original goal)",
          "properties": {
            "result": {
              "type": "integer"
            }
          },
          "required": [
            "result"
          ]
        }
      ]
    }
  },
  "required": [
    "original_goal",
    "thought_about_next_step_only",
    "next_step"
  ]
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
next-step ::= next-step-0 | next-step-1
next-step-0 ::= "{" space next-step-0-tool-calls-kv "}" space
next-step-0-tool-calls ::= "[" space next-step-0-tool-calls-tuple-0 "]" space
next-step-0-tool-calls-kv ::= "\"tool_calls\"" space ":" space next-step-0-tool-calls
next-step-0-tool-calls-tuple-0 ::= next-step-0-tool-calls-tuple-0-0 | next-step-0-tool-calls-tuple-0-1
next-step-0-tool-calls-tuple-0-0 ::= "{" space next-step-0-tool-calls-tuple-0-0-name-kv "," space next-step-0-tool-calls-tuple-0-0-arguments-kv "}" space
next-step-0-tool-calls-tuple-0-0-arguments ::= "{" space next-step-0-tool-calls-tuple-0-0-arguments-a-kv "," space next-step-0-tool-calls-tuple-0-0-arguments-b-kv "}" space
next-step-0-tool-calls-tuple-0-0-arguments-a-kv ::= "\"a\"" space ":" space integer
next-step-0-tool-calls-tuple-0-0-arguments-b-kv ::= "\"b\"" space ":" space integer
next-step-0-tool-calls-tuple-0-0-arguments-kv ::= "\"arguments\"" space ":" space next-step-0-tool-calls-tuple-0-0-arguments
next-step-0-tool-calls-tuple-0-0-name ::= "\"superSecretTool\""
next-step-0-tool-calls-tuple-0-0-name-kv ::= "\"name\"" space ":" space next-step-0-tool-calls-tuple-0-0-name
next-step-0-tool-calls-tuple-0-1 ::= "{" space next-step-0-tool-calls-tuple-0-1-name-kv "," space next-step-0-tool-calls-tuple-0-1-arguments-kv "}" space
next-step-0-tool-calls-tuple-0-1-arguments ::= "{" space next-step-0-tool-calls-tuple-0-1-arguments-text-kv "}" space
next-step-0-tool-calls-tuple-0-1-arguments-kv ::= "\"arguments\"" space ":" space next-step-0-tool-calls-tuple-0-1-arguments
next-step-0-tool-calls-tuple-0-1-arguments-text-kv ::= "\"text\"" space ":" space string
next-step-0-tool-calls-tuple-0-1-name ::= "\"say\""
next-step-0-tool-calls-tuple-0-1-name-kv ::= "\"name\"" space ":" space next-step-0-tool-calls-tuple-0-1-name
next-step-1 ::= "{" space next-step-1-result-kv "}" space
next-step-1-result-kv ::= "\"result\"" space ":" space integer
next-step-kv ::= "\"next_step\"" space ":" space next-step
root ::= "{" space thought-about-next-step-only-kv "," space next-step-kv "}" space
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
thought-about-next-step-only-kv ::= "\"thought_about_next_step_only\"" space ":" space string
```


### mistral_instruct_v0_1 / TOOLS_THOUGHTFUL_STEPS / without tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST]<tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call></s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST]The sum of 2535 and 32222000403 is 42.</s>
```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```


## mistral_instruct_v0_1 / TOOLS_MIXTRAL


### mistral_instruct_v0_1 / TOOLS_MIXTRAL / with tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST]<tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call></s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST]The sum of 2535 and 32222000403 is 42.</s>
```


Prompt:

```json
Call one or more functions to assist with the user query, every time this is possible. Don't make assumptions about what values to plug into functions. Here are the available tools:
<tools>
{
  "type": "function",
  "function": {
    "name": "superSecretTool",
    "description": "Adds two numbers",
    "parameters": {
      "properties": {
        "a": {
          "type": "integer"
        },
        "b": {
          "type": "integer"
        }
      },
      "required": [
        "a",
        "b"
      ]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "say",
    "description": "Says something out loud (TTS)",
    "parameters": {
      "properties": {
        "text": {
          "description": "The text to say out loud",
          "type": "string"
        }
      },
      "required": [
        "text"
      ]
    }
  }
}
</tools>

To call each function, give its name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"name": <function-name>, "arguments": <args-dict>}
</tool_call>
```


Grammar:

```js
content ::= [^<] | "<" [^t<] | "<t" [^o<]
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= content* tool-call?
say-tool-call ::= "{" space say-tool-call-name-kv "," space say-tool-call-arguments-kv "}" space
say-tool-call-arguments ::= "{" space say-tool-call-arguments-text-kv "}" space
say-tool-call-arguments-kv ::= "\"arguments\"" space ":" space say-tool-call-arguments
say-tool-call-arguments-text-kv ::= "\"text\"" space ":" space string
say-tool-call-name ::= "\"" "say" "\"" space
say-tool-call-name-kv ::= "\"name\"" space ":" space say-tool-call-name
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
superSecretTool-tool-call ::= "{" space superSecretTool-tool-call-name-kv "," space superSecretTool-tool-call-arguments-kv "}" space
superSecretTool-tool-call-arguments ::= "{" space superSecretTool-tool-call-arguments-a-kv "," space superSecretTool-tool-call-arguments-b-kv "}" space
superSecretTool-tool-call-arguments-a-kv ::= "\"a\"" space ":" space integer
superSecretTool-tool-call-arguments-b-kv ::= "\"b\"" space ":" space integer
superSecretTool-tool-call-arguments-kv ::= "\"arguments\"" space ":" space superSecretTool-tool-call-arguments
superSecretTool-tool-call-name ::= "\"" "superSecretTool" "\"" space
superSecretTool-tool-call-name-kv ::= "\"name\"" space ":" space superSecretTool-tool-call-name
tool-call ::= "<tool" "\\"? "_" "call>" space (superSecretTool-tool-call | say-tool-call)  space "</tool" "\\"? "_" "call>"
```


### mistral_instruct_v0_1 / TOOLS_MIXTRAL / without tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST]<tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call></s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST]The sum of 2535 and 32222000403 is 42.</s>
```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```


# functionary_v2_2


**Might Support Parallel Tool Calls**


Template:

```js
{#v2.2#}
{% for message in messages %}
{% if message['role'] == 'user' or message['role'] == 'system' %}
{{ '<|from|>' + message['role'] + '
<|recipient|>all
<|content|>' + message['content'] + '
' }}{% elif message['role'] == 'tool' %}
{{ '<|from|>' + message['name'] + '
<|recipient|>all
<|content|>' + message['content'] + '
' }}{% else %}
{% set contain_content='no'%}
{% if message['content'] is not none %}
{{ '<|from|>assistant
<|recipient|>all
<|content|>' + message['content'] }}{% set contain_content='yes'%}
{% endif %}
{% if 'tool_calls' in message and message['tool_calls'] is not none %}
{% for tool_call in message['tool_calls'] %}
{% set prompt='<|from|>assistant
<|recipient|>' + tool_call['function']['name'] + '
<|content|>' + tool_call['function']['arguments'] %}
{% if loop.index == 1 and contain_content == "no" %}
{{ prompt }}{% else %}
{{ '
' + prompt}}{% endif %}
{% endfor %}
{% endif %}
{{ '<|stop|>
' }}{% endif %}
{% endfor %}
{% if add_generation_prompt %}{{ '<|from|>assistant
<|recipient|>' }}{% endif %}
```


## functionary_v2_2 / TYPESCRIPT_FUNCTIONARY_V2


### functionary_v2_2 / TYPESCRIPT_FUNCTIONARY_V2 / with tools


Prompt:

```js
<|from|>user
<|recipient|>all
<|content|>Add two numbers for the purpose of this test.
<|from|>assistant
<|recipient|>all
<|content|><tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call><|stop|>
<|from|>superSecretTool
<|recipient|>all
<|content|>32222002938
<|from|>assistant
<|recipient|>all
<|content|>The sum of 2535 and 32222000403 is 42.<|stop|>
<|from|>assistant
<|recipient|>
```


Prompt:

```json
// Supported function definitions that should be called when necessary.
namespace functions {
// Adds two numbers
type superSecretTool = (_: {
a: number,
b: number
}) => any;

// Says something out loud (TTS)
type say = (_: {
// The text to say out loud
text: string
}) => any;
} // namespace functions
```


Grammar:

```js
content ::= start content-without-start
content-without-start ::= "all\n<|content|>" not-from*
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
not-from ::= ([^<] | "<" ([^|] | "|" ([^f] | "f" ([^r] | "r" ([^o] | "o" ([^m] | "m" ([^|] | "|" ([^>])?)?)?)?)?)?)?)
root ::= content-without-start   content*   (tool-call+ content*)? | tool-call-without-start tool-call* content*
say-args ::= "{" space say-args-text-kv "}" space
say-args-text-kv ::= "\"text\"" space ":" space string
say-call ::= "say" "\n<|content|>\n" say-args "\n"
space ::= " "?
start ::= "<|from|>assistant\n<|recipient|>"
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
superSecretTool-args ::= "{" space superSecretTool-args-a-kv "," space superSecretTool-args-b-kv "}" space
superSecretTool-args-a-kv ::= "\"a\"" space ":" space integer
superSecretTool-args-b-kv ::= "\"b\"" space ":" space integer
superSecretTool-call ::= "superSecretTool" "\n<|content|>\n" superSecretTool-args "\n"
tool-call ::= start tool-call-without-start
tool-call-without-start ::= superSecretTool-call | say-call
```


### functionary_v2_2 / TYPESCRIPT_FUNCTIONARY_V2 / without tools


Prompt:

```js
<|from|>user
<|recipient|>all
<|content|>Add two numbers for the purpose of this test.
<|from|>assistant
<|recipient|>all
<|content|><tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call><|stop|>
<|from|>superSecretTool
<|recipient|>all
<|content|>32222002938
<|from|>assistant
<|recipient|>all
<|content|>The sum of 2535 and 32222000403 is 42.<|stop|>
<|from|>assistant
<|recipient|>
```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```


# hermes_2_pro_mistral


Template:

```js
{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}
```


## hermes_2_pro_mistral / TOOLS_SHORT


### hermes_2_pro_mistral / TOOLS_SHORT / with tools


Prompt:

```js
<|im_start|>user
Add two numbers for the purpose of this test.<|im_end|>
<|im_start|>assistant
<tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call><|im_end|>
<|im_start|>tool
32222002938<|im_end|>
<|im_start|>assistant
The sum of 2535 and 32222000403 is 42.<|im_end|>
<|im_start|>assistant

```


Prompt:

```json
Here are the tools available:
<tools>
{
  "type": "function",
  "function": {
    "name": "superSecretTool",
    "description": "Adds two numbers",
    "parameters": {
      "properties": {
        "a": {
          "type": "integer"
        },
        "b": {
          "type": "integer"
        }
      },
      "required": [
        "a",
        "b"
      ]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "say",
    "description": "Says something out loud (TTS)",
    "parameters": {
      "properties": {
        "text": {
          "description": "The text to say out loud",
          "type": "string"
        }
      },
      "required": [
        "text"
      ]
    }
  }
}
</tools>
```


Grammar:

```js
content ::= [^<] | "<" [^t<] | "<t" [^o<]
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= content* tool-call?
say-tool-call ::= "{" space say-tool-call-name-kv "," space say-tool-call-arguments-kv "}" space
say-tool-call-arguments ::= "{" space say-tool-call-arguments-text-kv "}" space
say-tool-call-arguments-kv ::= "\"arguments\"" space ":" space say-tool-call-arguments
say-tool-call-arguments-text-kv ::= "\"text\"" space ":" space string
say-tool-call-name ::= "\"say\""
say-tool-call-name-kv ::= "\"name\"" space ":" space say-tool-call-name
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
superSecretTool-tool-call ::= "{" space superSecretTool-tool-call-name-kv "," space superSecretTool-tool-call-arguments-kv "}" space
superSecretTool-tool-call-arguments ::= "{" space superSecretTool-tool-call-arguments-a-kv "," space superSecretTool-tool-call-arguments-b-kv "}" space
superSecretTool-tool-call-arguments-a-kv ::= "\"a\"" space ":" space integer
superSecretTool-tool-call-arguments-b-kv ::= "\"b\"" space ":" space integer
superSecretTool-tool-call-arguments-kv ::= "\"arguments\"" space ":" space superSecretTool-tool-call-arguments
superSecretTool-tool-call-name ::= "\"superSecretTool\""
superSecretTool-tool-call-name-kv ::= "\"name\"" space ":" space superSecretTool-tool-call-name
tool-call ::= "<tool_call>" space (superSecretTool-tool-call | say-tool-call)  space "</tool_call>"
```


### hermes_2_pro_mistral / TOOLS_SHORT / without tools


Prompt:

```js
<|im_start|>user
Add two numbers for the purpose of this test.<|im_end|>
<|im_start|>assistant
<tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call><|im_end|>
<|im_start|>tool
32222002938<|im_end|>
<|im_start|>assistant
The sum of 2535 and 32222000403 is 42.<|im_end|>
<|im_start|>assistant

```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```


## hermes_2_pro_mistral / TOOLS_LONG


### hermes_2_pro_mistral / TOOLS_LONG / with tools


Prompt:

```js
<|im_start|>user
Add two numbers for the purpose of this test.<|im_end|>
<|im_start|>assistant
<tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call><|im_end|>
<|im_start|>tool
32222002938<|im_end|>
<|im_start|>assistant
The sum of 2535 and 32222000403 is 42.<|im_end|>
<|im_start|>assistant

```


Prompt:

```json
Call one or more functions to assist with the user query, every time this is possible. Don't make assumptions about what values to plug into functions. Here are the available tools:
<tools>
{
  "type": "function",
  "function": {
    "name": "superSecretTool",
    "description": "Adds two numbers",
    "parameters": {
      "properties": {
        "a": {
          "type": "integer"
        },
        "b": {
          "type": "integer"
        }
      },
      "required": [
        "a",
        "b"
      ]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "say",
    "description": "Says something out loud (TTS)",
    "parameters": {
      "properties": {
        "text": {
          "description": "The text to say out loud",
          "type": "string"
        }
      },
      "required": [
        "text"
      ]
    }
  }
}
</tools>

To call each function, give its name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"name": <function-name>, "arguments": <args-dict>}
</tool_call>
```


Grammar:

```js
content ::= [^<] | "<" [^t<] | "<t" [^o<]
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= content* tool-call?
say-tool-call ::= "{" space say-tool-call-name-kv "," space say-tool-call-arguments-kv "}" space
say-tool-call-arguments ::= "{" space say-tool-call-arguments-text-kv "}" space
say-tool-call-arguments-kv ::= "\"arguments\"" space ":" space say-tool-call-arguments
say-tool-call-arguments-text-kv ::= "\"text\"" space ":" space string
say-tool-call-name ::= "\"say\""
say-tool-call-name-kv ::= "\"name\"" space ":" space say-tool-call-name
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
superSecretTool-tool-call ::= "{" space superSecretTool-tool-call-name-kv "," space superSecretTool-tool-call-arguments-kv "}" space
superSecretTool-tool-call-arguments ::= "{" space superSecretTool-tool-call-arguments-a-kv "," space superSecretTool-tool-call-arguments-b-kv "}" space
superSecretTool-tool-call-arguments-a-kv ::= "\"a\"" space ":" space integer
superSecretTool-tool-call-arguments-b-kv ::= "\"b\"" space ":" space integer
superSecretTool-tool-call-arguments-kv ::= "\"arguments\"" space ":" space superSecretTool-tool-call-arguments
superSecretTool-tool-call-name ::= "\"superSecretTool\""
superSecretTool-tool-call-name-kv ::= "\"name\"" space ":" space superSecretTool-tool-call-name
tool-call ::= "<tool_call>" space (superSecretTool-tool-call | say-tool-call)  space "</tool_call>"
```


### hermes_2_pro_mistral / TOOLS_LONG / without tools


Prompt:

```js
<|im_start|>user
Add two numbers for the purpose of this test.<|im_end|>
<|im_start|>assistant
<tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call><|im_end|>
<|im_start|>tool
32222002938<|im_end|>
<|im_start|>assistant
The sum of 2535 and 32222000403 is 42.<|im_end|>
<|im_start|>assistant

```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```


## hermes_2_pro_mistral / TOOLS_THOUGHTFUL_STEPS


### hermes_2_pro_mistral / TOOLS_THOUGHTFUL_STEPS / with tools


Prompt:

```js
<|im_start|>user
Add two numbers for the purpose of this test.<|im_end|>
<|im_start|>assistant
{
  "thought_about_next_step_only": "",
  "next_step": {
    "tool_calls": [
      {
        "id": "call_531873",
        "type": "function",
        "function": {
          "name": "superSecretTool",
          "arguments": {
            "a": 2535,
            "b": 32222000403
          }
        }
      }
    ]
  }
}<|im_end|>
<|im_start|>tool
32222002938<|im_end|>
<|im_start|>assistant
{
  "thought_about_next_step_only": "",
  "next_step": {
    "result": "The sum of 2535 and 32222000403 is 42."
  }
}<|im_end|>
<|im_start|>assistant

```


Prompt:

```json
You are a function calling AI model.
Here are the tools available:
{
  "type": "function",
  "function": {
    "name": "superSecretTool",
    "description": "Adds two numbers",
    "parameters": {
      "properties": {
        "a": {
          "type": "integer"
        },
        "b": {
          "type": "integer"
        }
      },
      "required": [
        "a",
        "b"
      ]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "say",
    "description": "Says something out loud (TTS)",
    "parameters": {
      "properties": {
        "text": {
          "description": "The text to say out loud",
          "type": "string"
        }
      },
      "required": [
        "text"
      ]
    }
  }
}
Please respond in JSON format with the following schema: {
  "type": "object",
  "properties": {
    "thought_about_next_step_only": {
      "title": "Thought about next step",
      "type": "string"
    },
    "next_step": {
      "title": "Next Step: either a result or one or more tool calls to achieve the original goal",
      "oneOf": [
        {
          "properties": {
            "tool_calls": {
              "prefixItems": [
                {
                  "properties": {
                    "name": {
                      "title": "Name of the tool to call",
                      "type": "string"
                    },
                    "arguments": {
                      "title": "Arguments to pass to the tool",
                      "type": "object"
                    }
                  },
                  "required": [
                    "name",
                    "arguments"
                  ]
                }
              ]
            }
          },
          "required": [
            "tool_calls"
          ]
        },
        {
          "title": "Result (achieving original goal)",
          "properties": {
            "result": {
              "type": "integer"
            }
          },
          "required": [
            "result"
          ]
        }
      ]
    }
  },
  "required": [
    "original_goal",
    "thought_about_next_step_only",
    "next_step"
  ]
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
next-step ::= next-step-0 | next-step-1
next-step-0 ::= "{" space next-step-0-tool-calls-kv "}" space
next-step-0-tool-calls ::= "[" space next-step-0-tool-calls-tuple-0 "]" space
next-step-0-tool-calls-kv ::= "\"tool_calls\"" space ":" space next-step-0-tool-calls
next-step-0-tool-calls-tuple-0 ::= next-step-0-tool-calls-tuple-0-0 | next-step-0-tool-calls-tuple-0-1
next-step-0-tool-calls-tuple-0-0 ::= "{" space next-step-0-tool-calls-tuple-0-0-name-kv "," space next-step-0-tool-calls-tuple-0-0-arguments-kv "}" space
next-step-0-tool-calls-tuple-0-0-arguments ::= "{" space next-step-0-tool-calls-tuple-0-0-arguments-a-kv "," space next-step-0-tool-calls-tuple-0-0-arguments-b-kv "}" space
next-step-0-tool-calls-tuple-0-0-arguments-a-kv ::= "\"a\"" space ":" space integer
next-step-0-tool-calls-tuple-0-0-arguments-b-kv ::= "\"b\"" space ":" space integer
next-step-0-tool-calls-tuple-0-0-arguments-kv ::= "\"arguments\"" space ":" space next-step-0-tool-calls-tuple-0-0-arguments
next-step-0-tool-calls-tuple-0-0-name ::= "\"superSecretTool\""
next-step-0-tool-calls-tuple-0-0-name-kv ::= "\"name\"" space ":" space next-step-0-tool-calls-tuple-0-0-name
next-step-0-tool-calls-tuple-0-1 ::= "{" space next-step-0-tool-calls-tuple-0-1-name-kv "," space next-step-0-tool-calls-tuple-0-1-arguments-kv "}" space
next-step-0-tool-calls-tuple-0-1-arguments ::= "{" space next-step-0-tool-calls-tuple-0-1-arguments-text-kv "}" space
next-step-0-tool-calls-tuple-0-1-arguments-kv ::= "\"arguments\"" space ":" space next-step-0-tool-calls-tuple-0-1-arguments
next-step-0-tool-calls-tuple-0-1-arguments-text-kv ::= "\"text\"" space ":" space string
next-step-0-tool-calls-tuple-0-1-name ::= "\"say\""
next-step-0-tool-calls-tuple-0-1-name-kv ::= "\"name\"" space ":" space next-step-0-tool-calls-tuple-0-1-name
next-step-1 ::= "{" space next-step-1-result-kv "}" space
next-step-1-result-kv ::= "\"result\"" space ":" space integer
next-step-kv ::= "\"next_step\"" space ":" space next-step
root ::= "{" space thought-about-next-step-only-kv "," space next-step-kv "}" space
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
thought-about-next-step-only-kv ::= "\"thought_about_next_step_only\"" space ":" space string
```


### hermes_2_pro_mistral / TOOLS_THOUGHTFUL_STEPS / without tools


Prompt:

```js
<|im_start|>user
Add two numbers for the purpose of this test.<|im_end|>
<|im_start|>assistant
<tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call><|im_end|>
<|im_start|>tool
32222002938<|im_end|>
<|im_start|>assistant
The sum of 2535 and 32222000403 is 42.<|im_end|>
<|im_start|>assistant

```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```


## hermes_2_pro_mistral / TOOLS_HERMES_2_PRO


### hermes_2_pro_mistral / TOOLS_HERMES_2_PRO / with tools


Prompt:

```js
<|im_start|>user
Add two numbers for the purpose of this test.<|im_end|>
<|im_start|>assistant
<tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call><|im_end|>
<|im_start|>tool
32222002938<|im_end|>
<|im_start|>assistant
The sum of 2535 and 32222000403 is 42.<|im_end|>
<|im_start|>assistant

```


Prompt:

```json
You are a function calling AI agent with self-recursion. You can call only one function at a time and analyse data you get from function response. You are provided with function signatures within <tools></tools> XML tags. The current date is: 2024-03-30. You may use agentic frameworks for reasoning and planning to help with user query. Please call a function and wait for function results to be provided to you in the next iteration. Don't make assumptions about what values to plug into function arguments. Once you have called a function, results will be fed back to you within <tool_response></tool_response> XML tags. Don't make assumptions about tool results if <tool_response> XML tags are not present since function hasn't been executed yet. Analyze the data once you get the results and call another function. At each iteration please continue adding the your analysis to previous summary. Your final response should directly answer the user query with an anlysis or summary of the results of function calls. Here are the available tools: <tools> ['{"type":"function","function":{"name":"superSecretTool","description":"Adds two numbers","parameters":{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}}', '{"type":"function","function":{"name":"say","description":"Says something out loud (TTS)","parameters":{"properties":{"text":{"description":"The text to say out loud","type":"string"}},"required":["text"]}}}'] </tools> If the provided function signatures doesn't have the function you must call, you may write executable python code in markdown syntax and call code_interpreter() function as follows: <tool_call> {"arguments": {"code_markdown": <python-code>, "name": "code_interpreter"}} </tool_call> Make sure that the json object above with code markdown block is parseable with json.loads() and the XML block with XML ElementTree. Use the following pydantic model json schema for each tool call you will make: {'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name'], 'title': 'FunctionCall', 'type': 'object'} At the very first turn you don't have <tool_results> so you shouldn't not make up the results.
Please keep a running summary with analysis of previous function results and summaries from previous iterations.
Do not stop calling functions until the task has been accomplished or you've reached max iteration of 10.
Calling multiple functions at once can overload the system and increase cost so call one function at a time please.
If you plan to continue with analysis, always call another function.
For each function call return a valid json object (using doulbe quotes) with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>

```


Grammar:

```js
content ::= [^<] | "<" [^t<] | "<t" [^o<]
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= content* tool-call?
say-tool-call ::= "{" space say-tool-call-name-kv "," space say-tool-call-arguments-kv "}" space
say-tool-call-arguments ::= "{" space say-tool-call-arguments-text-kv "}" space
say-tool-call-arguments-kv ::= "\"arguments\"" space ":" space say-tool-call-arguments
say-tool-call-arguments-text-kv ::= "\"text\"" space ":" space string
say-tool-call-name ::= "\"say\""
say-tool-call-name-kv ::= "\"name\"" space ":" space say-tool-call-name
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
superSecretTool-tool-call ::= "{" space superSecretTool-tool-call-name-kv "," space superSecretTool-tool-call-arguments-kv "}" space
superSecretTool-tool-call-arguments ::= "{" space superSecretTool-tool-call-arguments-a-kv "," space superSecretTool-tool-call-arguments-b-kv "}" space
superSecretTool-tool-call-arguments-a-kv ::= "\"a\"" space ":" space integer
superSecretTool-tool-call-arguments-b-kv ::= "\"b\"" space ":" space integer
superSecretTool-tool-call-arguments-kv ::= "\"arguments\"" space ":" space superSecretTool-tool-call-arguments
superSecretTool-tool-call-name ::= "\"superSecretTool\""
superSecretTool-tool-call-name-kv ::= "\"name\"" space ":" space superSecretTool-tool-call-name
tool-call ::= "<tool_call>" space (superSecretTool-tool-call | say-tool-call)  space "</tool_call>"
```


### hermes_2_pro_mistral / TOOLS_HERMES_2_PRO / without tools


Prompt:

```js
<|im_start|>user
Add two numbers for the purpose of this test.<|im_end|>
<|im_start|>assistant
<tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call><|im_end|>
<|im_start|>tool
32222002938<|im_end|>
<|im_start|>assistant
The sum of 2535 and 32222000403 is 42.<|im_end|>
<|im_start|>assistant

```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```


# llama2


Template:

```js
{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}
```


## llama2 / TOOLS_SHORT


### llama2 / TOOLS_SHORT / with tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST] <tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call> </s><s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST] The sum of 2535 and 32222000403 is 42. </s>
```


Prompt:

```json
Here are the tools available:
<tools>
{
  "type": "function",
  "function": {
    "name": "superSecretTool",
    "description": "Adds two numbers",
    "parameters": {
      "properties": {
        "a": {
          "type": "integer"
        },
        "b": {
          "type": "integer"
        }
      },
      "required": [
        "a",
        "b"
      ]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "say",
    "description": "Says something out loud (TTS)",
    "parameters": {
      "properties": {
        "text": {
          "description": "The text to say out loud",
          "type": "string"
        }
      },
      "required": [
        "text"
      ]
    }
  }
}
</tools>
```


Grammar:

```js
content ::= [^<] | "<" [^t<] | "<t" [^o<]
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= content* tool-call?
say-tool-call ::= "{" space say-tool-call-name-kv "," space say-tool-call-arguments-kv "}" space
say-tool-call-arguments ::= "{" space say-tool-call-arguments-text-kv "}" space
say-tool-call-arguments-kv ::= "\"arguments\"" space ":" space say-tool-call-arguments
say-tool-call-arguments-text-kv ::= "\"text\"" space ":" space string
say-tool-call-name ::= "\"say\""
say-tool-call-name-kv ::= "\"name\"" space ":" space say-tool-call-name
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
superSecretTool-tool-call ::= "{" space superSecretTool-tool-call-name-kv "," space superSecretTool-tool-call-arguments-kv "}" space
superSecretTool-tool-call-arguments ::= "{" space superSecretTool-tool-call-arguments-a-kv "," space superSecretTool-tool-call-arguments-b-kv "}" space
superSecretTool-tool-call-arguments-a-kv ::= "\"a\"" space ":" space integer
superSecretTool-tool-call-arguments-b-kv ::= "\"b\"" space ":" space integer
superSecretTool-tool-call-arguments-kv ::= "\"arguments\"" space ":" space superSecretTool-tool-call-arguments
superSecretTool-tool-call-name ::= "\"superSecretTool\""
superSecretTool-tool-call-name-kv ::= "\"name\"" space ":" space superSecretTool-tool-call-name
tool-call ::= "<tool_call>" space (superSecretTool-tool-call | say-tool-call)  space "</tool_call>"
```


### llama2 / TOOLS_SHORT / without tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST] <tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call> </s><s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST] The sum of 2535 and 32222000403 is 42. </s>
```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```


## llama2 / TOOLS_LONG


### llama2 / TOOLS_LONG / with tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST] <tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call> </s><s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST] The sum of 2535 and 32222000403 is 42. </s>
```


Prompt:

```json
Call one or more functions to assist with the user query, every time this is possible. Don't make assumptions about what values to plug into functions. Here are the available tools:
<tools>
{
  "type": "function",
  "function": {
    "name": "superSecretTool",
    "description": "Adds two numbers",
    "parameters": {
      "properties": {
        "a": {
          "type": "integer"
        },
        "b": {
          "type": "integer"
        }
      },
      "required": [
        "a",
        "b"
      ]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "say",
    "description": "Says something out loud (TTS)",
    "parameters": {
      "properties": {
        "text": {
          "description": "The text to say out loud",
          "type": "string"
        }
      },
      "required": [
        "text"
      ]
    }
  }
}
</tools>

To call each function, give its name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"name": <function-name>, "arguments": <args-dict>}
</tool_call>
```


Grammar:

```js
content ::= [^<] | "<" [^t<] | "<t" [^o<]
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= content* tool-call?
say-tool-call ::= "{" space say-tool-call-name-kv "," space say-tool-call-arguments-kv "}" space
say-tool-call-arguments ::= "{" space say-tool-call-arguments-text-kv "}" space
say-tool-call-arguments-kv ::= "\"arguments\"" space ":" space say-tool-call-arguments
say-tool-call-arguments-text-kv ::= "\"text\"" space ":" space string
say-tool-call-name ::= "\"say\""
say-tool-call-name-kv ::= "\"name\"" space ":" space say-tool-call-name
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
superSecretTool-tool-call ::= "{" space superSecretTool-tool-call-name-kv "," space superSecretTool-tool-call-arguments-kv "}" space
superSecretTool-tool-call-arguments ::= "{" space superSecretTool-tool-call-arguments-a-kv "," space superSecretTool-tool-call-arguments-b-kv "}" space
superSecretTool-tool-call-arguments-a-kv ::= "\"a\"" space ":" space integer
superSecretTool-tool-call-arguments-b-kv ::= "\"b\"" space ":" space integer
superSecretTool-tool-call-arguments-kv ::= "\"arguments\"" space ":" space superSecretTool-tool-call-arguments
superSecretTool-tool-call-name ::= "\"superSecretTool\""
superSecretTool-tool-call-name-kv ::= "\"name\"" space ":" space superSecretTool-tool-call-name
tool-call ::= "<tool_call>" space (superSecretTool-tool-call | say-tool-call)  space "</tool_call>"
```


### llama2 / TOOLS_LONG / without tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST] <tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call> </s><s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST] The sum of 2535 and 32222000403 is 42. </s>
```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```


## llama2 / TOOLS_THOUGHTFUL_STEPS


### llama2 / TOOLS_THOUGHTFUL_STEPS / with tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST] {
  "thought_about_next_step_only": "",
  "next_step": {
    "tool_calls": [
      {
        "id": "call_531873",
        "type": "function",
        "function": {
          "name": "superSecretTool",
          "arguments": {
            "a": 2535,
            "b": 32222000403
          }
        }
      }
    ]
  }
} </s><s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST] {
  "thought_about_next_step_only": "",
  "next_step": {
    "result": "The sum of 2535 and 32222000403 is 42."
  }
} </s>
```


Prompt:

```json
You are a function calling AI model.
Here are the tools available:
{
  "type": "function",
  "function": {
    "name": "superSecretTool",
    "description": "Adds two numbers",
    "parameters": {
      "properties": {
        "a": {
          "type": "integer"
        },
        "b": {
          "type": "integer"
        }
      },
      "required": [
        "a",
        "b"
      ]
    }
  }
}
{
  "type": "function",
  "function": {
    "name": "say",
    "description": "Says something out loud (TTS)",
    "parameters": {
      "properties": {
        "text": {
          "description": "The text to say out loud",
          "type": "string"
        }
      },
      "required": [
        "text"
      ]
    }
  }
}
Please respond in JSON format with the following schema: {
  "type": "object",
  "properties": {
    "thought_about_next_step_only": {
      "title": "Thought about next step",
      "type": "string"
    },
    "next_step": {
      "title": "Next Step: either a result or one or more tool calls to achieve the original goal",
      "oneOf": [
        {
          "properties": {
            "tool_calls": {
              "prefixItems": [
                {
                  "properties": {
                    "name": {
                      "title": "Name of the tool to call",
                      "type": "string"
                    },
                    "arguments": {
                      "title": "Arguments to pass to the tool",
                      "type": "object"
                    }
                  },
                  "required": [
                    "name",
                    "arguments"
                  ]
                }
              ]
            }
          },
          "required": [
            "tool_calls"
          ]
        },
        {
          "title": "Result (achieving original goal)",
          "properties": {
            "result": {
              "type": "integer"
            }
          },
          "required": [
            "result"
          ]
        }
      ]
    }
  },
  "required": [
    "original_goal",
    "thought_about_next_step_only",
    "next_step"
  ]
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
next-step ::= next-step-0 | next-step-1
next-step-0 ::= "{" space next-step-0-tool-calls-kv "}" space
next-step-0-tool-calls ::= "[" space next-step-0-tool-calls-tuple-0 "]" space
next-step-0-tool-calls-kv ::= "\"tool_calls\"" space ":" space next-step-0-tool-calls
next-step-0-tool-calls-tuple-0 ::= next-step-0-tool-calls-tuple-0-0 | next-step-0-tool-calls-tuple-0-1
next-step-0-tool-calls-tuple-0-0 ::= "{" space next-step-0-tool-calls-tuple-0-0-name-kv "," space next-step-0-tool-calls-tuple-0-0-arguments-kv "}" space
next-step-0-tool-calls-tuple-0-0-arguments ::= "{" space next-step-0-tool-calls-tuple-0-0-arguments-a-kv "," space next-step-0-tool-calls-tuple-0-0-arguments-b-kv "}" space
next-step-0-tool-calls-tuple-0-0-arguments-a-kv ::= "\"a\"" space ":" space integer
next-step-0-tool-calls-tuple-0-0-arguments-b-kv ::= "\"b\"" space ":" space integer
next-step-0-tool-calls-tuple-0-0-arguments-kv ::= "\"arguments\"" space ":" space next-step-0-tool-calls-tuple-0-0-arguments
next-step-0-tool-calls-tuple-0-0-name ::= "\"superSecretTool\""
next-step-0-tool-calls-tuple-0-0-name-kv ::= "\"name\"" space ":" space next-step-0-tool-calls-tuple-0-0-name
next-step-0-tool-calls-tuple-0-1 ::= "{" space next-step-0-tool-calls-tuple-0-1-name-kv "," space next-step-0-tool-calls-tuple-0-1-arguments-kv "}" space
next-step-0-tool-calls-tuple-0-1-arguments ::= "{" space next-step-0-tool-calls-tuple-0-1-arguments-text-kv "}" space
next-step-0-tool-calls-tuple-0-1-arguments-kv ::= "\"arguments\"" space ":" space next-step-0-tool-calls-tuple-0-1-arguments
next-step-0-tool-calls-tuple-0-1-arguments-text-kv ::= "\"text\"" space ":" space string
next-step-0-tool-calls-tuple-0-1-name ::= "\"say\""
next-step-0-tool-calls-tuple-0-1-name-kv ::= "\"name\"" space ":" space next-step-0-tool-calls-tuple-0-1-name
next-step-1 ::= "{" space next-step-1-result-kv "}" space
next-step-1-result-kv ::= "\"result\"" space ":" space integer
next-step-kv ::= "\"next_step\"" space ":" space next-step
root ::= "{" space thought-about-next-step-only-kv "," space next-step-kv "}" space
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
thought-about-next-step-only-kv ::= "\"thought_about_next_step_only\"" space ":" space string
```


### llama2 / TOOLS_THOUGHTFUL_STEPS / without tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST] <tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call> </s><s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST] The sum of 2535 and 32222000403 is 42. </s>
```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```


## llama2 / TOOLS_HERMES_2_PRO


### llama2 / TOOLS_HERMES_2_PRO / with tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST] <tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call> </s><s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST] The sum of 2535 and 32222000403 is 42. </s>
```


Prompt:

```json
You are a function calling AI agent with self-recursion. You can call only one function at a time and analyse data you get from function response. You are provided with function signatures within <tools></tools> XML tags. The current date is: 2024-03-30. You may use agentic frameworks for reasoning and planning to help with user query. Please call a function and wait for function results to be provided to you in the next iteration. Don't make assumptions about what values to plug into function arguments. Once you have called a function, results will be fed back to you within <tool_response></tool_response> XML tags. Don't make assumptions about tool results if <tool_response> XML tags are not present since function hasn't been executed yet. Analyze the data once you get the results and call another function. At each iteration please continue adding the your analysis to previous summary. Your final response should directly answer the user query with an anlysis or summary of the results of function calls. Here are the available tools: <tools> ['{"type":"function","function":{"name":"superSecretTool","description":"Adds two numbers","parameters":{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}}', '{"type":"function","function":{"name":"say","description":"Says something out loud (TTS)","parameters":{"properties":{"text":{"description":"The text to say out loud","type":"string"}},"required":["text"]}}}'] </tools> If the provided function signatures doesn't have the function you must call, you may write executable python code in markdown syntax and call code_interpreter() function as follows: <tool_call> {"arguments": {"code_markdown": <python-code>, "name": "code_interpreter"}} </tool_call> Make sure that the json object above with code markdown block is parseable with json.loads() and the XML block with XML ElementTree. Use the following pydantic model json schema for each tool call you will make: {'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name'], 'title': 'FunctionCall', 'type': 'object'} At the very first turn you don't have <tool_results> so you shouldn't not make up the results.
Please keep a running summary with analysis of previous function results and summaries from previous iterations.
Do not stop calling functions until the task has been accomplished or you've reached max iteration of 10.
Calling multiple functions at once can overload the system and increase cost so call one function at a time please.
If you plan to continue with analysis, always call another function.
For each function call return a valid json object (using doulbe quotes) with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>

```


Grammar:

```js
content ::= [^<] | "<" [^t<] | "<t" [^o<]
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integer ::= ("-"? integral-part) space
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= content* tool-call?
say-tool-call ::= "{" space say-tool-call-name-kv "," space say-tool-call-arguments-kv "}" space
say-tool-call-arguments ::= "{" space say-tool-call-arguments-text-kv "}" space
say-tool-call-arguments-kv ::= "\"arguments\"" space ":" space say-tool-call-arguments
say-tool-call-arguments-text-kv ::= "\"text\"" space ":" space string
say-tool-call-name ::= "\"say\""
say-tool-call-name-kv ::= "\"name\"" space ":" space say-tool-call-name
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space
superSecretTool-tool-call ::= "{" space superSecretTool-tool-call-name-kv "," space superSecretTool-tool-call-arguments-kv "}" space
superSecretTool-tool-call-arguments ::= "{" space superSecretTool-tool-call-arguments-a-kv "," space superSecretTool-tool-call-arguments-b-kv "}" space
superSecretTool-tool-call-arguments-a-kv ::= "\"a\"" space ":" space integer
superSecretTool-tool-call-arguments-b-kv ::= "\"b\"" space ":" space integer
superSecretTool-tool-call-arguments-kv ::= "\"arguments\"" space ":" space superSecretTool-tool-call-arguments
superSecretTool-tool-call-name ::= "\"superSecretTool\""
superSecretTool-tool-call-name-kv ::= "\"name\"" space ":" space superSecretTool-tool-call-name
tool-call ::= "<tool_call>" space (superSecretTool-tool-call | say-tool-call)  space "</tool_call>"
```


### llama2 / TOOLS_HERMES_2_PRO / without tools


Prompt:

```js
<s>[INST] Add two numbers for the purpose of this test. [/INST] <tool_call>{"id": "call_531873", "type": "function", "function": {"name": "superSecretTool", "arguments": {"a": 2535, "b": 32222000403}}}</tool_call> </s><s>[INST] [TOOL(name=superSecretTool, id=call_531873)]32222002938[/TOOL] [/INST] The sum of 2535 and 32222000403 is 42. </s>
```


Prompt:

```json
Please respond in JSON format with the following schema: {
  "type": "integer"
}
```


Grammar:

```js
decimal-part ::= [0-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
integral-part ::= [0-9] | [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
root ::= ("-"? integral-part) space
space ::= " "?
```

