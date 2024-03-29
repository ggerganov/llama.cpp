#
#
# python -m examples.agent.test_chat_handlers | tee examples/agent/test_chat_handlers.md
    
import json
from pathlib import Path
import typer
from typing import Annotated

from examples.openai.api import ChatCompletionRequest, ChatCompletionResponse, Message, Tool, ToolFunction
from examples.openai.gguf_kvs import GGUFKeyValues, Keys
from examples.openai.prompting import ChatHandlerArgs, ChatTemplate, ToolsPromptStyle, get_chat_handler



TEST_MESSAGES = [
    Message(**{
        "role": "user",
        "name": None,
        "tool_call_id": None,
        "content": "What is the sum of 2535 squared and 32222000403 then multiplied by one and a half. What's a third of the result?",
        "tool_calls": None
    }),
    Message(**{
        "role": "assistant",
        "name": None,
        "tool_call_id": None,
        "content": "?",
        "tool_calls": [
            {
                "id": "call_531873",
                "type": "function",
                "function": {
                    "name": "add",
                    "arguments": {
                        "a": 2535,
                        "b": 32222000403
                    }
                }
            }
        ]
    }),
    Message(**{
        "role": "tool",
        "name": "add",
        "tool_call_id": "call_531873",
        "content": "32222002938",
        "tool_calls": None
    })
]

TEST_TOOLS = [
    Tool(
        type="function",
        function=ToolFunction(
            name="add",
            description="Adds two numbers",
            parameters={
                "properties": {
                  "a": {"type": "integer"},
                  "b": {"type": "integer"},
                },
                "required": ["a", "b"]
            }
        )
    ),
    Tool(
        type="function",
        function=ToolFunction(
            name="say",
            description="Says something out loud (TTS)",
            parameters={
                "properties": {
                  "text": {
                      "description": "The text to say out loud",
                      "type": "string"
                  },
                },
                "required": ["text"]
            }
        )
    )
]

TEST_OUTPUT_SCHEMA = {"type": "integer"}

if __name__ == "__main__":
   
    # chat_templates = {
    #   'mistral_instruct_v0_1': ChatTemplate.from_huggingface("mistralai/Mixtral-8x7B-Instruct-v0.1"),
    #   'functionary_v2_2': ChatTemplate.from_huggingface("meetkai/functionary-small-v2.2"),
    #   'hermes_2_pro_mistral': ChatTemplate.from_huggingface("NousResearch/Hermes-2-Pro-Mistral-7B"),
    #   'llama2': ChatTemplate.from_huggingface("meta-llama/Llama-2-7b-chat-hf"),
    # }
    # print(json.dumps({k: v.model_dump() for k, v in chat_templates.items()}, indent=2))
    # exit(0)

    chat_templates = {
        "mistral_instruct_v0_1": {
            "template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
            "eos_token": "</s>",
            "bos_token": "<s>"
        },
        "functionary_v2_2": {
            "template": "{#v2.2#}\n{% for message in messages %}\n{% if message['role'] == 'user' or message['role'] == 'system' %}\n{{ '<|from|>' + message['role'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}{% elif message['role'] == 'tool' %}\n{{ '<|from|>' + message['name'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}{% else %}\n{% set contain_content='no'%}\n{% if message['content'] is not none %}\n{{ '<|from|>assistant\n<|recipient|>all\n<|content|>' + message['content'] }}{% set contain_content='yes'%}\n{% endif %}\n{% if 'tool_calls' in message and message['tool_calls'] is not none %}\n{% for tool_call in message['tool_calls'] %}\n{% set prompt='<|from|>assistant\n<|recipient|>' + tool_call['function']['name'] + '\n<|content|>' + tool_call['function']['arguments'] %}\n{% if loop.index == 1 and contain_content == \"no\" %}\n{{ prompt }}{% else %}\n{{ '\n' + prompt}}{% endif %}\n{% endfor %}\n{% endif %}\n{{ '<|stop|>\n' }}{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}{{ '<|from|>assistant\n<|recipient|>' }}{% endif %}",
            "eos_token": "</s>",
            "bos_token": "<s>"
        },
        "hermes_2_pro_mistral": {
            "template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
            "eos_token": "<|im_end|>",
            "bos_token": "<s>"
        },
        "llama2": {
            "template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",
            "eos_token": "</s>",
            "bos_token": "<s>"
        },
    }
    chat_templates = {k: ChatTemplate(**v) for k, v in chat_templates.items()}

    print(f'\nMessages:\n\n```js\n{json.dumps([m.model_dump() for m in TEST_MESSAGES], indent=2)}\n```\n')

    for model_name, chat_template in chat_templates.items():
        print(f"\n# {model_name}\n")
        print(f'\nTemplate:\n\n```js\n{chat_template.template}\n```\n')

        print(f'\nPrompt:\n\n```js\n{chat_template.render(TEST_MESSAGES, add_generation_prompt=True)}\n```\n')

        argss = {
            "with tools": ChatHandlerArgs(
                chat_template=chat_template, #ChatTemplate.from_gguf(GGUFKeyValues(model)),
                response_schema=TEST_OUTPUT_SCHEMA,
                tools=TEST_TOOLS,
            ),
            "without tools": ChatHandlerArgs(
                chat_template=chat_template, #ChatTemplate.from_gguf(GGUFKeyValues(model)),
                response_schema=TEST_OUTPUT_SCHEMA,
                tools=[],
            ),
        }
        
        for style in ToolsPromptStyle:
            if (style == ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2) != (model_name.startswith("functionary")):
                continue

            if style == ToolsPromptStyle.TOOLS_MIXTRAL and model_name != "mistral_instruct_v0_1":
                continue

            if model_name == "mistral_instruct_v0_1" and style not in (ToolsPromptStyle.TOOLS_THOUGHTFUL_STEPS, ToolsPromptStyle.TOOLS_MIXTRAL):
                continue

            print(f'\n## {style}\n')

            for tn, args in argss.items():
                ch = get_chat_handler(args, parallel_calls=True, tool_style=style)
                
                print(f'\n### {tn}\n')
                
                print(f'\nPrompt:\n\n```json\n{ch.output_format_prompt.content}\n```\n')

                print(f'\nGrammar:\n\n```js\n{ch.grammar}\n```\n')


    # test_templates([
    #     Message(**{
    #         "role": "user",
    #         "name": None,
    #         "tool_call_id": None,
    #         "content": "What is the sum of 2535 squared and 32222000403 then multiplied by one and a half. What's a third of the result?",
    #         "tool_calls": None
    #     }),
    #     Message(**{
    #         "role": "assistant",
    #         # "name": None,
    #         "tool_call_id": None,
    #         "content": "?",
    #         "tool_calls": [
    #             {
    #                 # "id": "call_531873",
    #                 "type": "function",
    #                 "function": {
    #                     "name": "add",
    #                     "arguments": {
    #                         "a": 2535,
    #                         "b": 32222000403
    #                     }
    #                 }
    #             }
    #         ]
    #     }),
    #     Message(**{
    #         "role": "tool",
    #         "name": "add",
    #         "tool_call_id": "call_531873",
    #         "content": "32222002938",
    #         "tool_calls": None
    #     })
    # ])
