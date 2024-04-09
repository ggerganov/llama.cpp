#
#
# python -m examples.openai.test_chat_handlers | tee examples/openai/test_chat_handlers.md

import json
import sys

from examples.openai.api import FunctionCall, Message, Tool, ToolCall, ToolFunction
from examples.openai.prompting import ChatHandlerArgs, ChatTemplate, ToolsPromptStyle, get_chat_handler

TEST_ARG_A = 2535
TEST_ARG_B = 32222000403
TEST_SUM = 32222002938

QUESTION = "Add two numbers for the purpose of this test."
ANSWER = "The sum of 2535 and 32222000403 is 42."

PROMPT_MESSAGE = Message(
    role="user",
    content=QUESTION,
)
ASSIST_MESSAGE = Message(
    role="assistant",
    content=ANSWER,
)
TOOL_NAME = "superSecretTool"
TOOL_CALL = ToolCall(
    id="call_531873",
    type="function",
    function=FunctionCall(
        name=TOOL_NAME,
        arguments={
            "a": TEST_ARG_A,
            "b": TEST_ARG_B
        }
    )
)
TOOL_CALL_MESSAGE = Message(
    role="assistant",
    content=None,
    tool_calls=[TOOL_CALL],
)

TEST_THOUGHT = "I've thought a lot about this."
THOUGHTFUL_TOOL_CALL_MESSAGE = Message(
    role="assistant",
    content=TEST_THOUGHT,
    tool_calls=[TOOL_CALL],
)

# UNDERSCORE_ESCAPED_TOOL_CALL_MESSAGE = Message(**{
#     **TOOL_CALL_MESSAGE.model_dump(),
#     "tool_calls": [
#         json.loads(tc.model_dump_json().replace("_", "\\_"))
#         for tc in TOOL_CALL_MESSAGE.tool_calls
#     ],
# })
TOOL_MESSAGE = Message(
    role="tool",
    name=TOOL_NAME,
    tool_call_id="call_531873",
    content=f'{TEST_SUM}',
    tool_calls=None
)
TEST_MESSAGES = [
    PROMPT_MESSAGE,
    TOOL_CALL_MESSAGE,
    TOOL_MESSAGE,
    ASSIST_MESSAGE,
]
TEST_MESSAGES_THOUGHT = [
    PROMPT_MESSAGE,
    THOUGHTFUL_TOOL_CALL_MESSAGE,
    TOOL_MESSAGE,
    ASSIST_MESSAGE,
]


TEST_TOOLS = [
    Tool(
        type="function",
        function=ToolFunction(
            name=TOOL_NAME,
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

# Generate the JSON for TEST_TEMPLATES below by uncommenting this block:
#
# TEST_TEMPLATES = {
#   'mistral_instruct_v0_1': ChatTemplate.from_huggingface("mistralai/Mixtral-8x7B-Instruct-v0.1"),
#   'functionary_v2_2': ChatTemplate.from_huggingface("meetkai/functionary-small-v2.2"),
#   'hermes_2_pro_mistral': ChatTemplate.from_huggingface("NousResearch/Hermes-2-Pro-Mistral-7B"),
#   'llama2': ChatTemplate.from_huggingface("meta-llama/Llama-2-7b-chat-hf"),
# }
# print(json.dumps({k: v.model_dump() for k, v in TEST_TEMPLATES.items()}, indent=2))
# exit(0)

TEST_TEMPLATES = {
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
MODELS_WITH_PARALLEL_CALLS = set(["functionary_v2_2"])
TEST_TEMPLATES = {k: ChatTemplate(**v) for k, v in TEST_TEMPLATES.items()}

if __name__ == "__main__":

    failures = []

    print(f'\nMessages:\n\n```js\n{json.dumps([m.model_dump() for m in TEST_MESSAGES_THOUGHT], indent=2)}\n```\n')

    def check(b: bool, msg: str):
        if not b:
            sys.stderr.write(f'FAILURE: {msg}\n\n')
            failures.append(msg)

    functionary_v2_2 = TEST_TEMPLATES["functionary_v2_2"]
    check(functionary_v2_2.inferred_tool_style == ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2, "functionary_v2_2 should be inferred as TYPESCRIPT_FUNCTIONARY_V2")

    for model_name, chat_template in TEST_TEMPLATES.items():
        check(chat_template.potentially_supports_parallel_calls == (model_name in MODELS_WITH_PARALLEL_CALLS),
              f"{model_name} should {'not ' if model_name not in MODELS_WITH_PARALLEL_CALLS else ''} be detected as potentially supporting parallel calls")

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

        print(f"\n# {model_name}\n")

        if chat_template.potentially_supports_parallel_calls:
            print("\n**Might Support Parallel Tool Calls**\n")

        print(f'\nTemplate:\n\n```js\n{chat_template.template}\n```\n')

        for style in ToolsPromptStyle:
            if (style == ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2) != (model_name.startswith("functionary")):
                continue

            if style == ToolsPromptStyle.TOOLS_MIXTRAL and model_name != "mistral_instruct_v0_1":
                continue

            if model_name == "mistral_instruct_v0_1" and style not in (ToolsPromptStyle.TOOLS_THOUGHTFUL_STEPS, ToolsPromptStyle.TOOLS_MIXTRAL):
                continue

            print(f'\n## {model_name} / {style.name}\n')


            for tool_situation, args in argss.items():
                ch = get_chat_handler(args, parallel_calls=True, tool_style=style)

                print(f'\n### {model_name} / {style.name} / {tool_situation}\n')

                print(f'\nPrompt:\n\n```js\n{ch.render_prompt(TEST_MESSAGES_THOUGHT)}\n```\n')

                print(f'\nOutput format prompt:\n\n```json\n{ch.output_format_prompt.content}\n```\n')

                print(f'\nGrammar:\n\n```js\n{ch.grammar}\n```\n')


                # if model_name == 'hermes_2_pro_mistral':
                #     print("Skipping hermes_2_pro_mistral")
                #     continue
                def check_finds(msgs, strings_to_find):
                    prompt = ch.render_prompt(msgs)
                    for s in strings_to_find:
                        check(str(s) in prompt, f"Missing {s} in prompt for {model_name}:\n{prompt}")

                check_finds([PROMPT_MESSAGE], (QUESTION,))
                check_finds([ASSIST_MESSAGE], (ANSWER,))
                check_finds([TOOL_CALL_MESSAGE], (TEST_ARG_A, TEST_ARG_B, TOOL_NAME))
                check_finds([THOUGHTFUL_TOOL_CALL_MESSAGE], (TEST_THOUGHT, TEST_ARG_A, TEST_ARG_B, TOOL_NAME,))
                check_finds([TOOL_MESSAGE], (TEST_SUM,))
                if chat_template.potentially_supports_parallel_calls:
                    check_finds([TOOL_MESSAGE], (TOOL_NAME,))



    if failures:
        for f in failures:
            print(f'{f}\n\n')

        assert not failures
