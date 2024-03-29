from abc import ABC, abstractmethod
from enum import Enum
import jinja2
import json
from pathlib import Path
import random
import re
import sys
from typing import Optional
from pydantic import BaseModel

from examples.json_schema_to_grammar import SchemaConverter
from examples.openai.api import Tool, Message, FunctionCall, ToolCall
from examples.openai.gguf_kvs import GGUFKeyValues, Keys
from examples.openai.ts_converter import SchemaToTypeScriptConverter

# While the API will be usable with a generic tools usage like OpenAI,
# (see https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models),
# each model may need specific prompting (and/or constrained output,
# especially for models not fine-tuned for tool usage / function calling).
class ToolsPromptStyle(Enum):
    # Short prompt w/ <tools>schemas</tools>, <tool_call>...</tool_call> output
    TOOLS_SHORT = 1

    # Longer prompt w/ <tools>schemas</tools>, <tool_call>...</tool_call> output
    TOOLS_LONG = 2

    # Bespoke constrained output format that favours thought and reasoning
    # while allowing unambiguous parsing of parallel tool calling.
    TOOLS_BESPOKE = 3

    # Large prompt for https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B
    # <tool_call>...</tool_call> output
    # Requires:
    # - git clone https://github.com/NousResearch/Hermes-Function-Calling examples/openai/hermes_function_calling
    # - Set large context length as their prompts are super long
    TOOLS_HERMES_2_PRO = 4

    # Seems to want to escape underscores in tool names and in the <tool\_call>...</tool\_call> tags
    TOOLS_MISTRAL = 5

    # Short prompt w/ TypeScript definitions for https://github.com/MeetKai/functionary
    # https://github.com/MeetKai/functionary/blob/main/functionary/prompt_template/prompt_template_v2.py
    # Note: see this prior attempt to support Functionary: https://github.com/ggerganov/llama.cpp/pull/5695
    TYPESCRIPT_FUNCTIONARY_V2 = 6

def raise_exception(msg: str):
    raise Exception(msg)

class ChatTemplate(BaseModel):
    template: str

    @property
    def tool_style(self) -> 'ToolsPromptStyle':
        return self._tool_style

    def __init__(self, template: str, eos_token: str, bos_token: str):
        super().__init__(template=template
                         )
        env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
        self._template = env.from_string(template)
        self._eos_token = eos_token
        self._bos_token = bos_token

        self._strict_user_assistant_alternation = "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception" in template

        if "<|recipient|>' + tool_call['function']['name']" in template:
            self._tool_style = ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2
        else:
            self._tool_style = ToolsPromptStyle.TOOLS_BESPOKE
            # self._tool_style = ToolsPromptStyle.TOOLS_LONG
            # self._tool_style = ToolsPromptStyle.TOOLS_HERMES_2_PRO
            # self._tool_style = ToolsPromptStyle.TOOLS_MISTRAL

        # TODO: Test whether the template supports formatting tool_calls

        delimiter = '<%$[SAMPLE]$%>'
        user_msg = Message(role="user", content="Hey")
        empty_prompt = self.render([user_msg], add_generation_prompt=True).strip()
        planted_prompt = self.render([user_msg, Message(role="assistant", content=delimiter)], add_generation_prompt=False).strip()
        assert planted_prompt.startswith(empty_prompt), f"Planted prompt does not start with empty prompt: {planted_prompt} vs {empty_prompt}"
        [prefix, suffix] = planted_prompt[len(empty_prompt):].split(delimiter)

        # sys.stderr.write(f"\n# prefix={prefix}\n# suffix={suffix}\n\n")

        self._prefix = prefix
        self._suffix = suffix

    def strip_suffix(self, s: str) -> str:
        if s.endswith(self._suffix):
            return s[:-len(self._suffix)]
        else:
            sys.stderr.write(f"Expected suffix ({self._suffix}) not found: {s}\n")
            return s

    def __str__(self):
        return f"ChatTemplate(template={self.template}, eos_token={self._eos_token}, bos_token={self._bos_token})"

    def add_system_prompt(self, messages: list[Message], system_prompt: Message) -> list[Message]:
        assert system_prompt.role == "system"
        # TODO: add to last system message, or create a new one just before the last user message
        system_message = next(((i, m) for i, m in enumerate(messages) if m.role == "system"), None)
        if system_message is not None:
            (i, m) = system_message
            return messages[:i] + [Message(role="system", content=system_prompt.content + '\n' + m.content)] + messages[i+1:]
        else:
            return [system_prompt] + messages

    @staticmethod
    def from_gguf(metadata: GGUFKeyValues):
        if Keys.Tokenizer.CHAT_TEMPLATE not in metadata:
            raise NotImplementedError(f'Only supporting models with {Keys.Tokenizer.CHAT_TEMPLATE} entry in their GGUF key-values (TODO: add default template, maybe pick llama2\'s?)')

        tokens = metadata[Keys.Tokenizer.LIST]
        return ChatTemplate(
            template = metadata[Keys.Tokenizer.CHAT_TEMPLATE],
            bos_token = tokens[metadata[Keys.Tokenizer.BOS_ID]],
            eos_token = tokens[metadata[Keys.Tokenizer.EOS_ID]])

    @staticmethod
    def from_huggingface(model_id: str):
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        return ChatTemplate(
            template = tokenizer.chat_template or tokenizer.default_chat_template,
            bos_token = tokenizer.bos_token,
            eos_token = tokenizer.eos_token)

    def render(self, messages: list[Message], add_generation_prompt: bool, omit_bos: bool = False):
        if self._strict_user_assistant_alternation and any(m.role not in ('user', 'assistant') for m in messages):
            new_messages=[]
            i = 0
            n = len(messages)
            current_role = 'user'
            current_content = []

            def flush():
                nonlocal current_content
                nonlocal current_role
                new_messages.append(Message(
                    role=current_role,
                    content='\n'.join(current_content)
                ))
                current_content = []

            for i, message in enumerate(messages):
                if message.role == current_role:
                    current_content.append(message.content)
                elif message.role in ('user', 'assistant'):
                    flush()
                    current_role = 'assistant' if current_role == 'user' else 'user'
                    current_content.append(message.content)
                else:
                    if current_role == 'assistant':
                        flush()
                        current_role = 'user'
                    if message.role == 'system':
                        current_content.append(f'[SYS]{messages[i].content}[/SYS]')
                    elif message.role == 'tool':
                        current_content.append(f'[TOOL RESULT(name={messages[i].name}, id={messages[i].tool_call_id}]{messages[i].content}[/TOOL RESULT]')
                    else:
                        sys.stderr.write(f'Unexpected message role: {message.role}\n')
                        current_content.append(f'[ROLE={messages[i].role}]{messages[i].content}[/ROLE]')

                current_content.extend(
                    f'<tool_call>{json.dumps(tc.model_dump())}</tool_call>'
                    for tc in (message.tool_calls or [])
                )
            if current_content:
                flush()

            messages = new_messages

        result = self._template.render(
            messages=messages,
            eos_token=self._eos_token,
            bos_token='' if omit_bos else self._bos_token,
            raise_exception=raise_exception,
            add_generation_prompt=add_generation_prompt,
        )
        return result

class ChatHandlerArgs(BaseModel):
    chat_template: ChatTemplate
    response_schema: Optional[dict] = None
    tools: Optional[list[Tool]] = None

class ChatHandler(ABC):
    def __init__(self, args: ChatHandlerArgs):
        self.args = args
        self.output_format_prompt: Optional[Message] = None
        self.grammar: Optional[str] = None

    @abstractmethod
    def parse(self, s: str) -> Optional[Message]:
        raise NotImplementedError()

class NoToolsChatHandler(ChatHandler):
    def __init__(self, args: ChatHandlerArgs):
        super().__init__(args)
        assert not args.tools

        if args.response_schema:
            self.output_format_prompt = Message(
                role="system",
                content=_please_respond_with_schema(args.response_schema)
            )
            converter = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
            schema = converter.resolve_refs(args.response_schema, 'response')
            converter.visit(schema, '')
            self.grammar = converter.format_grammar()
        else:
            self.output_format_prompt = None
            self.grammar = None

    def parse(self, s: str) -> Optional[Message]:
        return Message(role="assistant", content=s)

class ToolCallTagsChatHandler(ChatHandler):
    def __init__(self, args: ChatHandlerArgs, escapes_underscores: bool, parallel_calls: bool):
        super().__init__(args)

        converter = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
        tool_rules = []
        for tool in self.args.tools:

            parameters_schema = tool.function.parameters
            parameters_schema = converter.resolve_refs(parameters_schema, tool.function.name)

            tool_rules.append(converter.visit(
                dict(
                    type="object",
                    properties=dict(
                        name=dict(type="string", pattern='^' + tool.function.name.replace('_', f'\\?_') + '$') if escapes_underscores \
                            else dict(const=tool.function.name),
                        arguments=parameters_schema,
                    ),
                    required=['name', 'arguments']
                ),
                f'{tool.function.name}-tool-call'
            ))

        def format_literal(s: str) -> str:
            if escapes_underscores:
                return ' "\\\\"? "_" '.join((converter._format_literal(part) for part in s.split('_')))
            else:
                return converter._format_literal(s)

        tool_call_rule = converter._add_rule(
            'tool_call',
            format_literal("<tool_call>") + " space (" +
            ' | '.join(tool_rules) +
            ")  space " + format_literal("</tool_call>"))# + ' space')

        # Ideally we'd want a negative lookahead of /<tool\\?_call>/, but it's just too hard to express in GBNF for now.
        # So we just over-constrain the content rule to not contain literals dangerously getting close to <tool_call>
        content_rule = converter._add_rule('content', '[^<] | "<" [^t<] | "<t" [^o<]')
        # content_rule = converter._add_rule('content', converter.not_literal('<tool_call>'))
        converter._add_rule(
            'root',
            # tool_call_rule)
            f'{content_rule}* ({tool_call_rule}+ {content_rule}*)?' if parallel_calls \
                else f'{content_rule}* {tool_call_rule}?')
        self.grammar = converter.format_grammar()

    def parse(self, s: str) -> Optional[Message]:
        s = self.args.chat_template.strip_suffix(s)

        if r'<tool\_call>' in s:
            # Some weird escaping of underscores is happening w/ Mixtral 8x7B Instruct
            s = s.replace(r'\_', '_')

        parts = _tool_call_re.split(s)
        if len(parts) == 1:
            return Message(role="assistant", content=s)
        else:
            content = []
            tool_calls = []
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    content.append(part)
                else:
                    try:
                        fc = json.loads(part)
                    except json.JSONDecodeError:
                        raise ValueError(f'Failed to parse tool call as JSON: {part}\nFull string: {s}')
                    tool_calls.append(
                        ToolCall(
                            id=gen_callid(),
                            function=FunctionCall(**fc)))

            content = '\n'.join(content).strip()
            return Message(role="assistant", content=content if content else None, tool_calls=tool_calls)


class TemplatedToolsChatHandler(ToolCallTagsChatHandler):
    def __init__(self, args: ChatHandlerArgs, template: str, parallel_calls: bool, escapes_underscores: bool = False):
        super().__init__(args, escapes_underscores=escapes_underscores, parallel_calls=parallel_calls)
        assert '{tools}' in template, 'Template must contain "{tools}"'

        self.output_format_prompt = Message(
            role="system",
            content=template.replace(
                '{tools}',
                '\n'.join(json.dumps(tool.model_dump(), indent=2) for tool in self.args.tools),
            )
        )

class Hermes2ProToolsChatHandler(ToolCallTagsChatHandler):
    def __init__(self, args: ChatHandlerArgs, parallel_calls: bool):
        super().__init__(args, escapes_underscores=False, parallel_calls=parallel_calls)

        # Hackily import https://github.com/NousResearch/Hermes-Function-Calling
        path = str(Path(__file__).parent / "hermes_function_calling")
        if path not in sys.path: sys.path.insert(0, path)
        try:
            from examples.openai.hermes_function_calling.prompter import PromptManager
        except ImportError:
            raise ImportError(f"Please `git clone https://github.com/NousResearch/Hermes-Function-Calling {path}`")

        prompt = PromptManager().generate_prompt(user_prompt=[], tools=[json.dumps(tool) for tool in args.tools])
        assert len(prompt) == 1 and prompt[0]["role"] == "system"
        self.output_format_prompt = Message(**prompt[0])

class FunctionaryToolsChatHandler(ChatHandler):
    def __init__(self, args: ChatHandlerArgs, parallel_calls: bool):
        super().__init__(args)

        # Only allowing a single tool call at a time for now.
        # Note that if there were more, they'd be separated by a '<|from|>assistant' literal

        self.output_format_prompt = Message(
            role="system",
            content= '// Supported function definitions that should be called when necessary.\n' +
                _tools_typescript_signatures(args.tools)
        )

        converter = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
        tool_rules = [
            converter._add_rule(
                tool.function.name + '-call',
                converter._format_literal(tool.function.name) + ' ' + converter._format_literal('\n<|content|>\n') + ' ' +
                converter.visit(tool.function.parameters, tool.function.name + '-args') + ' ' +
                converter._format_literal('\n'))
            for i, tool in enumerate(self.args.tools)
        ]

        not_from_rule = converter._add_rule('not_from', converter.not_literal("<|from|>"))
        content_without_start_rule = converter._add_rule(
            'content_without_start',
            converter._format_literal("all\n<|content|>") + ' ' + not_from_rule + '*')
        start_rule = converter._add_rule('start', converter._format_literal('<|from|>assistant\n<|recipient|>'))
        content_rule = converter._add_rule('content', start_rule + ' ' + content_without_start_rule)
        tool_call_without_start_rule = converter._add_rule(
            'tool_call_without_start',
            ' | '.join(tool_rules))
        tool_call_rule = converter._add_rule('tool_call', f'{start_rule} {tool_call_without_start_rule}')
        converter._add_rule(
            'root',
            f'{content_without_start_rule}   {content_rule}*   ({tool_call_rule}+ {content_rule}*)? | '
            f'{tool_call_without_start_rule} {tool_call_rule}* {content_rule}*' if parallel_calls \
                else f'{content_without_start_rule}  {tool_call_rule}? | {tool_call_without_start_rule}')

        self.grammar = converter.format_grammar()

    def parse(self, s: str) -> Optional[Message]:
        s = self.args.chat_template.strip_suffix(s)

        parts = _recipient_content_re.split(s)
        if len(parts) == 1:
            return Message(role="assistant", content=s)
        else:
            text_content = []
            tool_calls: list[ToolCall] = []
            for i in range((len(parts) - 1) // 3):
                assert parts[i * 3].strip() == '', f'Unexpected content before tool call: {parts[i * 3]}'
                recipient = parts[i * 3 + 1].strip()
                content = parts[i * 3 + 2]
                if recipient == 'all':
                    text_content.append(content)
                else:
                    try:
                        arguments = json.loads(content)
                    except json.JSONDecodeError:
                        raise ValueError(f'Failed to parse tool call content as JSON: {content}')
                    tool_calls.append(
                        ToolCall(
                            id=gen_callid(),
                            function=FunctionCall(name=recipient, arguments=arguments)))


            assert parts[-1].strip() in ('', '<|stop|>'), f'Unexpected content after tool calls: {parts[-1]}\nFull string: {s}'

            content = '\n'.join(text_content).strip()
            return Message(role="assistant", content=content if content else None, tool_calls=tool_calls if tool_calls else None)

def _make_bespoke_schema(response_schema, tool_call_schema, parallel_calls):
    return {
        "type": "object",
        "properties": {
            # "original_goal": {"title": "Original Goal", "type": "string"},
            "thought_about_next_step_only": {
                "title": "Thought about next step",
                # "title": "Thought about how the next step brings us closer to achieving the original goal",
                "type": "string"
            },
            "next_step": {
                "title": "Next Step: either a result or one or more tool calls to achieve the original goal",
                "oneOf": [
                    {
                        # "title": "Tool Calls",
                        "properties": {
                            # "type": {
                            #     "const": "tool_calls"
                            # },
                            "tool_calls": {
                                "prefixItems": tool_call_schema if parallel_calls \
                                    else [tool_call_schema],
                            }
                        },
                        "required": ["tool_calls"]
                    },
                    {
                        "title": "Result (achieving original goal)",
                        "properties": {
                            "result": response_schema,
                        },
                        "required": ["result"]
                    },
                ]
            },
        },
        "required": ["original_goal", "thought_about_next_step_only", "next_step"]
        # "required": ["next_step"]
    }

class BespokeToolsChatHandler(ChatHandler):
    def __init__(self, args: ChatHandlerArgs, parallel_calls: bool):
        super().__init__(args)

        # args.response_schema = args.response_schema or {}
        converter = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)

        response_schema = args.response_schema or {"type": "string"}
        converter.visit(
            _make_bespoke_schema(
                response_schema,
                {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {"const": tool.function.name},
                                "arguments": tool.function.parameters,
                            },
                            "required": ["name", "arguments"]
                        }
                        for tool in self.args.tools
                    ]
                },
                parallel_calls=parallel_calls,
            ),
            '',
        )
        self.grammar = converter.format_grammar()

        self.output_format_prompt = Message(
            role="system",
            content='\n'.join([
                'You are a function calling AI model.',
                'Here are the tools available:',
                _tools_schema_signatures(self.args.tools, indent=2),
                _please_respond_with_schema(
                    _make_bespoke_schema(
                        response_schema,
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
                            "required": ["name", "arguments"]
                        },
                        parallel_calls=parallel_calls,
                    )
                ),
            ])
        )

    def parse(self, s: str) -> Optional[Message]:
        s = self.args.chat_template.strip_suffix(s)
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            raise ValueError(f'Failed to parse data as JSON: {s}')

        next_step = data['next_step']
        if 'result' in next_step:
            return Message(role="assistant", content=json.dumps(next_step['result']))
        elif 'tool_calls' in next_step:
            return Message(
                role="assistant",
                content=data["thought_about_next_step_only"] if "thought_about_next_step_only" in data else None,
                tool_calls=[
                    ToolCall(id=gen_callid(), function=FunctionCall(**tc))
                    for tc in next_step['tool_calls']
                ]
            )
        else:
            raise ValueError(f'Unexpected data: {data}')

_SHORT_TEMPLATE='\n'.join([
    'Here are the tools available:',
    '<tools>',
    '{tools}',
    '</tools>',
])

_LONG_TEMPLATE='\n'.join([
    # '''You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.''',
    # 'You may call one or more functions to assist with the user query. Don\'t make assumptions about what values to plug into functions. Here are the available tools:',
    'Call one or more functions to assist with the user query, every time this is possible. Don\'t make assumptions about what values to plug into functions. Here are the available tools:',
    '<tools>',
    '{tools}',
    '</tools>',
    '',
    # 'Use the following json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"}',
    # '',
    # 'For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:',
    'To call each function, give its name and arguments within <tool_call></tool_call> XML tags as follows:',
    '<tool_call>',
    '{"name": <function-name>, "arguments": <args-dict>}',
    '</tool_call>',
    # 'This is not hypothetical, you're not asked what you would do. If you need a tool called, just call it with <tool_call>...</tool_call>.''',
])

def get_chat_handler(args: ChatHandlerArgs, parallel_calls: bool) -> ChatHandler:
    if not args.tools:
        return NoToolsChatHandler(args)
    elif args.chat_template.tool_style == ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2:
        return FunctionaryToolsChatHandler(args, parallel_calls=parallel_calls)
    elif args.chat_template.tool_style == ToolsPromptStyle.TOOLS_SHORT:
        return TemplatedToolsChatHandler(args, _SHORT_TEMPLATE, parallel_calls=parallel_calls)
    elif args.chat_template.tool_style == ToolsPromptStyle.TOOLS_LONG:
        return TemplatedToolsChatHandler(args, _LONG_TEMPLATE, parallel_calls=parallel_calls)
    elif args.chat_template.tool_style == ToolsPromptStyle.TOOLS_MISTRAL:
        return TemplatedToolsChatHandler(args, _LONG_TEMPLATE, parallel_calls=parallel_calls, escapes_underscores=True)
    elif args.chat_template.tool_style == ToolsPromptStyle.TOOLS_BESPOKE:
        return BespokeToolsChatHandler(args, parallel_calls=parallel_calls)
    elif args.chat_template.tool_style == ToolsPromptStyle.TOOLS_HERMES_2_PRO:
        return Hermes2ProToolsChatHandler(args)
    else:
        raise ValueError(f"Unsupported tool call style: {args.chat_template.tool_style}")

_ts_converter = SchemaToTypeScriptConverter()

def _please_respond_with_schema(schema: dict) -> str:
    # sig = json.dumps(schema, indent=2)
    sig = _ts_converter.visit(schema)
    return f'Please respond in JSON format with the following schema: {sig}'

def _tools_typescript_signatures(tools: list[Tool]) -> str:
    return 'namespace functions {' + '\n'.join(
        '// ' + tool.function.description.replace('\n', '\n// ') + '\n' + ''
        'type ' + tool.function.name + ' = (_: ' + _ts_converter.visit(tool.function.parameters) + ") => any;\n"
        for tool in tools
    ) + '} // namespace functions'

def _tools_schema_signatures(tools: list[Tool], indent=None) -> str:
    return '\n'.join(
        json.dumps(tool.model_dump(), indent=indent)
        for tool in tools
    )

_tool_call_re = re.compile(
    '<tool_call>(.*?)</tool_call>', re.DOTALL)
_recipient_content_re = re.compile(r'(?:(?:<\|(?:stop|from)\|>)+ *assistant\n<\|recipient\|>|^) *([^ <|>\n]+) *\n<\|content\|>(.*?)(?:$|<\|stop\|>\s*$|(?=(?:<\|(?:stop|from)\|>)+ *assistant\n))', re.DOTALL)

def gen_callid():
    return f'call_{random.randint(0, 1000000)}'
