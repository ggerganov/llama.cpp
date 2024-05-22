from abc import ABC, abstractmethod
from enum import Enum
import jinja2
import json
from pathlib import Path
import random
import re
import sys
from typing import Annotated, Any, Optional
from pydantic import BaseModel, Field, Json

from examples.json_schema_to_grammar import SchemaConverter
from examples.openai.api import Tool, Message, FunctionCall, ToolCall
from examples.openai.gguf_kvs import GGUFKeyValues, Keys  # type: ignore
from examples.openai.ts_converter import SchemaToTypeScriptConverter

# _THOUGHT_KEY = "thought"
_THOUGHT_KEY = "thought_about_next_step_only"

# While the API will be usable with a generic tools usage like OpenAI,
# (see https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models),
# each model may need specific prompting (and/or constrained output,
# especially for models not fine-tuned for tool usage / function calling).
class ToolsPromptStyle(str, Enum):
    # Short prompt w/ <tools>schemas</tools>, <tool_call>...</tool_call> output
    TOOLS_SHORT = "short"

    # Longer prompt w/ <tools>schemas</tools>, <tool_call>...</tool_call> output
    TOOLS_LONG = "long"

    # Bespoke constrained output format that favours thought and reasoning
    # while allowing unambiguous parsing of parallel tool calling.
    TOOLS_THOUGHTFUL_STEPS = "thoughtful_steps"

    # Large prompt for https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B
    # <tool_call>...</tool_call> output
    # Requires:
    # - git clone https://github.com/NousResearch/Hermes-Function-Calling examples/openai/hermes_function_calling
    # - Set large context length as their prompts are super long
    TOOLS_HERMES_2_PRO = "tools_hermes_2_pro"

    # Seems to want to escape underscores in tool names and in the <tool\_call>...</tool\_call> tags
    TOOLS_MIXTRAL = "mixtral"

    # Short prompt w/ TypeScript definitions for https://github.com/MeetKai/functionary
    # https://github.com/MeetKai/functionary/blob/main/functionary/prompt_template/prompt_template_v2.py
    # Note: see this prior attempt to support Functionary: https://github.com/ggerganov/llama.cpp/pull/5695
    TYPESCRIPT_FUNCTIONARY_V2 = "functionary_v2"

def raise_exception(msg: str):
    raise Exception(msg)

class ChatTemplate(BaseModel):
    template: str
    eos_token: str
    bos_token: str

    inferred_tool_style: Annotated[Optional['ToolsPromptStyle'], Field(exclude=True)] = None
    expects_strict_user_assistant_alternance: Annotated[Optional[bool], Field(exclude=True)] = None
    formats_tool_call: Annotated[Optional[bool], Field(exclude=True)] = None
    formats_tool_call_content: Annotated[Optional[bool], Field(exclude=True)] = None
    formats_tool_result: Optional[bool] = None
    formats_tool_name: Optional[bool] = None

    @property
    def potentially_supports_parallel_calls(self) -> bool:
        return bool(self.formats_tool_result and self.formats_tool_name)

    def __init__(self, template: str, eos_token: str, bos_token: str):
        super().__init__(template=template, eos_token=eos_token, bos_token=bos_token)
        env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
        self._template = env.from_string(template)
        # print(template)

        # self.expects_strict_user_assistant_alternance = "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception" in template

        self.probe_template_capabilities()
        self.extract_prefix_suffix_from_template()

        if "<|recipient|>' + tool_call['function']['name']" in template:
            self.inferred_tool_style = ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2
        else:
            self.inferred_tool_style = ToolsPromptStyle.TOOLS_THOUGHTFUL_STEPS
            # self.inferred_tool_style = ToolsPromptStyle.TOOLS_LONG
            # self.inferred_tool_style = ToolsPromptStyle.TOOLS_HERMES_2_PRO
            # self.inferred_tool_style = ToolsPromptStyle.TOOLS_MIXTRAL

    def probe_template_capabilities(self):

        def succeeds(messages: list[Message], strings_to_find = ()):
            try:
                result = self.raw_render(messages, add_generation_prompt=True)
                # print(result)
                for s in strings_to_find:
                    if s not in result:
                        return False
                return True
            except Exception as e:
                # print(e)
                return False

        # if self.inferred_tool_style == ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2:
        user_msg = Message(role="user", content="Hey")
        assistant_msg = Message(role="assistant", content="I, Robot")

        self.expects_strict_user_assistant_alternance = not succeeds([assistant_msg, user_msg]) and succeeds([user_msg, assistant_msg])

        thought = "Precious thought"
        fn_name = "callMeMaybe"
        toolcall = ToolCall(id="call_531873", type="function", function=FunctionCall(name=fn_name, arguments=json.dumps({"lol": 123})))
        toolcall_msg = Message(role="assistant", content=None, tool_calls=[toolcall])
        tool_result = "Tool result"
        tool_name = "additioner"
        tool_msg = Message(role="tool", name=tool_name, content=tool_result)
        stringified_toolcall_msg = Message(role="assistant", content=None, tool_calls=[ToolCall(function=FunctionCall(name=fn_name, arguments=json.dumps({"lol": 123})))])
        toolcall_content_msg = Message(role="assistant", content=thought, tool_calls=toolcall_msg.tool_calls)

        self.formats_tool_call = succeeds([user_msg, toolcall_msg], (fn_name,))
        if self.formats_tool_call:
            self.formats_tool_call_content = succeeds([user_msg, toolcall_content_msg], (thought,))

        self.formats_tool_result = succeeds([user_msg, assistant_msg, tool_msg], (tool_result,))
        self.formats_tool_name = succeeds([user_msg, assistant_msg, tool_msg], (tool_name,))
        # assert self.formats_tools or self.expects_strict_user_assistant_alternance

    def extract_prefix_suffix_from_template(self):

        delimiter = '<%$[SAMPLE]$%>'
        user_msg = Message(role="user", content="Hey")
        empty_prompt = self.raw_render([user_msg], add_generation_prompt=True).strip()
        planted_prompt = self.raw_render([user_msg, Message(role="assistant", content=delimiter)], add_generation_prompt=False).strip()
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
        from transformers import LlamaTokenizer  # type: ignore
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        return ChatTemplate(
            template = tokenizer.chat_template or tokenizer.default_chat_template,
            bos_token = tokenizer.bos_token,
            eos_token = tokenizer.eos_token)

    def raw_render(self, messages: list[Message], add_generation_prompt: bool, omit_bos: bool = False):
        result = self._template.render(
            messages=[messages.model_dump() for messages in messages],
            eos_token=self.eos_token,
            bos_token='' if omit_bos else self.bos_token,
            raise_exception=raise_exception,
            add_generation_prompt=add_generation_prompt,
        )
        return result

class ChatHandlerArgs(BaseModel):
    chat_template: ChatTemplate
    response_schema: Optional[dict[str,Any]] = None
    tools: Optional[list[Tool]] = None

class ChatHandler(ABC):
    def __init__(self, args: ChatHandlerArgs, style: Optional[ToolsPromptStyle]):
        self.args = args
        self.style = style
        self.output_format_prompt: Optional[Message] = None
        self.grammar: Optional[str] = None

    @abstractmethod
    def parse(self, s: str) -> Optional[Message]:
        raise NotImplementedError()


    def add_system_prompt(self, messages: list[Message], system_prompt: Message) -> list[Message]:
        assert system_prompt.role == "system"
        # TODO: add to last system message, or create a new one just before the last user message
        system_message = next(((i, m) for i, m in enumerate(messages) if m.role == "system"), None)
        if system_message:
            (i, m) = system_message
            return messages[:i] + [Message(role="system", content=(system_prompt.content + '\n' if system_prompt.content else '') + (m.content or ''))] + messages[i+1:]
        else:
            return [system_prompt] + messages

    def render_prompt(self, messages: list[Message]) -> str:

        if self.output_format_prompt:
            messages = self.add_system_prompt(messages, self.output_format_prompt)

        def normalize(m: Message):
            if self.style == ToolsPromptStyle.TOOLS_THOUGHTFUL_STEPS and m.role == "assistant":
                if m.tool_calls:
                    m = Message(
                        role=m.role,
                        content=json.dumps({
                            _THOUGHT_KEY: m.content or '',
                            "next_step": {
                                "tool_calls": [tc.model_dump() for tc in m.tool_calls]
                            }
                        }, indent=2)
                    )
                else:
                    m = Message(
                        role=m.role,
                        content=json.dumps({
                            _THOUGHT_KEY: '',
                            "next_step": {
                                "result": m.content
                            }
                        }, indent=2)
                    )
                # Fall through to benefit from role normalization

            if m.tool_calls:
                if not self.args.chat_template.formats_tool_call or not self.args.chat_template.formats_tool_call_content:
                    return Message(
                        role=m.role,
                        content='\n'.join([
                            *([m.content] if m.content else ()),
                            *([
                                f'<tool_call>{json.dumps(tc.model_dump())}</tool_call>'
                                for tc in m.tool_calls
                            ])
                        ])
                    )
                else:
                    return m
            elif self.args.chat_template.expects_strict_user_assistant_alternance and m.role not in ('user', 'assistant'):
                if m.role == "system":
                    return Message(role="user", content=f'[SYS]{m.content}[/SYS]')
                elif m.role == "tool":
                    return Message(role="user", content=f'[TOOL(name={m.name}, id={m.tool_call_id})]{m.content}[/TOOL]')
                else:
                    sys.stderr.write(f'Unexpected message role: {message.role}\n')
                    return Message(role="user", content=f'[{m.role.upper()}]{m.content}[/{m.role.upper()}]')
            else:
                return m

        messages=[normalize(m) for m in messages]

        if self.args.chat_template.expects_strict_user_assistant_alternance:
            new_messages=[]
            current_role = 'user'
            current_content: list[str] = []

            def flush():
                nonlocal current_content
                nonlocal current_role

                if self.args.chat_template.expects_strict_user_assistant_alternance or current_content:
                    new_messages.append(Message(
                        role=current_role,
                        content='\n'.join(current_content)
                    ))
                    current_content = []

            for i, message in enumerate(messages):
                assert message.role in ('user', 'assistant')

                if message.role == current_role:
                    if message.content:
                        current_content.append(message.content)
                else:
                    flush()
                    current_role = 'assistant' if current_role == 'user' else 'user'
                    if message.content:
                        current_content.append(message.content)
            if current_content:
                flush()
            messages = new_messages

        # JSON!
        # messages = [m.model_dump() for m in messages]

        return self.args.chat_template.raw_render(
            messages=messages,
            add_generation_prompt=True,
        )

class NoToolsChatHandler(ChatHandler):
    def __init__(self, args: ChatHandlerArgs):
        super().__init__(args, None)
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
    def __init__(self, args: ChatHandlerArgs, style: Optional[ToolsPromptStyle], escapes_underscores: bool, parallel_calls: bool):
        super().__init__(args, style)

        converter = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
        tool_rules = []
        for tool in self.args.tools or []:

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
            content: list[str] = []
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
                            function=FunctionCall(
                                name=fc["name"],
                                arguments=json.dumps(fc["arguments"]))))

            content_str = '\n'.join(content).strip()
            return Message(role="assistant", content=content_str if content_str else None, tool_calls=tool_calls)


class TemplatedToolsChatHandler(ToolCallTagsChatHandler):
    def __init__(self, args: ChatHandlerArgs, template: str, parallel_calls: bool, escapes_underscores: bool = False, style: Optional[ToolsPromptStyle] = None):
        super().__init__(args, style=style, escapes_underscores=escapes_underscores, parallel_calls=parallel_calls)
        assert '{tools}' in template, 'Template must contain "{tools}"'

        self.output_format_prompt = Message(
            role="system",
            content=template.replace(
                '{tools}',
                '\n'.join(json.dumps(tool.model_dump(), indent=2) for tool in (self.args.tools or [])),
            )
        )

class Hermes2ProToolsChatHandler(ToolCallTagsChatHandler):
    def __init__(self, args: ChatHandlerArgs, parallel_calls: bool):
        super().__init__(args, style=ToolsPromptStyle.TOOLS_HERMES_2_PRO, escapes_underscores=False, parallel_calls=parallel_calls)

        # Hackily import https://github.com/NousResearch/Hermes-Function-Calling
        path = str(Path(__file__).parent / "hermes_function_calling")
        if path not in sys.path: sys.path.insert(0, path)
        try:
            from examples.openai.hermes_function_calling.prompter import PromptManager  # type: ignore
        except ImportError:
            raise ImportError(f"Please `git clone https://github.com/NousResearch/Hermes-Function-Calling {path}`")

        prompt = PromptManager().generate_prompt(user_prompt=[], tools=[tool.model_dump_json() for tool in args.tools or []])
        assert len(prompt) == 1 and prompt[0]["role"] == "system"
        self.output_format_prompt = Message(**prompt[0])

class FunctionaryToolsChatHandler(ChatHandler):
    def __init__(self, args: ChatHandlerArgs, parallel_calls: bool):
        super().__init__(args, ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2)

        self.output_format_prompt = Message(
            role="system",
            content= '// Supported function definitions that should be called when necessary.\n' +
                _tools_typescript_signatures(args.tools or [])
        )

        converter = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
        tool_rules = [
            converter._add_rule(
                tool.function.name + '-call',
                converter._format_literal(tool.function.name) + ' ' + converter._format_literal('\n<|content|>\n') + ' ' +
                converter.visit(tool.function.parameters, tool.function.name + '-args') + ' ' +
                converter._format_literal('\n'))
            for i, tool in enumerate(self.args.tools or [])
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
            _THOUGHT_KEY: {
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
        "required": ["original_goal", _THOUGHT_KEY, "next_step"]
        # "required": ["next_step"]
    }

class ThoughtfulStepsToolsChatHandler(ChatHandler):
    def __init__(self, args: ChatHandlerArgs, parallel_calls: bool):
        super().__init__(args, ToolsPromptStyle.TOOLS_THOUGHTFUL_STEPS)

        # args.response_schema = args.response_schema or {}
        converter = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)

        response_schema = converter.resolve_refs(args.response_schema or {"type": "string"}, 'response')
        tool_parameter_schemas = {
            tool.function.name: converter.resolve_refs(tool.function.parameters, tool.function.name)
            for tool in self.args.tools or []
        }
        # sys.stderr.write(f"# RESOLVED RESPONSE SCHEMA: {json.dumps(response_schema, indent=2)}\n")
        # sys.stderr.write(f"# RESOLVED TOOL PARAMETER SCHEMA: {json.dumps(tool_parameter_schemas, indent=2)}\n")
        converter.visit(
            _make_bespoke_schema(
                response_schema,
                {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {"const": tool_name},
                                "arguments": tool_parameters,
                            },
                            "required": ["name", "arguments"]
                        }
                        for tool_name, tool_parameters in tool_parameter_schemas.items()
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
                _tools_schema_signatures(self.args.tools or [], indent=2),
                # _tools_typescript_signatures(self.args.tools),
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
                content=data.get(_THOUGHT_KEY),
                tool_calls=[
                    ToolCall(
                        id=gen_callid(),
                        function=FunctionCall(
                            name=tc["name"],
                            arguments=json.dumps(tc["arguments"])))
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

def get_chat_handler(args: ChatHandlerArgs, parallel_calls: bool, tool_style: Optional[ToolsPromptStyle] = None, verbose=False) -> ChatHandler:
    tool_style = tool_style if tool_style is not None else args.chat_template.inferred_tool_style

    if parallel_calls and not args.chat_template.potentially_supports_parallel_calls:
        sys.stderr.write(f"# WARNING: Disabled parallel_calls as model does not seem to support it (will fall back to sequential calls)\n")
        parallel_calls = False

    if verbose:
        sys.stderr.write(f"# Using tool style: {tool_style}\n")

    if not args.tools:
        return NoToolsChatHandler(args)

    elif tool_style == ToolsPromptStyle.TOOLS_THOUGHTFUL_STEPS:
        return ThoughtfulStepsToolsChatHandler(args, parallel_calls=parallel_calls)

    elif tool_style == ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2:
        return FunctionaryToolsChatHandler(args, parallel_calls=parallel_calls)

    elif tool_style == ToolsPromptStyle.TOOLS_SHORT:
        return TemplatedToolsChatHandler(args, _SHORT_TEMPLATE, parallel_calls=parallel_calls)

    elif tool_style == ToolsPromptStyle.TOOLS_LONG:
        return TemplatedToolsChatHandler(args, _LONG_TEMPLATE, parallel_calls=parallel_calls)

    elif tool_style == ToolsPromptStyle.TOOLS_MIXTRAL:
        return TemplatedToolsChatHandler(args, _LONG_TEMPLATE, parallel_calls=parallel_calls, escapes_underscores=True)

    elif tool_style == ToolsPromptStyle.TOOLS_HERMES_2_PRO:
        return Hermes2ProToolsChatHandler(args, parallel_calls=parallel_calls)
    else:
        raise ValueError(f"Unsupported tool call style: {tool_style}")

# os.environ.get('NO_TS')
def _please_respond_with_schema(schema: dict[str, Any]) -> str:
    sig = json.dumps(schema, indent=2)
    # _ts_converter = SchemaToTypeScriptConverter()
    # # _ts_converter.resolve_refs(schema, 'schema')
    # sig = _ts_converter.visit(schema)
    return f'Please respond in JSON format with the following schema: {sig}'

def _tools_typescript_signatures(tools: list[Tool]) -> str:
    _ts_converter = SchemaToTypeScriptConverter()
    # for tool in tools:
    #     _ts_converter.resolve_refs(tool.function.parameters, tool.function.name)

    return 'namespace functions {\n' + '\n'.join(
        '// ' + (tool.function.description or '').replace('\n', '\n// ') + '\n' + ''
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
