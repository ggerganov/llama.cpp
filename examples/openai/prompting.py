from enum import Enum
import jinja2
import json
from pathlib import Path
import sys
from typing import Optional, Tuple, Callable
from typeguard import typechecked

from examples.json_schema_to_grammar import SchemaConverter
from examples.openai.api import Tool, Message
from examples.openai.gguf_kvs import GGUFKeyValues, Keys
from examples.openai.ts_converter import SchemaToTypeScriptConverter

@typechecked
def raise_exception(msg: str):
    raise Exception(msg)

class ChatFormat:
    def __init__(self, template: str, eos_token: str, bos_token: str):
        env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
        self.template = env.from_string(template)
        self.eos_token = eos_token
        self.bos_token = bos_token

        self.strict_user_assistant_alternation = "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception" in template

        if "<|recipient|>' + tool_call['function']['name']" in template:
            self.tool_style = ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2
        else:
            self.tool_style = ToolsPromptStyle.TOOLS_LONG

    def __str__(self):
        return f"ChatFormat(template={self.template}, eos_token={self.eos_token}, bos_token={self.bos_token})"

    @staticmethod
    def from_gguf(metadata: GGUFKeyValues):
        return ChatFormat(
            template = metadata[Keys.Tokenizer.CHAT_TEMPLATE],
            bos_token = metadata[Keys.Tokenizer.BOS_ID],
            eos_token = metadata[Keys.Tokenizer.EOS_ID])

    def render(self, messages: list[dict], add_generation_prompt: bool, omit_bos: bool = False):
        return self.template.render(
            messages=messages,
            eos_token=self.eos_token,
            bos_token='' if omit_bos else self.bos_token,
            raise_exception=raise_exception,
            add_generation_prompt=add_generation_prompt,
        )

# While the API will be usable with a generic tools usage like OpenAI,
# (see https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models),
# each model may need specific prompting (and/or constrained output,
# especially for models not fine-tuned for tool usage / function calling).
class ToolsPromptStyle(Enum):
    # Short prompt w/ <tools>schemas</tools>
    TOOLS_SHORT = 1

    # Longer prompt w/ <tools>schemas</tools>
    TOOLS_LONG = 2

    # Large prompt for https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B
    # Requires:
    # - git clone https://github.com/NousResearch/Hermes-Function-Calling examples/openai/hermes_function_calling
    # - Set large context length as their prompts are super long
    TOOLS_HERMES_2_PRO = 3

    # Short prompt w/ TypeScript definitions for https://github.com/MeetKai/functionary
    # https://github.com/MeetKai/functionary/blob/main/functionary/prompt_template/prompt_template_v2.py
    # Note: see this prior attempt to support Functionary: https://github.com/ggerganov/llama.cpp/pull/5695
    TYPESCRIPT_FUNCTIONARY_V2 = 4

@typechecked
def make_tools_prompt(chat_format: ChatFormat, tools: list[Tool], indent=2) -> Message:

    if chat_format.tool_style == ToolsPromptStyle.TOOLS_SHORT:
        return Message(
            role="system",
            content='\n'.join([
                'Here are the tools available:',
                '<tools>',
                *(json.dumps(tool.model_dump(), indent=indent) for tool in tools),
                '</tools>',
            ])
        )
    
    elif chat_format.tool_style == ToolsPromptStyle.TOOLS_LONG:
        return Message(
            role="system",
            content='\n'.join([
                '''You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.''',
                '''You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:''',
                '''<tools>''',
                *(json.dumps(tool.model_dump(), indent=indent) for tool in tools),
                '''</tools>''',
                '',
                '''Use the following json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"}''',
                '',
                '''For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:''',
                '''<tool_call>''',
                '''{"arguments": <args-dict>, "name": <function-name>}''',
                '''</tool_call>''',
            ])
        )
    
    elif chat_format.tool_style == ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2:
        ts_converter = SchemaToTypeScriptConverter()
        
        return Message(
            role="system",
            content='\n'.join([
                '// Supported function definitions that should be called when necessary.'
                'namespace functions {',
                *[
                    '// ' + tool.function.description.replace('\n', '\n// ') + '\n' + ''
                    'type ' + tool.function.name + ' = (_: ' + ts_converter.visit(tool.function.parameters) + ") => any;\n"
                    for tool in tools
                ],
                '} // namespace functions',
            ])
        )
    
    elif chat_format.tool_style == ToolsPromptStyle.TOOLS_HERMES_2_PRO:
        # Hackily import https://github.com/NousResearch/Hermes-Function-Calling
        path = str(Path(__file__).parent / "hermes_function_calling")
        if path not in sys.path: sys.path.insert(0, path)
        try:
            from examples.openai.hermes_function_calling.prompter import PromptManager
        except ImportError:
            raise ImportError(f"Please `git clone https://github.com/NousResearch/Hermes-Function-Calling {path}`")
        
        prompt = PromptManager().generate_prompt(user_prompt=[], tools=[json.dumps(tool) for tool in tools])
        assert len(prompt) == 1 and prompt[0]["role"] == "system"
        return Message(**prompt[0])
    
    else:
        raise ValueError(f"Unsupported tool call style: {chat_format.tool_style}")
    
@typechecked
def _outputs_tool_call_tags(style: ToolsPromptStyle) -> bool:
    return style in (
        ToolsPromptStyle.TOOLS_SHORT,
        ToolsPromptStyle.TOOLS_LONG,
        ToolsPromptStyle.TOOLS_HERMES_2_PRO,
    )

@typechecked
def make_grammar(chat_format: ChatFormat, tools: list[Tool], response_schema: Optional[dict], indent=2) -> Tuple[Optional[str], Callable[[str], Optional[Message]]]:

    converter = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)

    response_rule = converter.visit(response_schema, "response") if response_schema else None
        
    delimiter = '<%$[SAMPLE]$%>'
    empty_prompt = chat_format.render([], add_generation_prompt=True).strip()
    planted_prompt = chat_format.render([{"role": "assistant", "content": delimiter}], add_generation_prompt=False).strip()
    assert planted_prompt.startswith(empty_prompt), f"Planted prompt does not start with empty prompt: {planted_prompt} vs {empty_prompt}"
    [prefix, suffix] = planted_prompt[len(empty_prompt):].split(delimiter)

    if tools:
        if _outputs_tool_call_tags(chat_format.tool_style):
            tool_rules = [
                converter.visit(
                    dict(
                        type="object",
                        properties=dict(
                            name=dict(const=tool.function.name),
                            arguments=tool.function.parameters,
                        ),
                        required=['name', 'arguments']
                    ),
                    f'{tool.function.name}-tool-call'
                )
                for tool in tools
            ]

            # Constrain the output to be a non-tool-call message (constrained to a JSON schema or not)
            # OR a tool-call message respecting the schema of any of the tools
            converter._add_rule(
                "root", 
                converter._format_literal(prefix) + " (" +
                    (response_rule or converter.not_literal("<tool_call>")) + " | " +
                    converter._format_literal("<tool_call>") + " (" +
                    ' | '.join(tool_rules) +
                    ") " + converter._format_literal("</tool_call>") +
                ")") # + converter._format_literal(suffix))
            
            @typechecked
            def parse(s: str) -> Optional[Message]:
                ls = s.lstrip()
                if '<tool_call>'.startswith(ls) or ls.startswith('<tool_call>'):
                    if ls.startswith('<tool_call>') and ls.endswith('</tool_call>' + suffix):
                        tool_call = ls[len('<tool_call>'):-len('</tool_call>' + suffix)]
                        return Message(role="assistant", content=None, tool_calls=[json.loads(tool_call)])
                    return None
                else:
                    return Message(role="assistant", content=s)
            
            return (converter.format_grammar(), parse)

        elif chat_format.tool_style == ToolsPromptStyle.TYPESCRIPT_FUNCTIONARY_V2:
            # Only allowing a single tool call at a time for now.
            # Note that if there were more, they'd be separated by a '<|from|>assistant' literal
            converter._add_rule(
                "root", 
                converter._format_literal(prefix) + " (" +
                    (response_rule or converter.not_literal("<|recipient|>")) + " | " +
                    (' | '.join(
                        converter._format_literal(f"<|recipient|>{tool.function.name}\n<|content|>") + " " +
                        converter.visit(tool.function.parameters, tool.function.name + '-args')
                        for tool in tools
                    )) +
                    ") " +
                ")") # + converter._format_literal(suffix))
    
            @typechecked
            def parse(s: str) -> Optional[Message]:
                raise NotImplementedError(f'TODO: parse tool_style {chat_format.tool_style}: {s}')

            return (converter.format_grammar(), parse)

    elif response_schema:
        converter._add_rule("root", response_rule + ' ' + converter._format_literal(suffix))

        @typechecked
        def parse(s: str) -> Optional[Message]:
            if response_rule.endswith(suffix):
                return Message(role="assistant", content=s[:-len(suffix)])
            
        return (converter.format_grammar(), parse)

    else:
        converter._add_rule("root", converter._format_literal(prefix) + ' ' + converter._format_literal(suffix))

        @typechecked
        def parse(s: str) -> Optional[Message]:
            if s.endswith(suffix):
                return Message(role="assistant", content=s[:-len(suffix)])
            return None

        return (None, parse)
        
