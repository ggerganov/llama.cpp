# Usage:
#! ./server -m some-model.gguf &
#! pip install pydantic
#! python examples/json-schema-pydantic-example.py
#
# TODO:
# - https://github.com/NousResearch/Hermes-Function-Calling
#
# <|im_start|>system
# You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags
# You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:
# <tools> {'type': 'function', 'function': {'name': 'get_stock_fundamentals',
# 'description': 'get_stock_fundamentals(symbol: str) -> dict - Get fundamental data for a given stock symbol using yfinance API.\n\n    Args:\n    symbol (str): The stock symbol.\n\n    Returns:\n    dict: A dictionary containing fundamental data.', 'parameters': {'type': 'object', 'properties': {'symbol': {'type': 'string'}}, 'required': ['symbol']}}} 
# </tools> Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
# <tool_call>
# {'arguments': <args-dict>, 'name': <function-name>}
# </tool_call><|im_end|>

from dataclasses import dataclass
import subprocess
import sys
from pydantic import BaseModel, TypeAdapter
from annotated_types import MinLen
from typing import Annotated, Callable, List, Union, Literal, Optional, Type, get_args, get_origin
import json, requests

from examples.openai.api import ToolCallsTypeAdapter

def type_to_str(t):
    origin = get_origin(t)
    if origin is None:
        return t.__name__
    args = get_args(t)
    return origin.__name__ + (
        f'[{", ".join(type_to_str(a) for a in args)}]' if args else ''
    )

def build_union_type_adapter(*types):
    src = '\n'.join([
        'from pydantic import TypeAdapter',
        'from typing import Union',
        f'_out = TypeAdapter(Union[{", ".join(type_to_str(t) for t in types)}])',
    ])
    globs = {
        **globals(),
        **{t.__name__: t for t in types},
    }
    exec(src, globs)
    return globs['_out']

class Thought(BaseModel):
    thought: str


def build_tool_call_adapter2(final_output_type, *tools):
    lines = [
        'from pydantic import BaseModel, TypeAdapter',
        'from typing import Literal, Union',
    ]
    globs = {
        **globals(),
        **locals(),
        final_output_type.__name__: final_output_type,
    }
    tool_calls = []
    for fn in tools:
        # TODO: escape fn.__doc__ and fn.__doc__ to avoid comment or metadata injection!
        fn_name = fn.__name__
        fn_doc = fn.__doc__.replace('"""', "'''") if fn.__doc__ else None
        name = fn_name.replace('_', ' ').title().replace(' ', '')
        lines += [
            f'class {name}ToolArgs(BaseModel):',
            *(f'  {k}: {type_to_str(v)}' for k, v in fn.__annotations__.items() if k != 'return'),
            f'class {name}ToolCall(BaseModel):',
            *([f'  """{fn_doc}"""'] if fn_doc else []),
            f'  name: Literal["{fn_name}"]',
            f'  arguments: {name}ToolArgs',
            f'class {name}Tool(BaseModel):',
            # *([f'  """{fn_doc}"""'] if fn_doc else []),
            f'  id: str',
            f'  type: Literal["function"]',
            f'  function: {name}ToolCall',
            f'  def __call__(self) -> {type_to_str(fn.__annotations__.get("return"))}:',
            f'    return {fn_name}(**self.function.arguments.dict())',
        ]
        tool_calls.append(f'{name}Tool')
    
    lines += [
        # 'class FinalResult(BaseModel):',
        # f'  result: {type_to_str(final_output_type)}',
        # 'class Response(BaseModel):',
        # f'  """A response that starts with a thought about whether we need tools or not, the plan about tool usage (maybe a sequence of tool calls), and then either a final result (of type {final_output_type.__name__}) or a first tool call"""',
        # f'  original_goal: str',
        # f'  thought_process: str',
        # # f'  thought: str',
        # f'  next_step: Union[FinalResult, {", ".join(tool_calls)}]',
        # f'response_adapter = TypeAdapter(Response)'
        f'response_adapter = TypeAdapter(Union[{", ".join(tool_calls)}])',
    ]

    exec('\n'.join(lines), globs)
    return globs['response_adapter']

def create_completion2(*, response_model=None, max_tool_iterations=None, tools=[], endpoint="http://localhost:8080/v1/chat/completions", messages, **kwargs):
    '''
    Creates a chat completion using an OpenAI-compatible endpoint w/ JSON schema support
    (llama.cpp server, llama-cpp-python, Anyscale / Together...)

    The response_model param takes a type (+ supports Pydantic) and behaves just as w/ Instructor (see below)
    '''
    if response_model:
        type_adapter = TypeAdapter(response_model)
        schema = type_adapter.json_schema()
        # messages = [{
        #     "role": "system",
        #     "content": f"Respond in JSON format with the following schema: {json.dumps(schema, indent=2)}"
        # }] + messages
        # print("Completion: ", json.dumps(messages, indent=2))
        # print("SCHEMA: " + json.dumps(schema, indent=2))
        response_format={"type": "json_object", "schema": schema }

    tool_call_adapter = build_tool_call_adapter2(response_model, *tools)
    tool_adapters = [(fn, TypeAdapter(fn)) for fn in tools]
    tools_schemas = [{
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": fn.__doc__,
            "parameters": ta.json_schema()
        }
    } for (fn, ta) in tool_adapters]

    # messages = [{
    #     "role": "system",
    #     "content": '\n'.join([
    # #         "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.",
    # #         "You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:",
    # #         f'<tools>{json.dumps(tools_schemas)}</tools>',
    #         'Before calling each tool, you think clearly and briefly about why and how you are using the tool.',
    #         f"Respond in JSON format with the following schema: {json.dumps(schema, indent=2)}" if schema else "",
    #     ])
    # }] + messages

    i = 0
    while (max_tool_iterations is None or i < max_tool_iterations):
        body=dict(
            messages=messages,
            response_format=response_format,
            tools=tools_schemas,
            **kwargs
        )
        # sys.stderr.write(f'# REQUEST: {json.dumps(body, indent=2)}\n')
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=body,
        )
        if response.status_code != 200:
            raise Exception(f"Request failed ({response.status_code}): {response.text}")

        # sys.stderr.write(f"\n# RESPONSE:\n\n<<<{response.text}>>>\n\n")
        data = response.json()
        if 'error' in data:
            raise Exception(data['error']['message'])

        # sys.stderr.write(f"\n# RESPONSE DATA:\n\n{json.dumps(data, indent=2)}\n\n")
        # print(json.dumps(data, indent=2))
        choice = data["choices"][0]

        content = choice["message"].get("content")
        if choice.get("finish_reason") == "tool_calls":
            # sys.stderr.write(f'\n# TOOL CALLS:\n{json.dumps(choice["message"]["tool_calls"], indent=2)}\n\n')
            # tool_calls =ToolCallsTypeAdapter.validate_json(json.dumps(choice["tool_calls"]))
            messages.append(choice["message"])
            for tool_call in choice["message"]["tool_calls"]:
                # id = tool_call.get("id")
                # if id:
                #     del tool_call["id"]

                if content:
                    print(f'💭 {content}')

                tc = tool_call_adapter.validate_json(json.dumps(tool_call))
                
                pretty_call = f'{tc.function.name}({", ".join(f"{k}={v}" for k, v in tc.function.arguments.model_dump().items())})'
                sys.stdout.write(f'⚙️  {pretty_call}')
                result = tc()
                sys.stdout.write(f" -> {result}\n")
                messages.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "name": tc.function.name,
                    # "content": f'{result}',
                    "content": f'{pretty_call} = {result}',
                })
        else:
            assert content
            # print(content)
            # print(json.dumps(json.loads(content), indent=2))
            result = type_adapter.validate_json(content) if type_adapter else content
            # if isinstance(result, Thought):
            #     print(f'💭 {result.thought}')
            #     messages.append({
            #         "role": "assistant",
            #         "content": json.dumps(result.model_dump(), indent=2),
            #     })
            # else:
            return result

        i += 1

    if max_tool_iterations is not None:
        raise Exception(f"Failed to get a valid response after {max_tool_iterations} tool calls")

if __name__ == '__main__':

    class QAPair(BaseModel):
        question: str
        concise_answer: str
        justification: str

    class PyramidalSummary(BaseModel):
        title: str
        summary: str
        question_answers: Annotated[List[QAPair], MinLen(2)]
        sub_sections: Optional[Annotated[List['PyramidalSummary'], MinLen(2)]]

    # print("# Summary\n", create_completion(
    #     model="...",
    #     response_model=PyramidalSummary,
    #     messages=[{
    #         "role": "user",
    #         "content": f"""
    #             You are a highly efficient corporate document summarizer.
    #             Create a pyramidal summary of an imaginary internal document about our company processes
    #             (starting high-level, going down to each sub sections).
    #             Keep questions short, and answers even shorter (trivia / quizz style).
    #         """
    #     }]))
    
    import math

    def eval_python_expression(expr: str) -> float:
        """
            Evaluate a Python expression reliably.
            This can be used to compute complex nested mathematical expressions, or any python, really.
        """
        print("# Evaluating expression: ", expr)
        return "0.0"

    def add(a: float, b: float) -> float:
        """
            Add a and b reliably.
            Don't use this tool to compute the square of a number (use multiply or pow instead)
        """
        return a + b
    
    # def say(something: str) -> str:
    #     """
    #         Just says something. Used to say each thought out loud
    #     """
    #     return subprocess.check_call(["say", something])

    def multiply(a: float, b: float) -> float:
        """Multiply a with b reliably"""
        return a * b

    def divide(a: float, b: float) -> float:
        """Divide a by b reliably"""
        return a / b

    def pow(value: float, power: float) -> float:
        """
            Raise a value to a power (exponent) reliably.
            The square of x is pow(x, 2), its cube is pow(x, 3), etc.
        """
        return math.pow(value, power)

    result = create_completion2(
        model="...",
        response_model=str,
        tools=[add, multiply, divide, pow], #, say],#, eval_python_expression],
        # tools=[eval_python_expression],
        temperature=0.0,
        # repetition_penalty=1.0,
        n_predict=1000,
        top_k=1,
        top_p=0.0,
        # logit_bias={
        #     i: 10.0
        #     for i in range(1, 259)
        # },
        messages=[{
        #     "role": "system",
        #     "content": f"""
        #         You are a reliable assistant. You think step by step and think before using tools
        #     """
        # }, {
            "role": "user",
            # "content": f"""
            #     What is 10 squared?
            # """
            "content": f"""
                What is the sum of 2535 squared and 32222000403 then multiplied by one and a half. What's a third of the result?

                Keep your goal in mind at every step.
            """
                # Think step by step, start expressing the problem as an arithmetic expression
        }])
    
    # result = create_completion(
    #     model="...",
    #     response_model=float,
    #     tools=[add, multiply, divide, pow], #, say],#, eval_python_expression],
    #     temperature=0.0,
    #     # logit_bias={
    #     #     i: 10.0
    #     #     for i in range(1, 259)
    #     # },
    #     messages=[{
    #         "role": "user",
    #         # "content": f"""
    #         #     What is 10 squared?
    #         # """
    #         "content": f"""
    #             What is the sum of 2535 squared and 32222000403 then multiplied by one and a half. What's a third of the result?
    #         """
    #             # Think step by step, start expressing the problem as an arithmetic expression
    #     }])
    
    # 💭 First, I need to square the number 2535. For this, I will use the 'pow' tool.
    # ⚙️  pow(args={'value': 2535.0, 'power': 2.0})-> 6426225.0
    # 💭 Now that I have the square of 2535, I need to add it to 32222000403.0 and store the result.
    # ⚙️  add(args={'a': 6426225.0, 'b': 32222000403.0})-> 32228426628.0
    # 💭 Now that I have the sum of 2535 squared and 32222000403, I need to multiply it by 1.5.
    # ⚙️  pow(args={'value': 32228426628.0, 'power': 1.5})-> 5785736571757004.0
    # 💭 Now that I have the result of the sum multiplied by 1.5, I need to divide it by 3 to get a third of the result.
    # ⚙️  divide(args={'a': 5785736571757004.0, 'b': 3.0})-> 1928578857252334.8
    # 💭 I have now calculated a third of the result, which is 1928578857252334.8. I can now share this as the final answer.
    # Result:  1928578857252334.8

    expected_result = (2535 ** 2 + 32222000403) * 1.5 / 3.0
    print("➡️", result)
    assert math.fabs(result - expected_result) < 0.0001, f"Expected {expected_result}, got {result}"
