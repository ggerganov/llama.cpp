from pathlib import Path
import sys
from time import sleep
import typer
from pydantic import BaseModel, Json, TypeAdapter
from pydantic_core import SchemaValidator, core_schema
from typing import Annotated, Any, Callable, Dict, List, Union, Optional, Type
import json, requests

from examples.agent.openapi_client import OpenAPIMethod, openapi_methods_from_endpoint
from examples.agent.tools.std_tools import StandardTools
from examples.openai.api import ChatCompletionRequest, ChatCompletionResponse, Message, ResponseFormat, Tool, ToolFunction
from examples.agent.utils import collect_functions, load_module
from examples.openai.prompting import ToolsPromptStyle
from examples.openai.subprocesses import spawn_subprocess

def make_call_adapter(ta: TypeAdapter, fn: Callable[..., Any]):
    args_validator = SchemaValidator(core_schema.call_schema(
        arguments=ta.core_schema['arguments_schema'],
        function=fn,
    ))
    return lambda **kwargs: args_validator.validate_python(kwargs)

def completion_with_tool_usage(
        *,
        response_model: Optional[Union[Json[Any], type]]=None,
        max_iterations: Optional[int]=None,
        tools: List[Callable[..., Any]],
        endpoint: str,
        messages: List[Message],
        auth: Optional[str],
        verbose: bool,
        assume_llama_cpp_server: bool = False,
        **kwargs):
    '''
    Creates a chat completion using an OpenAI-compatible endpoint w/ JSON schema support
    (llama.cpp server, llama-cpp-python, Anyscale / Together...)

    The response_model param takes a type (+ supports Pydantic) and behaves just as w/ Instructor (see below)
    '''
    response_format = None
    type_adapter = None
    if response_model:
        if isinstance(response_model, dict):
            schema = response_model
        else:
            type_adapter = TypeAdapter(response_model)
            schema = type_adapter.json_schema()
        response_format=ResponseFormat(type="json_object", schema=schema)

    tool_map = {}
    tools_schemas = []
    for fn in tools:
        if isinstance(fn, OpenAPIMethod):
            tool_map[fn.__name__] = fn
            parameters_schema = fn.parameters_schema
        else:
            ta = TypeAdapter(fn)
            tool_map[fn.__name__] = make_call_adapter(ta, fn)
            parameters_schema = ta.json_schema()
        if verbose:
            sys.stderr.write(f'# PARAMS SCHEMA ({fn.__name__}): {json.dumps(parameters_schema, indent=2)}\n')
        tools_schemas.append(
            Tool(
                type="function",
                function=ToolFunction(
                    name=fn.__name__,
                    description=fn.__doc__ or '',
                    parameters=parameters_schema,
                )
            )
        )

    i = 0
    while (max_iterations is None or i < max_iterations):
        request = ChatCompletionRequest(
            messages=messages,
            response_format=response_format,
            tools=tools_schemas if tools_schemas else None,
            cache_prompt=True,
            **kwargs,
        )
        if verbose:
            sys.stderr.write(f'# REQUEST: {request.model_dump_json(indent=2)}\n')
        headers = {
            "Content-Type": "application/json",
        }
        if auth:
            headers["Authorization"] = auth

        def drop_nones(o):
            if isinstance(o, BaseModel):
                return drop_nones(o.model_dump())
            if isinstance(o, list):
                return [drop_nones(i) for i in o if i is not None]
            if isinstance(o, dict):
                return {
                    k: drop_nones(v)
                    for k, v in o.items()
                    if v is not None
                }
            return o
        
        if assume_llama_cpp_server:
            body = request.model_dump()
        else:
            # request_dict = request.model_dump()
            # body = drop_nones(request)
            tools_arg = None
            tool_choice = request.tool_choice
            response_format = None
            if request.tools:
                tools_arg = drop_nones(request.tools)
            if request.response_format:
                response_format = {
                    'type': request.response_format.type,
                }
                if request.response_format.schema:
                    assert tools_arg is None
                    assert tool_choice is None
                    tools_arg = [{
                        "type": "function",
                        "function": {
                            "name": "output",
                            "description": "A JSON object",
                            "parameters": request.response_format.schema,
                        }
                    }]
                    tool_choice = "output"

            body = drop_nones(dict(
                messages=drop_nones(request.messages),
                model=request.model,
                tools=tools_arg,
                tool_choice=tool_choice,
                temperature=request.temperature,
                response_format=response_format,
            ))

        if verbose:
            sys.stderr.write(f'# POSTing to {endpoint}/v1/chat/completions\n')
            sys.stderr.write(f'# HEADERS: {headers}\n')
            sys.stderr.write(f'# BODY: {json.dumps(body, indent=2)}\n')

        response = requests.post(
            f'{endpoint}/v1/chat/completions',
            headers=headers,
            json=body,
        )
        response.raise_for_status()
        response_json = response.json()
        response = ChatCompletionResponse(**response_json)
        if verbose:
            sys.stderr.write(f'# RESPONSE: {response.model_dump_json(indent=2)}\n')
        if response.error:
            raise Exception(f'Inference failed: {response.error.message}')

        assert len(response.choices) == 1
        choice = response.choices[0]

        content = choice.message.content
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                if content:
                    print(f'💭 {content}')

                args = json.loads(tool_call.function.arguments)
                pretty_call = f'{tool_call.function.name}({", ".join(f"{k}={v.model_dump_json() if isinstance(v, BaseModel) else json.dumps(v)}" for k, v in args.items())})'
                sys.stdout.write(f'⚙️  {pretty_call}')
                sys.stdout.flush()
                tool_result = tool_map[tool_call.function.name](**args)
                sys.stdout.write(f" → {tool_result}\n")
                messages.append(Message(
                    tool_call_id=tool_call.id,
                    role="tool",
                    name=tool_call.function.name,
                    content=f'{tool_result}',
                    # content=f'{pretty_call} = {tool_result}',
                ))
        else:
            assert content
            result = type_adapter.validate_json(content) if type_adapter else content
            return result

        i += 1

    if max_iterations is not None:
        raise Exception(f"Failed to get a valid response after {max_iterations} tool calls")


def main(
    goal: Annotated[str, typer.Option()],
    tools: Optional[List[str]] = None,
    format: Annotated[Optional[str], typer.Option(help="The output format: either a Python type (e.g. 'float' or a Pydantic model defined in one of the tool files), or a JSON schema, e.g. '{\"format\": \"date\"}'")] = None,
    max_iterations: Optional[int] = 10,
    std_tools: Optional[bool] = False,
    auth: Optional[str] = None,
    parallel_calls: Optional[bool] = False,
    verbose: bool = False,
    style: Optional[ToolsPromptStyle] = None,
    assume_llama_cpp_server: Optional[bool] = None,

    model: Optional[Annotated[str, typer.Option("--model", "-m")]] = None,# = "models/7B/ggml-model-f16.gguf",
    model_url: Optional[Annotated[str, typer.Option("--model-url", "-mu")]] = None,
    hf_repo: Optional[Annotated[str, typer.Option("--hf-repo", "-hfr")]] = None,
    hf_file: Optional[Annotated[str, typer.Option("--hf-file", "-hff")]] = None,
    
    endpoint: Optional[str] = None,
    context_length: Optional[int] = None,
    # endpoint: str = 'http://localhost:8080/v1/chat/completions',

    greedy: Optional[bool] = True,

    n_predict: Optional[int] = 1000,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    tfs_z: Optional[float] = None,
    typical_p: Optional[float] = None,
    temperature: Optional[float] = 0,
    dynatemp_range: Optional[float] = None,
    dynatemp_exponent: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presense_penalty: Optional[float] = None,
    mirostat: Optional[bool] = None,
    mirostat_tau: Optional[float] = None,
    mirostat_eta: Optional[float] = None,
    penalize_nl: Optional[bool] = None,
    n_keep: Optional[int] = None,
    seed: Optional[int] = None,
    n_probs: Optional[int] = None,
    min_keep: Optional[int] = None,
):
    if greedy:
        top_k = 1
        top_p = 0.0

    if not endpoint:
        server_port = 8080
        server_host = 'localhost'
        assume_llama_cpp_server = True
        endpoint = f'http://{server_host}:{server_port}'
        if verbose:
            sys.stderr.write(f"# Starting C++ server with model {model} on {endpoint}\n")
        cmd = [
            "python", "-m", "examples.openai.server",
            "--model", model,
            *(['--verbose'] if verbose else []),
            *(['--parallel-calls'] if parallel_calls else []),
            *([f'--context-length={context_length}'] if context_length else []),
            *([f'--style={style.value}'] if style else []),
        ]
        spawn_subprocess(cmd)
        sleep(5)

    tool_functions = []
    types: Dict[str, type] = {}
    for f in (tools or []):
        if f.startswith('http://') or f.startswith('https://'):
            tool_functions.extend(openapi_methods_from_endpoint(f))
        else:
            module = load_module(f)
            tool_functions.extend(collect_functions(module))
            types.update({
                k: v
                for k, v in module.__dict__.items()
                if isinstance(v, type)
            })

    if std_tools:
        tool_functions.extend(collect_functions(StandardTools))

    sys.stdout.write(f'🛠️  {", ".join(fn.__name__ for fn in tool_functions)}\n')

    response_model: Union[type, Json[Any]] = None #str
    if format:
        if format in types:
            response_model = types[format]
        elif format == 'json':
            response_model = {}
        else:
            try:
                response_model = json.loads(format)
            except:
                response_model = eval(format)


    result = completion_with_tool_usage(
        model="gpt-4o",
        endpoint=endpoint,
        response_model=response_model,
        max_iterations=max_iterations,
        tools=tool_functions,
        auth=auth,
        verbose=verbose,
        assume_llama_cpp_server=assume_llama_cpp_server or False,

        n_predict=n_predict,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        tfs_z=tfs_z,
        typical_p=typical_p,
        temperature=temperature,
        dynatemp_range=dynatemp_range,
        dynatemp_exponent=dynatemp_exponent,
        repeat_last_n=repeat_last_n,
        repeat_penalty=repeat_penalty,
        frequency_penalty=frequency_penalty,
        presense_penalty=presense_penalty,
        mirostat=mirostat,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        penalize_nl=penalize_nl,
        n_keep=n_keep,
        seed=seed,
        n_probs=n_probs,
        min_keep=min_keep,
        messages=[Message(role="user", content=goal)],
    )
    print(result if response_model else f'➡️ {result}')
    # exit(0)

if __name__ == '__main__':
    typer.run(main)

