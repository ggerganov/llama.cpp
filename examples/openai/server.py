import json, sys, subprocess, atexit
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.openai.llama_cpp_server_api import LlamaCppServerCompletionRequest
from examples.json_schema_to_grammar import SchemaConverter

from typing import Optional
import httpx
from fastapi import Depends, FastAPI, Request, Response
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse
from jsonargparse import CLI

from examples.openai.ts_converter import SchemaToTypeScriptConverter
from examples.openai.gguf_kvs import GGUFKeyValues, Keys
from examples.openai.api import Message, Tool, ToolFunction, ResponseFormat, ChatCompletionRequest
from examples.openai.chat_format import ChatFormat, ToolStyle

def _add_system_prompt(messages: list['Message'], system_prompt: str):
    # TODO: add to last system message, or create a new one just before the last user message
    system_message = next(((i, m) for i, m in enumerate(messages) if m.role == "system"), None)
    if system_message is not None:
        (i, m) = system_message
        messages[i].content = m.content + '\n' + system_prompt
    else:
        messages.insert(0, Message(role="system", content=system_prompt))
    return messages

def main(
    model: Path = Path("/Users/ochafik/AI/Models/Hermes-2-Pro-Mistral-7B.Q8_0.gguf"),
    host: str = "localhost",
    port: int = 8080,
    main_server_endpoint: Optional[str] = None,
    main_server_host: str = "localhost",
    main_server_port: Optional[int] = 8081,
):
    import uvicorn

    metadata = GGUFKeyValues(model)
    context_length = metadata[Keys.LLM.CONTEXT_LENGTH]
    chat_format = ChatFormat.from_gguf(metadata)
    print(chat_format)

    if not main_server_endpoint:
        server_process = subprocess.Popen([
            "./server", "-m", model,
            "--host", main_server_host, "--port", f'{main_server_port}',
            '-ctk', 'q4_0', '-ctv', 'f16',
            "-c", f"8192",
            # "-c", f"{context_length}",
        ])
        atexit.register(server_process.kill)
        main_server_endpoint = f"http://{main_server_host}:{main_server_port}"

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, chat_request: ChatCompletionRequest):
        headers = {
            "Content-Type": "application/json",
            "Authorization": request.headers.get("Authorization"),
        }

        if chat_request.response_format is not None:
            assert chat_request.response_format.type == "json_object", f"Unsupported response format: {chat_request.response_format.type}"
            response_schema = chat_request.response_format.json_schema or {}
        else:
            response_schema = None

        messages = chat_request.messages
        parser=None
        grammar=None

        converter = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)

        response_rule = converter.visit(response_schema, "response") if response_schema else None
          
        
        delimiter = '<%$[SAMPLE]$%>'
        empty_prompt = chat_format.render([], add_generation_prompt=True)
        planted_prompt = chat_format.render([{"role": "assistant", "content": delimiter}], add_generation_prompt=False)
        assert planted_prompt.startswith(empty_prompt), f"Planted prompt does not start with empty prompt: {planted_prompt} vs {empty_prompt}"
        [prefix, suffix] = planted_prompt[len(empty_prompt):].split(delimiter)

        if chat_request.tools:
            if chat_format.tool_style in (ToolStyle.DEFAULT, ToolStyle.NOUS_RESEARCH_HERMES):
                messages = _add_system_prompt(messages, '\n'.join([
                    'Here are the tools available:',
                    '<tools>',
                    *(tool.model_dump_json() for tool in chat_request.tools),
                    '</tools>',
                ]))

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
                    for tool in chat_request.tools
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
                    ") " + converter._format_literal(suffix))
                grammar = converter.format_grammar()
                
                def parse(s: str):
                    if '<tool_call>'.startswith(s):
                        if s.startswith('<tool_call>') and s.endswith('</tool_call>' + suffix):
                            s = s[len('<tool_call>'):-len('</tool_call>' + suffix)]
                            return {"role": "assistant", "tool_calls": [json.loads(s)]}
                        return None
                    else:
                        return {"role": "assistant", "content": s}
                
                parser = parse

            elif chat_format.tool_style == ToolStyle.FUNCTIONARY_V2:
                
                ts_converter = SchemaToTypeScriptConverter()
                
                messages = _add_system_prompt(messages, '\n'.join([
                    '// Supported function definitions that should be called when necessary.'
                    'namespace functions {',
                    *[
                        '// ' + tool.function.description.replace('\n', '\n// ') + '\n' + ''
                        'type ' + tool.function.name + ' = (_: ' + ts_converter.visit(tool.function.parameters) + ") => any;\n"
                        for tool in chat_request.tools
                    ],
                    '} // namespace functions',
                ]))

                # Only allowing a single tool call at a time for now.
                # Note that if there were more, they'd be separated by a '<|from|>assistant' literal
                converter._add_rule(
                    "root", 
                    converter._format_literal(prefix) + " (" +
                        (response_rule or converter.not_literal("<|recipient|>")) + " | " +
                        (' | '.join(
                            converter._format_literal(f"<|recipient|>{tool.function.name}\n<|content|>") + " " +
                            converter.visit(tool.function.parameters, tool.function.name + '-args')
                            for tool in chat_request.tools
                        )) +
                        ") " +
                    ") " + converter._format_literal(suffix))
                grammar = converter.format_grammar()
            else:
                raise NotImplementedError(f'Unsupported tool_style: {chat_format.tool_style}')

        elif response_schema:
            converter._add_rule('root', response_rule)
            grammar = converter.format_grammar()

            def parse(s):
                if s.endswith(suffix):
                    s = s[:-len(suffix)]
                    return {"role": "assistant", "content": s}
                return None
            
            parser = parse

        if chat_format.strict_user_assistant_alternation:
            print("TODO: merge system messages into user messages")
            # new_messages = []

        # TODO: Test whether the template supports formatting tool_calls
            
        prompt = chat_format.render(messages, add_generation_prompt=True)
        # print(prompt)
        # print(grammar)
        print(json.dumps(dict(
            prompt=prompt,
            stream=chat_request.stream,
            grammar=grammar,
        ), indent=2))
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{main_server_endpoint}/completions",
                json=LlamaCppServerCompletionRequest(
                    prompt=prompt,
                    stream=chat_request.stream,
                    n_predict=100,
                    grammar=grammar,
                ).model_dump(),
                headers=headers,
                timeout=None)
        
        return StreamingResponse(generate_chunks(response), media_type="text/event-stream") if chat_request.stream \
            else JSONResponse(response.json())

    async def generate_chunks(response):
        async for chunk in response.aiter_bytes():
            yield chunk

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    CLI(main)

