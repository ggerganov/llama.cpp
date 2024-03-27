# https://gist.github.com/ochafik/a3d4a5b9e52390544b205f37fb5a0df3
# pip install "fastapi[all]" "uvicorn[all]" sse-starlette jsonargparse jinja2 pydantic

import json, sys, subprocess, atexit
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.openai.llama_cpp_server_api import LlamaCppServerCompletionRequest
from examples.openai.gguf_kvs import GGUFKeyValues, Keys
from examples.openai.api import ChatCompletionResponse, Choice, Message, ChatCompletionRequest, Usage
from examples.openai.prompting import ChatFormat, make_grammar, make_tools_prompt

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import random
from starlette.responses import StreamingResponse
from typing import Annotated, Optional
import typer
from typeguard import typechecked

def generate_id(prefix):
    return f"{prefix}{random.randint(0, 1 << 32)}"

def main(
    model: Annotated[Optional[Path], typer.Option("--model", "-m")] = "models/7B/ggml-model-f16.gguf",
    # model: Path = Path("/Users/ochafik/AI/Models/Hermes-2-Pro-Mistral-7B.Q8_0.gguf"),
    # model_url: Annotated[Optional[str], typer.Option("--model-url", "-mu")] = None,
    host: str = "localhost",
    port: int = 8080,
    cpp_server_endpoint: Optional[str] = None,
    cpp_server_host: str = "localhost",
    cpp_server_port: Optional[int] = 8081,
):
    import uvicorn

    metadata = GGUFKeyValues(model)
    context_length = metadata[Keys.LLM.CONTEXT_LENGTH]
    chat_format = ChatFormat.from_gguf(metadata)
    # print(chat_format)

    if not cpp_server_endpoint:
        sys.stderr.write(f"# Starting C++ server with model {model} on {cpp_server_host}:{cpp_server_port}\n")
        server_process = subprocess.Popen([
            "./server", "-m", model,
            "--host", cpp_server_host, "--port", f'{cpp_server_port}',
            '-ctk', 'q4_0', '-ctv', 'f16',
            "-c", f"{2*8192}",
            # "-c", f"{context_length}",
        ], stdout=sys.stderr)
        atexit.register(server_process.kill)
        cpp_server_endpoint = f"http://{cpp_server_host}:{cpp_server_port}"

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
        if chat_request.tools:
            messages = chat_format.add_system_prompt(messages, make_tools_prompt(chat_format, chat_request.tools))

        (grammar, parser) = make_grammar(chat_format, chat_request.tools, response_schema)

        # TODO: Test whether the template supports formatting tool_calls
        sys.stderr.write(f'\n{grammar}\n\n')

        prompt = chat_format.render(messages, add_generation_prompt=True)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{cpp_server_endpoint}/completions",
                json=LlamaCppServerCompletionRequest(
                    prompt=prompt,
                    stream=chat_request.stream,
                    n_predict=300,
                    grammar=grammar,
                ).model_dump(),
                headers=headers,
                timeout=None)
        
        if chat_request.stream:
            # TODO: Remove suffix from streamed response using partial parser.
            assert not chat_request.tools and not chat_request.response_format, "Streaming not supported yet with tools or response_format"
            return StreamingResponse(generate_chunks(response), media_type="text/event-stream")
        else:
            result = response.json()
            if 'content' not in result:
                # print(json.dumps(result, indent=2))
                return JSONResponse(result)

            sys.stderr.write(json.dumps(result, indent=2) + "\n")
            # print(json.dumps(result.get('content'), indent=2))
            message = parser(result["content"])
            assert message is not None, f"Failed to parse response:\n{response.text}\n\n"

            prompt_tokens=result['timings']['prompt_n']
            completion_tokens=result['timings']['predicted_n']
            return JSONResponse(ChatCompletionResponse(
                id=generate_id('chatcmpl-'),
                object="chat.completion",
                created=int(time.time()),
                model=chat_request.model,
                choices=[Choice(
                    index=0,
                    message=message,
                    finish_reason="stop" if message.tool_calls is None else "tool_calls",
                )],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                system_fingerprint='...'
            ).model_dump())

    async def generate_chunks(response):
        async for chunk in response.aiter_bytes():
            yield chunk

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    typer.run(main)
