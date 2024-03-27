# https://gist.github.com/ochafik/a3d4a5b9e52390544b205f37fb5a0df3
# pip install "fastapi[all]" "uvicorn[all]" sse-starlette jsonargparse jinja2 pydantic

import json, sys, subprocess, atexit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.openai.llama_cpp_server_api import LlamaCppServerCompletionRequest
from examples.openai.gguf_kvs import GGUFKeyValues, Keys
from examples.openai.api import Message, ChatCompletionRequest
from examples.openai.prompting import ChatFormat, make_grammar, make_tools_prompt

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
from starlette.responses import StreamingResponse
from typing import Annotated, Optional
import typer
from typeguard import typechecked

def main(
    model: Annotated[Optional[Path], typer.Option("--model", "-m")] = "models/7B/ggml-model-f16.gguf",
    # model: Path = Path("/Users/ochafik/AI/Models/Hermes-2-Pro-Mistral-7B.Q8_0.gguf"),
    # model_url: Annotated[Optional[str], typer.Option("--model-url", "-mu")] = None,
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
            "-c", f"{2*8192}",
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
        if chat_request.tools:
            messages = chat_format.add_system_prompt(messages, make_tools_prompt(chat_format, chat_request.tools))

        (grammar, parser) = make_grammar(chat_format, chat_request.tools, response_schema)

        if chat_format.strict_user_assistant_alternation:
            print("TODO: merge system messages into user messages")
            # new_messages = []

        # TODO: Test whether the template supports formatting tool_calls
            
        prompt = chat_format.render(messages, add_generation_prompt=True)
        print(json.dumps(dict(
            stream=chat_request.stream,
            prompt=prompt,
            grammar=grammar,
        ), indent=2))
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{main_server_endpoint}/completions",
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
            print(json.dumps(result, indent=2))
            message = parser(result["content"])
            assert message is not None, f"Failed to parse response: {response.text}"
            return JSONResponse(message.model_dump())
            # return JSONResponse(response.json())

    async def generate_chunks(response):
        async for chunk in response.aiter_bytes():
            yield chunk

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    typer.run(main)
