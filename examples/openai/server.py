import json, sys
from pathlib import Path
import time

from examples.openai.llama_cpp_server_api import LlamaCppServerCompletionRequest
from examples.openai.gguf_kvs import GGUFKeyValues, Keys  # type: ignore
from examples.openai.api import ChatCompletionResponse, Choice, ChatCompletionRequest, Usage
from examples.openai.prompting import ChatHandlerArgs, ChatTemplate, ToolsPromptStyle, get_chat_handler

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import random
from starlette.responses import StreamingResponse
from typing import Annotated, Optional
import typer

from examples.openai.subprocesses import spawn_subprocess

def generate_id(prefix):
    return f"{prefix}{random.randint(0, 1 << 32)}"

def main(
    model: Annotated[str, typer.Option("--model", "-m")] = "models/7B/ggml-model-f16.gguf",
    template_hf_model_id_fallback: Annotated[Optional[str], typer.Option(help="If the GGUF model does not contain a chat template, get it from this HuggingFace tokenizer")] = 'meta-llama/Llama-2-7b-chat-hf',
    # model_url: Annotated[Optional[str], typer.Option("--model-url", "-mu")] = None,
    host: str = "localhost",
    port: int = 8080,
    parallel_calls: bool = False,
    style: Optional[ToolsPromptStyle] = None,
    auth: Optional[str] = None,
    verbose: bool = False,
    context_length: Optional[int] = None,
    endpoint: Optional[str] = None,
    server_host: str = "localhost",
    server_port: Optional[int] = 8081,
):
    import uvicorn

    chat_template = None
    if model:
        metadata = GGUFKeyValues(Path(model))

        if not context_length:
            context_length = metadata[Keys.LLM.CONTEXT_LENGTH]

        if Keys.Tokenizer.CHAT_TEMPLATE in metadata:
            chat_template = ChatTemplate.from_gguf(metadata)
        else:
            sys.stderr.write(f"# WARNING: Model does not contain a chat template, fetching it from HuggingFace tokenizer of {template_hf_model_id_fallback}\n")
            assert template_hf_model_id_fallback, "template_hf_model_id_fallback is required when the model does not contain a chat template"
            chat_template = ChatTemplate.from_huggingface(template_hf_model_id_fallback)

        if verbose:
            sys.stderr.write(f"# CHAT TEMPLATE:\n\n{chat_template}\n\n")

    if not chat_template:
        sys.stderr.write(f"# WARNING: Unsure which model we're talking to, fetching its chat template from HuggingFace tokenizer of {template_hf_model_id_fallback}\n")
        assert template_hf_model_id_fallback or chat_template, "template_hf_model_id_fallback is required when using an endpoint without a model"
        chat_template = ChatTemplate.from_huggingface(template_hf_model_id_fallback)

    if not endpoint:
        if verbose:
            sys.stderr.write(f"# Starting C++ server with model {model} on {server_host}:{server_port}\n")
        cmd = [
            "./server", "-m", model,
            "--host", server_host, "--port", f'{server_port}',
            # TODO: pass these from JSON / BaseSettings?
            '-ctk', 'q4_0', '-ctv', 'f16',
            "-c", f"{context_length}",
            *([] if verbose else ["--log-disable"]),
        ]

        spawn_subprocess(cmd)
        endpoint = f"http://{server_host}:{server_port}"

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, chat_request: ChatCompletionRequest):
        headers = {
            "Content-Type": "application/json",
        }
        if (auth_value := request.headers.get("Authorization", auth)):
            headers["Authorization"] = auth_value

        if chat_request.response_format is not None:
            assert chat_request.response_format.type == "json_object", f"Unsupported response format: {chat_request.response_format.type}"
            response_schema = chat_request.response_format.schema or {}
        else:
            response_schema = None

        chat_handler = get_chat_handler(
            ChatHandlerArgs(chat_template=chat_template, response_schema=response_schema, tools=chat_request.tools),
            parallel_calls=parallel_calls,
            tool_style=style,
            verbose=verbose,
        )

        prompt = chat_handler.render_prompt(chat_request.messages) if chat_request.messages else chat_request.prompt
        assert prompt is not None, "One of prompt or messages field is required"

        if verbose:
            sys.stderr.write(f'\n# REQUEST:\n\n{chat_request.model_dump_json(indent=2)}\n\n')
            # sys.stderr.write(f'\n# MESSAGES:\n\n{TypeAdapter(list[Message]).dump_json(messages)}\n\n')
            sys.stderr.write(f'\n# PROMPT:\n\n{prompt}\n\n')
            sys.stderr.write(f'\n# GRAMMAR:\n\n{chat_handler.grammar}\n\n')

        data = LlamaCppServerCompletionRequest(
            **{
                k: v
                for k, v in chat_request.model_dump().items()
                if k not in (
                    "prompt",
                    "tools",
                    "messages",
                    "response_format",
                )
            },
            prompt=prompt,
            grammar=chat_handler.grammar,
        ).model_dump()
        # sys.stderr.write(json.dumps(data, indent=2) + "\n")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{endpoint}/completions',
                json=data,
                headers=headers,
                timeout=None)

        if chat_request.stream:
            # TODO: Remove suffix from streamed response using partial parser.
            assert not chat_request.tools and not chat_request.response_format, "Streaming not supported yet with tools or response_format"
            return StreamingResponse(generate_chunks(response), media_type="text/event-stream")
        else:
            result = response.json()
            if verbose:
                sys.stderr.write("# RESULT:\n\n" + json.dumps(result, indent=2) + "\n\n")
            if 'content' not in result:
                # print(json.dumps(result, indent=2))
                return JSONResponse(result)

            # print(json.dumps(result.get('content'), indent=2))
            message = chat_handler.parse(result["content"])
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
