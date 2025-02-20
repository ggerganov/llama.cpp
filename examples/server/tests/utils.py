#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# type: ignore[reportUnusedImport]

import subprocess
import os
import re
import json
import sys
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    List,
    Literal,
    Tuple,
    Set,
)
from re import RegexFlag
import wget


DEFAULT_HTTP_TIMEOUT = 12 if "LLAMA_SANITIZE" not in os.environ else 30


class ServerResponse:
    headers: dict
    status_code: int
    body: dict | Any


class ServerProcess:
    # default options
    debug: bool = False
    server_port: int = 8080
    server_host: str = "127.0.0.1"
    model_hf_repo: str = "ggml-org/models"
    model_hf_file: str | None = "tinyllamas/stories260K.gguf"
    model_alias: str = "tinyllama-2"
    temperature: float = 0.8
    seed: int = 42

    # custom options
    model_alias: str | None = None
    model_url: str | None = None
    model_file: str | None = None
    model_draft: str | None = None
    n_threads: int | None = None
    n_gpu_layer: int | None = None
    n_batch: int | None = None
    n_ubatch: int | None = None
    n_ctx: int | None = None
    n_ga: int | None = None
    n_ga_w: int | None = None
    n_predict: int | None = None
    n_prompts: int | None = 0
    slot_save_path: str | None = None
    id_slot: int | None = None
    cache_prompt: bool | None = None
    n_slots: int | None = None
    server_continuous_batching: bool | None = False
    server_embeddings: bool | None = False
    server_reranking: bool | None = False
    server_metrics: bool | None = False
    server_slots: bool | None = False
    pooling: str | None = None
    draft: int | None = None
    api_key: str | None = None
    lora_files: List[str] | None = None
    disable_ctx_shift: int | None = False
    draft_min: int | None = None
    draft_max: int | None = None
    no_webui: bool | None = None
    jinja: bool | None = None
    reasoning_format: Literal['deepseek', 'none'] | None = None
    chat_template: str | None = None
    chat_template_file: str | None = None

    # session variables
    process: subprocess.Popen | None = None

    def __init__(self):
        if "N_GPU_LAYERS" in os.environ:
            self.n_gpu_layer = int(os.environ["N_GPU_LAYERS"])
        if "DEBUG" in os.environ:
            self.debug = True
        if "PORT" in os.environ:
            self.server_port = int(os.environ["PORT"])

    def start(self, timeout_seconds: int | None = DEFAULT_HTTP_TIMEOUT) -> None:
        if "LLAMA_SERVER_BIN_PATH" in os.environ:
            server_path = os.environ["LLAMA_SERVER_BIN_PATH"]
        elif os.name == "nt":
            server_path = "../../../build/bin/Release/llama-server.exe"
        else:
            server_path = "../../../build/bin/llama-server"
        server_args = [
            "--host",
            self.server_host,
            "--port",
            self.server_port,
            "--temp",
            self.temperature,
            "--seed",
            self.seed,
        ]
        if self.model_file:
            server_args.extend(["--model", self.model_file])
        if self.model_url:
            server_args.extend(["--model-url", self.model_url])
        if self.model_draft:
            server_args.extend(["--model-draft", self.model_draft])
        if self.model_hf_repo:
            server_args.extend(["--hf-repo", self.model_hf_repo])
        if self.model_hf_file:
            server_args.extend(["--hf-file", self.model_hf_file])
        if self.n_batch:
            server_args.extend(["--batch-size", self.n_batch])
        if self.n_ubatch:
            server_args.extend(["--ubatch-size", self.n_ubatch])
        if self.n_threads:
            server_args.extend(["--threads", self.n_threads])
        if self.n_gpu_layer:
            server_args.extend(["--n-gpu-layers", self.n_gpu_layer])
        if self.draft is not None:
            server_args.extend(["--draft", self.draft])
        if self.server_continuous_batching:
            server_args.append("--cont-batching")
        if self.server_embeddings:
            server_args.append("--embedding")
        if self.server_reranking:
            server_args.append("--reranking")
        if self.server_metrics:
            server_args.append("--metrics")
        if self.server_slots:
            server_args.append("--slots")
        if self.pooling:
            server_args.extend(["--pooling", self.pooling])
        if self.model_alias:
            server_args.extend(["--alias", self.model_alias])
        if self.n_ctx:
            server_args.extend(["--ctx-size", self.n_ctx])
        if self.n_slots:
            server_args.extend(["--parallel", self.n_slots])
        if self.n_predict:
            server_args.extend(["--n-predict", self.n_predict])
        if self.slot_save_path:
            server_args.extend(["--slot-save-path", self.slot_save_path])
        if self.n_ga:
            server_args.extend(["--grp-attn-n", self.n_ga])
        if self.n_ga_w:
            server_args.extend(["--grp-attn-w", self.n_ga_w])
        if self.debug:
            server_args.append("--verbose")
        if self.lora_files:
            for lora_file in self.lora_files:
                server_args.extend(["--lora", lora_file])
        if self.disable_ctx_shift:
            server_args.extend(["--no-context-shift"])
        if self.api_key:
            server_args.extend(["--api-key", self.api_key])
        if self.draft_max:
            server_args.extend(["--draft-max", self.draft_max])
        if self.draft_min:
            server_args.extend(["--draft-min", self.draft_min])
        if self.no_webui:
            server_args.append("--no-webui")
        if self.jinja:
            server_args.append("--jinja")
        if self.reasoning_format is not None:
            server_args.extend(("--reasoning-format", self.reasoning_format))
        if self.chat_template:
            server_args.extend(["--chat-template", self.chat_template])
        if self.chat_template_file:
            server_args.extend(["--chat-template-file", self.chat_template_file])

        args = [str(arg) for arg in [server_path, *server_args]]
        print(f"bench: starting server with: {' '.join(args)}")

        flags = 0
        if "nt" == os.name:
            flags |= subprocess.DETACHED_PROCESS
            flags |= subprocess.CREATE_NEW_PROCESS_GROUP
            flags |= subprocess.CREATE_NO_WINDOW

        self.process = subprocess.Popen(
            [str(arg) for arg in [server_path, *server_args]],
            creationflags=flags,
            stdout=sys.stdout,
            stderr=sys.stdout,
            env={**os.environ, "LLAMA_CACHE": "tmp"} if "LLAMA_CACHE" not in os.environ else None,
        )
        server_instances.add(self)

        print(f"server pid={self.process.pid}, pytest pid={os.getpid()}")

        # wait for server to start
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                response = self.make_request("GET", "/health", headers={
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else None
                })
                if response.status_code == 200:
                    self.ready = True
                    return  # server is ready
            except Exception as e:
                pass
            print(f"Waiting for server to start...")
            time.sleep(0.5)
        raise TimeoutError(f"Server did not start within {timeout_seconds} seconds")

    def stop(self) -> None:
        if self in server_instances:
            server_instances.remove(self)
        if self.process:
            print(f"Stopping server with pid={self.process.pid}")
            self.process.kill()
            self.process = None

    def make_request(
        self,
        method: str,
        path: str,
        data: dict | Any | None = None,
        headers: dict | None = None,
        timeout: float | None = None,
    ) -> ServerResponse:
        url = f"http://{self.server_host}:{self.server_port}{path}"
        parse_body = False
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
            parse_body = True
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            parse_body = True
        elif method == "OPTIONS":
            response = requests.options(url, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Unimplemented method: {method}")
        result = ServerResponse()
        result.headers = dict(response.headers)
        result.status_code = response.status_code
        result.body = response.json() if parse_body else None
        print("Response from server", json.dumps(result.body, indent=2))
        return result

    def make_stream_request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        headers: dict | None = None,
    ) -> Iterator[dict]:
        url = f"http://{self.server_host}:{self.server_port}{path}"
        if method == "POST":
            response = requests.post(url, headers=headers, json=data, stream=True)
        else:
            raise ValueError(f"Unimplemented method: {method}")
        for line_bytes in response.iter_lines():
            line = line_bytes.decode("utf-8")
            if '[DONE]' in line:
                break
            elif line.startswith('data: '):
                data = json.loads(line[6:])
                print("Partial response from server", json.dumps(data, indent=2))
                yield data


server_instances: Set[ServerProcess] = set()


class ServerPreset:
    @staticmethod
    def tinyllama2() -> ServerProcess:
        server = ServerProcess()
        server.model_hf_repo = "ggml-org/models"
        server.model_hf_file = "tinyllamas/stories260K.gguf"
        server.model_alias = "tinyllama-2"
        server.n_ctx = 512
        server.n_batch = 32
        server.n_slots = 2
        server.n_predict = 64
        server.seed = 42
        return server

    @staticmethod
    def bert_bge_small() -> ServerProcess:
        server = ServerProcess()
        server.model_hf_repo = "ggml-org/models"
        server.model_hf_file = "bert-bge-small/ggml-model-f16.gguf"
        server.model_alias = "bert-bge-small"
        server.n_ctx = 512
        server.n_batch = 128
        server.n_ubatch = 128
        server.n_slots = 2
        server.seed = 42
        server.server_embeddings = True
        return server

    @staticmethod
    def tinyllama_infill() -> ServerProcess:
        server = ServerProcess()
        server.model_hf_repo = "ggml-org/models"
        server.model_hf_file = "tinyllamas/stories260K-infill.gguf"
        server.model_alias = "tinyllama-infill"
        server.n_ctx = 2048
        server.n_batch = 1024
        server.n_slots = 1
        server.n_predict = 64
        server.temperature = 0.0
        server.seed = 42
        return server

    @staticmethod
    def stories15m_moe() -> ServerProcess:
        server = ServerProcess()
        server.model_hf_repo = "ggml-org/stories15M_MOE"
        server.model_hf_file = "stories15M_MOE-F16.gguf"
        server.model_alias = "stories15m-moe"
        server.n_ctx = 2048
        server.n_batch = 1024
        server.n_slots = 1
        server.n_predict = 64
        server.temperature = 0.0
        server.seed = 42
        return server

    @staticmethod
    def jina_reranker_tiny() -> ServerProcess:
        server = ServerProcess()
        server.model_hf_repo = "ggml-org/models"
        server.model_hf_file = "jina-reranker-v1-tiny-en/ggml-model-f16.gguf"
        server.model_alias = "jina-reranker"
        server.n_ctx = 512
        server.n_batch = 512
        server.n_slots = 1
        server.seed = 42
        server.server_reranking = True
        return server


def parallel_function_calls(function_list: List[Tuple[Callable[..., Any], Tuple[Any, ...]]]) -> List[Any]:
    """
    Run multiple functions in parallel and return results in the same order as calls. Equivalent to Promise.all in JS.

    Example usage:

    results = parallel_function_calls([
        (func1, (arg1, arg2)),
        (func2, (arg3, arg4)),
    ])
    """
    results = [None] * len(function_list)
    exceptions = []

    def worker(index, func, args):
        try:
            result = func(*args)
            results[index] = result
        except Exception as e:
            exceptions.append((index, str(e)))

    with ThreadPoolExecutor() as executor:
        futures = []
        for i, (func, args) in enumerate(function_list):
            future = executor.submit(worker, i, func, args)
            futures.append(future)

        # Wait for all futures to complete
        for future in as_completed(futures):
            pass

    # Check if there were any exceptions
    if exceptions:
        print("Exceptions occurred:")
        for index, error in exceptions:
            print(f"Function at index {index}: {error}")

    return results


def match_regex(regex: str, text: str) -> bool:
    return (
        re.compile(
            regex, flags=RegexFlag.IGNORECASE | RegexFlag.MULTILINE | RegexFlag.DOTALL
        ).search(text)
        is not None
    )


def download_file(url: str, output_file_path: str | None = None) -> str:
    """
    Download a file from a URL to a local path. If the file already exists, it will not be downloaded again.

    output_file_path is the local path to save the downloaded file. If not provided, the file will be saved in the root directory.

    Returns the local path of the downloaded file.
    """
    file_name = url.split('/').pop()
    output_file = f'./tmp/{file_name}' if output_file_path is None else output_file_path
    if not os.path.exists(output_file):
        print(f"Downloading {url} to {output_file}")
        wget.download(url, out=output_file)
        print(f"Done downloading to {output_file}")
    else:
        print(f"File already exists at {output_file}")
    return output_file


def is_slow_test_allowed():
    return os.environ.get("SLOW_TESTS") == "1" or os.environ.get("SLOW_TESTS") == "ON"
