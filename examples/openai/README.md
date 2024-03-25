# examples.openai: OpenAI API-compatible server

A simple Python server that sits above the C++ [../server](examples/server) and offers improved OAI compatibility.

## Usage

```bash
python -m examples.openai -m some-model.gguf


```

## Features

The new examples/openai/server.py:

- Uses llama.cpp C++ server as a backend (spawns it or connects to existing)

- Uses actual jinja2 chat templates read from the models

- Supports grammar-constrained output for both JSON response format and tool calls

- Tool calling “works” w/ all models (even non-specialized ones like Mixtral 7x8B)

    - Optimised support for Functionary & Nous Hermes, easy to extend to other tool-calling fine-tunes

## TODO

- Embedding endpoint w/ distinct server subprocess

- Automatic/manual session caching

    - Spawns the main C++ CLI under the hood

    - Support precaching long prompts from CLI

    - Instant incremental inference in long threads

- Improve examples/agent:

    - Interactive agent CLI that auto-discovers tools from OpenAPI endpoints

    - Script that wraps any Python source as a container-sandboxed OpenAPI endpoint (allowing running ~unsafe code w/ tools)

    - Basic memory / RAG / python interpreter tools

- Follow-ups

    - Remove OAI support from server

    - Remove non-Python json schema to grammar converters

    - Reach out to frameworks to advertise new option. 
