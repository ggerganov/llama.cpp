# examples.openai: OpenAI API-compatible server + agent / tools examples

A simple Python server that sits above the C++ [../server](examples/server) and offers improved OAI compatibility.

## Usage

Run a simple test:

```bash
# Spawns a Python server (which spawns a C++ Server) then hits it w/ a tool-calling request
examples/openai/test.sh
```

To simply run the Python server (+ C++ server under the hood):

```bash
python -m examples.openai
```

## Tools usage (WIP)

```bash
git clone https://github.com/NousResearch/Hermes-Function-Calling examples/openai/hermes_function_calling
```

Then edit `examples/agents/hermes_function_calling/utils.py`:

```py
log_folder = os.environ.get('LOG_FOLDER', os.path.join(script_dir, "inference_logs"))
```

Then run tools in a sandbox:

```bash
REQUIREMENTS_FILE=<( cat examples/agents/hermes_function_calling/requirements.txt | grep -vE "bitsandbytes|flash-attn" ) \
  examples/agents/run_sandboxed_tools.sh \
    examples/agents/hermes_function_calling/functions.py \
    -e LOG_FOLDER=/data/inference_logs
```

TODO: reactor that reads OpenAPI definitions and does the tool calling

## Features

The new examples/openai/server.py:

- Uses llama.cpp C++ server as a backend (spawns it or connects to existing)

- Uses actual jinja2 chat templates read from the models

- Supports grammar-constrained output for both JSON response format and tool calls

- Tool calling “works” w/ all models (even non-specialized ones like Mixtral 7x8B)

    - Optimised support for Functionary & Nous Hermes, easy to extend to other tool-calling fine-tunes

## TODO

- Support tool result messages

- Reactor / 

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
