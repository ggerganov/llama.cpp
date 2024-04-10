# examples.agent: Interactive agent that can use Python tools!

Have any LLM use local (sandboxed) tools, with a simple CLI.

```bash
python -m examples.agent \
    --model ~/AI/Models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
    --tools examples/agent/tools/example_math_tools.py \
    --goal "What is the sum of 2535 squared and 32222000403 then multiplied by one and a half. What's a third of the result?" \
    --greedy
```

<details>
<summary>Show output</summary>

```bash
💭 First, I will calculate the square of 2535, then add it to 32222000403. After that, I will multiply the result by 1.5 and finally, I will divide the result by 3.
⚙️  pow(value=2535, power=2) -> 6426225.0
💭 Now that I have calculated the square of 2535, I will calculate the sum of 6426225 and 32222000403.
⚙️  add(a=6426225, b=32222000403) -> 32228426628
💭 Now that I have calculated the sum, I will multiply it by 1.5.
⚙️  multiply(a=32228426628, b=1.5) -> 48342639942.0
💭 Now that I have calculated the product, I will divide it by 3.
⚙️  divide(a=48342639942.0, b=3) -> 16114213314.0
➡️ "\nThe result of the calculation is 16114213314.0."
```

</details>

```bash
python -m examples.agent \
    --tools examples/agent/tools/fake_weather_tools.py \
    --goal "What is the weather going to be like in San Francisco and Glasgow over the next 4 days." \
    --greedy
```

<details>
<summary>Show output</summary>

```bash
💭 I will first get the current weather in San Francisco, then get the 4-day weather forecast for both San Francisco and Glasgow.
⚙️  get_current_weather(location=San Francisco, format=fahrenheit) -> ...
💭 I will first get the current weather in San Francisco, then get the 4-day weather forecast for both San Francisco and Glasgow.
⚙️  get_n_day_weather_forecast(location=San Francisco, format=fahrenheit, num_days=4) -> ...
💭 I will first get the current weather in San Francisco, then get the 4-day weather forecast for both San Francisco and Glasgow.
⚙️  get_n_day_weather_forecast(location=Glasgow, format=celsius, num_days=4) -> ...
The current weather in San Francisco is sunny and 87.8F. Here is the 4-day weather forecast:

For San Francisco:
- In 1 day: Cloudy, 60.8F
- In 2 days: Sunny, 73.4F
- In 3 days: Cloudy, 62.6F

For Glasgow:
- In 1 day: Cloudy, 16C
- In 2 days: Sunny, 23C
- In 3 days: Cloudy, 17C
```

</details>


```bash
python -m examples.agent \
    --model ~/AI/Models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
    --std_tools \
    --goal "Wait 10sec then say Hi out loud" \
    --greedy
```

<details>
<summary>Show output</summary>

```bash
```

</details>

## Prerequisites

Note: To get conda, just install Miniforge (it's OSS): https://github.com/conda-forge/miniforge

```bash
conda create -n agent python=3.11
conda activate agent
pip install -r examples/agent/requirements.txt
pip install -r examples/openai/requirements.txt
```

## Components

This example relies on the new [OpenAI compatibility server](../openai).

```
  agent.py  →  examples.openai  →  server.cpp
            →  safe_tools.py
            → ( run_sandboxed_tools.sh :  Docker  →  fastify.py )  →  unsafe_tools.py  →  code interpreter, etc...
```

The agent can use tools written in Python, or (soon) exposed under OpenAPI endpoints. Only has standard Python deps (e.g. no langchain)

- Can call into any OpenAI endpoint that supports tool calling, spawns a local one if `--endpoint` isn't specified
(can pass all llama.cpp params)

- [Standard tools](./tools/std.py) include "safe" TTS, wait for/until helpers, and *requesting user input*.

- Tools are often "unsafe" (e.g. [Python execution functions](./tools/unsafe_python_tools.py)),
so we provide a script to run them in a Docker-sandboxed environment, exposed as an OpenAPI server:

    ```bash
    # With limactl, the default sandbox location ~/.llama.cpp/sandbox won't be writable
    # (see https://github.com/lima-vm/lima/discussions/393)
    # export DATA_DIR=/tmp/lima/llama.cpp/sandbox
    PORT=9999 examples/agent/run_sandboxed_tools.sh \
        examples/agent/tools/unsafe_python_tools.py &

    python -m examples.agent \
        --tools http://localhost:9999 \
        --goal "Whats cos(123) / 23 * 12.6 ?"
    ```

    <details>
    <summary>Show output</summary>

    ```
    💭 Calculate the expression using Python
    ⚙️  execute_python(source="import math\nresult = math.cos(123) / 23 * 12.6") -> {'result': -0.4864525314920599}
    ➡️ "-0.4864525314920599"
    ```

    </details>

    - [fastify.py](./fastify.py) turns a python module into an [OpenAPI](https://www.openapis.org/) endpoint using [FastAPI](https://fastapi.tiangolo.com/)

    - [run_sandboxed_tools.sh](./run_sandboxed_tools.sh) builds and runs a Docker environment with fastify inside it, and exposes its port locally

- Beyond just "tools", output format can be constrained using [JSON schemas](https://json-schema.org/) or [Pydantic](https://docs.pydantic.dev/latest/) types

    ```bash
    python -m examples.agent \
        --tools examples/agent/tools/example_summaries.py \
        --format PyramidalSummary \
        --goal "Create a pyramidal summary of Mankind's recent advancements"
    ```

## Launch parts separately

If you'd like to debug each binary separately (rather than have an agent spawing an OAI compat proxy spawning a C++ server), you can run these commands:

```bash
# C++ server
make -j server
./server \
    --model mixtral.gguf \
    --metrics \
    -ctk q4_0 \
    -ctv f16 \
    -c 32768 \
    --port 8081

# OpenAI compatibility layer
python -m examples.openai \
    --port 8080 \
    --endpoint http://localhost:8081 \
    --template-hf-model-id-fallback mistralai/Mixtral-8x7B-Instruct-v0.1

# Or have the OpenAI compatibility layer spawn the C++ server under the hood:
#   python -m examples.openai --model mixtral.gguf

# Agent itself:
python -m examples.agent \
    --endpoint http://localhost:8080 \
    --tools examples/agent/tools/example_summaries.py \
    --format PyramidalSummary \
    --goal "Create a pyramidal summary of Mankind's recent advancements"
```

## Use existing tools (WIP)

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

## TODO

- Wait for spawned servers to be heathly

- Add model URL / HF loading support

- Add Embedding endpoint + storage / retrieval tools (Faiss? ScaNN?), or spontaneous RAG

- Auto discover tools exposed by an OpenAPI endpoint

- Add a Python notebook tool example

- Update `run_sandboxed_tools.sh` to support dev mode (`uvicorn fastify:app --reload`)

- Follow-ups (depending on the vibe)

    - Remove OAI support from server

    - Remove non-Python json schema to grammar converters

