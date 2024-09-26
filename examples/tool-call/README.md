# Agents / Tool Calling w/ llama.cpp

- Install prerequisite: [uv](https://docs.astral.sh/uv/) (used to simplify python deps)

- Run `llama-server` w/ jinja templates:

  ```bash
  # make -j LLAMA_CURL=1 llama-server
  ./llama-server \
    -mu https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    --jinja \
    -c 8192 -fa
  ```

- Run some tools inside a docker container

  ```bash
  docker run --rm -it \
    -p "8088:8088" \
    -v $PWD/examples/tool-call:/src \
    ghcr.io/astral-sh/uv:python3.12-alpine \
    uv run /src/fastify.py --port 8088 /src/tools.py
  ```

- Verify which tools have been exposed: http://localhost:8088/docs

- Run the agent with a given goal:

  ```bash
  uv run examples/tool-call/agent.py \
    --tool-endpoint http://localhost:8088 \
    --goal "What is the sum of 2535 squared and 32222000403 then multiplied by one and a half. What's a third of the result?"
  ```