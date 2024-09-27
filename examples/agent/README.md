# Agents / Tool Calling w/ llama.cpp

- Install prerequisite: [uv](https://docs.astral.sh/uv/) (used to simplify python deps)

- Run `llama-server` w/ jinja templates:

  ```bash
  make -j LLAMA_CURL=1 llama-server
  ./llama-server \
    -mu https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    --jinja \
    -c 8192 -fa
  ```

- Run some tools inside a docker container (check http://localhost:8088/docs once running):

  ```bash
  docker run -p 8088:8088 -w /src -v $PWD/examples/agent:/src \
    --rm -it ghcr.io/astral-sh/uv:python3.12-alpine \
    uv run fastify.py --port 8088 tools.py
  ```

  > [!WARNING]
  > The command above gives tools (and your agent) access to the web (and read-only access to `examples/agent/**`. If you're concerned about unleashing a rogue agent on the web, please explore setting up proxies for your docker (and contribute back!)

- Run the agent with a given goal:

  ```bash
  uv run examples/agent/run.py \
    --tool-endpoint http://localhost:8088 \
    --goal "What is the sum of 2535 squared and 32222000403?"
  ```
