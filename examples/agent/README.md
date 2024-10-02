# Agents / Tool Calling w/ llama.cpp

- Install prerequisite: [uv](https://docs.astral.sh/uv/) (used to simplify python deps)

- Run `llama-server` w/ jinja templates. Note that most models need a template override (the HF to GGUF conversion only retains a single `chat_template`, but sometimes the models only support tool calls in an alternative chat template).

  ```bash
  make -j LLAMA_CURL=1 llama-server

  # Nous Hermes 2 Pro Llama 3 8B
  ./llama-server --jinja -fa --verbose \
    -hfr NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF -hff Hermes-2-Pro-Llama-3-8B-Q8_0.gguf \
    --chat-template-file tests/chat/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja

  # Llama 3.1 8B
  ./llama-server --jinja -fa --verbose \
    -hfr lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF -hff Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf

  # Llama 3.1 70B
  ./llama-server --jinja -fa --verbose \
    -hfr lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF -hff Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf

  # functionary-small-v3
  ./llama-server --jinja -fa --verbose \
    -hfr meetkai/functionary-small-v3.2-GGUF -hff functionary-small-v3.2.Q4_0.gguf \
    --chat-template-file tests/chat/templates/meetkai-functionary-medium-v3.2.jinja

  ./llama-server --jinja -fa --verbose \
    -m ~/Downloads/functionary-small-v3.2.Q4_0.gguf \
    --chat-template-file tests/chat/templates/meetkai-functionary-medium-v3.2.jinja

  # Llama 3.2 3B (poor adherence)
  ./llama-server --jinja -fa --verbose \
    -hfr lmstudio-community/Llama-3.2-3B-Instruct-GGUF -hff Llama-3.2-3B-Instruct-Q6_K_L.gguf \
    --chat-template-file tests/chat/templates/meta-llama-Llama-3.2-3B-Instruct.jinja

  ./llama-server --jinja -fa --verbose \
    -m ~/Downloads/Llama-3.2-3B-Instruct-Q6_K_L.gguf \
    --chat-template-file tests/chat/templates/meta-llama-Llama-3.2-3B-Instruct.jinja

  # Llama 3.2 1B (very poor adherence)
  ./llama-server --jinja -fa --verbose \
    -hfr lmstudio-community/Llama-3.2-1B-Instruct-GGUF -hff Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --chat-template-file tests/chat/templates/meta-llama-Llama-3.2-3B-Instruct.jinja
  ```

- Run the tools in [examples/agent/tools](./examples/agent/tools) inside a docker container (check http://localhost:8088/docs once running):

  ```bash
  docker run -p 8088:8088 -w /src -v $PWD/examples/agent:/src \
    --env BRAVE_SEARCH_API_KEY=$BRAVE_SEARCH_API_KEY \
    --rm -it ghcr.io/astral-sh/uv:python3.12-alpine \
    uv run fastify.py --port 8088 tools/
  ```

  > [!WARNING]
  > The command above gives tools (and your agent) access to the web (and read-only access to `examples/agent/**`. If you're concerned about unleashing a rogue agent on the web, please explore setting up proxies for your docker (and contribute back!)

- Run the agent with a given goal:

  ```bash
  uv run examples/agent/run.py --tools http://localhost:8088 \
    "What is the sum of 2535 squared and 32222000403?"

  uv run examples/agent/run.py --tools http://localhost:8088 \
    "What is the best BBQ join in Laguna Beach?"

  uv run examples/agent/run.py --tools http://localhost:8088 \
    "Search for, fetch and summarize the homepage of llama.cpp"
  ```

## TODO

- Implement code_interpreter using whichever tools are builtin for a given model.
