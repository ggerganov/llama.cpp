# Agents / Tool Calling w/ llama.cpp

- Install prerequisite: [uv](https://docs.astral.sh/uv/) (used to simplify python deps)

- Run `llama-server` w/ jinja templates. Note that most models need a template override (the HF to GGUF conversion only retains a single `chat_template`, but sometimes the models only support tool calls in an alternative chat template).

  ```bash
  make -j LLAMA_CURL=1 llama-server

  # Mistral NeMo
  ./llama-server --jinja -fa --verbose \
    -hfr bartowski/Mistral-Nemo-Instruct-2407-GGUF -hff Mistral-Nemo-Instruct-2407-Q8_0.gguf \
    --chat-template "$( python scripts/get_hf_chat_template.py mistralai/Mistral-Nemo-Instruct-2407 )"

  # Nous Hermes 2 Pro Llama 3 8B
  ./llama-server --jinja -fa --verbose \
    -hfr NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF -hff Hermes-2-Pro-Llama-3-8B-Q8_0.gguf \
    --chat-template "$( python scripts/get_hf_chat_template.py NousResearch/Hermes-2-Pro-Llama-3-8B tool_use )"

  # Llama 3.1 8B
  ./llama-server --jinja -fa --verbose \
    -hfr lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF -hff Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf

  # Llama 3.1 70B
  ./llama-server --jinja -fa --verbose \
    -hfr lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF -hff Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf

  # functionary-small-v3
  ./llama-server --jinja -fa --verbose \
    -hfr meetkai/functionary-small-v3.2-GGUF -hff functionary-small-v3.2.Q4_0.gguf \
    --chat-template "$( python scripts/get_hf_chat_template.py meetkai/functionary-medium-v3.2 )"

  # Llama 3.2 3B (poor adherence)
  ./llama-server --jinja -fa --verbose \
    -hfr lmstudio-community/Llama-3.2-3B-Instruct-GGUF -hff Llama-3.2-3B-Instruct-Q6_K.gguf \
    --chat-template "$( python scripts/get_hf_chat_template.py meta-llama/Llama-3.2-3B-Instruct )"

  # Llama 3.2 1B (very poor adherence)
  ./llama-server --jinja -fa --verbose \
    -hfr lmstudio-community/Llama-3.2-1B-Instruct-GGUF -hff Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --chat-template "$( python scripts/get_hf_chat_template.py meta-llama/Llama-3.2-3B-Instruct )"
  ```

- Run the tools in [examples/agent/tools](./examples/agent/tools) inside a docker container for *some* level of isolation (+ sneaky logging of outgoing http and https traffic: you wanna watch over those agents' shoulders for the time being 🧐). Check http://localhost:8088/docs to see the tools exposed.

  ```bash
  export BRAVE_SEARCH_API_KEY=... # Get one at https://api.search.brave.com/
  ./examples/agent/serve_tools_inside_docker.sh
  ```

  > [!WARNING]
  > The command above gives tools (and your agent) access to the web (and read-only access to `examples/agent/**`. You can loosen / restrict web access in [examples/agent/squid/conf/squid.conf](./squid/conf/squid.conf).

- Run the agent with some goal

  ```bash
  uv run examples/agent/run.py --tools http://localhost:8088 \
    "What is the sum of 2535 squared and 32222000403?"
  ```

  <details><summary>See output w/ Hermes-3-Llama-3.1-8B</summary>

  ```
  🛠️  Tools: python, fetch_page, brave_search
  ⚙️  python(code="print(2535**2 + 32222000403)")
  → 15 chars
  The sum of 2535 squared and 32222000403 is 32228426628.
  ```

  </details>

  ```bash
  uv run examples/agent/run.py --tools http://localhost:8088 \
    "What is the best BBQ joint in Laguna Beach?"
  ```

  <details><summary>See output w/ Hermes-3-Llama-3.1-8B</summary>

  ```
  🛠️  Tools: python, fetch_page, brave_search
  ⚙️  brave_search(query="best bbq joint in laguna beach")
  → 4283 chars
  Based on the search results, Beach Pit BBQ seems to be a popular and highly-rated BBQ joint in Laguna Beach. They offer a variety of BBQ options, including ribs, pulled pork, brisket, salads, wings, and more. They have dine-in, take-out, and catering options available.
  ```

  </details>

  ```bash
  uv run examples/agent/run.py --tools http://localhost:8088 \
    "Search for, fetch and summarize the homepage of llama.cpp"
  ```

  <details><summary>See output w/ Hermes-3-Llama-3.1-8B</summary>

  ```
  🛠️  Tools: python, fetch_page, brave_search
  ⚙️  brave_search(query="llama.cpp")
  → 3330 chars
  Llama.cpp is an open-source software library written in C++ that performs inference on various Large Language Models (LLMs). Alongside the library, it includes a CLI and web server. It is co-developed alongside the GGML project, a general-purpose tensor library. Llama.cpp is also available with Python bindings, known as llama.cpp-python. It has gained popularity for its ability to run LLMs on local machines, such as Macs with NVIDIA RTX systems. Users can leverage this library to accelerate LLMs and integrate them into various applications. There are numerous resources available, including tutorials and guides, for getting started with Llama.cpp and llama.cpp-python.
  ```

  </details>


- To compare the above results w/ a cloud provider's tool usage behaviour, just set the `--provider` flag (accepts `openai`, `together`, `groq`) and/or use `--endpoint`, `--api-key`, and `--model`

  ```bash
  export LLAMA_API_KEY=...      # for --provider=llama.cpp https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md
  export OPENAI_API_KEY=...     # for --provider=openai    https://platform.openai.com/api-keys
  export TOGETHER_API_KEY=...   # for --provider=together  https://api.together.ai/settings/api-keys
  export GROQ_API_KEY=...       # for --provider=groq      https://console.groq.com/keys
  uv run examples/agent/run.py --tools http://localhost:8088 \
    "Search for, fetch and summarize the homepage of llama.cpp" \
    --provider=openai
  ```

## TODO

- Implement code_interpreter using whichever tools are builtin for a given model.
