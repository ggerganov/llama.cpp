# Agents / Tool Calling w/ llama.cpp

While *any model* should work (using some generic support), we only support the native call style of a few models:
- Llama 3.x
- Functionary 3.x
- Hermes 2/3, Qwen 2.5
- Mistral Nemo.

For natively supported models, it's important to have the right template (it might not be in the GGUF; note that we prefer the `tool_use` variant of the Jinja template if it's present in the GGUF metadata). You can check which template is defined by inspecting `http://localhost:8080/props`, and inspect the logs for `Tool call style: `.

Here's how to run an agent w/ local tool call:

- Install prerequisite: [uv](https://docs.astral.sh/uv/) (used to simplify python deps)

- Run `llama-server` w/ any model:

  ```bash
  make -j LLAMA_CURL=1 llama-server

  # Native support for Mistral Nemo, Qwen 2.5, Hermes 3, Functionary 3.x
  # Note that some of these GGUFs lack the right template, so we override it
  # (otherwise they'd use the generic tool call support, which may be less efficient
  # and consume more tokens)

  ./llama-server --jinja -fa -ctk q4_0 -ctv q4_0 --verbose \
    -hfr bartowski/Qwen2.5-7B-Instruct-GGUF -hff Qwen2.5-7B-Instruct-Q4_K_M.gguf

  ./llama-server --jinja -fa -ctk q4_0 -ctv q4_0 --verbose \
    -hfr NousResearch/Hermes-3-Llama-3.1-8B-GGUF -hff Hermes-3-Llama-3.1-8B.Q4_K_M.gguf \
    --chat-template-file <( python scripts/get_hf_chat_template.py NousResearch/Hermes-3-Llama-3.1-8B tool_use )

  ./llama-server --jinja -fa -ctk q4_0 -ctv q4_0 --verbose \
    -hfr meetkai/functionary-small-v3.2-GGUF -hff functionary-small-v3.2.Q8_0.gguf \
    --chat-template-file <( python scripts/get_hf_chat_template.py meetkai/functionary-medium-v3.2 )

  ./llama-server --jinja -fa -ctk q4_0 -ctv q4_0 --verbose \
    -hfr lmstudio-community/Llama-3.2-3B-Instruct-GGUF -hff Llama-3.2-3B-Instruct-Q6_K.gguf \
    --chat-template-file <( python scripts/get_hf_chat_template.py meta-llama/Llama-3.2-3B-Instruct )

  ./llama-server --jinja -fa -ctk q4_0 -ctv q4_0 --verbose \
    -hfr bartowski/Mistral-Nemo-Instruct-2407-GGUF -hff Mistral-Nemo-Instruct-2407-Q8_0.gguf \
    --chat-template-file <( python scripts/get_hf_chat_template.py mistralai/Mistral-Nemo-Instruct-2407 )

  # Generic support, e.g. Phi 3.5, Gemma 2b, but really anything goes

  ./llama-server --jinja -fa --verbose \
    -hfr bartowski/Phi-3.5-mini-instruct-GGUF -hff Phi-3.5-mini-instruct-Q4_K_M.gguf

  ./llama-server --jinja -fa --verbose \
    -hfr bartowski/gemma-2-2b-it-GGUF -hff gemma-2-2b-it-Q4_K_M.gguf
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
  uv run examples/agent/run.py "What is the sum of 2535 squared and 32222000403?"
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
  uv run examples/agent/run.py "What is the best BBQ joint in Laguna Beach?"
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
  uv run examples/agent/run.py "Search for, fetch and summarize the homepage of llama.cpp"
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
  uv run examples/agent/run.py "Search for, fetch and summarize the homepage of llama.cpp" --provider=openai
  ```

## TODO

- Implement code_interpreter using whichever tools are builtin for a given model.
