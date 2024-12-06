@llama.cpp
@server
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   BOS token is 1
    And   42 as server seed
    And   greedy sampling
    And   8192 KV cache size
    And   32 as batch size
    And   1 slots
    And   prometheus compatible metrics exposed
    And   jinja templates are enabled


  Scenario Outline: Template <template_name> + tinystories model w/ required tool_choice yields <tool_name> tool call
    Given a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a test chat template file named <template_name>
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   <n_predict> max tokens to predict
    And   a user prompt say hello world with python
    And   a tool choice required
    And   <tool_name> tool
    And   parallel tool calls is <parallel_tool_calls>
    And   an OAI compatible chat completions request with no api error
    Then  tool <tool_name> is called with arguments <tool_arguments>

    Examples: Prompts
      | template_name                                 | n_predict | tool_name | tool_arguments                                           | parallel_tool_calls |
      | meetkai-functionary-medium-v3.1               | 32        | test      | {}                                                       | disabled            |
      | meetkai-functionary-medium-v3.1               | 32        | python    | {"code": ". She was so excited to go to the park and s"} | disabled            |
      | meetkai-functionary-medium-v3.2               | 32        | test      | {}                                                       | disabled            |
      | meetkai-functionary-medium-v3.2               | 32        | python    | {"code": "Yes,"}                                         | disabled            |
      | NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use | 128       | test      | {}                                                       | disabled            |
      | NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use | 128       | python    | {"code": "Yes,"}                                         | disabled            |
      | NousResearch-Hermes-3-Llama-3.1-8B-tool_use   | 128       | test      | {}                                                       | disabled            |
      | NousResearch-Hermes-3-Llama-3.1-8B-tool_use   | 128       | python    | {"code": "Yes,"}                                         | disabled            |
      | meta-llama-Meta-Llama-3.1-8B-Instruct         | 128       | test      | {}                                                       | disabled            |
      | meta-llama-Meta-Llama-3.1-8B-Instruct         | 128       | python    | {"code": "It's a shark."}                                | disabled            |
      | meta-llama-Llama-3.2-3B-Instruct              | 128       | test      | {}                                                       | disabled            |
      | meta-llama-Llama-3.2-3B-Instruct              | 128       | python    | {"code": "It's a shark."}                                | disabled            |
      | mistralai-Mistral-Nemo-Instruct-2407          | 128       | test      | {}                                                       | disabled            |
      | mistralai-Mistral-Nemo-Instruct-2407          | 128       | python    | {"code": "It's a small cost."}                           | disabled            |


  Scenario Outline: Template <template_name> + tinystories model yields no tool call
    Given a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a test chat template file named <template_name>
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   <n_predict> max tokens to predict
    And   a user prompt say hello world with python
    And   tools [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]
    And   an OAI compatible chat completions request with no api error
    Then  no tool is called

    Examples: Prompts
      | template_name                         | n_predict |
      | meta-llama-Meta-Llama-3.1-8B-Instruct | 64        |
      | meetkai-functionary-medium-v3.1       | 128       |
      | meetkai-functionary-medium-v3.2       | 128       |


  Scenario: Tool call template + tinystories and no tool won't call any tool
    Given a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a test chat template file named meta-llama-Meta-Llama-3.1-8B-Instruct
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   16 max tokens to predict
    And   a user prompt say hello world with python
    And   tools []
    And   an OAI compatible chat completions request with no api error
    Then  no tool is called


  @slow
  Scenario Outline: Python hello world w/ <hf_repo> + <tool> tool yields python call
    Given a model file <hf_file> from HF repo <hf_repo>
    And   a test chat template file named <template_override>
    And   no warmup
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   256 max tokens to predict
    And   a user prompt say hello world with python
    And   <tool> tool
    And   parallel tool calls is disabled
    And   an OAI compatible chat completions request with no api error
    Then  tool python is called with arguments <tool_arguments>

    Examples: Prompts
      | tool             | tool_arguments                       | hf_repo                                              | hf_file                                 | template_override                             |
      | python           | {"code": "print('Hello, world!')"}   | bartowski/gemma-2-2b-it-GGUF                         | gemma-2-2b-it-Q4_K_M.gguf               |                                               |
      | python           | {"code": "print('Hello, World!')"}   | bartowski/Mistral-Nemo-Instruct-2407-GGUF            | Mistral-Nemo-Instruct-2407-Q4_K_M.gguf  |                                               |
      | python           | {"code": "print(\"Hello World\")"}   | bartowski/Qwen2.5-7B-Instruct-GGUF                   | Qwen2.5-7B-Instruct-Q4_K_M.gguf         |                                               |
      | python           | {"code": "print('Hello, World!')"}   | bartowski/Phi-3.5-mini-instruct-GGUF                 | Phi-3.5-mini-instruct-Q4_K_M.gguf       |                                               |
      | python           | {"code": "print('Hello, world!')"}   | NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF            | Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf     | NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use |
      | python           | {"code": "print('hello world')"}     | NousResearch/Hermes-3-Llama-3.1-8B-GGUF              | Hermes-3-Llama-3.1-8B.Q4_K_M.gguf       | NousResearch-Hermes-3-Llama-3.1-8B-tool_use   |
      | python           | {"code": "print('Hello, World!'}"}   | bartowski/Llama-3.2-1B-Instruct-GGUF                 | Llama-3.2-1B-Instruct-Q4_K_M.gguf       | meta-llama-Llama-3.2-3B-Instruct              |
      | python           | {"code": "print("}                   | bartowski/Llama-3.2-3B-Instruct-GGUF                 | Llama-3.2-3B-Instruct-Q4_K_M.gguf       | meta-llama-Llama-3.2-3B-Instruct              |
      | python           | {"code": "print("}                   | lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF   | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  |                                               |
      | python           | {"code": "print('Hello, World!')"}   | bartowski/functionary-small-v3.2-GGUF                | functionary-small-v3.2-Q8_0.gguf        | meetkai-functionary-medium-v3.2               |
      | code_interpreter | {"code": "print('Hello, world!')"}   | bartowski/gemma-2-2b-it-GGUF                         | gemma-2-2b-it-Q4_K_M.gguf               |                                               |
      | code_interpreter | {"code": "print('Hello, World!')"}   | bartowski/Mistral-Nemo-Instruct-2407-GGUF            | Mistral-Nemo-Instruct-2407-Q4_K_M.gguf  | mistralai-Mistral-Nemo-Instruct-2407          |
      | code_interpreter | {"code": "print(\"Hello World\")"}   | bartowski/Qwen2.5-7B-Instruct-GGUF                   | Qwen2.5-7B-Instruct-Q4_K_M.gguf         |                                               |
      | code_interpreter | {"code": "print('Hello, World!')"}   | bartowski/Phi-3.5-mini-instruct-GGUF                 | Phi-3.5-mini-instruct-Q4_K_M.gguf       |                                               |
      | code_interpreter | {"code": "print('Hello, world!')"}   | NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF            | Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf     | NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use |
      | code_interpreter | {"code": "print('hello world')"}     | NousResearch/Hermes-3-Llama-3.1-8B-GGUF              | Hermes-3-Llama-3.1-8B.Q4_K_M.gguf       | NousResearch-Hermes-3-Llama-3.1-8B-tool_use   |
      | code_interpreter | {"code": "print('Hello, World!'}"}   | lmstudio-community/Llama-3.2-1B-Instruct-GGUF        | Llama-3.2-1B-Instruct-Q4_K_M.gguf       | meta-llama-Llama-3.2-3B-Instruct              |
      | code_interpreter | {"code": "print("}                   | lmstudio-community/Llama-3.2-3B-Instruct-GGUF        | Llama-3.2-3B-Instruct-Q4_K_M.gguf       | meta-llama-Llama-3.2-3B-Instruct              |
      | code_interpreter | {"code": "print("}                   | lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF   | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf  |                                               |
      | code_interpreter | {"code": "print('Hello, World!')"}   | bartowski/functionary-small-v3.2-GGUF                | functionary-small-v3.2-Q8_0.gguf        | meetkai-functionary-medium-v3.2               |


  @slow
  Scenario Outline: Python hello world w/o tools yields no tool call
    Given a model file Phi-3.5-mini-instruct-Q4_K_M.gguf from HF repo bartowski/Phi-3.5-mini-instruct-GGUF
    And   no warmup
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   256 max tokens to predict
    And   a user prompt say hello world with python
    And   parallel tool calls is disabled
    And   an OAI compatible chat completions request with no api error
    Then  no tool is called


  @slow
  Scenario Outline: Python hello world w/o none tool_choice yields no tool call
    Given a model file Phi-3.5-mini-instruct-Q4_K_M.gguf from HF repo bartowski/Phi-3.5-mini-instruct-GGUF
    And   no warmup
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   256 max tokens to predict
    And   a user prompt say hello world with python
    And   a tool choice none
    And   python tool
    And   parallel tool calls is disabled
    And   an OAI compatible chat completions request with no api error
    Then  no tool is called


  @slow
  Scenario: Parallel tool calls
    Given a model file Mistral-Nemo-Instruct-2407-Q4_K_M.gguf from HF repo bartowski/Mistral-Nemo-Instruct-2407-GGUF
    And   a test chat template file named mistralai-Mistral-Nemo-Instruct-2407
    And   no warmup
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   512 max tokens to predict
    And   a user prompt get the weather in paris and search for llama.cpp's latest commits (don't write comments in the code)
    And   python tool
    And   parallel tool calls is enabled
    And   an OAI compatible chat completions request with no api error
    Then  receiving the following tool calls: [{"arguments": {"code": "import requests\nresponse = requests.get('https://api.openweathermap.org/data/2.9/weather?q=Paris&appid=YOUR_API_KEY')\nprint(response.json())"}, "name": "ipython" , "id": "123456789"}, {"arguments": {"code": "!git log --oneline --after 2024-01-01 --before 2024-12-31 llama.cpp" }, "name": "ipython" , "id": "987654321"}]
