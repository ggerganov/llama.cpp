@llama.cpp
@server
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a model file test-model.gguf
    And   a model alias tinyllama-2
    And   BOS token is 1
    And   42 as server seed
    And   8192 KV cache size
    And   32 as batch size
    And   2 slots
    And   prometheus compatible metrics exposed
    And   jinja templates are enabled


  Scenario Outline: OAI Compatibility w/ tools and required tool_choice
    Given a chat template file ../../../tests/chat/templates/<template_name>.jinja
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   <n_predict> max tokens to predict
    And   a user prompt write a hello world in python
    And   a tool choice required
    And   tools <tools>
    And   an OAI compatible chat completions request with no api error
    Then  tool <tool_name> is called with arguments <tool_arguments>

    Examples: Prompts
      | template_name                         | n_predict | tool_name | tool_arguments         | tools |
      | meetkai-functionary-medium-v3.1       | 128       | test      | {}                     | [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]                                                                       |
      | meetkai-functionary-medium-v3.1       | 128       | ipython   | {"code": "Yes, you can."} | [{"type":"function", "function": {"name": "ipython", "description": "", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": ""}}, "required": ["code"]}}}] |
      | meetkai-functionary-medium-v3.2       | 128       | test      | {}                     | [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]                                                                       |
      | meetkai-functionary-medium-v3.2       | 128       | ipython   | {"code": "Yes,"}       | [{"type":"function", "function": {"name": "ipython", "description": "", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": ""}}, "required": ["code"]}}}] |
      | meta-llama-Meta-Llama-3.1-8B-Instruct | 64        | test      | {}                     | [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]                                                                       |
      | meta-llama-Meta-Llama-3.1-8B-Instruct | 64        | ipython   | {"code": "it and realed at the otter. Asked Dave Dasty, Daisy is a big, shiny blue. As"}    | [{"type":"function", "function": {"name": "ipython", "description": "", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": ""}}, "required": ["code"]}}}] |
      | meta-llama-Llama-3.2-3B-Instruct      | 64        | test      | {}                     | [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]                                                                       |
      | meta-llama-Llama-3.2-3B-Instruct      | 64        | ipython   | {"code": "Yes,"}    | [{"type":"function", "function": {"name": "ipython", "description": "", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": ""}}, "required": ["code"]}}}] |


  Scenario Outline: OAI Compatibility w/ tools and auto tool_choice
    Given a chat template file ../../../tests/chat/templates/<template_name>.jinja
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   <n_predict> max tokens to predict
    And   a user prompt write a hello world in python
    And   tools [{"type":"function", "function": {"name": "test", "description": "", "parameters": {"type": "object", "properties": {}}}}]
    And   an OAI compatible chat completions request with no api error
    Then  no tool is called

    Examples: Prompts
      | template_name                         | n_predict |
      | meta-llama-Meta-Llama-3.1-8B-Instruct | 64        |
      | meetkai-functionary-medium-v3.1       | 128       |
      | meetkai-functionary-medium-v3.2       | 128       |


  Scenario: OAI Compatibility w/ no tool
    Given a chat template file ../../../tests/chat/templates/meta-llama-Meta-Llama-3.1-8B-Instruct.jinja
    And   the server is starting
    And   the server is healthy
    And   a model test
    And   16 max tokens to predict
    And   a user prompt write a hello world in python
    And   a tool choice <tool_choice>
    And   tools []
    And   an OAI compatible chat completions request with no api error
    Then  no tool is called

