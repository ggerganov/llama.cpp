Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080 with 2 slots, 42 as seed and llama.cpp as api key
    Then  the server is starting
    Then  the server is healthy

  @llama.cpp
  Scenario: Health
    When the server is healthy
    Then the server is ready
    And  all slots are idle

  @llama.cpp
  Scenario Outline: Completion
    Given a prompt <prompt>
    And   a user api key <api_key>
    And   <n_predict> max tokens to predict
    And   a completion request
    Then  <n_predict> tokens are predicted

    Examples: Prompts
      | prompt                           | n_predict | api_key   |
      | I believe the meaning of life is | 128       | llama.cpp |
      | Write a joke about AI            | 512       | llama.cpp |
      | say goodbye                      | 0         |           |

  @llama.cpp
  Scenario Outline: OAI Compatibility
    Given a system prompt <system_prompt>
    And   a user prompt <user_prompt>
    And   a model <model>
    And   <max_tokens> max tokens to predict
    And   streaming is <enable_streaming>
    And   a user api key <api_key>
    Given an OAI compatible chat completions request with an api error <api_error>
    Then  <max_tokens> tokens are predicted

    Examples: Prompts
      | model        | system_prompt               | user_prompt                          | max_tokens | enable_streaming | api_key   | api_error |
      | llama-2      | You are ChatGPT.            | Say hello.                           | 64         | false            | llama.cpp | none      |
      | codellama70b | You are a coding assistant. | Write the fibonacci function in c++. | 512        | true             | llama.cpp | none      |
      | John-Doe     | You are an hacker.          | Write segfault code in rust.         | 0          | true             | hackme    | raised    |

  @llama.cpp
  Scenario: Multi users
    Given a prompt:
      """
      Write a very long story about AI.
      """
    And a prompt:
      """
      Write another very long music lyrics.
      """
    And 32 max tokens to predict
    And a user api key llama.cpp
    Given concurrent completion requests
    Then the server is busy
    And  all slots are busy
    Then the server is idle
    And  all slots are idle
    Then all prompts are predicted

  @llama.cpp
  Scenario: Multi users OAI Compatibility
    Given a system prompt "You are an AI assistant."
    And   a model tinyllama-2
    Given a prompt:
      """
      Write a very long story about AI.
      """
    And a prompt:
      """
      Write another very long music lyrics.
      """
    And 32 max tokens to predict
    And streaming is enabled
    And a user api key llama.cpp
    Given concurrent OAI completions requests
    Then the server is busy
    And  all slots are busy
    Then the server is idle
    And  all slots are idle
    Then all prompts are predicted

  # FIXME: #3969 infinite loop on the CI, not locally, if n_prompt * n_predict > kv_size
  @llama.cpp
  Scenario: Multi users with total number of tokens to predict exceeds the KV Cache size
    Given a prompt:
      """
      Write a very long story about AI.
      """
    And a prompt:
      """
      Write another very long music lyrics.
      """
    And a prompt:
      """
      Write a very long poem.
      """
    And a prompt:
      """
      Write a very long joke.
      """
    And 512 max tokens to predict
    And a user api key llama.cpp
    Given concurrent completion requests
    Then the server is busy
    And  all slots are busy
    Then the server is idle
    And  all slots are idle
    Then all prompts are predicted


  @llama.cpp
  Scenario: Embedding
    When embeddings are computed for:
    """
    What is the capital of Bulgaria ?
    """
    Then embeddings are generated


  @llama.cpp
  Scenario: OAI Embeddings compatibility
    Given a model tinyllama-2
    When an OAI compatible embeddings computation request for:
    """
    What is the capital of Spain ?
    """
    Then embeddings are generated


  @llama.cpp
  Scenario: Tokenize / Detokenize
    When tokenizing:
    """
    What is the capital of France ?
    """
    Then tokens can be detokenize

  @llama.cpp
  Scenario Outline: CORS Options
    When an OPTIONS request is sent from <origin>
    Then CORS header <cors_header> is set to <cors_header_value>

    Examples: Headers
      | origin          | cors_header                      | cors_header_value |
      | localhost       | Access-Control-Allow-Origin      | localhost         |
      | web.mydomain.fr | Access-Control-Allow-Origin      | web.mydomain.fr   |
      | origin          | Access-Control-Allow-Credentials | true              |
      | web.mydomain.fr | Access-Control-Allow-Methods     | POST              |
      | web.mydomain.fr | Access-Control-Allow-Headers     | *                 |
