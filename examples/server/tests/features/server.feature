@llama.cpp
@server
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a model alias tinyllama-2
    And   42 as server seed
      # KV Cache corresponds to the total amount of tokens
      # that can be stored across all independent sequences: #4130
      # see --ctx-size and #5568
    And   32 KV cache size
    And   512 as batch size
    And   1 slots
    And   embeddings extraction
    And   32 server max tokens to predict
    And   prometheus compatible metrics exposed
    Then  the server is starting
    Then  the server is healthy

  Scenario: Health
    Then the server is ready
    And  all slots are idle

  Scenario Outline: Completion
    Given a prompt <prompt>
    And   <n_predict> max tokens to predict
    And   a completion request with no api error
    Then  <n_predicted> tokens are predicted matching <re_content>
    And   prometheus metrics are exposed

    Examples: Prompts
      | prompt                           | n_predict | re_content                       | n_predicted |
      | I believe the meaning of life is | 8         | (read\|going)+                   | 8           |
      | Write a joke about AI            | 64        | (park\|friends\|scared\|always)+ | 32          |

  Scenario Outline: OAI Compatibility
    Given a model <model>
    And   a system prompt <system_prompt>
    And   a user prompt <user_prompt>
    And   <max_tokens> max tokens to predict
    And   streaming is <enable_streaming>
    Given an OAI compatible chat completions request with no api error
    Then  <n_predicted> tokens are predicted matching <re_content>

    Examples: Prompts
      | model        | system_prompt               | user_prompt                          | max_tokens | re_content             | n_predicted | enable_streaming |
      | llama-2      | Book                        | What is the best book                | 8          | (Mom\|what)+           | 8           | disabled         |
      | codellama70b | You are a coding assistant. | Write the fibonacci function in c++. | 64         | (thanks\|happy\|bird)+ | 32          | enabled          |

  Scenario: Embedding
    When embeddings are computed for:
    """
    What is the capital of Bulgaria ?
    """
    Then embeddings are generated

  Scenario: OAI Embeddings compatibility
    Given a model tinyllama-2
    When an OAI compatible embeddings computation request for:
    """
    What is the capital of Spain ?
    """
    Then embeddings are generated

  Scenario: OAI Embeddings compatibility with multiple inputs
    Given a model tinyllama-2
    Given a prompt:
      """
      In which country Paris is located ?
      """
    And a prompt:
      """
      Is Madrid the capital of Spain ?
      """
    When an OAI compatible embeddings computation request for multiple inputs
    Then embeddings are generated

  Scenario: Tokenize / Detokenize
    When tokenizing:
    """
    What is the capital of France ?
    """
    Then tokens can be detokenize

  Scenario: Models available
    Given available models
    Then  1 models are supported
    Then  model 0 is identified by tinyllama-2
    Then  model 0 is trained on 128 tokens context
