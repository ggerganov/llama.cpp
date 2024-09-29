@llama.cpp
@server
Feature: Cancellation of llama.cpp server requests

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a model file test-model.gguf
    And   a model alias tinyllama-2
    And   BOS token is 1
    And   42 as server seed
      # KV Cache corresponds to the total amount of tokens
      # that can be stored across all independent sequences: #4130
      # see --ctx-size and #5568
    And   512 KV cache size
    And   32 as batch size
    And   2 slots
    And   64 server max tokens to predict
    And   prometheus compatible metrics exposed
    And   300 milliseconds delay in sampler for testing
    And   no warmup
    Then  the server is starting
    Then  the server is healthy
    # Then  the server is healthy with timeout 10 seconds


  Scenario Outline: Cancelling an OAI chat completion request frees up slot (streaming <enable_streaming>)
    Given a model llama-2
    And   a user prompt Once upon a time
    And   a system prompt You tell lengthy stories
    And   256 max tokens to predict
    And   256 server max tokens to predict
    And   streaming is <enable_streaming>
    And   disconnect after 100 milliseconds
    Given concurrent OAI completions requests
    And   wait for 700 milliseconds
    Then  all slots are idle

    Examples: Prompts
      | enable_streaming |
      | disabled         |
      | enabled          |


  Scenario Outline: Cancelling a completion request frees up slot (streaming <enable_streaming>)
    Given a model llama-2
    Given a prompt Once upon a time
    And   256 max tokens to predict
    And   256 server max tokens to predict
    And   streaming is <enable_streaming>
    And   disconnect after 100 milliseconds
    Given a completion request with no api error
    And   wait for 700 milliseconds
    Then  all slots are idle

    Examples: Prompts
      | enable_streaming |
      | disabled         |
      | enabled          |
