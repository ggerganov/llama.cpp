@llama.cpp
@server
Feature: Cancellation of llama.cpp server requests

  Background: Server startup
    Given a server listening on localhost:8080
    And   500 milliseconds delay in sampler for testing
    And   a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a model file test-model.gguf
    And   a model alias tinyllama-2
    And   BOS token is 1
    And   42 as server seed
      # KV Cache corresponds to the total amount of tokens
      # that can be stored across all independent sequences: #4130
      # see --ctx-size and #5568
    And   256 KV cache size
    And   32 as batch size
    And   1 slots
    And   64 server max tokens to predict
    Then  the server is starting
    Then  the server is healthy

  # Scenario: Health
  #   Then the server is ready
  #   And  all slots are idle

  @wip
  Scenario Outline: Cancelling completion request frees up slot
    Given a prompt:
    """
    Once upon
    """
    And   256 max tokens to predict
    And   256 server max tokens to predict
    And   streaming is <enable_streaming>
    And   a completion request cancelled after 100 milliseconds
    # And   wait for 50 milliseconds
    Then  all slots are idle

    Examples: Prompts
      | enable_streaming |
      | disabled         |
      | enabled          |
