@llama.cpp
@results
Feature: Results

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file tinyllamas/split/stories15M-00001-of-00003.gguf from HF repo ggml-org/models
    And   a model file test-model-00001-of-00003.gguf
    And   128 as batch size
    And   256 KV cache size
    And   128 max tokens to predict

  Scenario Outline: Multi users completion
    Given <n_slots> slots
    And   continuous batching
    Then  the server is starting
    Then  the server is healthy

    Given 42 as seed
    And a prompt:
      """
      Write a very long story about AI.
      """

    Given 42 as seed
    And a prompt:
      """
      Write a very long story about AI.
      """

    Given 42 as seed
    And a prompt:
      """
      Write a very long story about AI.
      """

    Given 42 as seed
    And a prompt:
      """
      Write a very long story about AI.
      """

    Given 42 as seed
    And a prompt:
      """
      Write a very long story about AI.
      """

    Given concurrent completion requests
    Then the server is busy
    Then the server is idle
    And  all slots are idle
    Then all predictions are equal
    Examples:
      | n_slots |
      | 1       |
      | 2       |
