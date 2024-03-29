@llama.cpp
@server
Feature: llama.cpp server slot management

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   prompt caching is enabled
    And   2 slots
    And   . as slot save path
    And   2048 KV cache size
    And   42 as server seed
    And   24 max tokens to predict
    Then  the server is starting
    Then  the server is healthy

  Scenario: Save and Restore Slot
    Given a user prompt "What is the capital of France?"
    And   using slot id 1
    And   a completion request with no api error
    Then  24 tokens are predicted matching Lily
    And   22 prompt tokens are processed
    When  the slot 1 is saved with filename "slot1.bin"
    Then  the server responds with status code 200
    Given a user prompt "What is the capital of Germany?"
    And   a completion request with no api error
    Then  24 tokens are predicted matching Thank
    And   7 prompt tokens are processed
    When  the slot 2 is restored with filename "slot1.bin"
    Then  the server responds with status code 200
    Given a user prompt "What is the capital of France?"
    And   using slot id 2
    And   a completion request with no api error
    Then  24 tokens are predicted matching Lily
    And   1 prompt tokens are processed

  Scenario: Erase Slot
    Given a user prompt "What is the capital of France?"
    And   using slot id 1
    And   a completion request with no api error
    Then  24 tokens are predicted matching Lily
    And   22 prompt tokens are processed
    When  the slot 1 is erased
    Then  the server responds with status code 200
    Given a user prompt "What is the capital of France?"
    And   a completion request with no api error
    Then  24 tokens are predicted matching Lily
    And   22 prompt tokens are processed
