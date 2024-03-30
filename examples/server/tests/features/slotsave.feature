@llama.cpp
@slotsave
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
    # First prompt in slot 1 should be fully processed
    Given a user prompt "What is the capital of France?"
    And   using slot id 1
    And   a completion request with no api error
    Then  24 tokens are predicted matching (Lily|cake)
    And   22 prompt tokens are processed
    When  the slot 1 is saved with filename "slot1.bin"
    Then  the server responds with status code 200
    # Since we have cache, this should only process the last tokens
    Given a user prompt "What is the capital of Germany?"
    And   a completion request with no api error
    Then  24 tokens are predicted matching (Thank|special)
    And   7 prompt tokens are processed
    # Loading the original cache into slot 0,
    # we should only be processing 1 prompt token and get the same output
    When  the slot 0 is restored with filename "slot1.bin"
    Then  the server responds with status code 200
    Given a user prompt "What is the capital of France?"
    And   using slot id 0
    And   a completion request with no api error
    Then  24 tokens are predicted matching (Lily|cake)
    And   1 prompt tokens are processed
    # For verification that slot 1 was not corrupted during slot 0 load, same thing
    Given a user prompt "What is the capital of Germany?"
    And   using slot id 1
    And   a completion request with no api error
    Then  24 tokens are predicted matching (Thank|special)
    And   1 prompt tokens are processed

  Scenario: Erase Slot
    Given a user prompt "What is the capital of France?"
    And   using slot id 1
    And   a completion request with no api error
    Then  24 tokens are predicted matching (Lily|cake)
    And   22 prompt tokens are processed
    When  the slot 1 is erased
    Then  the server responds with status code 200
    Given a user prompt "What is the capital of France?"
    And   a completion request with no api error
    Then  24 tokens are predicted matching (Lily|cake)
    And   22 prompt tokens are processed
