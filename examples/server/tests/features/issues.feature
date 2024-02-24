# List of ongoing issues
@bug
Feature: Issues
    # Issue #5655
  Scenario: Multi users embeddings
    Given a server listening on localhost:8080
    And   a model file stories260K.gguf
    And   a model alias tinyllama-2
    And   42 as server seed
    And   64 KV cache size
    And   2 slots
    And   continuous batching
    And   embeddings extraction
    Then  the server is starting
    Then  the server is healthy

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
    Given concurrent embedding requests
    Then the server is busy
    Then the server is idle
    Then all embeddings are generated
