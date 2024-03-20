@llama.cpp
@embeddings
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model url https://huggingface.co/ggml-org/models/resolve/main/bert-bge-small/ggml-model-f16.gguf
    And   a model file ggml-model-f16.gguf
    And   a model alias bert-bge-small
    And   42 as server seed
    And   2 slots
    And   1024 as batch size
    And   1024 as ubatch size
    And   2048 KV cache size
    And   embeddings extraction
    Then  the server is starting
    Then  the server is healthy

  Scenario: Embedding
    When embeddings are computed for:
    """
    What is the capital of Bulgaria ?
    """
    Then embeddings are generated

  Scenario: OAI Embeddings compatibility
    Given a model bert-bge-small
    When an OAI compatible embeddings computation request for:
    """
    What is the capital of Spain ?
    """
    Then embeddings are generated

  Scenario: OAI Embeddings compatibility with multiple inputs
    Given a model bert-bge-small
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

  Scenario: Multi users embeddings
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

  Scenario: Multi users OAI compatibility embeddings
    Given a prompt:
      """
      In which country Paris is located ?
      """
    And a prompt:
      """
      Is Madrid the capital of Spain ?
      """
    And a prompt:
      """
      What is the biggest US city ?
      """
    And a prompt:
      """
      What is the capital of Bulgaria ?
      """
    And   a model bert-bge-small
    Given concurrent OAI embedding requests
    Then the server is busy
    Then the server is idle
    Then all embeddings are generated

  Scenario: All embeddings should be the same
    Given 10 fixed prompts
    And   a model bert-bge-small
    Given concurrent OAI embedding requests
    Then all embeddings are the same
