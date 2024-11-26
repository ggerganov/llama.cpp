@llama.cpp
@n_predict
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file test-model.gguf
    And   a model alias tinyllama-2
    And   42 as server seed
    And   64 KV cache size

  Scenario: Generate N tokens
    And   12 max tokens to predict
    Then  the server is starting
    Then  the server is healthy
    Given a prompt:
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    """
    And   a completion request with no api error
    Then  12 tokens are predicted

  Scenario: Generate tokens until context is full
    And   -2 server max tokens to predict
    Then  the server is starting
    Then  the server is healthy
    Given a prompt:
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    """
    And   a completion request with no api error
    Then  11 tokens are predicted
