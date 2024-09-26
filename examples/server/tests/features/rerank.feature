@llama.cpp
@rerank
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model url https://huggingface.co/ggml-org/models/resolve/main/jina-reranker-v1-tiny-en/ggml-model-f16.gguf
    And   a model file jina-reranker-v1-tiny-en.gguf
    And   a model alias jina-reranker-v1-tiny-en
    And   42 as server seed
    And   2 slots
    And   128 as batch size
    And   128 as ubatch size
    And   512 KV cache size
    And   embeddings extraction
    Then  the server is starting
    Then  the server is healthy

# TODO: implement some tests
#       https://github.com/ggerganov/llama.cpp/pull/9510
#  Scenario: Rerank
#    Given a prompt:
#      """
#      What is panda?
#      """
#    And a prompt:
#      """
#      Hi.
#      """
#    And a prompt:
#      """
#      It's a bear.
#      """
#    And a prompt:
#      """
#      The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.
#      """
#    When reranking request
#    Then reranking results are returned
