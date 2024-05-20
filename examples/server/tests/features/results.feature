@llama.cpp
@results
Feature: Results

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file tinyllamas/split/stories15M-00001-of-00003.gguf from HF repo ggml-org/models
    And   a model file test-model-00001-of-00003.gguf
    And   128 as batch size
    And   1024 KV cache size
    And   128 max tokens to predict
    And   continuous batching

  Scenario Outline: consistent results with same seed
    Given <n_slots> slots
    And   1.0 temperature
    Then  the server is starting
    Then  the server is healthy

    Given 4 prompts "Title: Little Red Riding Hood But In Space\n\nSummary:" with seed 42

    Given concurrent completion requests
    Then the server is busy
    Then the server is idle
    And  all slots are idle
    Then all predictions are equal
    Examples:
      | n_slots |
      | 1       |
      # FIXME: unified KV cache nondeterminism
      # | 2       |

  Scenario Outline: different results with different seed
    Given <n_slots> slots
    And   1.0 temperature
    Then  the server is starting
    Then  the server is healthy

    Given 1 prompts "Title: Little Red Riding Hood But In Space\n\nSummary:" with seed 42
    Given 1 prompts "Title: Little Red Riding Hood But In Space\n\nSummary:" with seed 43
    Given 1 prompts "Title: Little Red Riding Hood But In Space\n\nSummary:" with seed 44
    Given 1 prompts "Title: Little Red Riding Hood But In Space\n\nSummary:" with seed 45

    Given concurrent completion requests
    Then the server is busy
    Then the server is idle
    And  all slots are idle
    Then all predictions are different
    Examples:
      | n_slots |
      | 1       |
      | 2       |

  Scenario Outline: consistent results with same seed and varying batch size
    Given 4 slots
    And   <temp> temperature
    # And   0 as draft
    Then  the server is starting
    Then  the server is healthy

    Given 1 prompts "Write a very long story about AI." with seed 42
    And   concurrent completion requests
    # Then the server is busy # Not all slots will be utilized.
    Then  the server is idle
    And   all slots are idle

    Given <n_parallel> prompts "Write a very long story about AI." with seed 42
    And   concurrent completion requests
    # Then the server is busy # Not all slots will be utilized.
    Then the server is idle
    And  all slots are idle

    Then all predictions are equal
    Examples:
      | n_parallel | temp |
      | 1          | 0.0  |
      | 1          | 1.0  |
      # FIXME: unified KV cache nondeterminism
      # See https://github.com/ggerganov/whisper.cpp/issues/1941#issuecomment-1986923227
      # and https://github.com/ggerganov/llama.cpp/pull/6122#discussion_r1531405574
      # and https://github.com/ggerganov/llama.cpp/pull/7347 .
      # | 2          | 0.0  |
      # | 4          | 0.0  |
      # | 2          | 1.0  |
      # | 4          | 1.0  |

  Scenario Outline: consistent token probs with same seed and prompt
    Given <n_slots> slots
    And   <n_kv> KV cache size
    And   1.0 temperature
    And   <n_predict> max tokens to predict
    Then  the server is starting
    Then  the server is healthy

    Given 1 prompts "The meaning of life is" with seed 42
    And   concurrent completion requests
    # Then the server is busy # Not all slots will be utilized.
    Then  the server is idle
    And   all slots are idle

    Given <n_parallel> prompts "The meaning of life is" with seed 42
    And   concurrent completion requests
    # Then the server is busy # Not all slots will be utilized.
    Then the server is idle
    And  all slots are idle

    Then all token probabilities are equal
    Examples:
      | n_slots | n_kv | n_predict | n_parallel |
      | 4       | 1024 | 1         | 1          |
      # FIXME: unified KV cache nondeterminism
      # See https://github.com/ggerganov/whisper.cpp/issues/1941#issuecomment-1986923227
      # and https://github.com/ggerganov/llama.cpp/pull/6122#discussion_r1531405574
      # and https://github.com/ggerganov/llama.cpp/pull/7347 .
      # | 4       | 1024 | 1         | 4          |
      # | 4       | 1024 | 100       | 1          |
      # This test still fails even the above patches; the first token probabilities are already different.
      # | 4       | 1024 | 100       | 4          |
