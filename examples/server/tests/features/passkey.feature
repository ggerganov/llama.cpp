#@llama.cpp
@passkey
@wip
@slow
@bug
Feature: Passkey / Self-extend with context shift

  Background: Server startup
    Given a server listening on localhost:8080

  # Generates a long text of junk and inserts a secret passkey number inside it.
  # We process the entire prompt using batches of n_batch and shifting the cache
  # when it is full and then we query the LLM for the secret passkey.
  # see #3856 and #4810
  Scenario Outline: Passkey
    Given a model file <hf_file> from HF repo <hf_repo>
    And   <n_batch> as batch size
    And   <n_junk> as number of junk
    And   <n_predicted> server max tokens to predict
    And   a self-extend context with a factor of <n_grp>
    And   <seed> as seed
    And   a KV cache size based on the model trained context <n_ctx_train> extended by <n_grp> with additional <n_keep> tokens
    And   <n_slots> slots
    And   <n_ga> group attention factor to extend context size through self-extend
    And   <n_ga_w> group attention width to extend context size through self-extend
    # Can be override with N_GPU_LAYERS
    And   <ngl> GPU offloaded layers
    Then  the server is starting
    Then  the server is healthy
    Given available models
    Then  model 0 is trained on <n_ctx_train> tokens context
    Given a prefix prompt:
    """
    here is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.
    """
    And a passkey prompt template:
    """
    The pass key is <passkey> Remember it. <passkey> is the pass key.
    """
    And a junk suffix prompt:
    """
    The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
    """
    And a suffix prompt:
    """
    What is the pass key? The pass key is
    """
    Given a "<passkey>" passkey challenge prompt with the passkey inserted every <i_pos> junk
    And  a completion request with no api error
    Then <n_predicted> tokens are predicted matching <re_content>

    Examples:
      | hf_repo             | hf_file           | n_ctx_train | ngl | n_batch | n_slots | n_ga | n_ga_w | n_junk | n_grp | i_pos | seed | n_keep | passkey | n_predicted | re_content          |
      | TheBloke/phi-2-GGUF | phi-2.Q4_K_M.gguf | 2048        | 5   | 512     | 1       | 8    | 512    | 250    | 4     | 50    | 86   | 32     | 42      | 5           | The passkey is 42\. |
