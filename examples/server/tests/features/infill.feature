@llama.cpp
@infill
Feature: llama.cpp server

  # The current model is made by adding FIM tokens to the existing stories260K
  # We may want to use a better model in the future, maybe something like SmolLM 360M

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file tinyllamas/stories260K-infill.gguf from HF repo ggml-org/models
    And   a model file test-model-infill.gguf
    And   a model alias tinyllama-infill
    And   42 as server seed
    And   1024 as batch size
    And   1024 as ubatch size
    And   2048 KV cache size
    And   64 max tokens to predict
    And   0.0 temperature
    Then  the server is starting
    Then  the server is healthy

  Scenario: Infill without input_extra
    Given a prompt "Complete this"
    And   an infill input extra none none
    And   an infill input prefix "#include <cstdio>\n#include \"llama.h\"\n\nint main() {\n    int n_threads = llama_"
    And   an infill input suffix "}\n"
    And   an infill request with no api error
    Then  64 tokens are predicted matching One|day|she|saw|big|scary|bird

  Scenario: Infill with input_extra
    Given a prompt "Complete this"
    And   an infill input extra "llama.h" "LLAMA_API int32_t llama_n_threads();\n"
    And   an infill input prefix "#include <cstdio>\n#include \"llama.h\"\n\nint main() {\n    int n_threads = llama_"
    And   an infill input suffix "}\n"
    And   an infill request with no api error
    Then  64 tokens are predicted matching cuts|Jimmy|mom|came|into|the|room"
