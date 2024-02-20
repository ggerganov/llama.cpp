# Server Integration Test

Server tests scenario using [BDD](https://en.wikipedia.org/wiki/Behavior-driven_development) with [behave](https://behave.readthedocs.io/en/latest/).

### Install dependencies
`pip install -r requirements.txt`

### Run tests
1. Build the server
2. download a GGUF model: `./scripts/hf.sh --repo ggml-org/models --file tinyllamas/stories260K.gguf`
3. Start the test: `./tests.sh stories260K.gguf -ngl 23`

### Skipped scenario

Scenario must be annotated with `@llama.cpp` to be included in the scope.
`@bug` annotation aims to link a scenario with a GitHub issue.
