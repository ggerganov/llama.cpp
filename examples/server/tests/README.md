# Server Integration Test

Server tests scenario using [BDD](https://en.wikipedia.org/wiki/Behavior-driven_development) with [behave](https://behave.readthedocs.io/en/latest/).

### Install dependencies
`pip install -r requirements.txt`

### Run tests
1. Build the server
2. download required models:
   1. `../../../scripts/hf.sh --repo ggml-org/models --file tinyllamas/stories260K.gguf`
3. Start the test: `./tests.sh`

It's possible to override some scenario steps values with environment variables:
 - `PORT` -> `context.server_port` to set the listening port of the server during scenario, default: `8080`
 - `LLAMA_SERVER_BIN_PATH` -> to change the server binary path, default: `../../../build/bin/server`
 - `DEBUG` -> "ON" to enable server verbose mode `--verbose`   

### Run @bug, @wip or @wrong_usage annotated scenario

Feature or Scenario must be annotated with `@llama.cpp` to be included in the default scope.
- `@bug` annotation aims to link a scenario with a GitHub issue.
- `@wrong_usage` are meant to show user issue that are actually an expected behavior
- `@wip` to focus on a scenario working in progress

To run a scenario annotated with `@bug`, start:
`DEBUG=ON ./tests.sh --no-skipped --tags bug`