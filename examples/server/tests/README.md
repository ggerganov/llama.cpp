# Server Integration Test

Functional server tests suite.

### Install dependencies
`pip install -r requirements.txt`

### Run tests
1. Build the server
2. download a GGUF model: `./scripts/hf.sh --repo ggml-org/models --file tinyllamas/stories260K.gguf`
3. Start the test: `./tests.sh stories260K.gguf -ngl 23`
