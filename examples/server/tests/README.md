# Server Integration Test

Functional server tests suite.

### Install dependencies
`pip install -r requirements.txt`

### Run tests
1. Build the server
2. download a GGUF model: `../../../scripts/hf.sh --repo ngxson/dummy-llama --file llama_xs_q4.bin`
3. Start the test: `./tests.sh tinyllama-2-1b-miniguanaco.Q2_K.gguf -ngl 23 --log-disable`
