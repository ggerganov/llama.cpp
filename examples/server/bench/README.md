### Server benchmark tools

Benchmark is using [k6](https://k6.io/).

##### Install k6

Follow instruction from: https://k6.io/docs/get-started/installation/

Example for ubuntu:
```shell
snap install k6
```

#### Download a dataset

This dataset was originally proposed in [vLLM benchmarks](https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md).

```shell
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

#### Download a model
Example for PHI-2

```shell
../../../scripts/hf.sh --repo ggml-org/models --file phi-2/ggml-model-q4_0.gguf
```

#### Start the server
The server must answer OAI Chat completion requests on `http://localhost:8080/v1` or according to the environment variable `SERVER_BENCH_URL`.

Example:
```shell
server --host localhost --port 8080 \
  --model ggml-model-q4_0.gguf \
  --cont-batching \
  --metrics \
  --parallel 8 \
  --batch-size 512 \
  --ctx-size 4096 \
  --log-format text \
  -ngl 33
```

#### Run the benchmark

```shell
k6 run script.js
```

The benchmark values can be overridden with:
- `SERVER_BENCH_URL` server url prefix for chat completions, default `http://localhost:8080/v1`
- `SERVER_BENCH_N_PROMPTS` total prompts to randomly select in the benchmark, default `480`
- `SERVER_BENCH_MODEL_ALIAS` model alias to pass in the completion request, default `my-model`

Or with [k6 options](https://k6.io/docs/using-k6/k6-options/reference/):

```shell
SERVER_BENCH_N_PROMPTS=500 k6 run script.js --duration 10m --iterations 500 --vus 8
```

#### Metrics

Following metrics are available:
- `llamacpp_prompt_tokens` Gauge of OAI response `usage.prompt_tokens`
- `llamacpp_prompt_tokens_total_counter` Counter of OAI response `usage.prompt_tokens`
- `llamacpp_completion_tokens` Gauge of OAI response `usage.completion_tokens`
- `llamacpp_completion_tokens_total_counter` Counter of OAI response `usage.completion_tokens`
- `llamacpp_completions_tokens_seconds` Gauge of `usage.completion_tokens` divided by the request time in second
- `llamacpp_completions_truncated_rate` Rate of completions truncated, i.e. if `finish_reason === 'length'`
- `llamacpp_completions_stop_rate` Rate of completions truncated, i.e. if `finish_reason === 'stop'`

The script will fail if too many completions are truncated, see `llamacpp_completions_truncated_rate`.

K6 metrics might be compared against [server metrics](../README.md), with:

```shell
curl http://localhost:8080/metrics
```
