import http from 'k6/http'
import {check, sleep} from 'k6'
import {SharedArray} from 'k6/data'
import {Counter, Rate, Trend} from 'k6/metrics'
import exec from 'k6/execution';

// Server chat completions prefix
const server_url = __ENV.SERVER_BENCH_URL ? __ENV.SERVER_BENCH_URL : 'http://localhost:8080/v1'

// Number of total prompts in the dataset - default 10m / 10 seconds/request * number of users
const n_prompt = __ENV.SERVER_BENCH_N_PROMPTS ? parseInt(__ENV.SERVER_BENCH_N_PROMPTS) : 600 / 10 * 8

// Model name to request
const model = __ENV.SERVER_BENCH_MODEL_ALIAS ? __ENV.SERVER_BENCH_MODEL_ALIAS : 'my-model'

// Dataset path
const dataset_path = __ENV.SERVER_BENCH_DATASET ? __ENV.SERVER_BENCH_DATASET : './ShareGPT_V3_unfiltered_cleaned_split.json'

// Max tokens to predict
const max_tokens = __ENV.SERVER_BENCH_MAX_TOKENS ? parseInt(__ENV.SERVER_BENCH_MAX_TOKENS) : 512

// Max prompt tokens
const n_prompt_tokens = __ENV.SERVER_BENCH_MAX_PROMPT_TOKENS ? parseInt(__ENV.SERVER_BENCH_MAX_PROMPT_TOKENS) : 1024

// Max slot context
const n_ctx_slot = __ENV.SERVER_BENCH_MAX_CONTEXT ? parseInt(__ENV.SERVER_BENCH_MAX_CONTEXT) : 2048

export function setup() {
    console.info(`Benchmark config: server_url=${server_url} n_prompt=${n_prompt} model=${model} dataset_path=${dataset_path} max_tokens=${max_tokens}`)
}

const data = new SharedArray('conversations', function () {
    const tokenizer = (message) => message.split(/[\s,'".?]/)

    return JSON.parse(open(dataset_path))
        // Filter out the conversations with less than 2 turns.
        .filter(data => data["conversations"].length >= 2)
        .filter(data => data["conversations"][0]["from"] === "human")
        .map(data => {
            return {
                prompt: data["conversations"][0]["value"],
                n_prompt_tokens: tokenizer(data["conversations"][0]["value"]).length,
                n_completion_tokens: tokenizer(data["conversations"][1]["value"]).length,
            }
        })
        // Filter out too short sequences
        .filter(conv => conv.n_prompt_tokens >= 4 && conv.n_completion_tokens >= 4)
        // Filter out too long sequences.
        .filter(conv => conv.n_prompt_tokens <= n_prompt_tokens && conv.n_prompt_tokens + conv.n_completion_tokens <= n_ctx_slot)
        // Keep only first n prompts
        .slice(0, n_prompt)
})

const llamacpp_prompt_tokens = new Trend('llamacpp_prompt_tokens')
const llamacpp_completion_tokens = new Trend('llamacpp_completion_tokens')
const llamacpp_tokens_second = new Trend('llamacpp_tokens_second')

const llamacpp_prompt_tokens_total_counter = new Counter('llamacpp_prompt_tokens_total_counter')
const llamacpp_completion_tokens_total_counter = new Counter('llamacpp_completion_tokens_total_counter')

const llamacpp_completions_truncated_rate = new Rate('llamacpp_completions_truncated_rate')
const llamacpp_completions_stop_rate = new Rate('llamacpp_completions_stop_rate')

export const options = {
    thresholds: {
        llamacpp_completions_truncated_rate: [
            // more than 80% of truncated input will abort the test
            {threshold: 'rate < 0.8', abortOnFail: true, delayAbortEval: '1m'},
        ],
    },
    duration: '10m',
    vus: 8,
}

export default function () {
    const conversation = data[exec.scenario.iterationInInstance % data.length]
    const payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are ChatGPT, an AI assistant.",
            },
            {
                "role": "user",
                "content": conversation.prompt,
            }
        ],
        "model": model,
        "stream": false,
        "max_tokens": max_tokens
    }

    const body = JSON.stringify(payload)

    let res = http.post(`${server_url}/chat/completions`, body, {
        headers: {'Content-Type': 'application/json'},
        timeout: '300s'
    })

    check(res, {'success completion': (r) => r.status === 200})

    if (res.status === 200) {
        const completions = res.json()

        llamacpp_prompt_tokens.add(completions.usage.prompt_tokens)
        llamacpp_prompt_tokens_total_counter.add(completions.usage.prompt_tokens)

        llamacpp_completion_tokens.add(completions.usage.completion_tokens)
        llamacpp_completion_tokens_total_counter.add(completions.usage.completion_tokens)

        llamacpp_completions_truncated_rate.add(completions.choices[0].finish_reason === 'length')
        llamacpp_completions_stop_rate.add(completions.choices[0].finish_reason === 'stop')

        llamacpp_tokens_second.add(completions.usage.total_tokens / res.timings.duration * 1.e3)
    } else {
        console.error(`response: ${res.body} request=${payload}`)
    }

    sleep(0.3)
}
