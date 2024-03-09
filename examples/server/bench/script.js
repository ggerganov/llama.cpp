import http from 'k6/http'
import {check, sleep} from 'k6'
import {SharedArray} from 'k6/data'
import {Counter, Gauge, Rate} from 'k6/metrics'

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

export function setup() {
    console.info(`Benchmark config: server_url=${server_url} n_prompt=${n_prompt} model=${model} dataset_path=${dataset_path} max_tokens=${max_tokens}`)
}

const data = new SharedArray('conversations', function () {
    return JSON.parse(open(dataset_path))
        // Filter out the conversations with less than 2 turns.
        .filter(data => data["conversations"].length >= 2)
        // Only keep the first two turns of each conversation.
        .map(data => Array(data["conversations"][0]["value"], data["conversations"][1]["value"]))
        // Keep only first n prompts
        .slice(0, n_prompt)
})

const llamacpp_prompt_tokens = new Gauge('llamacpp_prompt_tokens')
const llamacpp_completion_tokens = new Gauge('llamacpp_completion_tokens')

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
    const conversation = data[Math.floor(Math.random() * data.length)]
    const payload = {
        "messages": [
            {
                "role": "system",
                "content": conversation[0],
            },
            {
                "role": "user",
                "content": conversation[1],
            }
        ],
        "model": model,
        "stream": false,
        "max_tokens": max_tokens
    }

    const body = JSON.stringify(payload)

    console.debug(`request: ${body}`)

    let res = http.post(`${server_url}/chat/completions`, body, {
        headers: {'Content-Type': 'application/json'},
        timeout: '300s'
    })

    check(res, {'success completion': (r) => r.status === 200})

    if (res.status === 200) {
        console.debug(`response: ${res.body}`)

        const completions = res.json()

        llamacpp_prompt_tokens.add(completions.usage.prompt_tokens)
        llamacpp_prompt_tokens_total_counter.add(completions.usage.prompt_tokens)

        llamacpp_completion_tokens.add(completions.usage.completion_tokens)
        llamacpp_completion_tokens_total_counter.add(completions.usage.completion_tokens)

        llamacpp_completions_truncated_rate.add(completions.choices[0].finish_reason === 'length')
        llamacpp_completions_stop_rate.add(completions.choices[0].finish_reason === 'stop')
    } else {
        console.error(`response: ${res.body}`)
    }

    sleep(0.3)
}
