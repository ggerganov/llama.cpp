import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import { Counter, Gauge, Rate } from 'k6/metrics';

const data = new SharedArray('conversations', function () {
    return JSON.parse(open('./ShareGPT_V3_unfiltered_cleaned_split.json'))

        // Filter out the conversations with less than 2 turns.
        .filter(data => data["conversations"].length >= 2)
        // Only keep the first two turns of each conversation.
        .map(data => Array(data["conversations"][0]["value"], data["conversations"][1]["value"]));
});

const llamacpp_prompt_tokens = new Gauge('llamacpp_prompt_tokens');
const llamacpp_completion_tokens = new Gauge('llamacpp_completion_tokens');

const llamacpp_completions_tokens_seconds = new Gauge('llamacpp_completions_tokens_seconds');

const llamacpp_prompt_tokens_total_counter = new Counter('llamacpp_prompt_tokens_total_counter');
const llamacpp_completion_tokens_total_counter = new Counter('llamacpp_completion_tokens_total_counter');

const llamacpp_completions_truncated_rate = new Rate('llamacpp_completions_truncated_rate');
const llamacpp_completions_stop_rate = new Rate('llamacpp_completions_stop_rate');

export const options = {
    thresholds: {
        llamacpp_completions_truncated_rate: [
            // more than 10% of truncated input will abort the test
            { threshold: 'rate < 0.1', abortOnFail: true, delayAbortEval: '1m' },
        ],
    },
    scenarios: {
        completions: {
            executor: 'ramping-vus',
            startVUs: 1,
            stages: [
                {duration: '1m', target: 8},
                {duration: '3m', target: 8},
                {duration: '1m', target: 0},
            ],
            gracefulRampDown: '30s',
        },
    },
};

export default function () {
    const conversation = data[0]
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
        "model": "model",
        "stream": false,
    }
    let res = http.post('http://localhost:8080/v1/chat/completions', JSON.stringify(payload), {
        headers: { 'Content-Type': 'application/json' },
    })

    check(res, {'success completion': (r) => r.status === 200})

    const completions = res.json()

    llamacpp_prompt_tokens.add(completions.usage.prompt_tokens)
    llamacpp_prompt_tokens_total_counter.add(completions.usage.prompt_tokens)

    llamacpp_completion_tokens.add(completions.usage.completion_tokens)
    llamacpp_completion_tokens_total_counter.add(completions.usage.completion_tokens)

    llamacpp_completions_tokens_seconds.add(completions.usage.completion_tokens / res.timings.duration * 1e3)

    llamacpp_completions_truncated_rate.add(completions.choices[0].finish_reason === 'length')
    llamacpp_completions_stop_rate.add(completions.choices[0].finish_reason === 'stop')


    sleep(0.3)
}