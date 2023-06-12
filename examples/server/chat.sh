#!/bin/bash

API_URL="http://127.0.0.1:8080"

CHAT=(
    "Hello, Assistant."
    "Hello. How may I help you today?"
    "Please tell me the largest city in Europe."
    "Sure. The largest city in Europe is Moscow, the capital of Russia."
)

INSTRUCTION="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

format_prompt() {
    echo -n "${INSTRUCTION}"
    printf "\n### Human: %s\n### Assistant: %s" "${CHAT[@]}" "$1"
}

tokenize() {
    echo -n "$1" | jq -Rs '{content:.}' | curl \
        --silent \
        --request POST \
        --url "${API_URL}/tokenize" \
        --data "@-" | jq '.tokens[]'
}

N_KEEP=$(tokenize "${INSTRUCTION}" | wc -l)

chat_completion() {
    CONTENT=$(format_prompt "$1" | jq -Rs --argjson n_keep $N_KEEP '{
        prompt: .,
        temperature: 0.2,
        top_k: 40,
        top_p: 0.9,
        n_keep: $n_keep,
        n_predict: 256,
        stop: ["\n### Human:"]
    }' | curl \
        --silent \
        --request POST \
        --url "${API_URL}/completion" \
        --data "@-" | jq -r '.content | sub("^\\s*"; "")')

    printf "$CONTENT\n"

    CHAT+=("$1" "$CONTENT")
}

while true; do
    read -p "> " QUESTION
    chat_completion "${QUESTION}"
done
