# Chatbot UI with local LLMs

This example makes use of three projects [llama.cpp](https://github.com/ggerganov/llama.cpp), [gpt-llama.cpp](https://github.com/keldenl/gpt-llama.cpp) and [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui) to provide a ChatGPT UI like experience with llama.cpp.

## How to use
1. Edit the volume bind in `compose.yaml` with the path to the mode you wish to use

        volumes:
          - type:   bind
            source: /llm_models/something.ggml.q4_0.bin
            target: /llama.cpp/models/model.bin

1. Start services with `docker-compose`

        docker-compose up --build

1. When updating use the following `docker-compose` command to make sure everything gets updated

        docker-compose up --no-cache --build --pull-always --force-recreate
