# llama.cpp/example/server-parallel

This example demonstrates a PoC HTTP API server that handles simulataneus requests. Long prompts are not supported.

## Quick Start

To get started right away, run the following command, making sure to use the correct path for the model you have:

### Unix-based systems (Linux, macOS, etc.):

```bash
./server-parallel -m models/7B/ggml-model.gguf --ctx_size 2048 -t 4 -ngl 33 --batch-size 512 --parallel 3 -n 512 --cont-batching
```

### Windows:

```powershell
server-parallel.exe -m models\7B\ggml-model.gguf --ctx_size 2048 -t 4 -ngl 33 --batch-size 512 --parallel 3 -n 512 --cont-batching
```
The above command will start a server that by default listens on `127.0.0.1:8080`.

## API Endpoints

-   **GET** `/props`: Return the user and assistant name for generate the prompt.

*Response:*
```json
{
    "user_name": "User:",
    "assistant_name": "Assistant:"
}
```

-   **POST** `/completion`: Given a prompt, it returns the predicted completion, just streaming mode.

    *Options:*

    `temperature`: Adjust the randomness of the generated text (default: 0.1).

    `prompt`: Provide a prompt as a string, It should be a coherent continuation of the system prompt.

    `system_prompt`: Provide a system prompt as a string.

    `anti_prompt`: Provide the name of the user coherent with the system prompt.

    `assistant_name`: Provide the name of the assistant coherent with the system prompt.

*Example request:*
```json
{
    // this changes the system prompt on runtime
    "system_prompt": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

Human: Hello
Assistant: Hi, how may I help you?
Human:",
    "anti_prompt": "Human:",
    "assistant_name": "Assistant:",

    // required options
    "prompt": "When is the day of independency of US?",
    "temperature": 0.2
}
```

*Response:*
```json
{
    "content": "<token_str>"
}
```

# This example is a Proof of Concept, have some bugs and unexpected behaivors, this not supports long prompts.
