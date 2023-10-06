# llama.cpp/example/server-parallel

This example demonstrates a PoC HTTP API server that handles simulataneus requests. Long prompts are not supported.

Command line options:

-   `--threads N`, `-t N`: Set the number of threads to use during generation.
-   `-tb N, --threads-batch N`: Set the number of threads to use during batch and prompt processing. If not specified, the number of threads will be set to the number of threads used for generation.
-   `-m FNAME`, `--model FNAME`: Specify the path to the LLaMA model file (e.g., `models/7B/ggml-model.gguf`).
-   `-m ALIAS`, `--alias ALIAS`: Set an alias for the model. The alias will be returned in API responses.
-   `-c N`, `--ctx-size N`: Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048, which will provide better results for longer input/inference. The size may differ in other models, for example, baichuan models were build with a context of 4096.
-   `-ngl N`, `--n-gpu-layers N`: When compiled with appropriate support (currently CLBlast or cuBLAS), this option allows offloading some layers to the GPU for computation. Generally results in increased performance.
-   `-mg i, --main-gpu i`: When using multiple GPUs this option controls which GPU is used for small tensors for which the overhead of splitting the computation across all GPUs is not worthwhile. The GPU in question will use slightly more VRAM to store a scratch buffer for temporary results. By default GPU 0 is used. Requires cuBLAS.
-   `-ts SPLIT, --tensor-split SPLIT`: When using multiple GPUs this option controls how large tensors should be split across all GPUs. `SPLIT` is a comma-separated list of non-negative values that assigns the proportion of data that each GPU should get in order. For example, "3,2" will assign 60% of the data to GPU 0 and 40% to GPU 1. By default the data is split in proportion to VRAM but this may not be optimal for performance. Requires cuBLAS.
-   `-b N`, `--batch-size N`: Set the batch size for prompt processing. Default: `512`.
-   `--memory-f32`: Use 32-bit floats instead of 16-bit floats for memory key+value. Not recommended.
-   `--mlock`: Lock the model in memory, preventing it from being swapped out when memory-mapped.
-   `--no-mmap`: Do not memory-map the model. By default, models are mapped into memory, which allows the system to load only the necessary parts of the model as needed.
-   `--numa`: Attempt optimizations that help on some NUMA systems.
-   `--lora FNAME`: Apply a LoRA (Low-Rank Adaptation) adapter to the model (implies --no-mmap). This allows you to adapt the pretrained model to specific tasks or domains.
-   `--lora-base FNAME`: Optional model to use as a base for the layers modified by the LoRA adapter. This flag is used in conjunction with the `--lora` flag, and specifies the base model for the adaptation.
-   `-to N`, `--timeout N`: Server read/write timeout in seconds. Default `600`.
-   `--host`: Set the hostname or ip address to listen. Default `127.0.0.1`.
-   `--port`: Set the port to listen. Default: `8080`.
-   `--path`: path from which to serve static files (default examples/server/public)
-   `-np N`, `--parallel N`: Set the number of slots for process requests (default: 1)
-   `-cb`, `--cont-batching`: enable continuous batching (a.k.a dynamic batching) (default: disabled)
-   `-r ANTI_PROMPT`, `--reverse-prompt ANTI_PROMPT`: Set a anti prompt, used as user name in prompt generation

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
    "system_prompt": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nHuman: Hello\nAssistant: Hi, how may I help you?\nHuman:",
    "anti_prompt": "Human:",
    "assistant_name": "Assistant:",
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
