# llama.cpp/example/server-parallel

This example demonstrates a PoC HTTP API server that handles simulataneus requests.

## Quick Start

To get started right away, run the following command, making sure to use the correct path for the model you have:

### Unix-based systems (Linux, macOS, etc.):

```bash
./server-parallel -m models/7B/ggml-model.gguf --ctx_size 2048 -t 4 -ngl 33 --batch-size 512 --parallel 3 -n 512 --cont-batching --reverse-prompt "User:"
```

### Windows:

```powershell
server-parallel.exe -m models\7B\ggml-model.gguf --ctx_size 2048 -t 4 -ngl 33 --batch-size 512 --parallel 3 -n 512 --cont-batching --reverse-prompt "User:"
```
The above command will start a server that by default listens on `127.0.0.1:8080`.
You can consume the endpoints with Postman or NodeJS with axios library. You can visit the web front end at the same url.

# This example is a Proof of Concept, have bugs and unexpected behaivors
