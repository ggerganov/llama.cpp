# llama.cpp/example/server

This example allow you to have a llama.cpp http server to interact from a web page or consume the API.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Node JS Test](#node-js-test)
3. [API Endpoints](#api-endpoints)
4. [Common Options](#common-options)
5. [Performance Tuning and Memory Options](#performance-tuning-and-memory-options)

## Quick Start

To get started right away, run the following command, making sure to use the correct path for the model you have:

#### Unix-based systems (Linux, macOS, etc.):

```bash
./server -m models/7B/ggml-model.bin --keep -1 --ctx_size 2048
```

#### Windows:

```powershell
server.exe -m models\7B\ggml-model.bin --keep -1 --ctx_size 2048
```

That will start a server that by default listens on `127.0.0.1:8080`. You can consume the endpoints with Postman or NodeJS with axios library.

## Node JS Test

You need to have [Node.js](https://nodejs.org/en) installed.

```bash
mkdir llama-client
cd llama-client
npm init
npm install axios
```

Create a index.js file and put inside this:

```javascript
const axios = require('axios');

async function LLamaTest() {
    let result = await axios.post("http://127.0.0.1:8080/setting-context", {
        context: [
            { role: "system", content: "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions." },
            { role: "user", content: "Hello, Assistant." },
            { role: "assistant", content: "Hello. How may I help you today?" },
            { role: "user", content: "Please tell me the largest city in Europe." },
            { role: "assistant", content: "Sure. The largest city in Europe is Moscow, the capital of Russia." }
        ],
        batch_size: 64,
        temperature: 0.2,
        top_k: 40,
        top_p: 0.9,
        n_predict: 2048,
        threads: 5
    });
    result = await axios.post("http://127.0.0.1:8080/set-message", {
        message: ' What is linux?'
    });
    if(result.data.can_inference) {
        result = await axios.get("http://127.0.0.1:8080/completion?stream=true", { responseType: 'stream' });
        result.data.on('data', (data) => {
            let completion = JSON.parse(data.toString());
            // token by token completion like Chat GPT
            process.stdout.write(completion.content);
        });

        /*
        Wait the entire completion (takes long time for response)

        result = await axios.get("http://127.0.0.1:8080/completion");
        console.log(result.data.content);
        */
    }
}

LLamaTest();
```

And run it:

```bash
node .
```

## API Endpoints

You can interact with this API Endpoints. This implementations just support chat style interaction.

-    `POST hostname:port/setting-context`: Setting up the Llama Context to begin the completions tasks.

Options:
`batch_size`: Set the batch size for prompt processing (default: 512).

`temperature`: Adjust the randomness of the generated text (default: 0.8).

`top_k`: Limit the next token selection to the K most probable tokens (default: 40).

`top_p`: Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P (default: 0.9).

`n_predict`: Set the number of tokens to predict when generating text (default: 128, -1 = infinity).

`threads`: Set the number of threads to use during computation.

`context`: Set a short conversation as context.

Insert items to an array of this form: `{ role: "user", content: "Hello, Assistant." }`, where:

`role` can be `system`, `assistant` and `user`.

`content` the message content.

-   `POST hostname:port/set-message`: Set the message of the user to Llama.

`message`: Set the message content.

-   `GET hostname:port/completion`: Receive the response, it can be a stream or wait until finish the completion.

`stream`: Set `true` if you want to receive a stream response.

## Common Options

-   `-m FNAME, --model FNAME`: Specify the path to the LLaMA model file (e.g., `models/7B/ggml-model.bin`).
-   `-c N, --ctx_size N`: Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048, which will provide better results for longer input/inference.
-   `--host`: Set the hostname or ip address to listen. Default `127.0.0.1`;

-   `--port`: Set the port to listen. Default: `8080`.

### Keep Prompt

The `--keep` option allows users to retain the original prompt when the model runs out of context, ensuring a connection to the initial instruction or conversation topic is maintained.

-   `--keep N`: Specify the number of tokens from the initial prompt to retain when the model resets its internal context. By default, this value is set to 0 (meaning no tokens are kept). Use `-1` to retain all tokens from the initial prompt.

By utilizing context management options like `--ctx_size` and `--keep`, you can maintain a more coherent and consistent interaction with the LLaMA models, ensuring that the generated text remains relevant to the original prompt or conversation.

### RNG Seed

-   `-s SEED, --seed SEED`: Set the random number generator (RNG) seed (default: -1, < 0 = random seed).

The RNG seed is used to initialize the random number generator that influences the text generation process. By setting a specific seed value, you can obtain consistent and reproducible results across multiple runs with the same input and settings. This can be helpful for testing, debugging, or comparing the effects of different options on the generated text to see when they diverge. If the seed is set to a value less than 0, a random seed will be used, which will result in different outputs on each run.

## Performance Tuning and Memory Options

### No Memory Mapping

-   `--no-mmap`: Do not memory-map the model. By default, models are mapped into memory, which allows the system to load only the necessary parts of the model as needed. However, if the model is larger than your total amount of RAM or if your system is low on available memory, using mmap might increase the risk of pageouts, negatively impacting performance. Disabling mmap results in slower load times but may reduce pageouts if you're not using `--mlock`. Note that if the model is larger than the total amount of RAM, turning off mmap would prevent the model from loading at all.

### Memory Float 32

-   `--memory_f32`: Use 32-bit floats instead of 16-bit floats for memory key+value, allowing higher quality inference at the cost of higher memory usage.

## Limitations:
* The actual implementation of llama.cpp need a `llama-state` for support multiple contexts and clients.
* The context can't be reset during runtime.
