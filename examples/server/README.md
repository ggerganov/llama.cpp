# llama.cpp/example/server

This example allow you to have a llama.cpp http server to interact from a web page or consume the API.

Command line options:

-   `--threads N`, `-t N`: use N threads.
-   `-m FNAME`, `--model FNAME`: Specify the path to the LLaMA model file (e.g., `models/7B/ggml-model.bin`).
-   `-c N`, `--ctx-size N`: Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048, which will provide better results for longer input/inference.
-   `-ngl N`, `--n-gpu-layers N`: When compiled with appropriate support (currently CLBlast or cuBLAS), this option allows offloading some layers to the GPU for computation. Generally results in increased performance.
-   `--embedding`: Enable the embedding mode. **Completion function doesn't work in this mode**.
-   `--host`: Set the hostname or ip address to listen. Default `127.0.0.1`;
-   `--port`: Set the port to listen. Default: `8080`.


## Quick Start

To get started right away, run the following command, making sure to use the correct path for the model you have:

### Unix-based systems (Linux, macOS, etc.):

```bash
./server -m models/7B/ggml-model.bin -c 2048
```

### Windows:

```powershell
server.exe -m models\7B\ggml-model.bin -c 2048
```

That will start a server that by default listens on `127.0.0.1:8080`.
You can consume the endpoints with Postman or NodeJS with axios library.

## Testing with CURL

Using [curl](https://curl.se/). On Windows `curl.exe` should be available in the base OS.

```sh
curl --request POST \
    --url http://localhost:8080/completion \
    --data '{"prompt": "Building a website can be done in 10 simple steps:","n_predict": 128}'
```

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
const axios = require("axios");

const prompt = `Building a website can be done in 10 simple steps:`;

async function Test() {
    let result = await axios.post("http://127.0.0.1:8080/completion", {
        prompt,
        n_predict: 512,
    });

    // the response is received until completion finish
    console.log(result.data.content);
}

Test();
```

And run it:

```bash
node .
```

## API Endpoints

You can interact with this API Endpoints.
This implementations just support chat style interaction.

-   **POST** `hostname:port/completion`: Setting up the Llama Context to begin the completions tasks.

    *Options:*

    `temperature`: Adjust the randomness of the generated text (default: 0.8).

    `top_k`: Limit the next token selection to the K most probable tokens (default: 40).

    `top_p`: Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P (default: 0.9).

    `n_predict`: Set the number of tokens to predict when generating text (default: 128, -1 = infinity).

    `n_keep`: Specify the number of tokens from the initial prompt to retain when the model resets its internal context.
    By default, this value is set to 0 (meaning no tokens are kept). Use `-1` to retain all tokens from the initial prompt.

    `stream`: It allows receiving each predicted token in real-time instead of waiting for the completion to finish. To enable this, set to `true`.

    `prompt`: Provide a prompt. Internally, the prompt is compared, and it detects if a part has already been evaluated, and the remaining part will be evaluate.

    `stop`: Specify the strings that indicate a stop.
    These words will not be included in the completion, so make sure to add them to the prompt for the next iteration.
    Default: `[]`

-   **POST** `hostname:port/tokenize`: Tokenize a given text

    *Options:*

    `content`: Set the text to tokenize.

## More examples

### Interactive mode

Check the sample in [chat.mjs](chat.mjs).
Run with node:

```sh
node chat.mjs
```
