# llama.cpp/example/server

This example demonstrates a simple HTTP API server and a simple web front end to interact with llama.cpp.

Command line options:

-   `--threads N`, `-t N`: Set the number of threads to use during computation.
-   `-m FNAME`, `--model FNAME`: Specify the path to the LLaMA model file (e.g., `models/7B/ggml-model.bin`).
-   `-m ALIAS`, `--alias ALIAS`: Set an alias for the model. The alias will be returned in API responses.
-   `-c N`, `--ctx-size N`: Set the size of the prompt context. The default is 512, but LLaMA models were built with a context of 2048, which will provide better results for longer input/inference. The size may differ in other models, for example, baichuan models were build with a context of 4096.
-   `-ngl N`, `--n-gpu-layers N`: When compiled with appropriate support (currently CLBlast or cuBLAS), this option allows offloading some layers to the GPU for computation. Generally results in increased performance.
-   `-mg i, --main-gpu i`: When using multiple GPUs this option controls which GPU is used for small tensors for which the overhead of splitting the computation across all GPUs is not worthwhile. The GPU in question will use slightly more VRAM to store a scratch buffer for temporary results. By default GPU 0 is used. Requires cuBLAS.
-   `-ts SPLIT, --tensor-split SPLIT`: When using multiple GPUs this option controls how large tensors should be split across all GPUs. `SPLIT` is a comma-separated list of non-negative values that assigns the proportion of data that each GPU should get in order. For example, "3,2" will assign 60% of the data to GPU 0 and 40% to GPU 1. By default the data is split in proportion to VRAM but this may not be optimal for performance. Requires cuBLAS.
-   `-lv, --low-vram`: Do not allocate a VRAM scratch buffer for holding temporary results. Reduces VRAM usage at the cost of performance, particularly prompt processing speed. Requires cuBLAS.
-   `-b N`, `--batch-size N`: Set the batch size for prompt processing. Default: `512`.
-   `--memory-f32`: Use 32-bit floats instead of 16-bit floats for memory key+value. Not recommended.
-   `--mlock`: Lock the model in memory, preventing it from being swapped out when memory-mapped.
-   `--no-mmap`: Do not memory-map the model. By default, models are mapped into memory, which allows the system to load only the necessary parts of the model as needed.
-   `--lora FNAME`: Apply a LoRA (Low-Rank Adaptation) adapter to the model (implies --no-mmap). This allows you to adapt the pretrained model to specific tasks or domains.
-   `--lora-base FNAME`: Optional model to use as a base for the layers modified by the LoRA adapter. This flag is used in conjunction with the `--lora` flag, and specifies the base model for the adaptation.
-   `-to N`, `--timeout N`: Server read/write timeout in seconds. Default `600`.
-   `--host`: Set the hostname or ip address to listen. Default `127.0.0.1`.
-   `--port`: Set the port to listen. Default: `8080`.
-   `--path`: path from which to serve static files (default examples/server/public)
-   `--embedding`: Enable embedding extraction, Default: disabled.

## Build

server is build alongside everything else from the root of the project

- Using `make`:

  ```bash
  make
  ```

- Using `CMake`:

  ```bash
  cmake --build . --config Release
  ```

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

The above command will start a server that by default listens on `127.0.0.1:8080`.
You can consume the endpoints with Postman or NodeJS with axios library. You can visit the web front end at the same url.

## Testing with CURL

Using [curl](https://curl.se/). On Windows `curl.exe` should be available in the base OS.

```sh
curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
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

-   **POST** `/completion`: Given a prompt, it returns the predicted completion.

    *Options:*

    `temperature`: Adjust the randomness of the generated text (default: 0.8).

    `top_k`: Limit the next token selection to the K most probable tokens (default: 40).

    `top_p`: Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P (default: 0.9).

    `n_predict`: Set the number of tokens to predict when generating text. **Note:** May exceed the set limit slightly if the last token is a partial multibyte character. When 0, no tokens will be generated but the prompt is evaluated into the cache. (default: 128, -1 = infinity).

    `n_keep`: Specify the number of tokens from the initial prompt to retain when the model resets its internal context.
    By default, this value is set to 0 (meaning no tokens are kept). Use `-1` to retain all tokens from the initial prompt.

    `stream`: It allows receiving each predicted token in real-time instead of waiting for the completion to finish. To enable this, set to `true`.

    `prompt`: Provide a prompt. Internally, the prompt is compared, and it detects if a part has already been evaluated, and the remaining part will be evaluate. A space is inserted in the front like main.cpp does.

    `stop`: Specify a JSON array of stopping strings.
    These words will not be included in the completion, so make sure to add them to the prompt for the next iteration (default: []).

    `tfs_z`: Enable tail free sampling with parameter z (default: 1.0, 1.0 = disabled).

    `typical_p`: Enable locally typical sampling with parameter p (default: 1.0, 1.0 = disabled).

    `repeat_penalty`: Control the repetition of token sequences in the generated text (default: 1.1).

    `repeat_last_n`: Last n tokens to consider for penalizing repetition (default: 64, 0 = disabled, -1 = ctx-size).

    `penalize_nl`: Penalize newline tokens when applying the repeat penalty (default: true).

    `presence_penalty`: Repeat alpha presence penalty (default: 0.0, 0.0 = disabled).

    `frequency_penalty`: Repeat alpha frequency penalty (default: 0.0, 0.0 = disabled);

    `mirostat`: Enable Mirostat sampling, controlling perplexity during text generation (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).

    `mirostat_tau`: Set the Mirostat target entropy, parameter tau (default: 5.0).

    `mirostat_eta`: Set the Mirostat learning rate, parameter eta (default: 0.1).

    `grammar`: Set grammar for grammar-based sampling (default: no grammar)

    `seed`: Set the random number generator (RNG) seed (default: -1, -1 = random seed).

    `ignore_eos`: Ignore end of stream token and continue generating (default: false).

    `logit_bias`: Modify the likelihood of a token appearing in the generated text completion. For example, use `"logit_bias": [[15043,1.0]]` to increase the likelihood of the token 'Hello', or `"logit_bias": [[15043,-1.0]]` to decrease its likelihood. Setting the value to false, `"logit_bias": [[15043,false]]` ensures that the token `Hello` is never produced (default: []).

-   **POST** `/tokenize`: Tokenize a given text.

    *Options:*

    `content`: Set the text to tokenize.

    Note that the special `BOS` token is not added in front of the text and also a space character is not inserted automatically as it is for `/completion`.

-   **POST** `/embedding`: Generate embedding of a given text just as [the embedding example](../embedding) does.

    *Options:*

    `content`: Set the text to process.

## More examples

### Interactive mode

Check the sample in [chat.mjs](chat.mjs).
Run with NodeJS version 16 or later:

```sh
node chat.mjs
```

Another sample in [chat.sh](chat.sh).
Requires [bash](https://www.gnu.org/software/bash/), [curl](https://curl.se) and [jq](https://jqlang.github.io/jq/).
Run with bash:

```sh
bash chat.sh
```

### API like OAI

API example using Python Flask: [api_like_OAI.py](api_like_OAI.py)
This example must be used with server.cpp

```sh
python api_like_OAI.py
```

After running the API server, you can use it in Python by setting the API base URL.
```python
openai.api_base = "http://<Your api-server IP>:port"
```

Then you can utilize llama.cpp as an OpenAI's **chat.completion** or **text_completion** API

### Extending or building alternative Web Front End

The default location for the static files is `examples/server/public`. You can extend the front end by running the server binary with `--path` set to `./your-directory` and importing `/completion.js` to get access to the llamaComplete() method.

Read the documentation in `/completion.js` to see convenient ways to access llama.

A simple example is below:

```html
<html>
  <body>
    <pre>
      <script type="module">
        import { llama } from '/completion.js'

        const prompt = `### Instruction:
Write dad jokes, each one paragraph.
You can use html formatting if needed.

### Response:`

        for await (const chunk of llama(prompt)) {
          document.write(chunk.data.content)
        }
      </script>
    </pre>
  </body>
</html>
```
