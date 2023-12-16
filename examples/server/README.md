# llama.cpp/example/server

This example demonstrates a simple HTTP API server and a simple web front end to interact with llama.cpp.

Command line options:

-   `--threads N`, `-t N`: Set the number of threads to use during generation.
-   `-tb N, --threads-batch N`: Set the number of threads to use during batch and prompt processing. If not specified, the number of threads will be set to the number of threads used for generation.
-   `-m FNAME`, `--model FNAME`: Specify the path to the LLaMA model file (e.g., `models/7B/ggml-model.gguf`).
-   `-a ALIAS`, `--alias ALIAS`: Set an alias for the model. The alias will be returned in API responses.
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
-   `--embedding`: Enable embedding extraction, Default: disabled.
-   `-np N`, `--parallel N`: Set the number of slots for process requests (default: 1)
-   `-cb`, `--cont-batching`: enable continuous batching (a.k.a dynamic batching) (default: disabled)
-   `-spf FNAME`, `--system-prompt-file FNAME` Set a file to load "a system prompt (initial prompt of all slots), this is useful for chat applications. [See more](#change-system-prompt-on-runtime)
-   `--mmproj MMPROJ_FILE`: Path to a multimodal projector file for LLaVA.

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
./server -m models/7B/ggml-model.gguf -c 2048
```

### Windows:

```powershell
server.exe -m models\7B\ggml-model.gguf -c 2048
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
```

Create a index.js file and put inside this:

```javascript
const prompt = `Building a website can be done in 10 simple steps:`;

async function Test() {
    let response = await fetch("http://127.0.0.1:8080/completion", {
        method: 'POST',
        body: JSON.stringify({
            prompt,
            n_predict: 512,
        })
    })
    console.log((await response.json()).content)
}

Test()
```

And run it:

```bash
node index.js
```

## API Endpoints

-   **POST** `/completion`: Given a `prompt`, it returns the predicted completion.

    *Options:*

    `prompt`: Provide the prompt for this completion as a string or as an array of strings or numbers representing tokens. Internally, the prompt is compared to the previous completion and only the "unseen" suffix is evaluated. If the prompt is a string or an array with the first element given as a string, a `bos` token is inserted in the front like `main` does.

    `temperature`: Adjust the randomness of the generated text (default: 0.8).

    `top_k`: Limit the next token selection to the K most probable tokens (default: 40).

    `top_p`: Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P (default: 0.95).

    `min_p`: The minimum probability for a token to be considered, relative to the probability of the most likely token (default: 0.05).

    `n_predict`: Set the maximum number of tokens to predict when generating text. **Note:** May exceed the set limit slightly if the last token is a partial multibyte character. When 0, no tokens will be generated but the prompt is evaluated into the cache. (default: -1, -1 = infinity).

    `n_keep`: Specify the number of tokens from the prompt to retain when the context size is exceeded and tokens need to be discarded.
    By default, this value is set to 0 (meaning no tokens are kept). Use `-1` to retain all tokens from the prompt.

    `stream`: It allows receiving each predicted token in real-time instead of waiting for the completion to finish. To enable this, set to `true`.

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

    `n_probs`: If greater than 0, the response also contains the probabilities of top N tokens for each generated token (default: 0)

    `image_data`: An array of objects to hold base64-encoded image `data` and its `id`s to be reference in `prompt`. You can determine the place of the image in the prompt as in the following: `USER:[img-12]Describe the image in detail.\nASSISTANT:` In this case, `[img-12]` will be replaced by the embeddings of the image id 12 in the following `image_data` array: `{..., "image_data": [{"data": "<BASE64_STRING>", "id": 12}]}`. Use `image_data` only with multimodal models, e.g., LLaVA.

    *Result JSON:*

    Note: When using streaming mode (`stream`) only `content` and `stop` will be returned until end of completion.

    `content`: Completion result as a string (excluding `stopping_word` if any). In case of streaming mode, will contain the next token as a string.

    `stop`: Boolean for use with `stream` to check whether the generation has stopped (Note: This is not related to stopping words array `stop` from input options)

    `generation_settings`: The provided options above excluding `prompt` but including `n_ctx`, `model`

    `model`: The path to the model loaded with `-m`

    `prompt`: The provided `prompt`

    `stopped_eos`: Indicating whether the completion has stopped because it encountered the EOS token

    `stopped_limit`: Indicating whether the completion stopped because `n_predict` tokens were generated before stop words or EOS was encountered

    `stopped_word`: Indicating whether the completion stopped due to encountering a stopping word from `stop` JSON array provided

    `stopping_word`: The stopping word encountered which stopped the generation (or "" if not stopped due to a stopping word)

    `timings`: Hash of timing information about the completion such as the number of tokens `predicted_per_second`

    `tokens_cached`: Number of tokens from the prompt which could be re-used from previous completion (`n_past`)

    `tokens_evaluated`: Number of tokens evaluated in total from the prompt

    `truncated`: Boolean indicating if the context size was exceeded during generation, i.e. the number of tokens provided in the prompt (`tokens_evaluated`) plus tokens generated (`tokens predicted`) exceeded the context size (`n_ctx`)

    `slot_id`: Assign the completion task to an specific slot. If is -1 the task will be assigned to a Idle slot (default: -1)

    `cache_prompt`: Save the prompt and generation for avoid reprocess entire prompt if a part of this isn't change (default: false)

    `system_prompt`: Change the system prompt (initial prompt of all slots), this is useful for chat applications. [See more](#change-system-prompt-on-runtime)

-   **POST** `/tokenize`: Tokenize a given text.

    *Options:*

    `content`: Set the text to tokenize.

    Note that the special `BOS` token is not added in front of the text and also a space character is not inserted automatically as it is for `/completion`.

-   **POST** `/detokenize`: Convert tokens to text.

    *Options:*

    `tokens`: Set the tokens to detokenize.

-   **POST** `/embedding`: Generate embedding of a given text just as [the embedding example](../embedding) does.

    *Options:*

    `content`: Set the text to process.

-   **POST** `/infill`: For code infilling. Takes a prefix and a suffix and returns the predicted completion as stream.

    *Options:*

    `input_prefix`: Set the prefix of the code to infill.

    `input_suffix`: Set the suffix of the code to infill.

    It also accepts all the options of `/completion` except `stream` and `prompt`.

-   **GET** `/props`: Return the required assistant name and anti-prompt to generate the prompt in case you have specified a system prompt for all slots.

-   **POST** `/v1/chat/completions`: OpenAI-compatible Chat Completions API. Given a ChatML-formatted json description in `messages`, it returns the predicted completion. Both synchronous and streaming mode are supported, so scripted and interactive applications work fine. While no strong claims of compatibility with OpenAI API spec is being made, in our experience it suffices to support many apps. Only ChatML-tuned models, such as Dolphin, OpenOrca, OpenHermes, OpenChat-3.5, etc can be used with this endpoint. Compared to `api_like_OAI.py` this API implementation does not require a wrapper to be served.

    *Options:*

    See [OpenAI Chat Completions API documentation](https://platform.openai.com/docs/api-reference/chat). While some OpenAI-specific features such as function calling aren't supported, llama.cpp `/completion`-specific features such are `mirostat` are supported.

    *Examples:*

    You can use either Python `openai` library with appropriate checkpoints:

    ```python
    import openai

    client = openai.OpenAI(
        base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
        api_key = "sk-no-key-required"
    )

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
        {"role": "user", "content": "Write a limerick about python exceptions"}
    ]
    )

    print(completion.choices[0].message)
    ```
    ... or raw HTTP requests:

    ```shell
    curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer no-key" \
    -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
    {
        "role": "system",
        "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
    },
    {
        "role": "user",
        "content": "Write a limerick about python exceptions"
    }
    ]
    }'
    ```

## More examples

### Change system prompt on runtime

To use the server example to serve multiple chat-type clients while keeping the same system prompt, you can utilize the option `system_prompt` to achieve that. This only needs to be done once to establish it.

`prompt`: Specify a context that you want all connecting clients to respect.

`anti_prompt`: Specify the word you want to use to instruct the model to stop. This must be sent to each client through the `/props` endpoint.

`assistant_name`: The bot's name is necessary for each customer to generate the prompt. This must be sent to each client through the `/props` endpoint.

```json
{
    "system_prompt": {
        "prompt": "Transcript of a never ending dialog, where the User interacts with an Assistant.\nThe Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.\nUser: Recommend a nice restaurant in the area.\nAssistant: I recommend the restaurant \"The Golden Duck\". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.\nUser: Who is Richard Feynman?\nAssistant: Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including \"Surely You're Joking, Mr. Feynman!\" and \"What Do You Care What Other People Think?\".\nUser:",
        "anti_prompt": "User:",
        "assistant_name": "Assistant:"
    }
}
```

**NOTE**: You can do this automatically when starting the server by simply creating a .json file with these options and using the CLI option `-spf FNAME` or `--system-prompt-file FNAME`.

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
