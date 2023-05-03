## llama.cpp/example/server

This example allow you to have a llama.cpp http server to interact from a web page or consume the API.

It doesn't require external dependencies.

## Limitations:
* Just tested in Windows and Linux
* Only CMake build.
* Only one context at a time.
* Just vicuna support for interaction.

## Endpoints

You can interact with this API Endpoints.

`POST hostname:port/setting-context`

`POST hostname:port/set-message`

`GET hostname:port/completion`

## Usage
### Get Code
```bash
git clone https://github.com/FSSRepo/llama.cpp.git
cd llama.cpp
```
### Build
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
### Run
Model tested: [Vicuna](https://huggingface.co/chharlesonfire/ggml-vicuna-7b-4bit/blob/main/ggml-vicuna-7b-q4_0.bin)
```bash
server -m ggml-vicuna-7b-q4_0.bin --keep -1 --ctx_size 2048
```

### Node JS Test the endpoints

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

async function Test() {
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
            // token by token completion
            let dat = JSON.parse(data.toString());
            process.stdout.write(dat.content);
        });

        /*
        Wait the entire completion (takes long time for response)

        result = await axios.get("http://127.0.0.1:8080/completion");
        console.log(result.data.content);
        */
    }
}

Test();
```

And run it:

```bash
node .
```