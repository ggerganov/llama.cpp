# llama.cpp

Inference of [Facebook's LLaMA](https://github.com/facebookresearch/llama) model in pure C/C++

## Description

The main goal is to run the model using 4-bit quantization on a MacBook.

- Plain C/C++ implementation without dependencies
- Apple silicon first-class citizen - optimized via Arm Neon and Accelerate framework
- Mixed F16 / F32 precision
- 4-bit quantization support
- Runs on the CPU

This was hacked in an evening - I have no idea if it works correctly.

So far, I've tested just the 7B model and the generated text starts coherently, but typically degrades significanlty after ~30-40 tokens.
Here is a "typical" run:

```java
make -j && ./main -m ./models/7B/ggml-model-q4_0.bin -t 8 -n 128
I llama.cpp build info: 
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:   -framework Accelerate
I CC:       Apple clang version 14.0.0 (clang-1400.0.29.202)
I CXX:      Apple clang version 14.0.0 (clang-1400.0.29.202)

c++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread main.cpp ggml.o utils.o -o main  -framework Accelerate
./main -h
usage: ./main [options]

options:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  RNG seed (default: -1)
  -t N, --threads N     number of threads to use during computation (default: 4)
  -p PROMPT, --prompt PROMPT
                        prompt to start generation with (default: random)
  -n N, --n_predict N   number of tokens to predict (default: 128)
  --top_k N             top-k sampling (default: 40)
  --top_p N             top-p sampling (default: 0.9)
  --temp N              temperature (default: 0.8)
  -b N, --batch_size N  batch size for prompt processing (default: 8)
  -m FNAME, --model FNAME
                        model path (default: models/llama-7B/ggml-model.bin)

main: seed = 1678476633
llama_model_load: loading model from './models/7B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 64
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: ggml ctx size = 4529.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: .................................... done
llama_model_load: model size =  4017.27 MB / num tensors = 291

main: prompt: 'If'
main: number of tokens in prompt = 2
     1 -> ''
  3644 -> 'If'

sampling parameters: temp = 0.800000, top_k = 40, top_p = 0.950000


If you are a fan of the original Star Wars trilogy, then you'll want to see this.
If you don't know your Star Wars lore, this will be a huge eye-opening and you will be a little confusing.
Awesome movie. [end of text]


main: mem per token = 14434244 bytes
main:     load time =  1313.77 ms
main:   sample time =     6.17 ms
main:  predict time =  3271.53 ms / 54.53 ms per token
main:    total time =  4797.98 ms
```

## Usage

```bash
# build this repo
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# obtain the original LLaMA model weights and place them in ./models
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model

# convert the 7B model to ggml FP16 format
python3 convert-pth-to-ggml.py models/7B/ 1

# quantize the model to 4-bits
./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin 2

# run the inference
./main -m ./models/7B/ggml-model-q4_0.bin -t 8 -n 128
```

## Limitations

- Currently, only LLaMA-7B is supported since I haven't figured out how to merge the tensors of the bigger models. However, in theory, you should be able to run 65B on a 64GB MacBook
- Not sure if my tokenizer is correct. There are a few places where we might have a mistake:
  - https://github.com/ggerganov/llama.cpp/blob/26c084662903ddaca19bef982831bfb0856e8257/convert-pth-to-ggml.py#L79-L87
  - https://github.com/ggerganov/llama.cpp/blob/26c084662903ddaca19bef982831bfb0856e8257/utils.h#L65-L69
  In general, it seems to work, but I think it fails for unicode character support. Hopefully, someone can help with that
- I don't know yet how much the quantization affects the quality of the generated text
- Probably the token sampling can be improved
- x86 quantization support [not yet ready](https://github.com/ggerganov/ggml/pull/27). Basically, you want to run this on Apple Silicon

