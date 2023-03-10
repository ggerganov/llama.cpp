# llama.cpp

Inference of [Facebook's LLaMA](https://github.com/facebookresearch/llama) model in pure C/C++

**TEMPORARY NOTICE:**
If you observe garbage results, make sure to update to latest master. There was a bug and it was fixed here: https://github.com/ggerganov/llama.cpp/commit/70bc0b8b15b98dca23b28f0c8f5e34b27e424cda

## Description

The main goal is to run the model using 4-bit quantization on a MacBook.

- Plain C/C++ implementation without dependencies
- Apple silicon first-class citizen - optimized via Arm Neon and Accelerate framework
- Mixed F16 / F32 precision
- 4-bit quantization support
- Runs on the CPU

This was hacked in an evening - I have no idea if it works correctly.

So far, I've tested just the 7B model.
Here is a "typical" run:

```java
make -j && ./main -m ../LLaMA-4bit/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -t 8 -n 512
I llama.cpp build info: 
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:   -framework Accelerate
I CC:       Apple clang version 14.0.0 (clang-1400.0.29.202)
I CXX:      Apple clang version 14.0.0 (clang-1400.0.29.202)

make: Nothing to be done for `default'.
main: seed = 1678486056
llama_model_load: loading model from '../LLaMA-4bit/7B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: ggml ctx size = 4529.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: .................................... done
llama_model_load: model size =  4017.27 MB / num tensors = 291

main: prompt: 'Building a website can be done in 10 simple steps:'
main: number of tokens in prompt = 15
     1 -> ''
  8893 -> 'Build'
   292 -> 'ing'
   263 -> ' a'
  4700 -> ' website'
   508 -> ' can'
   367 -> ' be'
  2309 -> ' done'
   297 -> ' in'
 29871 -> ' '
 29896 -> '1'
 29900 -> '0'
  2560 -> ' simple'
  6576 -> ' steps'
 29901 -> ':'

sampling parameters: temp = 0.800000, top_k = 40, top_p = 0.950000


Building a website can be done in 10 simple steps:
1) Select a domain name and web hosting plan
2) Complete a sitemap
3) List your products
4) Write product descriptions
5) Create a user account
6) Build the template
7) Start building the website
8) Advertise the website
9) Provide email support
10) Submit the website to search engines
A website is a collection of web pages that are formatted with HTML. HTML is the code that defines what the website looks like and how it behaves.
The HTML code is formatted into a template or a format. Once this is done, it is displayed on the user's browser.
The web pages are stored in a web server. The web server is also called a host. When the website is accessed, it is retrieved from the server and displayed on the user's computer.
A website is known as a website when it is hosted. This means that it is displayed on a host. The host is usually a web server.
A website can be displayed on different browsers. The browsers are basically the software that renders the website on the user's screen.
A website can also be viewed on different devices such as desktops, tablets and smartphones.
Hence, to have a website displayed on a browser, the website must be hosted.
A domain name is an address of a website. It is the name of the website.
The website is known as a website when it is hosted. This means that it is displayed on a host. The host is usually a web server.
A website can be displayed on different browsers. The browsers are basically the software that renders the website on the user’s screen.
A website can also be viewed on different devices such as desktops, tablets and smartphones. Hence, to have a website displayed on a browser, the website must be hosted.
A domain name is an address of a website. It is the name of the website.
A website is an address of a website. It is a collection of web pages that are formatted with HTML. HTML is the code that defines what the website looks like and how it behaves.
The HTML code is formatted into a template or a format. Once this is done, it is displayed on the user’s browser.
A website is known as a website when it is hosted

main: mem per token = 14434244 bytes
main:     load time =  1332.48 ms
main:   sample time =  1081.40 ms
main:  predict time = 31378.77 ms / 61.41 ms per token
main:    total time = 34036.74 ms
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

