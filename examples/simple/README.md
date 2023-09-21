# llama.cpp/example/simple

The purpose of this example is to demonstrate a minimal usage of llama.cpp for generating text with a given prompt.
The example demonstrates single-batch as well as parallel generation.

## Single-batch generation

```bash
./simple ./models/llama-7b-v2/ggml-model-f16.gguf "Hello my name is" 1

...

main: n_len = 32, n_ctx = 2048, n_parallel = 1, n_kv_req = 32

 Hello my name is Shawn and I'm a 20 year old male from the United States. I'm a 20 year old

main: decoded 27 tokens in 2.31 s, speed: 11.68 t/s

llama_print_timings:        load time =   579.15 ms
llama_print_timings:      sample time =     0.72 ms /    28 runs   (    0.03 ms per token, 38888.89 tokens per second)
llama_print_timings: prompt eval time =   655.63 ms /    10 tokens (   65.56 ms per token,    15.25 tokens per second)
llama_print_timings:        eval time =  2180.97 ms /    27 runs   (   80.78 ms per token,    12.38 tokens per second)
llama_print_timings:       total time =  2891.13 ms
```

## Parallel generation

```bash
./simple ./models/llama-7b-v2/ggml-model-f16.gguf "Hello my name is" 4

...

main: n_len = 32, n_ctx = 2048, n_parallel = 4, n_kv_req = 113

 Hello my name is

main: generating 4 sequences ...

main: stream 0 finished
main: stream 1 finished
main: stream 2 finished
main: stream 3 finished

sequence 0:

Hello my name is Shirley. I am a 25-year-old female who has been working for over 5 years as a b

sequence 1:

Hello my name is Renee and I'm a 32 year old female from the United States. I'm looking for a man between

sequence 2:

Hello my name is Diana. I am looking for a housekeeping job. I have experience with children and have my own transportation. I am

sequence 3:

Hello my name is Cody. I am a 3 year old neutered male. I am a very friendly cat. I am very playful and

main: decoded 108 tokens in 3.57 s, speed: 30.26 t/s

llama_print_timings:        load time =   587.00 ms
llama_print_timings:      sample time =     2.56 ms /   112 runs   (    0.02 ms per token, 43664.72 tokens per second)
llama_print_timings: prompt eval time =  4089.11 ms /   118 tokens (   34.65 ms per token,    28.86 tokens per second)
llama_print_timings:        eval time =     0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =  4156.04 ms
```
