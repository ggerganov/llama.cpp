# llama.cpp/example/batched

The example demonstrates batched generation from a given prompt

```bash
./batched ./models/llama-7b-v2/llama-2-7b-chat.Q8_0.gguf "Hello my name is" 4

...

main: n_len = 32, n_ctx = 128, n_batch = 32, n_parallel = 4, n_kv_req = 113, n_threads = 16, n_threads_batch = 16

<s> Hello my name is

main: generating 4 sequences ...

main: stream 0 finished at n_cur = 32
main: stream 1 finished at n_cur = 32
main: stream 2 finished at n_cur = 32
main: stream 3 finished at n_cur = 32

sequence 0:

Hello my name is [Your Name], and I am a [Your Profession] with [Number of Years] of experience in the [Your Industry

sequence 1:

Hello my name is Drew and I am a 31 year old man from the United States. I have been a fan of anime for as

sequence 2:

Hello my name is Tiffany and I am a 30 year old female. I have been experiencing some symptoms that I am concerned about

sequence 3:

Hello my name is John and I am a 26 year old man from the United States. I have been experiencing some strange symptoms over the

main: decoded 108 tokens in 4.19 s, speed: 25.76 t/s

llama_print_timings:        load time =     549.60 ms
llama_print_timings:      sample time =       4.14 ms /   112 runs   (    0.04 ms per token, 27027.03 tokens per second)
llama_print_timings: prompt eval time =    4333.64 ms /   113 tokens (   38.35 ms per token,    26.08 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =    4742.24 ms /   114 tokens
```
