# llama.cpp/example/simple-token-healing

This example extends [simple](../simple/README.md) with [token healing](https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb).

Without token healing:
```bash
./simple ./models/phi-2/ggml-model-q4_0.gguf "print('Hel"
...
main: n_len = 32, n_ctx = 2048, n_kv_req = 32

print('Helping the customer')
...
```

Heal the last token (`1`):
```bash
./simple-token-healing ./models/phi-2/ggml-model-q4_0.gguf "print('Hel" 1
...
token_healing: prefix = 'Hel' (1 tokens)
 [ 12621] 'Hel'
 [ 15496] 'Hello'
 [ 22087] 'Help'
 [ 28254] 'Hell'
 [ 47429] 'Helper'

main: n_len = 32, n_ctx = 2048, n_kv_req = 32

print('Hello, World!')
...
```

Backtrack multiple tokens until there doesn't exist a token which can cover the prompt's suffix (`n`):
```bash
./simple-token-healing ./models/phi-2/ggml-model-q4_0.gguf "print('Hello, worl" n
...
token_healing: prefix = ' worl' (2 tokens)
 [   995] ' world'
 [  8688] ' worldwide'
 [ 11621] ' worlds'
 [ 29081] ' worldview'
 [ 43249] ' worldly'

main: n_len = 32, n_ctx = 2048, n_kv_req = 32

print('Hello, world!')
...
```

Backtrack multiple tokens but don't constrain the decoding to a single token (`m`):
```bash
./simple-token-healing ./models/phi-2/ggml-model-q4_0.gguf "print('Hello, worl" m
...
token_healing: prefix = ' worl' (2 tokens)

main: n_len = 32, n_ctx = 2048, n_kv_req = 32

print('Hello,
token_healing: prefix = ' worl'
 [   220] ' '
 [   266] ' w'
 [   476] ' wor'
 [   995] ' world'
 [  8688] ' worldwide'
 [ 11621] ' worlds'
 [ 24486] ' wo'
 [ 29081] ' worldview'
 [ 43249] ' worldly'
 world!')
...
```
