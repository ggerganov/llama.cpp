# llama.cpp/example/simple-token-healing

This example extends [simple](../simple/README.md) with token healing (aka. token alignment).

`usage: ./simple-token-healing MODEL_PATH [PROMPT] [TOKEN_HEALING 0|1|d1|d|r[N]]`

## Examples
`0`: Without token healing (same as running `./simple ...`):
```bash
./simple-token-healing ./models/phi-2/ggml-model-q4_0.gguf "print('Hel" 0
...
main: n_len = 32, n_ctx = 2048, n_kv_req = 32

print('Helping the customer')
...
```

`1`: Roll back the last token and constrain the bytes of the next token to start with the chopped off last token [0, 2]:
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

`d1`: Roll back multiple tokens until there doesn't exist a token which can cover the prompt's suffix and do a single constrained decoding step [2]:
```bash
./simple-token-healing ./models/phi-2/ggml-model-q4_0.gguf "print('Hello, worl" d1
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

`d`: Roll back multiple tokens until there doesn't exist a token which can cover the prompt's suffix but allow multiple decoding steps:
```bash
./simple-token-healing ./models/phi-2/ggml-model-q4_0.gguf "print('Hello, worl" d
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

`r[N]`: Roll back `N` tokens and constrain the decoding to the bytes of those tokens (multiple decoding steps) [1].
The paper [1] recommends `N=3`:
```bash
./simple-token-healing ./models/phi-2/ggml-model-q4_0.gguf "print('Hello, worl" r3
...
token_healing: prefix = ', worl' (3 tokens)

main: n_len = 32, n_ctx = 2048, n_kv_req = 32

print('Hello
token_healing: prefix = ', worl'
 [    11] ','
,
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

## Sources
- [0] https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb 
- [1] https://arxiv.org/abs/2403.08688
- [2] https://arxiv.org/abs/2402.01035
