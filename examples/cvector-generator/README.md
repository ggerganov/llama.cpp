# cvector-generator

This example demonstrates how to generate a control vector using gguf models.

Related PRs:
- [Add support for control vectors](https://github.com/ggerganov/llama.cpp/pull/5970)
- (Issue) [Generate control vector using llama.cpp](https://github.com/ggerganov/llama.cpp/issues/6880)
- [Add cvector-generator example](https://github.com/ggerganov/llama.cpp/pull/7514)

## Examples

```sh
# CPU only
./cvector-generator -m ./dolphin-2.0-mistral-7b.Q4_K_M.gguf

# With GPU
./cvector-generator -m ./dolphin-2.0-mistral-7b.Q4_K_M.gguf -ngl 99

# With advanced options
./cvector-generator -m ./dolphin-2.0-mistral-7b.Q4_K_M.gguf -ngl 99 --completions 128 --pca-iter 2000 --pca-batch 100

# To see help message
./cvector-generator -h
# Then, have a look at "cvector" section
```

## Tips and tricks

If you have multiple lines per prompt, you can escape the newline character (change it to `\n`). For example:

```
<|im_start|>system\nAct like a person who is extremely happy.<|im_end|>
<|im_start|>system\nYou are in a very good mood today<|im_end|>
```
