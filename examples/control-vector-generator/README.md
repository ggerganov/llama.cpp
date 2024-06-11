# control-vector-generator

This example demonstrates how to generate a control vector using gguf models.

Related PRs:
- [Add support for control vectors](https://github.com/ggerganov/llama.cpp/pull/5970)
- (Issue) [Generate control vector using llama.cpp](https://github.com/ggerganov/llama.cpp/issues/6880)
- [Add control-vector-generator](https://github.com/ggerganov/llama.cpp/pull/7514)

Example:

```sh
# CPU only
./control-vector-generator -m ./dolphin-2.0-mistral-7b.Q4_K_M.gguf

# With GPU
./control-vector-generator --num-completions 2 --pca-iter 40 -m ./dolphin-2.0-mistral-7b.Q4_K_M.gguf -ngl 99

# With advanced options
# Please note that the ORDER of arguments does matter
# example-related options (i.e., --num-completions, --pca-iter) always come before model options (i.e., -m, -ngl)
./control-vector-generator --num-completions 128 --pca-iter 2000 --batch-pca 100 -m ./dolphin-2.0-mistral-7b.Q4_K_M.gguf -ngl 99

# To see help message
./control-vector-generator -h
```
