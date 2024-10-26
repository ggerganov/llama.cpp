# Migration notice for binary filenames

> [!IMPORTANT]
[2024 Jun 12] Binaries have been renamed w/ a `jarvis-` prefix. `main` is now `jarvis-cli`, `server` is `jarvis-server`, etc (https://github.com/ggerganov/jarvis.cpp/pull/7809)

This migration was important, but it is a breaking change that may not always be immediately obvious to users.

Please update all scripts and workflows to use the new binary names.

| Old Filename | New Filename |
| ---- | ---- |
| main | jarvis-cli |
| server | jarvis-server |
| jarvis-bench | jarvis-bench |
| embedding | jarvis-embedding |
| quantize | jarvis-quantize |
| tokenize | jarvis-tokenize |
| export-lora | jarvis-export-lora |
| libllava.a | libllava.a |
| baby-jarvis | jarvis-baby-jarvis |
| batched | jarvis-batched |
| batched-bench | jarvis-batched-bench |
| benchmark-matmult | jarvis-benchmark-matmult |
| convert-jarvis2c-to-ggml | jarvis-convert-jarvis2c-to-ggml |
| eval-callback | jarvis-eval-callback |
| gbnf-validator | jarvis-gbnf-validator |
| gguf | jarvis-gguf |
| gguf-split | jarvis-gguf-split |
| gritlm | jarvis-gritlm |
| imatrix | jarvis-imatrix |
| infill | jarvis-infill |
| llava-cli | jarvis-llava-cli |
| lookahead | jarvis-lookahead |
| lookup | jarvis-lookup |
| lookup-create | jarvis-lookup-create |
| lookup-merge | jarvis-lookup-merge |
| lookup-stats | jarvis-lookup-stats |
| parallel | jarvis-parallel |
| passkey | jarvis-passkey |
| perplexity | jarvis-perplexity |
| q8dot | jarvis-q8dot |
| quantize-stats | jarvis-quantize-stats |
| retrieval | jarvis-retrieval |
| save-load-state | jarvis-save-load-state |
| simple | jarvis-simple |
| speculative | jarvis-speculative |
| vdot | jarvis-vdot |
| tests/test-c.o | tests/test-c.o |

