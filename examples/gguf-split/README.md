## GGUF split Example

CLI to split / merge GGUF files.

**Command line options:**

- `--split`: split GGUF to multiple GGUF, default operation.
- `--split-max-tensors`: maximum tensors in each split: default(128)
- `--merge`: merge multiple GGUF to a single GGUF.


### Build command
Windows: `g++ -o gguf-split gguf-split.cpp -I ..\..\..\llama.cpp\ -I ..\..\common ..\..\ggml.c ..\..\llama.cpp`
