# llama.cpp

This repo is cloned from llama.cpp [commit 74d73dc85cc2057446bf63cc37ff649ae7cebd80](https://github.com/ggerganov/llama.cpp/tree/74d73dc85cc2057446bf63cc37ff649ae7cebd80). It is compatible with llama-cpp-python [commit 7ecdd944624cbd49e4af0a5ce1aa402607d58dcc](https://github.com/abetlen/llama-cpp-python/commit/7ecdd944624cbd49e4af0a5ce1aa402607d58dcc)

## Customize quantization group size at compilation (CPU inference only)

The only thing that is different is to add -DQK4_0 flag when cmake.

```bash
cmake -B build_cpu_g128 -DQK4_0=128
cmake --build build_cpu_g128
```

To quantize the model with the customized group size, run

```bash
./build_cpu_g128/bin/llama-quantize <model_path.gguf> <quantization_type>
```

To run the quantized model, run

```bash
./build_cpu_g128/bin/llama-cli -m <quantized_model_path.gguf>
```

### Note:

You should make sure that the model you run is quantized to the same group size as the one you compile with.
Or you'll receive a runtime error when loading the model.
