# llama.cpp/examples/ggml-debug

A simple example which demonstrates how to use callback during the inference.
It simply prints to the console all operations and tensor data.

Usage:

```shell
ggml-debug \
  --hf-repo ggml-org/models \
  --hf-file phi-2/ggml-model-q4_0.gguf \
  --model phi-2-q4_0.gguf \
  --prompt hello \
  --seed 42 \
  -ngl 33
```

Will print:

```shell
llm_load_tensors: offloaded 33/33 layers to GPU
...
llama_new_context_with_model: n_ctx      = 512
...
llama_new_context_with_model:      CUDA0 compute buffer size =   105.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =     6.01 MiB
llama_new_context_with_model: graph nodes  = 1225
llama_new_context_with_model: graph splits = 2

system_info: n_threads = 6 / 12 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | 
ggml_debug:                 inp_embd =   GET_ROWS(token_embd.weight{2560, 51200, 1, 1}, inp_tokens{1, 1, 1, 1}}) = {2560, 1, 1, 1} 
                                     [
                                      [
                                       [ -0.0181,  -0.0181,   0.0453, ...],
                                      ],
                                     ]
ggml_debug:                   norm-0 =       NORM(CUDA0#inp_embd#0{2560, 1, 1, 1}, }) = {2560, 1, 1, 1} 
                                     [
                                      [
                                       [ -0.6989,  -0.6989,   1.7686, ...],
                                      ],
                                     ]
ggml_debug:                 norm_w-0 =        MUL(norm-0{2560, 1, 1, 1}, blk.0.attn_norm.weight{2560, 1, 1, 1}}) = {2560, 1, 1, 1} 
                                     [
                                      [
                                       [ -0.1800,  -0.1788,   0.4663, ...],
                                      ],
                                     ]
ggml_debug:              attn_norm-0 =        ADD(norm_w-0{2560, 1, 1, 1}, blk.0.attn_norm.bias{2560, 1, 1, 1}}) = {2560, 1, 1, 1} 
                                     [
                                      [
                                       [ -0.1863,  -0.1712,   0.4750, ...],
                                      ],
                                     ]
ggml_debug:                   wqkv-0 =    MUL_MAT(blk.0.attn_qkv.weight{2560, 7680, 1, 1}, attn_norm-0{2560, 1, 1, 1}}) = {7680, 1, 1, 1} 
                                     [
                                      [
                                       [ -1.1238,  -2.3523,  -1.6938, ...],
                                      ],
                                     ]
ggml_debug:                   bqkv-0 =        ADD(wqkv-0{7680, 1, 1, 1}, blk.0.attn_qkv.bias{7680, 1, 1, 1}}) = {7680, 1, 1, 1} 
                                     [
                                      [
                                       [ -1.1135,  -2.5451,  -1.8321, ...],
                                      ],
                                     ]
ggml_debug:            bqkv-0 (view) =       VIEW(bqkv-0{7680, 1, 1, 1}, }) = {2560, 1, 1, 1} 
                                     [
                                      [
                                       [ -1.1135,  -2.5451,  -1.8321, ...],
                                      ],
                                     ]
ggml_debug:                   Qcur-0 =       CONT(bqkv-0 (view){2560, 1, 1, 1}, }) = {2560, 1, 1, 1} 
                                     [
                                      [
                                       [ -1.1135,  -2.5451,  -1.8321, ...],
                                      ],
                                     ]
ggml_debug:        Qcur-0 (reshaped) =    RESHAPE(Qcur-0{2560, 1, 1, 1}, }) = {80, 32, 1, 1} 
                                     [
                                      [
                                       [ -1.1135,   0.8348,   0.8010, ...],
                                       [ -2.5451,  -1.1920,   0.0546, ...],
                                       [ -1.8321,  -0.0515,   0.8186, ...],
                                       ...
                                      ],
                                     ]
ggml_debug:                   Qcur-0 =       ROPE(Qcur-0 (reshaped){80, 32, 1, 1}, CUDA0#inp_pos#0{1, 1, 1, 1}}) = {80, 32, 1, 1} 
                                     [
                                      [
                                       [ -1.1135,   0.8348,   0.8010, ...],
                                       [ -2.5451,  -1.1920,   0.0546, ...],
                                       [ -1.8321,  -0.0515,   0.8186, ...],
                                       ...
                                      ],
                                     ]
ggml_debug:                   Qcur-0 =      SCALE(Qcur-0{80, 32, 1, 1}, }) = {80, 32, 1, 1} 
                                     [
                                      [
                                       [ -0.1245,   0.0933,   0.0896, ...],
                                       [ -0.2845,  -0.1333,   0.0061, ...],
                                       [ -0.2048,  -0.0058,   0.0915, ...],
                                       ...
                                      ],
                                     ]
```
