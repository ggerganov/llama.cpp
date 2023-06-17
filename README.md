llama.cpp modification to run Falcon (work in progress)

Status:  
* Quantization works except for Q_K_ types  
* CUDA not yet functional  


It appears the Q5 Falcon 40B inference time on CPU is as fast as the A100 fp16 inference time at 2 tk/second  
CPU inference examples:  
```
 Q:\ggllm.cpp> .\build\bin\Release\falcon_main.exe -t 31 -m Q:\models\falcon-40b\q5_1 -p "Love relates to hate like" -n 50 -ngl 0
main: build = 677 (dd3d346)
main: seed  = 1687010794
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090
falcon.cpp: loading model from Q:\models\falcon-40b\q5_1
falcon_model_load_internal: format     = ggjt v3 (latest)
falcon_model_load_internal: n_vocab    = 65024
falcon_model_load_internal: n_ctx      = 512
falcon_model_load_internal: n_embd     = 8192
falcon_model_load_internal: n_head     = 128
falcon_model_load_internal: n_head_kv     = 8
falcon_model_load_internal: n_layer    = 60
falcon_model_load_internal: version      = 40
falcon_model_load_internal: ftype      = 9 (mostly Q5_1)
falcon_model_load_internal: n_ff       = 32768
falcon_model_load_internal: n_parts    = 1
falcon_model_load_internal: model size = 40B
falcon_model_load_internal: ggml ctx size =    0.00 MB (mmap size = 29929.00 MB)
falcon_model_load_internal: using CUDA for GPU acceleration
falcon_model_load_internal: mem required  = 33513.70 MB (+  120.00 MB per state)
falcon_model_load_internal: offloading 0 layers to GPU
falcon_model_load_internal: total VRAM used: 512 MB
...................................................................................................
falcon_init_from_file: kv self size  =  120.00 MB

system_info: n_threads = 31 / 32 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | VSX = 0 |
sampling: repeat_last_n = 64, repeat_penalty = 1.100000, presence_penalty = 0.000000, frequency_penalty = 0.000000, top_k = 40, tfs_z = 1.000000, top_p = 0.950000, typical_p = 1.000000, temp = 0.800000, mirostat = 0, mirostat_lr = 0.100000, mirostat_ent = 5.000000
generate: n_ctx = 512, n_batch = 512, n_predict = 50, n_keep = 0


Love relates to hate like light relates to darkness.
Love is the strongest thing in the world, but hate is the second strongest force.
Love is a force multiplier.
For every moment of love, there is a parallel moment of hate.
You can’t
falcon_print_timings:        load time =  4420.23 ms
falcon_print_timings:      sample time =    11.34 ms /    50 runs   (    0.23 ms per token)
falcon_print_timings: prompt eval time =   785.42 ms /     5 tokens (  157.08 ms per token)
falcon_print_timings:        eval time = 27512.23 ms /    49 runs   (  561.47 ms per token)
falcon_print_timings:       total time = 28315.91 ms
```


Below are Falcon 7B tests:
**Q5_1 is working, comes with ggml v3 as a bonus (mmap support)**
```
falcon_model_load_internal: ftype      = 9 (mostly Q5_1)
falcon_print_timings:        load time =   952.24 ms
falcon_print_timings:      sample time =    67.91 ms /   300 runs   (    0.23 ms per token)
falcon_print_timings: prompt eval time =   370.94 ms /    14 tokens (   26.50 ms per token)
falcon_print_timings:        eval time = 50367.68 ms /   299 runs   (  168.45 ms per token)
```
**Q4_1 is working as well**
```
falcon_print_timings:        load time =   864.40 ms
falcon_print_timings:      sample time =    22.68 ms /   100 runs   (    0.23 ms per token)
falcon_print_timings: prompt eval time =   287.00 ms /    14 tokens (   20.50 ms per token)
falcon_print_timings:        eval time = 12233.39 ms /    99 runs   (  123.57 ms per token)
```

Q_K_*: not working (no segfaults anymore, looks like an error in qkv handling as it's outputting garbage.
