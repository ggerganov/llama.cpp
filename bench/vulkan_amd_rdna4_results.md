# AMD RDNA4 Vulkan Performance Fix - Benchmark Results

## Test Environment

| Component | Value |
|---|---|
| GPU | AMD Radeon RX 9070 XT (gfx1201 / RDNA4) |
| VRAM | 16304 MiB |
| ROCm | 7.16.26323 (G:\ROCM10RT-gfx1201) |
| Vulkan SDK | 1.4.357.0 (C:\VulkanSDK\1.4.357.0) |
| Build | d7bd05774 (10829) |

## Benchmark Results

### Qwen3-8B Q4_K_M (4.68 GiB)

| Backend | Threads | Batch | Prompt (pp512/128) | Decode (tg128/16) | Pre-fix tg128 |
|---|---|---|---|---|---|
| Vulkan (patched) | 4 | 512 | 1607.79 t/s | **89.41 t/s** | 5.26 t/s |
| ROCm (HIP) | 4 | 128 | 1826.14 t/s | 92.41 t/s | N/A |

**Improvement**: Vulkan decode tg128 improved from 5.26 to 89.41 t/s (**16.9x speedup**).

### Llama-3.2-1B Q8_0 (1.22 GiB)

| Backend | Threads | Batch | Prompt (pp64) | Decode (tg8) |
|---|---|---|---|---|
| ROCm (HIP) | 4 | 64 | 4986.83 t/s | 261.13 t/s |

## Summary

- Vulkan decode performance on RX 9070 XT: **5.26 t/s -> 89.41 t/s** (16.9x)
- Vulkan now within 3% of ROCm for decode throughput (89.41 vs 92.41 t/s)
- ROCm remains ~13% ahead on prompt processing (1826 vs 1608 t/s)

## GPU Detection

```
Vulkan: AMD Radeon RX 9070 XT (AMD proprietary driver) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 32 | shared memory: 32768 | int dot: 1 | matrix cores: none

ROCm: AMD Radeon RX 9070 XT, gfx1201 (0x1201), VMM: no, Wave Size: 32, VRAM: 16304 MiB
```

## Reproduction

```sh
# Vulkan (patched)
llama-bench -m Qwen3-8B-Q4_K_M.gguf -r 3 -n 128 -p 512 -ngl 99 -t 4 -b 512 -ub 512 -o md

# ROCm (HIP)
llama-bench -m Qwen3-8B-Q4_K_M.gguf -r 1 -n 16 -p 128 -ngl 99 -t 4 -b 128 -ub 128 -o md
```
