#include "common.cuh"

void ggml_cuda_flash_attn_ext(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V,
        const ggml_tensor * mask, ggml_tensor * KQV);
