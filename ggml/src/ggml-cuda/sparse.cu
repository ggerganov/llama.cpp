#include "sparse.cuh"

__global__ void sparse_mm_kernel(
        const uint8_t * __restrict__ blob,
        const float   * __restrict__ b,
        float         * __restrict__ out,
        const int64_t n_rows, const int64_t n_embd, const int64_t n_tokens) {
    // one warp per (row, token): lanes split the row's entries, then reduce
    const int lane = (int) (threadIdx.x & 31);
    const int64_t warp = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= n_rows * n_tokens) {
        return;
    }

    const uint32_t * hdr     = (const uint32_t *) blob;
    const uint32_t * row_off = hdr + 2;
    const uint16_t * ents    = (const uint16_t *) (row_off + n_rows + 1);

    const int64_t row = warp % n_rows;
    const int64_t t   = warp / n_rows;

    const uint32_t e0 = row_off[row];
    const uint32_t e1 = row_off[row + 1];

    float sum = 0.0f;
    const float * bt = b + n_embd * t;
    for (uint32_t e = e0 + lane; e < e1; e += 32) {
        const uint32_t col = ents[2*e + 0];
        const float    v   = __uint_as_float((uint32_t) ents[2*e + 1] << 16);
        sum += v * bt[col];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }
    if (lane == 0) {
        out[row + n_rows * t] = sum;
    }
}

__global__ void sparse_rows_kernel(
        const uint8_t * __restrict__ blob,
        const int32_t * __restrict__ ids,
        float         * __restrict__ out,
        const int64_t n_embd, const int64_t n_tok) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tok) {
        return;
    }

    const uint32_t * hdr     = (const uint32_t *) blob;
    const int64_t    n_rows  = hdr[0];
    const uint32_t * row_off = hdr + 2;
    const uint16_t * ents    = (const uint16_t *) (row_off + n_rows + 1);

    float * row_out = out + n_embd * t;
    for (int64_t c = 0; c < n_embd; ++c) {
        row_out[c] = 0.0f;
    }

    const int32_t row = ids[t];
    if (row < 0 || row >= n_rows) {
        return;
    }
    for (uint32_t e = row_off[row]; e < row_off[row + 1]; ++e) {
        row_out[ents[2*e + 0]] += __uint_as_float((uint32_t) ents[2*e + 1] << 16);
    }
}

void ggml_cuda_op_sparse_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * as = dst->src[0];
    const ggml_tensor * b  = dst->src[1];

    const int64_t n_rows   = dst->ne[0];
    const int64_t n_embd   = b->ne[0];
    const int64_t n_tokens = b->ne[1] * b->ne[2] * b->ne[3];

    const int64_t total = n_rows * n_tokens;
    const int block = 256;
    const int grid  = (int) ((total * 32 + block - 1) / block);

    sparse_mm_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const uint8_t *) as->data, (const float *) b->data, (float *) dst->data,
        n_rows, n_embd, n_tokens);
}

void ggml_cuda_op_sparse_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * as  = dst->src[0];
    const ggml_tensor * ids = dst->src[1];

    const int64_t n_embd = dst->ne[0];
    const int64_t n_tok  = ids->ne[0];

    const int block = 128;
    const int grid  = (int) ((n_tok + block - 1) / block);

    sparse_rows_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const uint8_t *) as->data, (const int32_t *) ids->data, (float *) dst->data,
        n_embd, n_tok);
}
