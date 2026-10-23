#include "common.cuh"

// BF16X lossless overflow side-table corrections.
// Blob layout (I8 tensor, see ggml_sparse_mm / ggml_sparse_rows):
//   u32 n_rows, u32 n_entries, u32 row_off[n_rows+1], { u16 col, u16 bf16_bits }[n_entries]

void ggml_cuda_op_sparse_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sparse_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
