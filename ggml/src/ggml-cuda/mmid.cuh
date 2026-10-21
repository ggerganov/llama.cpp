#pragma once

// copies ids to out, [n_expert_used, n_tokens] contiguous, giving each id < 0 an expert that its token does not use
void ggml_cuda_fix_neg_ids(const ggml_tensor * ids, int32_t * out, int n_experts, cudaStream_t stream);

// zeroes the rows of dst that belong to ids < 0
void ggml_cuda_zero_neg_rows(const ggml_tensor * ids, ggml_tensor * dst, cudaStream_t stream);

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);
