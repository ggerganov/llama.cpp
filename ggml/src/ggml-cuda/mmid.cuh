#pragma once

#include "common.cuh"

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// MoE matrix multiplications pack the column tiles of all experts into one index space, expert e owning ceil(ncols_e/J)
// consecutive tiles, so that the number of tiles depends on the total number of columns rather than the max. per expert.
// Finds the expert zt and its column tile jt for the packed tile jt_packed, expert_bounds is from ggml_cuda_launch_mm_ids_helper.
// The whole warp must call this with the same jt_packed. Returns false past the last tile, jt is then the total number of tiles.
template <int J>
static __device__ __forceinline__ bool mm_ids_find_tile(
        const int32_t * __restrict__ expert_bounds, const int n_experts, const int jt_packed, int & zt, int & jt) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    int ntiles_prev = 0;
    for (int e0 = 0; e0 < n_experts; e0 += warp_size) {
        const int e = e0 + threadIdx.x;
        const int ntiles_e = e < n_experts ? (expert_bounds[e + 1] - expert_bounds[e] + J - 1) / J : 0;
        const int ntiles_incl = ntiles_prev + warp_prefix_inclusive_sum<int, warp_size>(ntiles_e);

        // The inclusive sums are monotonic, so the lane holding jt_packed is the number of lanes whose tiles end at or before it:
        const int lane = warp_reduce_sum<warp_size>(ntiles_incl <= jt_packed);
        if (lane < warp_size) {
            zt = e0 + lane;
            jt = jt_packed - __shfl_sync(0xFFFFFFFF, ntiles_incl - ntiles_e, lane, warp_size);
            return true;
        }
        ntiles_prev = __shfl_sync(0xFFFFFFFF, ntiles_incl, warp_size - 1, warp_size);
    }
    jt = ntiles_prev;
    return false;
}
