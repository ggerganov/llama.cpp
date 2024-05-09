#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-vec-f32.cuh"

template<int D, int ncols, int parallel_blocks> // D == head size
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_vec_ext_f32(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.y              + nb01*ic0);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.y / gqa_ratio));
    const half   * V_h   = (const half   *) (V    + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half   * maskh = (const half   *)  mask + ne11*ic0;

    const int stride_KV  = nb11 / sizeof(half);
    const int stride_KV2 = nb11 / sizeof(half2);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ float KQ[ncols*D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*D + tid] = -FLT_MAX/2.0f;
    }

    float kqmax[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -FLT_MAX/2.0f;
    }
    float kqsum[ncols] = {0.0f};

    __shared__ float kqmax_shared[ncols][WARP_SIZE];
    __shared__ float kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    // Convert Q to half2 and store in registers:
    float2 Q_h2[ncols][D/(2*WARP_SIZE)];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            Q_h2[j][i0/WARP_SIZE]    = Q_f2[j*(nb01/sizeof(float2)) + i];
            Q_h2[j][i0/WARP_SIZE].x *= scale;
            Q_h2[j][i0/WARP_SIZE].y *= scale;
        }
    }

    float VKQ[ncols] = {0.0f};

    const int k_start = parallel_blocks == 1 ? 0 : ip*D;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*D) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        float kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

            float sum[ncols] = {0.0f};
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
                const int k_KQ = k_KQ_0 + threadIdx.x;

                const half2 K_ik = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ];
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    sum[j] +=  __low2float(K_ik) * Q_h2[j][k_KQ_0/WARP_SIZE].x;
                    sum[j] += __high2float(K_ik) * Q_h2[j][k_KQ_0/WARP_SIZE].y;
                }
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                sum[j] = warp_reduce_sum(sum[j]);
                sum[j] += mask ? __half2float(maskh[j*ne11 + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new_arr[j] = fmaxf(kqmax_new_arr[j], sum[j]);

                if (threadIdx.x == 0) {
                    KQ[j*D + i_KQ] = sum[j];
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_new_arr[j];

            kqmax_new_j = warp_reduce_max(kqmax_new_j);
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const float val = expf(KQ[j*D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*D + tid] = val;

            VKQ[j] *= KQ_max_scale;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < D; ++k) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k >= ne11) {
                break;
            }

            const float V_ki = __half2float(V_h[(k_VKQ_0 + k)*stride_KV + tid]);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_ki*KQ[j*D + k];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum(kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum(kqsum[j_VKQ]);

        float dst_val = VKQ[j_VKQ];
        if (parallel_blocks == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
        dst[j_dst*D*gridDim.y + D*blockIdx.y + tid] = dst_val;
    }

    if (parallel_blocks != 1 && tid != 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            dst_meta[(ic0 + j)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax[j], kqsum[j]);
        }
    }
}

template <int D, int cols_per_block, int parallel_blocks> void launch_fattn_vec_f32(
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V, ggml_tensor * KQV, const ggml_tensor * mask,
        ggml_cuda_pool & pool, cudaStream_t main_stream
) {
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
    }

    constexpr int  nwarps = (D + WARP_SIZE - 1) / WARP_SIZE;
    const     dim3 block_dim(WARP_SIZE, nwarps, 1);
    const     dim3 blocks_num(parallel_blocks*((Q->ne[1] + cols_per_block - 1) / cols_per_block), Q->ne[2], Q->ne[3]);
    const     int  shmem = 0;

    float scale;
    memcpy(&scale, KQV->op_params, sizeof(float));

    flash_attn_vec_ext_f32<D, cols_per_block, parallel_blocks>
        <<<blocks_num, block_dim, shmem, main_stream>>> (
                (const char *) Q->data,
                (const char *) K->data,
                (const char *) V->data,
                mask ? ((const char *) mask->data) : nullptr,
                parallel_blocks == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
                scale,
                Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                Q->nb[1], Q->nb[2], Q->nb[3],
                K->nb[1], K->nb[2], K->nb[3],
                KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                );
    CUDA_CHECK(cudaGetLastError());

    if (parallel_blocks == 1) {
        return;
    }

    const dim3 block_dim_combine(D, 1, 1);
    const dim3 blocks_num_combine(Q->ne[1], blocks_num.y, blocks_num.z);
    const int  shmem_combine = 0;

    flash_attn_combine_results<D, parallel_blocks>
        <<<blocks_num_combine, block_dim_combine, shmem_combine, main_stream>>>
        (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext_vec_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_tensor * mask = dst->src[3];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->ne[0] == 64 || Q->ne[0] == 128 && "FlashAttention without tensor cores only supports head sizes 64 and 128.");

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        constexpr int parallel_blocks = 4;
        switch (Q->ne[0]) {
            case 64:
                launch_fattn_vec_f32< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_vec_f32<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block = 2;
        constexpr int parallel_blocks = 4;
        switch (Q->ne[0]) {
            case 64:
                launch_fattn_vec_f32< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_vec_f32<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block = 4;
        constexpr int parallel_blocks = 4;
        switch (Q->ne[0]) {
            case 64:
                launch_fattn_vec_f32< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_vec_f32<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    if (Q->ne[1] <= 8) {
        constexpr int cols_per_block = 8;
        constexpr int parallel_blocks = 4;
        switch (Q->ne[0]) {
            case 64:
                launch_fattn_vec_f32< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_vec_f32<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    constexpr int cols_per_block = 8;
    constexpr int parallel_blocks = 1;
    switch (Q->ne[0]) {
        case 64:
            launch_fattn_vec_f32< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
            break;
        case 128:
            launch_fattn_vec_f32<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
}
