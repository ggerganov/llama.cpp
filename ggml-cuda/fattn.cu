#include "fattn.cuh"

#include <mma.h>

static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
#pragma unroll
   for (int mask = 16; mask > 0; mask >>= 1) {
       a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, mask, 32));
   }
   return a;
#else
   GGML_UNUSED(a);
   NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
}

static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL && CUDART_VERSION >= CUDART_HMAX
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = __hmax2(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
#else
    GGML_UNUSED(x);
    NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL && CUDART_VERSION >= CUDART_HMAX
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_vec_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float * __restrict__ dst,
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
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.y              + nb01*blockIdx.x);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.y / gqa_ratio));
    const half   * V_h   = (const half   *) (V    + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half   * maskh = (const half   *)  mask + ne31*blockIdx.x;

    const int stride_KV  = nb11 / sizeof(half);
    const int stride_KV2 = nb11 / sizeof(half2);

    constexpr int nwarps = D/WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ half KQ[D];
    KQ[tid] = 0.0f;
    half2 * KQ2 = (half2 *) KQ;

    half kqmax = -INFINITY;
    half kqsum = 0.0f;

    __shared__ half kqmax_shared[WARP_SIZE];
    __shared__ half kqsum_shared[WARP_SIZE];
    if (threadIdx.y == 0) {
        kqmax_shared[threadIdx.x] = -INFINITY;
        kqsum_shared[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Convert Q to half2 and store in registers:
    half2 Q_h2[(D/2 + WARP_SIZE - 1) / WARP_SIZE];
#pragma unroll
    for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
        const int i = i0 + threadIdx.x;
        if (i0 + WARP_SIZE > D/2 && i >= D/2) {
            break;
        }

        Q_h2[i0/WARP_SIZE] = make_half2(scale, scale) * make_half2(Q_f2[i].x, Q_f2[i].y);
    }

    half2 VKQ = make_half2(0.0f, 0.0f); // Each thread calculates a single VKQ value.

    for (int k_VKQ_0 = 0; k_VKQ_0 < ne11; k_VKQ_0 += D) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        half kqmax_new = kqmax;
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if (256 % D != 0 && k_VKQ_0 + i_KQ >= ne11) {
                break;
            }

            half2 sum2 = make_half2(0.0f, 0.0f);
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
                const int k_KQ = k_KQ_0 + threadIdx.x;
                if (k_KQ_0 + WARP_SIZE > D/2 && k_KQ >= D/2) {
                    break;
                }

                const half2 K_ik = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ];
                sum2 += K_ik * Q_h2[k_KQ_0/WARP_SIZE];
            }

            sum2 = warp_reduce_sum(sum2);
            half sum = __low2half(sum2) + __high2half(sum2);
            sum += mask ? maskh[k_VKQ_0 + i_KQ] : __float2half(0.0f);
            kqmax_new = __hmax(kqmax_new, sum);
            if (threadIdx.x == 0) {
                KQ[i_KQ] = sum;
            }
        }

        kqmax_new = warp_reduce_max(kqmax_new);
        if (threadIdx.x == 0) {
            kqmax_shared[threadIdx.y] = kqmax_new;
        }
        __syncthreads();
        kqmax_new = kqmax_shared[threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);

        const half KQ_max_scale = hexp(kqmax - kqmax_new);
        kqmax = kqmax_new;

        const half val = hexp(KQ[tid] - kqmax);
        kqsum = kqsum*KQ_max_scale + val;
        KQ[tid] = val;

        VKQ *= __half2half2(KQ_max_scale);

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < D; k0 += 2) {
            if (256 % D != 0 && k_VKQ_0 + k0 >= ne11) {
                break;
            }

            half2 V_k;
            reinterpret_cast<half&>(V_k.x) = V_h[(k_VKQ_0 + k0 + 0)*stride_KV + tid];
            reinterpret_cast<half&>(V_k.y) = V_h[(k_VKQ_0 + k0 + 1)*stride_KV + tid];
            VKQ += V_k*KQ2[k0/2];
        }
    }

    kqsum = warp_reduce_sum(kqsum);
    if (threadIdx.x == 0) {
        kqsum_shared[threadIdx.y] = kqsum;
    }
    __syncthreads();
    kqsum = kqsum_shared[threadIdx.x];
    kqsum = warp_reduce_sum(kqsum);

    dst[D*gridDim.y*blockIdx.x + D*blockIdx.y + tid] = (__low2half(VKQ) + __high2half(VKQ)) / kqsum;
}

template<int D, int ncols> // D == head size
__launch_bounds__(ncols == 8 || D > 128 ? D : 2*D, 1)
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float * __restrict__ dst,
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
    static_assert(ncols == 8 || ncols % 16 == 0, "ncols must be 8 or a multiple of 16.");
    constexpr int frag_m = ncols == 8 ? 32 : 16;
    constexpr int frag_n = ncols == 8 ?  8 : 16;
    static_assert(D % frag_m == 0, "If ncols == 8 then D % frag_m must be 0.");
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::row_major> frag_a_K;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_a_V;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_b;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, frag_m, frag_n, 16, half>                          frag_c;

    constexpr int nwarps = (D <= 128 || ncols == 8 ? D : D/2) / frag_m;
    constexpr int nthreads = nwarps*WARP_SIZE;
    static_assert(nthreads % D == 0, "nthreads not divisible by D.");
    constexpr int tc_vals_per_iter = nwarps*frag_m;
    static_assert(D % tc_vals_per_iter == 0, "D not divisible by tensor core vals per iter.");
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);
    constexpr int D_padded = D + 8; // Pad internal representation of KQ, KQV to reduce shared memory bank conflicts.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f   = (const float *) (Q + nb02* blockIdx.y                + ncols*nb01*blockIdx.x);
    const half  * K_h   = (const half  *) (K + nb12*(blockIdx.y / gqa_ratio));
    const half  * V_h   = (const half  *) (V + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half2 * mask2 = (half2       *)  mask + ncols*ne11*blockIdx.x/2;

    const int stride_Q  = nb01 / sizeof(float);
    const int stride_KV = nb11 / sizeof(half);

    frag_b Q_b[D/16][ncols/frag_n];

    __shared__ half KQ[ncols*D_padded]; // Buffer for temporarily holding tiles of KQ.
    half2 * KQ2 = (half2 *) KQ;

    half2    KQ_rowsum[(ncols + nwarps - 1) / nwarps] = {{0.0f,      0.0f}};
    half2       KQ_max[(ncols + nwarps - 1) / nwarps] = {{-INFINITY, -INFINITY}};
    half2 KQ_max_scale[(ncols + nwarps - 1) / nwarps] = {{0.0f,      0.0f}};

    __shared__ half VKQ[ncols*D_padded]; // Accumulator for final VKQ slice.
    half2 * VKQ2 = (half2 *) VKQ;
#pragma unroll
    for (int i0 = 0; i0 < ncols*D_padded/2; i0 += nthreads) {
        const int i = i0 + tid;
        if (i0 + nthreads > ncols*D_padded/2 && i >= ncols*D_padded/2) {
            break;
        }

        VKQ2[i] = make_half2(0.0f, 0.0f);
    }

    // Convert Q to half and apply scale, temporarily store in KQ:
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nthreads/D) {
        const int j = j0 + tid/D;
        const int i = tid % D;
        KQ[j*D_padded + i] = ncols*blockIdx.x + j < ne01 ? Q_f[j*stride_Q + i] * scale : 0.0f;
    }

    __syncthreads();

    // Load Q into tensor core fragments/registers since it will be used frequently:
#pragma unroll
    for (int i0 = 0; i0 < D; i0 += 16) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
            nvcuda::wmma::load_matrix_sync(Q_b[i0/16][j0/frag_n], KQ + j0*D_padded + i0, D_padded);
        }
    }

    __syncthreads();

    // Iterate over ne11 == previous tokens:
    for (int k_VKQ_0 = 0; k_VKQ_0 < ne11; k_VKQ_0 += D) {
        const bool has_valid_data = 256 % D == 0 || k_VKQ_0 + frag_m*threadIdx.y < ne11;

        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += tc_vals_per_iter) {
            frag_c KQ_c[ncols/frag_n];
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                nvcuda::wmma::fill_fragment(KQ_c[j], 0.0f);
            }
            if (has_valid_data) {
#pragma unroll
                for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 16) {
                    frag_a_K K_a;
                    nvcuda::wmma::load_matrix_sync(K_a, K_h + (k_VKQ_0 + i_KQ_0 + frag_m*threadIdx.y)*stride_KV + k_KQ_0, stride_KV);
#pragma unroll
                    for (int j = 0; j < ncols/frag_n; ++j) {
                        nvcuda::wmma::mma_sync(KQ_c[j], K_a, Q_b[k_KQ_0/16][j], KQ_c[j]);
                    }
                }
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                nvcuda::wmma::store_matrix_sync(KQ + j0*D_padded + i_KQ_0 + frag_m*threadIdx.y, KQ_c[j0/frag_n], D_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            half2 KQ_max_new = KQ_max[j0/nwarps];
#pragma unroll
            for (int k0 = 0; k0 < D/2; k0 += WARP_SIZE) {
                const int k = k0 + threadIdx.x;
                if (k0 + WARP_SIZE > D/2 && k >= D/2) {
                    break;
                }
                KQ_max_new = __hmax2(KQ_max_new, KQ2[j*(D_padded/2) + k]);
            }
            KQ_max_new = __half2half2(warp_reduce_max(__hmax(__low2half(KQ_max_new), __high2half(KQ_max_new))));
            KQ_max_scale[j0/nwarps] = h2exp(KQ_max[j0/nwarps] - KQ_max_new);
            KQ_max[j0/nwarps] = KQ_max_new;

            half2 KQ_rowsum_add = make_half2(0.0f, 0.0f);
#pragma unroll
            for (int k0 = 0; k0 < D/2; k0 += WARP_SIZE) {
                const int k = k0 + threadIdx.x;
                if (k0 + WARP_SIZE > D/2 && k >= D/2) {
                    break;
                }
                if (256 % D != 0 && k_VKQ_0 + 2*k >= ne11) {
                    break;
                }

                half2 val = KQ2[j*(D_padded/2) + k];
                val += mask ? mask2[(j*ne11 + k_VKQ_0)/2 + k] : make_half2(0.0f, 0.0f);
                val = h2exp(val - KQ_max[j0/nwarps]);
                KQ_rowsum_add += val;
                KQ2[j*(D_padded/2) + k] = val;
            }
            KQ_rowsum_add = warp_reduce_sum(KQ_rowsum_add);

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[j0/nwarps] = KQ_max_scale[j0/nwarps]*KQ_rowsum[j0/nwarps] + KQ_rowsum_add;
        }

        __syncthreads();

        frag_b KQ_b[D/16][ncols/frag_n];
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
#pragma unroll
            for (int k0 = 0; k0 < D; k0 += 16) {
                nvcuda::wmma::load_matrix_sync(KQ_b[k0/16][j0/frag_n], KQ + j0*D_padded + k0, D_padded);
            }
        }

        frag_c VKQ_c[D/tc_vals_per_iter][ncols/frag_n];
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += tc_vals_per_iter) {
    #pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                nvcuda::wmma::fill_fragment(VKQ_c[i_KQ_0/tc_vals_per_iter][j], 0.0f);
            }

    #pragma unroll
            for (int k0 = 0; k0 < D; k0 += 16) {
                if (256 % D != 0 && k_VKQ_0 + k0 >= ne11) {
                    break;
                }

                frag_a_V v_a;
                nvcuda::wmma::load_matrix_sync(v_a, V_h + (k_VKQ_0 + k0)*stride_KV + i_KQ_0 + frag_m*threadIdx.y, stride_KV);
    #pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    nvcuda::wmma::mma_sync(VKQ_c[i_KQ_0/tc_vals_per_iter][j], v_a, KQ_b[k0/16][j], VKQ_c[i_KQ_0/tc_vals_per_iter][j]);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += tc_vals_per_iter) {
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                nvcuda::wmma::store_matrix_sync(
                    KQ + j0*D_padded + i_KQ_0 + frag_m*threadIdx.y,
                    VKQ_c[i_KQ_0/tc_vals_per_iter][j0/frag_n],
                    D_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                if (i0 + WARP_SIZE > D/2 && i >= D/2) {
                    break;
                }
                VKQ2[j*(D_padded/2) + i] = KQ_max_scale[j0/nwarps]*VKQ2[j*(D_padded/2) + i] + KQ2[j*(D_padded/2) + i];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if ((j0 + nwarps > ncols && j >= ncols) || ncols*blockIdx.x + j >= ne01) {
            return;
        }
        const float KQ_rowsum_j = __low2float(KQ_rowsum[j0/nwarps]) + __high2float(KQ_rowsum[j0/nwarps]);
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D && i >= D) {
                break;
            }
            dst[D*gridDim.y*(ncols*blockIdx.x + j) + D*blockIdx.y + i] = __half2float(VKQ[j*D_padded + i]) / KQ_rowsum_j;
        }
    }
}


void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_tensor * mask = dst->src[3];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(K->type == GGML_TYPE_F16);
    GGML_ASSERT(V->type == GGML_TYPE_F16);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!mask || mask->ne[1] >= GGML_PAD(Q->ne[1], 16) &&
                                "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

    ggml_cuda_set_device(ctx.device);

    const cudaStream_t main_stream = ctx.stream();

    float scale;
    memcpy(&scale, KQV->op_params, sizeof(float));

    if (Q->ne[0] % WARP_SIZE == 0 && Q->ne[0] >= 128 && Q->ne[1] == 1) {
        const int nwarps = Q->ne[0] / WARP_SIZE;
        const dim3 blocks_num(Q->ne[1], Q->ne[2], Q->ne[3]);
        const dim3 block_dim(WARP_SIZE, nwarps, 1);
        const int shmem = 0;
        switch (Q->ne[0]) {
            // case 64:
            //     flash_attn_vec_ext_f16<64>
            //         <<<blocks_num, block_dim, shmem, main_stream>>> (
            //                 (const char *) Q->data, // Query
            //                 (const char *) K->data, // Key
            //                 (const char *) V->data, // Value
            //                 mask ? ((const char *) mask->data) : nullptr, // Mask
            //                 (float *) KQV->data, // dst
            //                 scale,
            //                 Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
            //                 K->ne[0], K->ne[1], K->ne[2], K->ne[3],
            //                 mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
            //                 Q->nb[1], Q->nb[2], Q->nb[3],
            //                 K->nb[1], K->nb[2], K->nb[3],
            //                 KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
            //                 );
            //     break;
            // case 80:
            //     flash_attn_vec_ext_f16<80>
            //         <<<blocks_num, block_dim, shmem, main_stream>>> (
            //                 (const char *) Q->data, // Query
            //                 (const char *) K->data, // Key
            //                 (const char *) V->data, // Value
            //                 mask ? ((const char *) mask->data) : nullptr, // Mask
            //                 (float *) KQV->data, // dst
            //                 scale,
            //                 Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
            //                 K->ne[0], K->ne[1], K->ne[2], K->ne[3],
            //                 mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
            //                 Q->nb[1], Q->nb[2], Q->nb[3],
            //                 K->nb[1], K->nb[2], K->nb[3],
            //                 KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
            //                 );
            //     break;
            // case 96:
            //     flash_attn_vec_ext_f16<96>
            //         <<<blocks_num, block_dim, shmem, main_stream>>> (
            //                 (const char *) Q->data, // Query
            //                 (const char *) K->data, // Key
            //                 (const char *) V->data, // Value
            //                 mask ? ((const char *) mask->data) : nullptr, // Mask
            //                 (float *) KQV->data, // dst
            //                 scale,
            //                 Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
            //                 K->ne[0], K->ne[1], K->ne[2], K->ne[3],
            //                 mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
            //                 Q->nb[1], Q->nb[2], Q->nb[3],
            //                 K->nb[1], K->nb[2], K->nb[3],
            //                 KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
            //                 );
            //     break;
            // case 112:
            //     flash_attn_vec_ext_f16<112>
            //         <<<blocks_num, block_dim, shmem, main_stream>>> (
            //                 (const char *) Q->data, // Query
            //                 (const char *) K->data, // Key
            //                 (const char *) V->data, // Value
            //                 mask ? ((const char *) mask->data) : nullptr, // Mask
            //                 (float *) KQV->data, // dst
            //                 scale,
            //                 Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
            //                 K->ne[0], K->ne[1], K->ne[2], K->ne[3],
            //                 mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
            //                 Q->nb[1], Q->nb[2], Q->nb[3],
            //                 K->nb[1], K->nb[2], K->nb[3],
            //                 KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
            //                 );
            //     break;
            case 128:
                flash_attn_vec_ext_f16<128>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 256:
                flash_attn_vec_ext_f16<256>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int cols_per_block;
    if (Q->ne[1] >= 128 && Q->ne[0] <= 128 && Q->ne[0] % 32 == 0) {
        cols_per_block = 64;
    } else if (Q->ne[1] >= 64 && (Q->ne[0] <= 128 || ggml_cuda_info().devices[ctx.device].cc >= CC_AMPERE)) {
        cols_per_block = 32;
    } else if (Q->ne[1] >= 32 || Q->ne[0] % 32 != 0) {
        cols_per_block = 16;
    } else  {
        cols_per_block = 8;
    }
    const int frag_m = cols_per_block == 8 ? 32 : 16;
    const int nwarps = (Q->ne[0] <= 128 || cols_per_block == 8 ? Q->ne[0] : Q->ne[0]/2) / frag_m;
    const dim3 blocks_num((Q->ne[1] + cols_per_block - 1) / cols_per_block, Q->ne[2], Q->ne[3]);
    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    const size_t shmem = 0;

    switch (Q->ne[0]) {
        case 64: switch (cols_per_block) {
            case 8:
                flash_attn_ext_f16<64, 8>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 16:
                flash_attn_ext_f16<64, 16>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 32:
                flash_attn_ext_f16<64, 32>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 64:
                flash_attn_ext_f16<64, 64>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        case 80: switch (cols_per_block) {
            // case 8:
            //     fused_attn_vec_ext_f16<80, 8>
            //         <<<blocks_num, block_dim, shmem, main_stream>>> (
            //                 (const char *) Q->data, // Query
            //                 (const char *) K->data, // Key
            //                 (const char *) V->data, // Value
            //                 mask ? ((const char *) mask->data) : nullptr, // Mask
            //                 (float *) KQV->data, // dst
            //                 scale,
            //                 Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
            //                 K->ne[0], K->ne[1], K->ne[2], K->ne[3],
            //                 mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
            //                 Q->nb[1], Q->nb[2], Q->nb[3],
            //                 K->nb[1], K->nb[2], K->nb[3],
            //                 KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
            //                 );
            //     break;
            case 16:
                flash_attn_ext_f16<80, 16>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 32:
                flash_attn_ext_f16<80, 32>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 64:
                flash_attn_ext_f16<80, 64>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        case 96: switch (cols_per_block) {
            case 8:
                flash_attn_ext_f16<96, 8>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 16:
                flash_attn_ext_f16<96, 16>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 32:
                flash_attn_ext_f16<96, 32>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 64:
                flash_attn_ext_f16<96, 64>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        case 112: switch (cols_per_block) {
            // case 8:
            //     fused_attn_vec_ext_f16<112, 8>
            //         <<<blocks_num, block_dim, shmem, main_stream>>> (
            //                 (const char *) Q->data, // Query
            //                 (const char *) K->data, // Key
            //                 (const char *) V->data, // Value
            //                 mask ? ((const char *) mask->data) : nullptr, // Mask
            //                 (float *) KQV->data, // dst
            //                 scale,
            //                 Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
            //                 K->ne[0], K->ne[1], K->ne[2], K->ne[3],
            //                 mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
            //                 Q->nb[1], Q->nb[2], Q->nb[3],
            //                 K->nb[1], K->nb[2], K->nb[3],
            //                 KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
            //                 );
            //     break;
            case 16:
                flash_attn_ext_f16<112, 16>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 32:
                flash_attn_ext_f16<112, 32>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 64:
                flash_attn_ext_f16<112, 64>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        case 128: switch (cols_per_block) {
            case 8:
                flash_attn_ext_f16<128, 8>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 16:
                flash_attn_ext_f16<128, 16>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 32:
                flash_attn_ext_f16<128, 32>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 64:
                flash_attn_ext_f16<128, 64>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        case 256: switch (cols_per_block) {
            case 8:
                flash_attn_ext_f16<256, 8>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 16:
                flash_attn_ext_f16<256, 16>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            case 32:
                flash_attn_ext_f16<256, 32>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                break;
            // case 64:
            //     flash_attn_ext_f16<256, 64>
            //         <<<blocks_num, block_dim, shmem, main_stream>>> (
            //                 (const char *) Q->data, // Query
            //                 (const char *) K->data, // Key
            //                 (const char *) V->data, // Value
            //                 mask ? ((const char *) mask->data) : nullptr, // Mask
            //                 (float *) KQV->data, // dst
            //                 scale,
            //                 Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
            //                 K->ne[0], K->ne[1], K->ne[2], K->ne[3],
            //                 mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
            //                 Q->nb[1], Q->nb[2], Q->nb[3],
            //                 K->nb[1], K->nb[2], K->nb[3],
            //                 KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
            //                 );
            //     break;
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        default:
            GGML_ASSERT(false);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}
