#include "common.cuh"
#include "fattn.cuh"

#include <cstdint>

#if FP16_MMA_AVAILABLE
#include <mma.h>
#endif

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

template<int D, int ncols, int parallel_blocks> // D == head size
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_vec_ext_f16(
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
#if FP16_AVAILABLE
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

    __shared__ half KQ[ncols*D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*D + tid] = -HALF_MAX_HALF;
    }
    half2 * KQ2 = (half2 *) KQ;

    half kqmax[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -HALF_MAX_HALF;
    }
    half kqsum[ncols] = {0.0f};

    __shared__ half kqmax_shared[ncols][WARP_SIZE];
    __shared__ half kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -HALF_MAX_HALF;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    // Convert Q to half2 and store in registers:
    half2 Q_h2[ncols][D/(2*WARP_SIZE)];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const float2 tmp = Q_f2[j*(nb01/sizeof(float2)) + i];
            Q_h2[j][i0/WARP_SIZE] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
        }
    }

    half2 VKQ[ncols] = {{0.0f, 0.0f}};

    const int k_start = parallel_blocks == 1 ? 0 : ip*D;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*D) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        // For unknown reasons using a half array of size 1 for kqmax_new causes a performance regression,
        // see https://github.com/ggerganov/llama.cpp/pull/7061 .
        // Therefore this variable is defined twice but only used once (so that the compiler can optimize out the unused variable).
        half kqmax_new = kqmax[0];
        half kqmax_new_arr[ncols];
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

            half2 sum2[ncols] = {{0.0f, 0.0f}};
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
                const int k_KQ = k_KQ_0 + threadIdx.x;

                const half2 K_ik = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ];
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    sum2[j] += K_ik * Q_h2[j][k_KQ_0/WARP_SIZE];
                }
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                sum2[j] = warp_reduce_sum(sum2[j]);
                half sum = __low2half(sum2[j]) + __high2half(sum2[j]);
                sum += mask ? maskh[j*ne11 + k_VKQ_0 + i_KQ] : __float2half(0.0f);

                if (ncols == 1) {
                    kqmax_new        = ggml_cuda_hmax(kqmax_new,        sum);
                } else {
                    kqmax_new_arr[j] = ggml_cuda_hmax(kqmax_new_arr[j], sum);
                }

                if (threadIdx.x == 0) {
                    KQ[j*D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = ncols == 1 ? kqmax_new : kqmax_new_arr[j];

            kqmax_new_j = warp_reduce_max(kqmax_new_j);
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(KQ[j*D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*D + tid] = val;

            VKQ[j] *= __half2half2(KQ_max_scale);
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < D; k0 += 2) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                break;
            }

            half2 V_k;
            reinterpret_cast<half&>(V_k.x) = V_h[(k_VKQ_0 + k0 + 0)*stride_KV + tid];
            reinterpret_cast<half&>(V_k.y) = V_h[(k_VKQ_0 + k0 + 1)*stride_KV + tid];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_k*KQ2[j*(D/2) + k0/2];
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

        half dst_val = (__low2half(VKQ[j_VKQ]) + __high2half(VKQ[j_VKQ]));
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
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// D == head size, VKQ_stride == num VKQ rows calculated in parallel:
template<int D, int ncols, int nwarps, int VKQ_stride, int parallel_blocks, typename KQ_acc_t>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_ext_f16(
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
#if FP16_MMA_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = ncols*(blockIdx.x / parallel_blocks); // Index of the first Q/QKV column to work on.
    const int ip  =        blockIdx.x % parallel_blocks;  // Index in group of blocks running for the same column in parallel.

    static_assert(D <= FATTN_KQ_STRIDE, "D must be <= FATTN_KQ_STRIDE.");
    static_assert(ncols == 8 || ncols % 16 == 0, "ncols must be 8 or a multiple of 16.");
    constexpr int frag_m = ncols == 8 ? 32 : 16;
    constexpr int frag_n = ncols == 8 ?  8 : 16;
    static_assert(D % frag_m == 0, "If ncols == 8 then D % frag_m must be 0.");
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::row_major> frag_a_K;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_a_V;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_b;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, frag_m, frag_n, 16, KQ_acc_t>                      frag_c_KQ;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, frag_m, frag_n, 16, half>                          frag_c_VKQ;

    constexpr int KQ_stride_tc  = nwarps*frag_m; // Number of KQ rows calculated in parallel.
    constexpr int VKQ_ratio = KQ_stride_tc/VKQ_stride; // Number of parallel VKQ accumulators needed to keep all warps busy.
    static_assert(VKQ_ratio <= nwarps, "VKQ_ratio must be <= nwarps.");

    // Pad internal representation of KQ, KQV to reduce shared memory bank conflicts:
    constexpr int D_padded = D + 8;
    constexpr int kqs_padded = FATTN_KQ_STRIDE + 8;
    constexpr int kqar = sizeof(KQ_acc_t)/sizeof(half);

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f   = (const float *) (Q + nb02* blockIdx.y              + nb01*ic0);
    const half  * K_h   = (const half  *) (K + nb12*(blockIdx.y / gqa_ratio));
    const half  * V_h   = (const half  *) (V + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half  * maskh = (const half  *)  mask + (nb31/sizeof(half))* ic0;
    const half2 * mask2 = (const half2 *)  mask + (nb31/sizeof(half))*(ic0/2);

    const int stride_Q  = nb01 / sizeof(float);
    const int stride_KV = nb11 / sizeof(half);

    frag_b Q_b[D/16][ncols/frag_n];

    // A single buffer for temporarily holding tiles of KQ and VKQ parts:
    constexpr int mem_KQ = ncols*kqs_padded*kqar;
    constexpr int mem_VKQ_parts = VKQ_ratio*ncols*D_padded;
    __shared__ half KQ[mem_KQ >= mem_VKQ_parts ? mem_KQ : mem_VKQ_parts];
    float * KQ_f = (float *) KQ;
    half2 * KQ2 = (half2 *) KQ;

    float    KQ_rowsum_f[ncols/nwarps] = {0.0f};
    float       KQ_max_f[ncols/nwarps];
    float KQ_max_scale_f[ncols/nwarps] = {0.0f};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_f[j] = -FLT_MAX/2.0f;
    }

    half2    KQ_rowsum_h2[ncols/nwarps] = {{0.0f, 0.0f}};
    half2       KQ_max_h2[ncols/nwarps];
    half2 KQ_max_scale_h2[ncols/nwarps] = {{0.0f, 0.0f}};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_h2[j] = make_half2(-HALF_MAX_HALF, -HALF_MAX_HALF);
    }

    __shared__ half VKQ[ncols*D_padded]; // Accumulator for final VKQ slice.
    half2 * VKQ2 = (half2 *) VKQ;
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D/2 && i >= D/2) {
                break;
            }
            VKQ2[j*(D_padded/2) + i] = make_half2(0.0f, 0.0f);
        }
    }

    // Convert Q to half and apply scale, temporarily store in KQ:
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D && i >= D) {
                break;
            }
            KQ[j*D_padded + i] = ic0 + j < ne01 ? Q_f[j*stride_Q + i] * scale : 0.0f;
        }
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
    for (int k_VKQ_0 = ip*FATTN_KQ_STRIDE; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE) {
        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE; i_KQ_0 += KQ_stride_tc) {
            frag_c_KQ KQ_c[ncols/frag_n];
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                nvcuda::wmma::fill_fragment(KQ_c[j], 0.0f);
            }
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 16) {
                frag_a_K K_a;
                nvcuda::wmma::load_matrix_sync(K_a, K_h + (k_VKQ_0 + i_KQ_0 + frag_m*threadIdx.y)*stride_KV + k_KQ_0, stride_KV);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    nvcuda::wmma::mma_sync(KQ_c[j], K_a, Q_b[k_KQ_0/16][j], KQ_c[j]);
                }
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                nvcuda::wmma::store_matrix_sync((KQ_acc_t *) KQ + j0*kqs_padded + i_KQ_0 + frag_m*threadIdx.y, KQ_c[j0/frag_n], kqs_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (std::is_same<KQ_acc_t, float>::value) {
                float KQ_f_tmp[FATTN_KQ_STRIDE / WARP_SIZE];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0/WARP_SIZE] = KQ_f[j*kqs_padded + k];
                }

                float KQ_max_new = KQ_max_f[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0/WARP_SIZE] += mask ? __half2float(maskh[j*(nb31/sizeof(half)) + k_VKQ_0 + k]) : 0.0f;
                    KQ_max_new = max(KQ_max_new, KQ_f_tmp[k0/WARP_SIZE]);
                }
                KQ_max_new = warp_reduce_max(KQ_max_new);

                const float diff = KQ_max_f[j0/nwarps] - KQ_max_new;
                KQ_max_scale_f[j0/nwarps] = expf(diff);
                if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                    KQ_max_scale_f[j0/nwarps] = 0.0f;
                }
                KQ_max_f[j0/nwarps] = KQ_max_new;

                float KQ_rowsum_add = 0.0f;
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    const float diff = KQ_f_tmp[k0/WARP_SIZE] - KQ_max_f[j0/nwarps];
                    KQ_f_tmp[k0/WARP_SIZE] = expf(diff);
                    if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                        KQ_f_tmp[k0/WARP_SIZE] = 0.0f;
                    }
                    KQ_rowsum_add += KQ_f_tmp[k0/WARP_SIZE];
                    KQ[j*(kqar*kqs_padded) + k] = KQ_f_tmp[k0/WARP_SIZE];
                }
                KQ_rowsum_add = warp_reduce_sum(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_f[j0/nwarps] = KQ_max_scale_f[j0/nwarps]*KQ_rowsum_f[j0/nwarps] + KQ_rowsum_add;
            } else {
                half2 KQ2_tmp[FATTN_KQ_STRIDE/(2*WARP_SIZE)];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0/WARP_SIZE] = KQ2[j*(kqs_padded/2) + k];
                }

                half2 KQ_max_new = KQ_max_h2[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0/WARP_SIZE] += mask ? mask2[(j*ne11 + k_VKQ_0)/2 + k] : make_half2(0.0f, 0.0f);
                    KQ_max_new = ggml_cuda_hmax2(KQ_max_new, KQ2_tmp[k0/WARP_SIZE]);
                }
                KQ_max_new = __half2half2(warp_reduce_max(ggml_cuda_hmax(__low2half(KQ_max_new), __high2half(KQ_max_new))));
                const half2 diff = KQ_max_h2[j0/nwarps] - KQ_max_new;
                KQ_max_scale_h2[j0/nwarps] = h2exp(diff);
                const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                *((uint32_t *) &KQ_max_scale_h2[j0/nwarps]) &= ftz_mask;
                KQ_max_h2[j0/nwarps] = KQ_max_new;

                half2 KQ_rowsum_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                    const int k = k0 + threadIdx.x;

                    const half2 diff = KQ2_tmp[k0/WARP_SIZE] - KQ_max_h2[j0/nwarps];
                    KQ2_tmp[k0/WARP_SIZE] = h2exp(diff);
                    const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                    *((uint32_t *) &KQ2_tmp[k0/WARP_SIZE]) &= ftz_mask;
                    KQ_rowsum_add += KQ2_tmp[k0/WARP_SIZE];
                    KQ2[j*(kqs_padded/2) + k] = KQ2_tmp[k0/WARP_SIZE];
                }
                KQ_rowsum_add = warp_reduce_sum(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_h2[j0/nwarps] = KQ_max_scale_h2[j0/nwarps]*KQ_rowsum_h2[j0/nwarps] + KQ_rowsum_add;
            }
        }

        __syncthreads();

        frag_b KQ_b[FATTN_KQ_STRIDE/(VKQ_ratio*16)][ncols/frag_n];
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio)*16;
                nvcuda::wmma::load_matrix_sync(
                    KQ_b[k0/(VKQ_ratio*16)][j0/frag_n],
                    KQ + j0*(kqar*kqs_padded) + k,
                    kqar*kqs_padded);
            }
        }

        frag_c_VKQ VKQ_c[D/VKQ_stride][ncols/frag_n];
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += VKQ_stride) {
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                nvcuda::wmma::fill_fragment(VKQ_c[i_VKQ_0/VKQ_stride][j], 0.0f);
            }

#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio)*16;

                frag_a_V v_a;
                nvcuda::wmma::load_matrix_sync(v_a, V_h + (k_VKQ_0 + k)*stride_KV + i_VKQ_0 + frag_m*(threadIdx.y/VKQ_ratio), stride_KV);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    nvcuda::wmma::mma_sync(VKQ_c[i_VKQ_0/VKQ_stride][j], v_a, KQ_b[k0/(VKQ_ratio*16)][j], VKQ_c[i_VKQ_0/VKQ_stride][j]);
                }
            }
        }

        __syncthreads();

        const int offset_k = (threadIdx.y % VKQ_ratio) * (ncols*D_padded);
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += VKQ_stride) {
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                nvcuda::wmma::store_matrix_sync(
                    KQ + offset_k + j0*D_padded + i_KQ_0 + frag_m*(threadIdx.y/VKQ_ratio),
                    VKQ_c[i_KQ_0/VKQ_stride][j0/frag_n],
                    D_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            half2 VKQ_scale;
            if (std::is_same<KQ_acc_t, float>::value) {
                VKQ_scale = make_half2(KQ_max_scale_f[j0/nwarps], KQ_max_scale_f[j0/nwarps]);
            } else {
                VKQ_scale = KQ_max_scale_h2[j0/nwarps];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                if (i0 + WARP_SIZE > D/2 && i >= D/2) {
                    break;
                }

                half2 VKQ_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int l = 0; l < VKQ_ratio; ++l) {
                    VKQ_add += KQ2[l*(ncols*D_padded/2) + j*(D_padded/2) + i];
                }
                VKQ2[j*(D_padded/2) + i] = VKQ_scale*VKQ2[j*(D_padded/2) + i] + VKQ_add;
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j_VKQ = j0 + threadIdx.y;
        if (ic0 + j_VKQ >= ne01) {
            return;
        }
        const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;

        float KQ_rowsum_j;
        if (std::is_same<KQ_acc_t, float>::value) {
            KQ_rowsum_j = KQ_rowsum_f[j0/nwarps];
        } else {
            KQ_rowsum_j = __low2float(KQ_rowsum_h2[j0/nwarps]) + __high2float(KQ_rowsum_h2[j0/nwarps]);
        }

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (i0 + WARP_SIZE > D && i >= D) {
                break;
            }
            float dst_val = VKQ[j_VKQ*D_padded + i];
            if (parallel_blocks == 1) {
                dst_val /= KQ_rowsum_j;
            }
            dst[j_dst*gridDim.y*D + blockIdx.y*D + i] = dst_val;
        }

        if (parallel_blocks == 1 || threadIdx.x != 0) {
            continue;
        }

        float2 dst_meta_val;
        if (std::is_same<KQ_acc_t, float>::value) {
            dst_meta_val.x = KQ_max_f[j0/nwarps];
        } else {
            dst_meta_val.x = __low2float(KQ_max_h2[j0/nwarps]);
        }
        dst_meta_val.y = KQ_rowsum_j;
        dst_meta[(ic0 + j_VKQ)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = dst_meta_val;
    }
#else
   NO_DEVICE_CODE;
#endif // FP16_MMA_AVAILABLE
}

template<int D, int parallel_blocks> // D == head size
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst) {
#if FP16_AVAILABLE
    VKQ_parts += parallel_blocks*D * gridDim.y*blockIdx.x;
    VKQ_meta  += parallel_blocks   * gridDim.y*blockIdx.x;
    dst       +=                 D * gridDim.y*blockIdx.x;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ float2 meta[parallel_blocks];
    if (tid < 2*parallel_blocks) {
        ((float *) meta)[threadIdx.x] = ((const float *)VKQ_meta) [blockIdx.y*(2*parallel_blocks) + tid];
    }

    __syncthreads();

    float kqmax = meta[0].x;
#pragma unroll
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
#pragma unroll
    for (int l = 0; l < parallel_blocks; ++l) {
        const float diff = meta[l].x - kqmax;
        const float KQ_max_scale = expf(diff);
        const uint32_t ftz_mask = 0xFFFFFFFF * (diff > SOFTMAX_FTZ_THRESHOLD);
        *((uint32_t *) &KQ_max_scale) &= ftz_mask;

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*gridDim.y*D + blockIdx.y*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[blockIdx.y*D + tid] = VKQ_numerator / VKQ_denominator;
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

constexpr int get_max_power_of_2(int x) {
    return x % 2 == 0 ? 2*get_max_power_of_2(x/2) : 1;
}

static_assert(get_max_power_of_2(1) == 1, "Test failed.");
static_assert(get_max_power_of_2(2) == 2, "Test failed.");
static_assert(get_max_power_of_2(4) == 4, "Test failed.");
static_assert(get_max_power_of_2(6) == 2, "Test failed.");

// Number of VKQ rows calculated in parallel:
constexpr int get_VKQ_stride(int D, int nwarps, int frag_m) {
    return (get_max_power_of_2(D/frag_m) < nwarps ? get_max_power_of_2(D/frag_m) : nwarps)*frag_m;
}

static_assert(get_VKQ_stride(128, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride(128, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride(128, 4, 32) == 128, "Test failed.");
static_assert(get_VKQ_stride( 64, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride( 64, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 64, 4, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 80, 1, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 2, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 4, 16) ==  16, "Test failed.");

template <int D, int cols_per_block, int parallel_blocks> void launch_fattn_vec_f16(
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

    flash_attn_vec_ext_f16<D, cols_per_block, parallel_blocks>
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

template <int D, int cols_per_block, int nwarps, int parallel_blocks, typename KQ_acc_t> void launch_fattn_f16_impl(
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V, ggml_tensor * KQV, const ggml_tensor * mask,
        ggml_cuda_pool & pool, cudaStream_t main_stream
) {
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
    }

    constexpr int  frag_m = (cols_per_block) == 8 && (D) % 32 == 0 ? 32 : 16;
    const     dim3 block_dim(WARP_SIZE, nwarps, 1);
    const     dim3 blocks_num(parallel_blocks*(Q->ne[1] + cols_per_block - 1) / cols_per_block, Q->ne[2], Q->ne[3]);
    const     int  shmem = 0;

    float scale;
    memcpy(&scale, KQV->op_params, sizeof(float));

    flash_attn_ext_f16<D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m), parallel_blocks, KQ_acc_t>
        <<<blocks_num, block_dim, shmem, main_stream>>> (
                (const char *) Q->data,
                (const char *) K->data,
                (const char *) V->data,
                mask ? ((const char *) mask->data) : nullptr,
                (parallel_blocks) == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
                scale,
                Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                Q->nb[1], Q->nb[2], Q->nb[3],
                K->nb[1], K->nb[2], K->nb[3],
                KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                );
    CUDA_CHECK(cudaGetLastError());

    if ((parallel_blocks) == 1) {
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

template <int D, int cols_per_block, int nwarps, typename KQ_acc_t> void launch_fattn_f16(
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V, ggml_tensor * KQV, const ggml_tensor * mask,
        const int nsm, ggml_cuda_pool & pool, cudaStream_t main_stream
) {
    const int blocks_num_pb1 = ((Q->ne[1] + cols_per_block - 1) / cols_per_block)*Q->ne[2]*Q->ne[3];

    if (4*blocks_num_pb1 < 2*nsm) {
        launch_fattn_f16_impl<D, cols_per_block, nwarps, 4, KQ_acc_t>(Q, K, V, KQV, mask, pool, main_stream);
        return;
    }
    if (2*blocks_num_pb1 < 2*nsm) {
        launch_fattn_f16_impl<D, cols_per_block, nwarps, 2, KQ_acc_t>(Q, K, V, KQV, mask, pool, main_stream);
        return;
    }
    launch_fattn_f16_impl<D, cols_per_block, nwarps, 1, KQ_acc_t>(Q, K, V, KQV, mask, pool, main_stream);
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

    GGML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");

    ggml_cuda_set_device(ctx.device);

    const int cc  = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    const int32_t precision = KQV->op_params[1];

    if (!fp16_mma_available(cc)) {
        GGML_ASSERT(precision == GGML_PREC_DEFAULT);
        GGML_ASSERT(Q->ne[0] == 64 || Q->ne[0] == 128 && "FlashAttention without tensor cores only supports head sizes 64 and 128.");

        if (Q->ne[1] == 1) {
            constexpr int cols_per_block = 1;
            constexpr int parallel_blocks = 4;
            switch (Q->ne[0]) {
                case 64:
                    launch_fattn_vec_f16< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                    break;
                case 128:
                    launch_fattn_vec_f16<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
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
                    launch_fattn_vec_f16< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                    break;
                case 128:
                    launch_fattn_vec_f16<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
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
                    launch_fattn_vec_f16< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                    break;
                case 128:
                    launch_fattn_vec_f16<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
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
                    launch_fattn_vec_f16< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                    break;
                case 128:
                    launch_fattn_vec_f16<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
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
                launch_fattn_vec_f16< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_vec_f16<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    if (precision != GGML_PREC_DEFAULT) {
        if (Q->ne[1] <= 32 || Q->ne[0] > 128) {
            constexpr int cols_per_block = 16;
            constexpr int nwarps         =  4;
            switch (Q->ne[0]) {
                case 64:
                    launch_fattn_f16< 64, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 80:
                    launch_fattn_f16< 80, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 96:
                    launch_fattn_f16< 96, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 112:
                    launch_fattn_f16<112, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 128:
                    launch_fattn_f16<128, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 256:
                    launch_fattn_f16<256, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                default:
                    GGML_ASSERT(false);
                    break;
            }
        } else {
            constexpr int cols_per_block = 32;
            constexpr int nwarps         =  4;
            switch (Q->ne[0]) {
                case 64:
                    launch_fattn_f16< 64, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 80:
                    launch_fattn_f16< 80, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 96:
                    launch_fattn_f16< 96, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 112:
                    launch_fattn_f16<112, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                case 128:
                    launch_fattn_f16<128, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                    break;
                // case 256:
                //     launch_fattn_f16<256, cols_per_block, nwarps, float>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                //     break;
                default:
                    GGML_ASSERT(false);
                    break;
            }
        }
        return;
    }

    if (Q->ne[1] == 1 && Q->ne[0] % (2*WARP_SIZE) == 0) {
        constexpr int cols_per_block = 1;
        constexpr int parallel_blocks = 4;
        switch (Q->ne[0]) {
            case 64:
                launch_fattn_vec_f16< 64, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_vec_f16<128, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            case 256:
                launch_fattn_vec_f16<256, cols_per_block, parallel_blocks>(Q, K, V, KQV, mask, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    if (Q->ne[1] <= 8 && Q->ne[0] % WARP_SIZE == 0) {
        constexpr int cols_per_block = 8;
        constexpr int nwarps         = 4;
        switch (Q->ne[0]) {
            case 64:
                launch_fattn_f16< 64, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 96:
                launch_fattn_f16< 96, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_f16<128, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 256:
                launch_fattn_f16<256, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    if (Q->ne[1] <= 32) {
        constexpr int cols_per_block = 16;
        constexpr int nwarps         =  4;
        switch (Q->ne[0]) {
            case 64:
                launch_fattn_f16< 64, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 80:
                launch_fattn_f16< 80, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 96:
                launch_fattn_f16< 96, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 112:
                launch_fattn_f16<112, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 128:
                launch_fattn_f16<128, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            case 256:
                launch_fattn_f16<256, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        return;
    }

    constexpr int cols_per_block = 32;
    constexpr int nwarps         =  4;
    switch (Q->ne[0]) {
        case 64:
            launch_fattn_f16< 64, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        case 80:
            launch_fattn_f16< 80, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        case 96:
            launch_fattn_f16< 96, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        case 112:
            launch_fattn_f16<112, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        case 128:
            launch_fattn_f16<128, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        case 256:
            launch_fattn_f16<256, cols_per_block, nwarps, half>(Q, K, V, KQV, mask, nsm, ctx.pool(), ctx.stream());
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
    return;
}
