#include "common.cuh"
#include "fattn.cuh"

#include <mma.h>

#define FATTN_KQ_STRIDE 256
#define HALF_MAX_HALF   __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.

template<int D, int parallel_blocks> // D == head size
__launch_bounds__(((D + WARP_SIZE - 1) / WARP_SIZE)*WARP_SIZE, 1)
static __global__ void flash_attn_vec_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float * __restrict__ dst,
        half2 * __restrict__ dst_meta,
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
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.y);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.y / gqa_ratio));
    const half   * V_h   = (const half   *) (V    + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half   * maskh = (const half   *)  mask;

    if (parallel_blocks == 1) {
        Q_f2  += blockIdx.x*nb01/sizeof(float2);
        maskh += blockIdx.x*ne11;
    }

    const int stride_KV  = nb11 / sizeof(half);
    const int stride_KV2 = nb11 / sizeof(half2);

    constexpr int nwarps = (D + WARP_SIZE - 1) / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nwarps*WARP_SIZE);

    __shared__ half KQ[nwarps*WARP_SIZE];
    KQ[tid] = -INFINITY;
    half2 * KQ2 = (half2 *) KQ;

    half kqmax = -HALF_MAX_HALF;
    half kqsum = 0.0f;

    __shared__ half kqmax_shared[WARP_SIZE];
    __shared__ half kqsum_shared[WARP_SIZE];
    if (threadIdx.y == 0) {
        kqmax_shared[threadIdx.x] = -HALF_MAX_HALF;
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

    const int k_start  = parallel_blocks == 1 ? 0 : blockIdx.x*D;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*D) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        half kqmax_new = kqmax;
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
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

        if (tid < D) {
#pragma unroll
            for (int k0 = 0; k0 < D; k0 += 2) {
                if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k0 >= ne11) {
                    break;
                }

                half2 V_k;
                reinterpret_cast<half&>(V_k.x) = V_h[(k_VKQ_0 + k0 + 0)*stride_KV + tid];
                reinterpret_cast<half&>(V_k.y) = V_h[(k_VKQ_0 + k0 + 1)*stride_KV + tid];
                VKQ += V_k*KQ2[k0/2];
            }
        }
    }

    if (tid >= D) {
        kqsum = 0.0f;
    }

    kqsum = warp_reduce_sum(kqsum);
    if (threadIdx.x == 0) {
        kqsum_shared[threadIdx.y] = kqsum;
    }
    __syncthreads();
    kqsum = kqsum_shared[threadIdx.x];
    kqsum = warp_reduce_sum(kqsum);

    if (tid >= D) {
        return;
    }

    if (parallel_blocks == 1) {
        dst[D*gridDim.y*blockIdx.x + D*blockIdx.y + tid] = (__low2half(VKQ) + __high2half(VKQ)) / kqsum;
    } else {
        dst[D*gridDim.y*blockIdx.x + D*blockIdx.y + tid] = (__low2half(VKQ) + __high2half(VKQ));

        if (tid == 0) {
            dst_meta[blockIdx.y*parallel_blocks + blockIdx.x] = make_half2(kqmax, kqsum);
        }
    }
#else
   NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
}

template<int D, int ncols, int nwarps, int VKQ_stride, int parallel_blocks> // D == head size, VKQ_stride == num VKQ rows calculated in parallel
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float * __restrict__ dst,
        half2 * __restrict__ dst_meta,
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
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_VOLTA
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.
    static_assert(D <= FATTN_KQ_STRIDE, "D must be <= FATTN_KQ_STRIDE.");
    static_assert(ncols == 8 || ncols % 16 == 0, "ncols must be 8 or a multiple of 16.");
    constexpr int frag_m = ncols == 8 ? 32 : 16;
    constexpr int frag_n = ncols == 8 ?  8 : 16;
    static_assert(D % frag_m == 0, "If ncols == 8 then D % frag_m must be 0.");
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::row_major> frag_a_K;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_a_V;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    frag_m, frag_n, 16, half, nvcuda::wmma::col_major> frag_b;
    typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, frag_m, frag_n, 16, half>                          frag_c;

    constexpr int KQ_stride_tc  = nwarps*frag_m; // Number of KQ rows calculated in parallel.
    constexpr int VKQ_ratio = KQ_stride_tc/VKQ_stride; // Number of parallel VKQ accumulators needed to keep all warps busy.
    static_assert(VKQ_ratio <= nwarps, "VKQ_ratio must be <= nwarps.");

    // Pad internal representation of KQ, KQV to reduce shared memory bank conflicts:
    constexpr int D_padded = D + 8;
    constexpr int kqs_padded = FATTN_KQ_STRIDE + 8;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f   = (const float *) (Q + nb02* blockIdx.y);
    const half  * K_h   = (const half  *) (K + nb12*(blockIdx.y / gqa_ratio));
    const half  * V_h   = (const half  *) (V + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half2 * mask2 = (half2       *)  mask;

    if (parallel_blocks == 1) {
        Q_f   += blockIdx.x * ncols*nb01/sizeof(float);
        mask2 += blockIdx.x * ncols*ne11/2;
    }

    const int stride_Q  = nb01 / sizeof(float);
    const int stride_KV = nb11 / sizeof(half);

    frag_b Q_b[D/16][ncols/frag_n];

    // A single buffer for temporarily holding tiles of KQ and VKQ parts:
    constexpr int mem_KQ = ncols*kqs_padded;
    constexpr int mem_VKQ_parts = VKQ_ratio*ncols*D_padded;
    __shared__ half KQ[mem_KQ >= mem_VKQ_parts ? mem_KQ : mem_VKQ_parts];
    half2 * KQ2 = (half2 *) KQ;

    half2    KQ_rowsum[ncols/nwarps] = {{ 0.0f,           0.0f}};
    half2       KQ_max[ncols/nwarps] = {{-HALF_MAX_HALF, -HALF_MAX_HALF}};
    half2 KQ_max_scale[ncols/nwarps] = {{ 0.0f,           0.0f}};

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
            if (parallel_blocks == 1) {
                KQ[j*D_padded + i] = ncols*blockIdx.x + j < ne01 ? Q_f[j*stride_Q + i] * scale : 0.0f;
            } else {
                KQ[j*D_padded + i] = j == 0 ? Q_f[j*stride_Q + i] * scale : 0.0f;
            }
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
    const int k_start  = parallel_blocks == 1 ? 0 : blockIdx.x*FATTN_KQ_STRIDE;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE) {
        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE; i_KQ_0 += KQ_stride_tc) {
            frag_c KQ_c[ncols/frag_n];
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
                nvcuda::wmma::store_matrix_sync(KQ + j0*kqs_padded + i_KQ_0 + frag_m*threadIdx.y, KQ_c[j0/frag_n], kqs_padded, nvcuda::wmma::mem_col_major);
            }
        }

        __syncthreads();

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            half2 KQ_max_new = KQ_max[j0/nwarps];
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                const int k = k0 + threadIdx.x;
                KQ_max_new = __hmax2(KQ_max_new, KQ2[j*(kqs_padded/2) + k]);
            }
            KQ_max_new = __half2half2(warp_reduce_max(__hmax(__low2half(KQ_max_new), __high2half(KQ_max_new))));
            KQ_max_scale[j0/nwarps] = h2exp(KQ_max[j0/nwarps] - KQ_max_new);
            KQ_max[j0/nwarps] = KQ_max_new;

            half2 KQ_rowsum_add = make_half2(0.0f, 0.0f);
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += WARP_SIZE) {
                const int k = k0 + threadIdx.x;

                half2 val = KQ2[j*(kqs_padded/2) + k];
                val += mask ? mask2[(j*ne11 + k_VKQ_0)/2 + k] : make_half2(0.0f, 0.0f);
                val = h2exp(val - KQ_max[j0/nwarps]);
                KQ_rowsum_add += val;
                KQ2[j*(kqs_padded/2) + k] = val;
            }
            KQ_rowsum_add = warp_reduce_sum(KQ_rowsum_add);

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[j0/nwarps] = KQ_max_scale[j0/nwarps]*KQ_rowsum[j0/nwarps] + KQ_rowsum_add;
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
                    KQ + j0*kqs_padded + k,
                    kqs_padded);
            }
        }

        frag_c VKQ_c[D/VKQ_stride][ncols/frag_n];
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
                VKQ2[j*(D_padded/2) + i] = KQ_max_scale[j0/nwarps]*VKQ2[j*(D_padded/2) + i] + VKQ_add;
            }
        }

        __syncthreads();
    }

    if (parallel_blocks == 1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            if (ncols*blockIdx.x + j >= ne01) {
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
    } else {
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += nwarps*WARP_SIZE) {
            const int i = i0 + threadIdx.y*WARP_SIZE + threadIdx.x;
            if (i0 + nwarps*WARP_SIZE > D && i >= D) {
                return;
            }
            dst[D*gridDim.y*blockIdx.x + D*blockIdx.y + i] = VKQ[i];
        }

        if (threadIdx.y == 0 && threadIdx.x == 0) {
            dst_meta[blockIdx.y*parallel_blocks + blockIdx.x] = make_half2(
                __low2half(KQ_max[0]), __low2half(KQ_rowsum[0]) + __high2half(KQ_rowsum[0]));
        }
    }
#else
   NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_VOLTA
}

template<int D, int parallel_blocks> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float * __restrict__ VKQ_parts,
        const half2 * __restrict__ VKQ_meta,
        float * __restrict__ dst) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ half2 meta[parallel_blocks];
    if (tid < parallel_blocks) {
        meta[threadIdx.x] = VKQ_meta[blockIdx.y*parallel_blocks + tid];
    }

    __syncthreads();

    half kqmax = __low2half(meta[0]);
#pragma unroll
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = __hmax(kqmax, __low2half(meta[l]));
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
#pragma unroll
    for (int l = 0; l < parallel_blocks; ++l) {
        float KQ_max_scale = hexp(__low2half(meta[l]) - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*gridDim.y*D + blockIdx.y*D + tid];
        VKQ_denominator += KQ_max_scale * __high2float(meta[l]);
    }

    dst[blockIdx.y*D + tid] = VKQ_numerator / VKQ_denominator;
#else
   NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
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

#define FATTN_SWITCH_CASE(D, ncols, nwarps)                                        \
    case ncols: {                                                                  \
        constexpr int frag_m = (ncols) == 8 && (D) % 32 == 0 ? 32 : 16;            \
        flash_attn_ext_f16<D, ncols, nwarps, get_VKQ_stride(D, nwarps, frag_m), 1> \
            <<<blocks_num, block_dim, shmem, main_stream>>> (                      \
                    (const char *) Q->data,                                        \
                    (const char *) K->data,                                        \
                    (const char *) V->data,                                        \
                    mask ? ((const char *) mask->data) : nullptr,                  \
                    (float *) KQV->data, nullptr,                                  \
                    scale,                                                         \
                    Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],                        \
                    K->ne[0], K->ne[1], K->ne[2], K->ne[3],                        \
                    mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,               \
                    Q->nb[1], Q->nb[2], Q->nb[3],                                  \
                    K->nb[1], K->nb[2], K->nb[3],                                  \
                    KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]                 \
                    );                                                             \
        }                                                                          \
        break;                                                                     \

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

    const cudaStream_t main_stream = ctx.stream();

    float scale;
    memcpy(&scale, KQV->op_params, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int parallel_blocks = 4;

        ggml_cuda_pool_alloc<float> dst_tmp(ctx.pool());
        ggml_cuda_pool_alloc<half2> dst_tmp_meta(ctx.pool());

        const int nwarps = (Q->ne[0] + WARP_SIZE - 1) / WARP_SIZE;
        const dim3 blocks_num(parallel_blocks*Q->ne[1], Q->ne[2], Q->ne[3]);
        const dim3 block_dim(WARP_SIZE, nwarps, 1);
        const int shmem = 0;

        // Performance of the vector kernel is very bad for head sizes 80 and 112, use the tensor core kernel instead:
        constexpr int nwarps_tc = 4;
        constexpr dim3 block_dim_tc(WARP_SIZE, nwarps_tc, 1);

        const dim3 blocks_num_combine(1, blocks_num.y, blocks_num.z);
        const dim3 block_dim_combine(Q->ne[0], 1, 1);
        const int shmem_combine = 0;

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }

        switch (Q->ne[0]) {
            case 64:
                flash_attn_vec_ext_f16<64, parallel_blocks>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            parallel_blocks == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                if (parallel_blocks == 1) {
                    break;
                }
                CUDA_CHECK(cudaGetLastError());
                flash_attn_combine_results<64, parallel_blocks>
                    <<<blocks_num_combine, block_dim_combine, shmem_combine, main_stream>>>
                    (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
                break;
            case 80:
                flash_attn_ext_f16<80, 16, nwarps_tc, get_VKQ_stride(80, nwarps_tc, 16), parallel_blocks>
                    <<<blocks_num, block_dim_tc, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            parallel_blocks == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                if (parallel_blocks == 1) {
                    break;
                }
                CUDA_CHECK(cudaGetLastError());
                flash_attn_combine_results<80, parallel_blocks>
                    <<<blocks_num_combine, block_dim_combine, shmem_combine, main_stream>>>
                    (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
                break;
            case 96:
                flash_attn_vec_ext_f16<96, parallel_blocks>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            parallel_blocks == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                if (parallel_blocks == 1) {
                    break;
                }
                CUDA_CHECK(cudaGetLastError());
                flash_attn_combine_results<96, parallel_blocks>
                    <<<blocks_num_combine, block_dim_combine, shmem_combine, main_stream>>>
                    (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
                break;
            case 112:
                flash_attn_vec_ext_f16<112, parallel_blocks>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            parallel_blocks == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                if (parallel_blocks == 1) {
                    break;
                }
                CUDA_CHECK(cudaGetLastError());
                flash_attn_combine_results<112, parallel_blocks>
                    <<<blocks_num_combine, block_dim_combine, shmem_combine, main_stream>>>
                    (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
                break;
            case 128:
                flash_attn_vec_ext_f16<128, parallel_blocks>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            parallel_blocks == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                if (parallel_blocks == 1) {
                    break;
                }
                CUDA_CHECK(cudaGetLastError());
                flash_attn_combine_results<128, parallel_blocks>
                    <<<blocks_num_combine, block_dim_combine, shmem_combine, main_stream>>>
                    (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
                break;
            case 256:
                flash_attn_vec_ext_f16<256, parallel_blocks>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? ((const char *) mask->data) : nullptr, // Mask
                            parallel_blocks == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
                if (parallel_blocks == 1) {
                    break;
                }
                CUDA_CHECK(cudaGetLastError());
                flash_attn_combine_results<256, parallel_blocks>
                    <<<blocks_num_combine, block_dim_combine, shmem_combine, main_stream>>>
                    (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data);
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int cols_per_block;
    if (Q->ne[1] >= 64 && (Q->ne[0] <= 128 || ggml_cuda_info().devices[ctx.device].cc >= CC_AMPERE)) {
        cols_per_block = 32;
    } else if (Q->ne[1] >= 32 || Q->ne[0] % 32 != 0) {
        cols_per_block = 16;
    } else  {
        cols_per_block = 8;
    }
    constexpr int nwarps = 4;
    const dim3 blocks_num((Q->ne[1] + cols_per_block - 1) / cols_per_block, Q->ne[2], Q->ne[3]);
    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    const size_t shmem = 0;

    switch (Q->ne[0]) {
        case 64: switch (cols_per_block) {
            FATTN_SWITCH_CASE(64,  8, nwarps);
            FATTN_SWITCH_CASE(64, 16, nwarps);
            FATTN_SWITCH_CASE(64, 32, nwarps);
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        case 80: switch (cols_per_block) {
            // FATTN_SWITCH_CASE(80,  8, nwarps);
            FATTN_SWITCH_CASE(80, 16, nwarps);
            FATTN_SWITCH_CASE(80, 32, nwarps);
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        case 96: switch (cols_per_block) {
            FATTN_SWITCH_CASE(96,  8, nwarps);
            FATTN_SWITCH_CASE(96, 16, nwarps);
            FATTN_SWITCH_CASE(96, 32, nwarps);
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        case 112: switch (cols_per_block) {
            // FATTN_SWITCH_CASE(112,  8, nwarps);
            FATTN_SWITCH_CASE(112, 16, nwarps);
            FATTN_SWITCH_CASE(112, 32, nwarps);
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        case 128: switch (cols_per_block) {
            FATTN_SWITCH_CASE(128,  8, nwarps);
            FATTN_SWITCH_CASE(128, 16, nwarps);
            FATTN_SWITCH_CASE(128, 32, nwarps);
            default:
                fprintf(stderr, "cols_per_block == %d not implemented.\n", cols_per_block);
                GGML_ASSERT(false);
                break;
        } break;
        case 256: switch (cols_per_block) {
            FATTN_SWITCH_CASE(256,  8, nwarps);
            FATTN_SWITCH_CASE(256, 16, nwarps);
            FATTN_SWITCH_CASE(256, 32, nwarps);
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
