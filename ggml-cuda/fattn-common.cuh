#include "common.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
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
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3);

typedef half (*vec_dot_KQ_f16_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);
typedef float (*vec_dot_KQ_f32_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

template<typename T, int D>
__device__ __forceinline__ T vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
#if __CUDA_ARCH__ > MIN_CC_DP4A

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    half sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        const int v = (get_int_from_uint8(K_q4_0[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/WARP_SIZE];

        const int sumi = __dp4a(v, u, 0);

#if FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 sum2 = __half2half2(K_q4_0[ib].d) * Q_ds[k_KQ_0/WARP_SIZE];
            sum += (T) (((half) sumi)*__low2half(sum2) - __high2half(sum2) /* *8/QI8_1 == 1 */);
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            sum += (T) (__half2float(K_q4_0[ib].d) * (sumi*Q_ds[k_KQ_0/WARP_SIZE].x - (8/QI8_1)*Q_ds[k_KQ_0/WARP_SIZE].y));
        }
    }

    return sum;
#else
    GGML_UNUSED(K_c);
    GGML_UNUSED(Q_v);
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);
    NO_DEVICE_CODE;
#endif  // __CUDA_ARCH__ > MIN_CC_DP4A
}

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
#if __CUDA_ARCH__ > MIN_CC_DP4A

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        const int v = (get_int_from_uint8_aligned(K_q4_1[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/WARP_SIZE];

        const int sumi = __dp4a(v, u, 0);

#if FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 d4d8_m4s8 = K_q4_1[ib].dm * Q_ds[k_KQ_0/WARP_SIZE];
            const half2 sumid4d8_m4s8scaled = d4d8_m4s8 * make_half2(sumi, 1.0f/QI8_1);
            sum += (T) (__low2half(sumid4d8_m4s8scaled) + __high2half(sumid4d8_m4s8scaled));
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            const float sumid4d8   =  __low2float(K_q4_1[ib].dm)*Q_ds[k_KQ_0/WARP_SIZE].x * sumi;
            const float m4s8scaled = __high2float(K_q4_1[ib].dm)*Q_ds[k_KQ_0/WARP_SIZE].y / QI8_1;

            sum += (T) (sumid4d8 + m4s8scaled);
        }
    }

    return sum;
#else
    GGML_UNUSED(K_c);
    GGML_UNUSED(Q_v);
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);
    NO_DEVICE_CODE;
#endif  // __CUDA_ARCH__ > MIN_CC_DP4A
}

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
#if __CUDA_ARCH__ > MIN_CC_DP4A

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v = (get_int_from_uint8(K_q5_0[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int vh = get_int_from_uint8(K_q5_0[ib].qh, 0) >> (iqs8 * QI5_0);
        v |= (vh <<  4) & 0x00000010; // 0 ->  4
        v |= (vh << 11) & 0x00001000; // 1 -> 12
        v |= (vh << 18) & 0x00100000; // 2 -> 20
        v |= (vh << 25) & 0x10000000; // 3 -> 28

        const int u = Q_q8[k_KQ_0/WARP_SIZE];

        const int sumi = __dp4a(v, u, 0);

#if FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 sum2 = __half2half2(K_q5_0[ib].d) * Q_ds[k_KQ_0/WARP_SIZE];
            sum += (T) (((half) sumi)*__low2half(sum2) - __high2half(sum2)*__float2half(2.0f)) /* *16/QI8_1 == 2 */;
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            sum += (T) (__half2float(K_q5_0[ib].d) * (sumi*Q_ds[k_KQ_0/WARP_SIZE].x - (16/QI8_1)*Q_ds[k_KQ_0/WARP_SIZE].y));
        }
    }

    return sum;
#else
    GGML_UNUSED(K_c);
    GGML_UNUSED(Q_v);
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);
    NO_DEVICE_CODE;
#endif  // __CUDA_ARCH__ > MIN_CC_DP4A
}

template<typename T, int D>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
#if __CUDA_ARCH__ > MIN_CC_DP4A

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v = (get_int_from_uint8(K_q5_1[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int vh = get_int_from_uint8(K_q5_1[ib].qh, 0) >> (iqs8 * QI5_1);
        v |= (vh <<  4) & 0x00000010; // 0 ->  4
        v |= (vh << 11) & 0x00001000; // 1 -> 12
        v |= (vh << 18) & 0x00100000; // 2 -> 20
        v |= (vh << 25) & 0x10000000; // 3 -> 28

        const int u = Q_q8[k_KQ_0/WARP_SIZE];

        const int sumi = __dp4a(v, u, 0);

#if FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 d5d8_m5s8 = K_q5_1[ib].dm * Q_ds[k_KQ_0/WARP_SIZE];
            const half2 sumid5d8_m5s8scaled = d5d8_m5s8 * make_half2(sumi, 1.0f/QI8_1);
            sum += (T) (__low2half(sumid5d8_m5s8scaled) + __high2half(sumid5d8_m5s8scaled));
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            const float sumid5d8   =  __low2float(K_q5_1[ib].dm)*Q_ds[k_KQ_0/WARP_SIZE].x * sumi;
            const float m5s8scaled = __high2float(K_q5_1[ib].dm)*Q_ds[k_KQ_0/WARP_SIZE].y / QI8_1;

            sum += (T) (sumid5d8 + m5s8scaled);
        }
    }

    return sum;
#else
    GGML_UNUSED(K_c);
    GGML_UNUSED(Q_v);
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);
    NO_DEVICE_CODE;
#endif  // __CUDA_ARCH__ > MIN_CC_DP4A
}

template <typename T, int D>
__device__ __forceinline__ T vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
#if __CUDA_ARCH__ > MIN_CC_DP4A

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/sizeof(int); k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        const int v = get_int_from_int8(K_q8_0[ib].qs, iqs);

        T Q_d;
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;
            Q_d = __low2half(Q_ds[k_KQ_0/WARP_SIZE]);
        } else {
            const float2 * Q_ds = (const float2 *) Q_ds_v;
            Q_d = Q_ds[k_KQ_0/WARP_SIZE].x;
        }

        sum += vec_dot_q8_0_q8_1_impl<T, 1>(&v, &Q_q8[k_KQ_0/WARP_SIZE], K_q8_0[ib].d, Q_d);
    }

    return sum;
#else
    GGML_UNUSED(K_c);
    GGML_UNUSED(Q_v);
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);
    NO_DEVICE_CODE;
#endif  // __CUDA_ARCH__ > MIN_CC_DP4A
}

template <typename T, int D>
__device__ __forceinline__ T vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

#if FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        const half2 * Q_h2 = (const half2 *) Q_v;

        half2 sum2 = make_half2(0.0f, 0.0f);

#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
            const int k_KQ = k_KQ_0 + threadIdx.x;

            const half2 K_ik = K_h2[k_KQ];
            sum2 += K_ik * Q_h2[k_KQ_0/WARP_SIZE];
        }

        return __low2half(sum2) + __high2half(sum2);
    }
#endif // FP16_AVAILABLE

    const float2 * Q_f2 = (const float2 *) Q_v;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const half2 K_ik = K_h2[k_KQ];
        sum +=  __low2float(K_ik) * Q_f2[k_KQ_0/WARP_SIZE].x;
        sum += __high2float(K_ik) * Q_f2[k_KQ_0/WARP_SIZE].y;
    }

    return sum;
}

template <typename Tds>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < sizeof(int); ++l) {
        vals[l] = scale * x[4*threadIdx.x + l];
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < sizeof(int); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < sizeof(int); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef half  (*dequantize_1_f16_t)(const void *, const int64_t);
typedef float (*dequantize_1_f32_t)(const void *, const int64_t);

template <typename T>
__device__ __forceinline__ T dequantize_1_q4_0(const void * __restrict__ vx, const int64_t i) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i          /  QK4_0;
    const int     iqs   =  i          % (QK4_0/2);
    const int     shift = (i % QK4_0) / (QK4_0/2);

    const T   d  = x[ib].d;
    const int q0 = x[ib].qs[iqs];
    const int q  = ((q0 >> (4*shift)) & 0x0F) - 8;

#if FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half) d)*((half) q);
    }
#endif // FP16_AVAILABLE

    return ((float) d)*((float) q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q4_1(const void * __restrict__ vx, const int64_t i) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i          /  QK4_1;
    const int     iqs   =  i          % (QK4_1/2);
    const int     shift = (i % QK4_1) / (QK4_1/2);

    const half2 dm = x[ib].dm;
    const int   q0 = x[ib].qs[iqs];
    const int   q  = ((q0 >> (4*shift)) & 0x0F);

#if FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return __low2half(dm)*((half) q) + __high2half(dm);
    }
#endif // FP16_AVAILABLE

    return __low2float(dm)*((float) q) + __high2float(dm);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q5_0(const void * __restrict__ vx, const int64_t i) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i          /  QK5_0;
    const int     idq   =  i          %  QK5_0;
    const int     iqs   =  i          % (QK5_0/2);
    const int     shift = (i % QK5_0) / (QK5_0/2);

    const T   d   = x[ib].d;
    const int ql0 = x[ib].qs[iqs];
    const int qh0 = get_int_from_uint8(x[ib].qh, 0);
    const int ql  = ((ql0 >> (4*shift)) & 0x0F);
    const int qh  = ((qh0 >> idq) << 4) & 0x10;
    const int q   = (ql | qh) - 16;

#if FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half) d)*((half) q);
    }
#endif // FP16_AVAILABLE

    return ((float) d)*((float) q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q5_1(const void * __restrict__ vx, const int64_t i) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i          /  QK5_1;
    const int     idq   =  i          %  QK5_1;
    const int     iqs   =  i          % (QK5_1/2);
    const int     shift = (i % QK5_1) / (QK5_1/2);

    const half2 dm  = x[ib].dm;
    const int   ql0 = x[ib].qs[iqs];
    const int   qh0 = get_int_from_uint8_aligned(x[ib].qh, 0);
    const int   ql  = ((ql0 >> (4*shift)) & 0x0F);
    const int   qh  = ((qh0 >> idq) << 4) & 0x10;
    const int   q   = (ql | qh);

#if FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return __low2half(dm)*((half) q) + __high2half(dm);
    }
#endif // FP16_AVAILABLE

    return __low2float(dm)*((float) q) + __high2float(dm);
}

template <typename T>
__device__ __forceinline__ T dequantize_1_q8_0(const void * __restrict__ vx, const int64_t i) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i / QK8_0;
    const int     iqs = i % QK8_0;

    const T   d = x[ib].d;
    const int q = x[ib].qs[iqs];

#if FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half) d)*((half) q);
    }
#endif // FP16_AVAILABLE

    return ((float) d)*((float) q);
}

template <typename T>
__device__ __forceinline__ T dequantize_1_f16(const void * __restrict__ vx, const int64_t i) {
    const half * x = (const half *) vx;

    return x[i];
}

template<int D, int parallel_blocks> // D == head size
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst) {
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
}

// Aliases for FATTN_VEC_CASE macro:
static constexpr ggml_type ggml_type_q4_0 = GGML_TYPE_Q4_0;
static constexpr ggml_type ggml_type_q4_1 = GGML_TYPE_Q4_1;
static constexpr ggml_type ggml_type_q5_0 = GGML_TYPE_Q5_0;
static constexpr ggml_type ggml_type_q5_1 = GGML_TYPE_Q5_1;
static constexpr ggml_type ggml_type_q8_0 = GGML_TYPE_Q8_0;
static constexpr ggml_type ggml_type_f16  = GGML_TYPE_F16;

typedef half f16;
typedef float f32;

#define FATTN_VEC_CASE(type_VKQ, D, type_suffix_K, type_suffix_V)                                                 \
    if (Q->ne[0] == (D) && K->type == ggml_type_##type_suffix_K && V->type == ggml_type_##type_suffix_V) {        \
        constexpr int nwarps = (D)/WARP_SIZE;                                                                     \
        constexpr bool Q_q8_1 = ggml_type_##type_suffix_K != GGML_TYPE_F16;                                       \
        fattn_kernel_t fattn_kernel = flash_attn_vec_ext_##type_VKQ<                                              \
            (D), cols_per_block, parallel_blocks,                                                                 \
            vec_dot_fattn_vec_KQ_##type_suffix_K<type_VKQ, (D)>, Q_q8_1, dequantize_1_##type_suffix_V<type_VKQ>>; \
        launch_fattn<(D), parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);                       \
        return;                                                                                                   \
    }                                                                                                             \

static void on_no_fattn_vec_case(const int D) {
    if (D == 64) {
        fprintf(stderr, "Unsupported KV type combination for head_size 64.\n");
        fprintf(stderr, "By default only f16 KV cache is supported.\n");
        fprintf(stderr, "Compile with LLAMA_CUDA_FA_ALL_QUANTS for V cache quantization support.\n");
        GGML_ASSERT(false);
    } else {
        fprintf(stderr, "Unsupported KV type combination for head_size 128.\n");
        fprintf(stderr, "Supported combinations:\n");
        fprintf(stderr, "  - K == q4_0, V == q4_0,  4.50 BPV\n");
        fprintf(stderr, "  - K == q8_0, V == q8_0,  8.50 BPV\n");
        fprintf(stderr, "  - K == f16,  V == f16,  16.00 BPV\n");
        fprintf(stderr, "Compile with LLAMA_CUDA_FA_ALL_QUANTS for all combinations of q4_0, q4_1, q5_0, q5_1, q8_0, and f16.\n");
        GGML_ASSERT(false);
    }
}

template <int D, int parallel_blocks>
void launch_fattn(ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, int nwarps, int cols_per_block) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_tensor * mask = dst->src[3];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!mask || mask->ne[1] >= GGML_PAD(Q->ne[1], 16) &&
                                "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

    GGML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();

    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
    }

    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    const dim3 blocks_num(parallel_blocks*((Q->ne[1] + cols_per_block - 1) / cols_per_block), Q->ne[2], Q->ne[3]);
    const int  shmem = 0;

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) KQV->op_params + 1, sizeof(float));

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    fattn_kernel<<<blocks_num, block_dim, shmem, main_stream>>>(
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        mask ? ((const char *) mask->data) : nullptr,
        (parallel_blocks) == 1 ? (float *) KQV->data : dst_tmp.ptr, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2,
        Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3],
        mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
        Q->nb[1], Q->nb[2], Q->nb[3],
        K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3],
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

template<int D, int ncols, int parallel_blocks, vec_dot_KQ_f16_t vec_dot_KQ, bool Q_q8_1, dequantize_1_f16_t dequantize_1_v> // D == head size
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__global__ void flash_attn_vec_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
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
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
#if FP16_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb02* blockIdx.y              + nb01*ic0;
    K += nb12*(blockIdx.y / gqa_ratio);
    V += nb22*(blockIdx.y / gqa_ratio);

    const half * maskh = (const half   *)  mask + ne11*ic0;

    const float slopef = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ half KQ[ncols*D];
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

    // Convert Q to half2 (f16 K) or q8_1 (quantized K) and store in registers:
    half2  Q_h2[ncols][D/(2*WARP_SIZE)];
    int   Q_i32[ncols][D/(sizeof(int)*QK8_1) == 0 ? 1 : D/(sizeof(int)*QK8_1)];
    half2  Q_ds[ncols][D/QK8_1 == 0 ? 1 : D/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int   * tmp_q_i32 = (int   *) &KQ[j*D];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_half2(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<half2>(Q_f + 4*i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int   * tmp_q_i32 = (int   *) &KQ[j*D];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_f2_j = (const float2 *) (Q + j*nb01);

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float2 tmp = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_h2[j][i0/WARP_SIZE] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
            }
        }
    }


#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*D + tid] = -HALF_MAX_HALF;
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

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                half sum = vec_dot_KQ(K + (k_VKQ_0 + i_KQ)*nb11, Q_h2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum(sum);
                sum += mask ? slopeh*maskh[j*ne11 + k_VKQ_0 + i_KQ] : __float2half(0.0f);

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
            reinterpret_cast<half&>(V_k.x) = dequantize_1_v(V + (k_VKQ_0 + k0 + 0)*nb21, tid);
            reinterpret_cast<half&>(V_k.y) = dequantize_1_v(V + (k_VKQ_0 + k0 + 1)*nb21, tid);
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
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum(kqsum[j_VKQ]);

        half dst_val = (__low2half(VKQ[j_VKQ]) + __high2half(VKQ[j_VKQ]));
        if (parallel_blocks == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
        dst[j_dst*D*gridDim.y + D*blockIdx.y + tid] = dst_val;
    }

    if (parallel_blocks != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[(ic0 + tid)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax[tid], kqsum[tid]);
    }
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

#define DECL_FATTN_VEC_F16_INST(D, ncols, parallel_blocks, vec_dot_KQ, Q_q8_1, dequantize_1_v) \
    template __global__ void flash_attn_vec_ext_f16<D, ncols, parallel_blocks, vec_dot_KQ, Q_q8_1, dequantize_1_v>( \
        const char * __restrict__ Q, \
        const char * __restrict__ K, \
        const char * __restrict__ V, \
        const char * __restrict__ mask, \
        float      * __restrict__ dst, \
        float2     * __restrict__ dst_meta, \
        const float scale, \
        const float max_bias, \
        const float m0, \
        const float m1, \
        const uint32_t n_head_log2, \
        const int ne00, \
        const int ne01, \
        const int ne02, \
        const int ne03, \
        const int ne10, \
        const int ne11, \
        const int ne12, \
        const int ne13, \
        const int ne31, \
        const int nb31, \
        const int nb01, \
        const int nb02, \
        const int nb03, \
        const int nb11, \
        const int nb12, \
        const int nb13, \
        const int nb21, \
        const int nb22, \
        const int nb23, \
        const int ne0, \
        const int ne1, \
        const int ne2, \
        const int ne3)


extern DECL_FATTN_VEC_F16_INST(64, 1, 4, (vec_dot_fattn_vec_KQ_f16<half, 64>), false, dequantize_1_f16<half>);
extern DECL_FATTN_VEC_F16_INST(128, 1, 4, (vec_dot_fattn_vec_KQ_f16<half, 128>), false, dequantize_1_f16<half>);
extern DECL_FATTN_VEC_F16_INST(256, 1, 4, (vec_dot_fattn_vec_KQ_f16<half, 256>), false, dequantize_1_f16<half>);

#define DECL_FATTN_VEC_INST(type_VKQ, D, cols_per_block, parallel_blocks, type_suffix_K, type_suffix_V) \
    template __global__ void flash_attn_vec_ext_##type_VKQ< \
        (D), cols_per_block, parallel_blocks, \
        vec_dot_fattn_vec_KQ_##type_suffix_K<type_VKQ, (D)>, ggml_type_##type_suffix_K != GGML_TYPE_F16, dequantize_1_##type_suffix_V<type_VKQ>>( \
        const char * __restrict__ Q, \
        const char * __restrict__ K, \
        const char * __restrict__ V, \
        const char * __restrict__ mask, \
        float      * __restrict__ dst, \
        float2     * __restrict__ dst_meta, \
        const float scale, \
        const float max_bias, \
        const float m0, \
        const float m1, \
        const uint32_t n_head_log2, \
        const int ne00, \
        const int ne01, \
        const int ne02, \
        const int ne03, \
        const int ne10, \
        const int ne11, \
        const int ne12, \
        const int ne13, \
        const int ne31, \
        const int nb31, \
        const int nb01, \
        const int nb02, \
        const int nb03, \
        const int nb11, \
        const int nb12, \
        const int nb13, \
        const int nb21, \
        const int nb22, \
        const int nb23, \
        const int ne0, \
        const int ne1, \
        const int ne2, \
        const int ne3)


extern DECL_FATTN_VEC_INST(f16, 128, 1, 4, q4_0, q4_0);
extern DECL_FATTN_VEC_INST(f16, 128, 1, 4, q8_0, q8_0);
