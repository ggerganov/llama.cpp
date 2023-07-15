// kernels for ggml-cuda
#include <cuda.h>
#include <cuda_fp16.h>


template<typename dst_t>
using to_t_cuda_t = void (*)(const void * x, dst_t * y, int k, cudaStream_t stream);

// support for vector types in generic code
template<typename T> struct vec2_t_impl;
template<> struct vec2_t_impl<half>   { typedef half2 type; };
template<> struct vec2_t_impl<float>  { typedef float2 type; };

template<typename T> using vec2_t = typename vec2_t_impl<T>::type;

template<typename T> inline __host__ __device__ vec2_t<T> make_vec2_t(const T & x, const T & y);
template<> inline __host__ __device__ vec2_t<half>  make_vec2_t(const  half & x, const  half & y) { return make_half2 (x, y); }
template<> inline __host__ __device__ vec2_t<float> make_vec2_t(const float & x, const float & y) { return make_float2(x, y); }

// the cuda headers define operators for half2, but not for float2
// they are defined here to simplify generic code
inline __host__ __device__ float2   operator+(const float2 & a, const float2 & b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ float2   operator-(const float2 & a, const float2 & b) { return make_float2(a.x - b.x, a.y - b.y); }
inline __host__ __device__ float2   operator*(const float2 & a, const float2 & b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __host__ __device__ float2   operator/(const float2 & a, const float2 & b) { return make_float2(a.x / b.x, a.y / b.y); }
inline __host__ __device__ float2 & operator+=(     float2 & a, const float2 & b) { a.x += b.x; a.y += b.y; return a; }
inline __host__ __device__ float2 & operator-=(     float2 & a, const float2 & b) { a.x -= b.x; a.y -= b.y; return a; }
inline __host__ __device__ float2 & operator*=(     float2 & a, const float2 & b) { a.x *= b.x; a.y *= b.y; return a; }
inline __host__ __device__ float2 & operator/=(     float2 & a, const float2 & b) { a.x /= b.x; a.y /= b.y; return a; }

template<typename dst_t>
using dequantize_kernel_t = void (*)(const void * vx, const int ib, const int iqs, vec2_t<dst_t> & v);

__device__ half  sqrt(const half x) { return hsqrt(x); }
__device__ half  exp(const half x) { return hexp(x); }
__device__ half2 exp(const half2 x) { return h2exp(x); }
__device__ half  cos(const half x) { return hcos(x); }
__device__ half  sin(const half x) { return hsin(x); }
__device__ half  max(const half x, const half y) { return __hmax(x, y); }
__device__ half2 max(const half2 x, const half2 y) { return __hmax2(x, y); }


template<typename T> struct op_max { __device__ T operator()(T a, T b) const { return max(a, b); } };
template<typename T> struct op_sum { __device__ T operator()(T a, T b) const { return a + b; } };

template<template<typename> class op_t, typename T>
static inline __device__ T warp_reduce_all(T val) {
    op_t<T> op;
#pragma unroll
    for (int mask = warpSize/2; mask > 0; mask /= 2)  {
        val = op(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}

template<typename T>
static __device__ T zero_init() { return T(0); }
template<>
__device__ half2 zero_init() { return half2(0.0f, 0.0f); }

template<template<typename> class op_t, typename T>
static __device__ T block_reduce_all(const T val, const T init = zero_init<T>()) {
    const int warp_id = threadIdx.x / warpSize; // warp id within the block
    const int lane_id = threadIdx.x % warpSize; // lane id within the warp
    const int num_warps = blockDim.x / warpSize; // number of warps in the block

    __shared__ T lane_result[32]; // max 32 warps per block

    // reduce warps
    T warp_reduction = warp_reduce_all<op_t>(val);

    __syncthreads();

    // first thread within a warp writes reduction to shared memory
    if (lane_id == 0) {
        lane_result[warp_id] = warp_reduction;
    }

    // wait for all warps to finish writing their reductions
    __syncthreads();

    // reduce the results of all warps
    T block_reduction = init;
    if (lane_id < num_warps) {
        block_reduction = lane_result[lane_id];
    }

    block_reduction = warp_reduce_all<op_t>(block_reduction);

    return block_reduction;
}

template<typename dst_t>
static __device__ void convert_fp16(const void * vx, const int ib, const int iqs, vec2_t<dst_t> & v) {
    const half * x = (const half *) vx;

    v.x = (dst_t)(x[ib + iqs + 0]);
    v.y = (dst_t)(x[ib + iqs + 1]);
}

template<typename dst_t>
static __device__ void convert_fp32(const void * vx, const int ib, const int iqs, vec2_t<dst_t> & v) {
    const float * x = (const float *) vx;

    v.x = (dst_t)(x[ib + iqs + 0]);
    v.y = (dst_t)(x[ib + iqs + 1]);
}

template<typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_mul_mat_p021(const src0_t * vx, const src1_t * y, dst_t * dst, const int ncols_x, const int nrows_x, const int nchannels_x) {
    const src0_t * x = vx;
    // const int col_x = blockDim.x*blockIdx.x + threadIdx.x;
    // const int row_x = blockDim.y*blockIdx.y + threadIdx.y;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    dst_t tmp = 0;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel*ncols_x + col_x;
        const dst_t xi = (dst_t)(x[ix]);

        const int row_y = col_x;

        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

template<typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_mul_mat_vec_nc(
    const src0_t * vx, const src1_t * y, dst_t * dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int nchannels_x, const int channel_stride_x) {

    const src0_t * x = vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    const int idst = channel*nrows_dst + row_dst;

    dst_t tmp = 0;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        const int ix = channel*channel_stride_x + row_x*row_stride_x + col_x;
        const dst_t xi = (dst_t)(x[ix]);

        const int row_y = col_x;

        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

template <typename src_t, typename dst_t>
static __global__ void k_cpy(const char * cx, char * cdst, const int ne,
                                   const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
                                   const int ne10, const int ne11, const int nb10, const int nb11, const int nb12) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const int i02 = i / (ne00*ne01);
    const int i01 = (i - i02*ne01*ne00) / ne00;
    const int i00 = i - i02*ne01*ne00 - i01*ne00;
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02;

    const int i12 = i / (ne10*ne11);
    const int i11 = (i - i12*ne10*ne11) / ne10;
    const int i10 = i - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = i10*nb10 + i11*nb11 + i12*nb12;

    *(dst_t *)(cdst + dst_offset) = *(const src_t *)(cx + x_offset);
}

template<typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_add(const src0_t * x, const src1_t * y, dst_t * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = (dst_t)x[i] + (dst_t)y[i];
}

template<typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_mul(const src0_t * x, const src1_t * y, dst_t * dst, const int kx, const int ky) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= kx) {
        return;
    }
    dst[i] = (dst_t)x[i] * (dst_t)y[i%ky];
}

template<typename src0_t, typename dst_t>
static __global__ void k_silu(const src0_t * x, dst_t * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (src0_t(1) + exp(-x[i]));
}

// TODO: unstable with f16 compute, using f32 compute for now
template<typename src0_t, typename dst_t>
static __global__ void k_rms_norm(const src0_t * x, dst_t * dst, const int ncols) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const float eps  = 1e-6;

    float tmp = 0; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const float mean = tmp / (float)ncols;
    const float scale = 1.0f / sqrtf(mean + eps);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[row*ncols + col] = scale * (float)x[row*ncols + col];
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_rope(const src0_t * x, dst_t * dst, const int ncols, const float p, const float theta_scale) {
    const int col = 2*(blockDim.x*blockIdx.x + threadIdx.x);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;

    const dst_t theta = p * powf(theta_scale, col/2);
    const dst_t sin_theta = sin(theta);
    const dst_t cos_theta = cos(theta);

    const dst_t x0 = x[i + 0];
    const dst_t x1 = x[i + 1];

    dst[i + 0] = (dst_t)x0*cos_theta - (dst_t)x1*sin_theta;
    dst[i + 1] = (dst_t)x0*sin_theta + (dst_t)x1*cos_theta;
}

template<typename src0_t, typename dst_t>
static __global__ void k_diag_mask_inf(const src0_t * x, dst_t * dst, const int ncols, const int rows_per_channel, const int n_past) {
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    const int row = blockDim.y*blockIdx.y + threadIdx.y;

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? (dst_t)-INFINITY : (dst_t)x[i];
    dst[i] = (dst_t)x[i] - (dst_t)((col > n_past + row % rows_per_channel) * INT_MAX); // equivalent within rounding error but slightly faster on GPU
}

// TODO: numerically stable version - low prio since the softmax is computed in the fused attention kernel
// check: https://arxiv.org/pdf/2001.04438.pdf
template<typename src0_t, typename dst_t>
static __global__ void k_soft_max_orig(const src0_t * x, dst_t * dst, const int ncols) {
    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int block_size = blockDim.x;
    const int tid = threadIdx.x;

    float tmp = 0;

    for (int block_start = 0; block_start < ncols; block_start += block_size) {
        const int col = block_start + tid;

        if (col >= ncols) {
            break;
        }

        const int i = row*ncols + col;
        const float val = expf(x[i]);
        tmp += val;
        dst[i] = val;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    for (int block_start = 0; block_start < ncols; block_start += block_size) {
        const int col = block_start + tid;

        if (col >= ncols) {
            break;
        }

        const int i = row*ncols + col;
        dst[i] /= tmp;
    }
}

template<typename src_t, typename dst_t, int pack_size, int block_size>
static __global__ void k_soft_max(const src_t * x, dst_t * dst, const int64_t nrows, const int64_t ncols) {
    //assert(ncols % pack_size == 0);
    const int tid = threadIdx.x;
    const int num_packs = ncols / pack_size;

    for (int row = blockIdx.x; row < nrows; row += gridDim.x) {
        src_t th_max = -INFINITY;
        // row max thread
        #pragma unroll
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            // load pack
            src_t pack[pack_size];
            #pragma unroll
            for (int i = 0; i < pack_size; i++) {
                pack[i] = x[row * ncols + pack_id * pack_size + i];
            }
            // reduce max pack
            #pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                th_max = max(th_max, pack[i]);
            }
        }
        // reduce max row warp threads
        src_t row_max = block_reduce_all<op_max>(th_max, (src_t)-INFINITY);

        // row exp sum thread
        src_t th_sum = 0;
        #pragma unroll
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            // load pack
            src_t pack[pack_size];
            #pragma unroll
            for (int i = 0; i < pack_size; i++) {
                pack[i] = x[row * ncols + pack_id * pack_size + i];
            }
            // reduce pack
            #pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                th_sum += exp(pack[i] - row_max);
            }
        }

        // reduce row exp sum all threads
        src_t row_sum = block_reduce_all<op_sum>(th_sum);

        // store (row - row_max) / row exp sum
        #pragma unroll
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            // load pack
            src_t pack[pack_size];
            #pragma unroll
            for (int i = 0; i < pack_size; i++) {
                pack[i] = x[row * ncols + pack_id * pack_size + i];
            }
            // reduce pack
            #pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                pack[i] = exp(pack[i] - row_max) / row_sum;
            }

            // store pack
            #pragma unroll
            for (int i = 0; i < pack_size; i++) {
                dst[row * ncols + pack_id * pack_size + i] = pack[i];
            }
        }
    }
}

template<typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_scale(const src0_t * x, dst_t * dst, const src1_t * scale, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (dst_t)(*scale) * (dst_t)x[i];
}

template<typename dst_t, int qk, int qr, dequantize_kernel_t<dst_t> dequantize_kernel>
static __global__ void k_get_rows(const void * x, const int * y, dst_t * dst, const int ncols) {
    const int col = (blockIdx.x*blockDim.x + threadIdx.x)*2;
    const int row = blockDim.y*blockIdx.y + threadIdx.y;

    if (col >= ncols) {
        return;
    }

    const int r = y[row];

    // copy x[r*ncols + col] to dst[row*ncols + col]
    const int xi = r*ncols + col;
    const int di = row*ncols + col;

    const int ib = xi/qk; // block index
    const int iqs = (xi%qk)/qr; // quant index
    const int iybs = di - di%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    vec2_t<dst_t> v;
    dequantize_kernel(x, ib, iqs, v);
    dst[iybs + iqs + 0]        = v.x;
    dst[iybs + iqs + y_offset] = v.y;
}
