static const int GGML_CUDA_MAX_SUBSTREAMS = 1;
static const bool GGML_CUDA_SEQ_COMPUTE = true;

#define WARP_SIZE 32
#define CUDA_ADD_BLOCK_SIZE 256
#define CUDA_MUL_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_CPY_BLOCK_SIZE 32
#define CUDA_SCALE_BLOCK_SIZE 256
#define CUDA_ROPE_BLOCK_SIZE 256
#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BLOCK_SIZE 256
#define CUDA_QUANTIZE_BLOCK_SIZE 256

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_CUDA_DMMV_X
#define GGML_CUDA_DMMV_X 32
#endif
#ifndef GGML_CUDA_DMMV_Y
#define GGML_CUDA_DMMV_Y 1
#endif
#ifndef GGML_CUDA_MMV_Y
#define GGML_CUDA_MMV_Y 1
#endif


#ifndef K_QUANTS_PER_ITERATION
#define K_QUANTS_PER_ITERATION 2
#else
static_assert(K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#endif

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <queue>
#include <stdint.h>
#include <stdio.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <nvtx3/nvToolsExt.h>

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-cuda-kern.h"
#include "ggml-cuda-quant.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define CUDA_CHECK(err)                                                                 \
    do {                                                                                \
        cudaError_t err_ = (err);                                                       \
        if (err_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %d at %s (%s:%d): %s\n", err_,                  \
                __func__, __FILE__, __LINE__, cudaGetErrorString(err_));                \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#if CUDART_VERSION >= 12000
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s (%s:%d): %s\n", err_,              \
                __func__, __FILE__, __LINE__, cublasGetStatusString(err_));             \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#else
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);  \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#endif // CUDART_VERSION >= 12000

#define UNUSED(x) (void)(x)

typedef void (*ggml_cuda_op_t)(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t cudaStream_main);

struct cuda_pool_buffer {
    void * ptr;
    size_t size;
};

static std::unordered_map<cudaStream_t, std::vector<cuda_pool_buffer>> g_cuda_stream_pools;
static size_t g_cuda_pool_size = 0;

static void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size, cudaStream_t stream) {
    std::vector<cuda_pool_buffer>& pool = g_cuda_stream_pools[stream];

    // find existing
    for (size_t i = 0; i < pool.size(); ++i) {
        cuda_pool_buffer& b = pool[i];
        if (b.size >= size && b.ptr != nullptr) {
            void * ptr = b.ptr;
            *actual_size = b.size;
            pool.erase(pool.begin() + i);
            return ptr;
        }
    }

    // allocate new
    void * ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    *actual_size = size;

    g_cuda_pool_size += size;

    //fprintf(stderr, "cuda pool size: %.2f MB (allocating now: %.2f MB)\n", g_cuda_pool_size / 1024.0 / 1024.0, size / 1024.0 / 1024.0);

    return ptr;
}

static void ggml_cuda_pool_free(void * ptr, size_t size, cudaStream_t stream) {
    std::vector<cuda_pool_buffer>& pool = g_cuda_stream_pools[stream];

    pool.push_back({ ptr, size });
}

static void ggml_cuda_pool_free_all() {
    for (auto& p : g_cuda_stream_pools) {
        for (auto& b : p.second) {
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
            }
        }
    }
    g_cuda_stream_pools.clear();
}

template<typename src_t>
static void quantize_row_q8_1_cuda(const src_t * x, void * vy, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    quantize_q8_1<<<num_blocks, CUDA_QUANTIZE_BLOCK_SIZE, 0, stream>>>(x, vy, k);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<dst_t, QK4_0, QR4_0, dequantize_q4_0<dst_t>><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<dst_t, QK4_1, QR4_1, dequantize_q4_1<dst_t>><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static void dequantize_row_q5_0_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<dst_t, QK5_0, QR5_0, dequantize_q5_0<dst_t>><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static void dequantize_row_q5_1_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<dst_t, QK5_1, QR5_1, dequantize_q5_1<dst_t>><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static void dequantize_row_q8_0_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<dst_t, QK8_0, QR8_0, dequantize_q8_0<dst_t>><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

/*
static void dequantize_row_q2_K_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

static void dequantize_row_q3_K_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

static void dequantize_row_q5_K_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

*/
template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename src1_t, typename dst_t>
static void dequantize_mul_mat_vec_q4_0_cuda(const void * vx, const src1_t * y, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_DMMV_Y - 1) / GGML_CUDA_DMMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<src1_t, dst_t, QK4_0, QR4_0, dequantize_q4_0<dst_t>>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

template<typename src1_t, typename dst_t>
static void dequantize_mul_mat_vec_q4_1_cuda(const void * vx, const src1_t * y, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_DMMV_Y - 1) / GGML_CUDA_DMMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<src1_t, dst_t, QK4_1, QR4_1, dequantize_q4_1<dst_t>>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

template<typename src1_t, typename dst_t>
static void dequantize_mul_mat_vec_q5_0_cuda(const void * vx, const src1_t * y, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_DMMV_Y - 1) / GGML_CUDA_DMMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<src1_t, dst_t, QK5_0, QR5_0, dequantize_q5_0<dst_t>>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

template<typename src1_t, typename dst_t>
static void dequantize_mul_mat_vec_q5_1_cuda(const void * vx, const src1_t * y, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_DMMV_Y - 1) / GGML_CUDA_DMMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<src1_t, dst_t, QK5_1, QR5_1, dequantize_q5_1<dst_t>>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

template<typename src1_t, typename dst_t>
static void dequantize_mul_mat_vec_q8_0_cuda(const void * vx, const src1_t * y, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_DMMV_Y - 1) / GGML_CUDA_DMMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<src1_t, dst_t, QK8_0, QR8_0, dequantize_q8_0<dst_t>>
        <<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}
/*
template<typename src1_t, typename dst_t>
static void dequantize_mul_mat_vec_q2_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2;
    const int block_num_y = (nrows + ny - 1) / ny;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_q2_k<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

template<typename src1_t, typename dst_t>
static void dequantize_mul_mat_vec_q3_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const dim3 block_dims(32, 1, 1);
    dequantize_mul_mat_vec_q3_k<<<nrows, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

template<typename src1_t, typename dst_t>
static void dequantize_mul_mat_vec_q4_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const dim3 block_dims(32, 1, 1);
    dequantize_mul_mat_vec_q4_k<<<nrows, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

template<typename src1_t, typename dst_t>
static void dequantize_mul_mat_vec_q5_K_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const dim3 block_dims(32, 1, 1);
    dequantize_mul_mat_vec_q5_k<<<nrows, block_dims, 0, stream>>>(vx, y, dst, ncols);
}
*/

template<typename src1_t, typename dst_t>
static void dequantize_mul_mat_vec_q6_K_cuda(const void * vx, const src1_t * y, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(32, ny, 1);
    dequantize_mul_mat_vec_q6_k<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

template<typename src1_t, typename dst_t>
static void convert_mul_mat_vec_f16_cuda(const void * vx, const src1_t * y, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_DMMV_Y - 1) / GGML_CUDA_DMMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<src1_t, dst_t, 1, 1, convert_fp16<dst_t>><<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols, nrows);
}

template<typename dst_t>
static void mul_mat_vec_q4_0_q8_1_cuda(const void * vx, const void * vy, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<dst_t, QK4_0, QI4_0, block_q4_0, vec_dot_q4_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

template<typename dst_t>
static void mul_mat_vec_q4_1_q8_1_cuda(const void * vx, const void * vy, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<dst_t, QK4_0, QI4_1, block_q4_1, vec_dot_q4_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

template<typename dst_t>
static void mul_mat_vec_q5_0_q8_1_cuda(const void * vx, const void * vy, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<dst_t, QK5_0, QI5_0, block_q5_0, vec_dot_q5_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

template<typename dst_t>
static void mul_mat_vec_q5_1_q8_1_cuda(const void * vx, const void * vy, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<dst_t, QK5_1, QI5_1, block_q5_1, vec_dot_q5_1_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

template<typename dst_t>
static void mul_mat_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_nums(1, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    mul_mat_vec_q<dst_t, QK8_0, QI8_0, block_q8_0, vec_dot_q8_0_q8_1>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);
}

template<typename dst_t>
static void convert_fp16_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<dst_t, 1, 1, convert_fp16<dst_t>><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static to_t_cuda_t<dst_t> ggml_get_to_t_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_row_q5_0_cuda;
        case GGML_TYPE_Q5_1:
            return dequantize_row_q5_1_cuda;
        case GGML_TYPE_Q8_0:
            return dequantize_row_q8_0_cuda;
        /*
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        */
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_F16:
            return convert_fp16_cuda;
        default:
            return nullptr;
    }
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_mul_mat_p021_cuda(const src0_t * vx, const src1_t * y, dst_t * dst, const int ncols_x, const int nrows_x, const int nchannels_x, cudaStream_t stream) {
    const dim3 block_nums(1, nrows_x, nchannels_x);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    k_mul_mat_p021<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, nchannels_x);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_mul_mat_vec_nc_cuda(
    const src0_t * vx, const src1_t * y, dst_t * dst, const int ncols_x, const int nrows_x, const int row_stride_x,
    const int nchannels_x, const int channel_stride_x, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_x);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    k_mul_mat_vec_nc<<<block_nums, block_dims, 0, stream>>>
        (vx, y, dst, ncols_x, nrows_x, row_stride_x, nchannels_x, channel_stride_x);
}

template<typename src_t, typename dst_t>
static void ggml_cpy_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
    const int ne10, const int ne11, const int nb10, const int nb11, const int nb12, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    k_cpy<src_t, dst_t><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void add_cuda(const src0_t * x, const src1_t * y, dst_t * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_ADD_BLOCK_SIZE - 1) / CUDA_ADD_BLOCK_SIZE;
    k_add<<<num_blocks, CUDA_ADD_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void mul_cuda(const src0_t * x, const src1_t * y, dst_t * dst, const int kx, const int ky, cudaStream_t stream) {
    const int num_blocks = (kx + CUDA_MUL_BLOCK_SIZE - 1) / CUDA_MUL_BLOCK_SIZE;
    k_mul<<<num_blocks, CUDA_MUL_BLOCK_SIZE, 0, stream>>>(x, y, dst, kx, ky);
}

template<typename src0_t, typename dst_t>
static void silu_cuda(const src0_t * x, dst_t * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    k_silu<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template<typename src0_t, typename dst_t>
static void rms_norm_cuda(const src0_t * x, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    k_rms_norm<<<nrows, block_dims, 0, stream>>>(x, dst, ncols);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void scale_cuda(const src0_t * x, dst_t * dst, const src1_t * scale, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    k_scale<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, k);
}

template<typename src0_t, typename dst_t>
static void rope_cuda(const src0_t * x, dst_t * dst, const int ncols, const int nrows, const float p, const float theta_scale, cudaStream_t stream) {
    GGML_ASSERT(nrows % 2 == 0);
    const dim3 block_dims(2*CUDA_ROPE_BLOCK_SIZE, 1, 1);
    const int num_blocks_x = (ncols + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(num_blocks_x, nrows, 1);
    k_rope<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, p, theta_scale);
}

template<typename src0_t, typename dst_t>
static void diag_mask_inf_cuda(const src0_t * x, dst_t * dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream) {
    const dim3 block_dims(CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ncols_x + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, nrows_x, 1);
    k_diag_mask_inf<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols_x, rows_per_channel, n_past);
}

template<typename src0_t, typename dst_t>
static void soft_max_cuda(const src0_t * x, dst_t * dst, const int ncols, const int nrows, cudaStream_t stream) {
    // TODO: implement fast numerically stable version for small ncols
    //if (ncols >= 1024) {
        int num_blocks = nrows;
        if (ncols % 2 == 0) {
            k_soft_max<src0_t, dst_t, 2 , 1024>
                <<<num_blocks, 1024, 0, stream>>>(x, dst, nrows, ncols);
        }
        else {
            k_soft_max<src0_t, dst_t, 1, 1024>
                <<<num_blocks, 1024, 0, stream>>>(x, dst, nrows, ncols);
        }
    //}
    //else {
    //    const dim3 block_dims(WARP_SIZE, 1, 1);
    //    const dim3 block_nums(1, nrows, 1);
    //    k_soft_max_orig<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    //}
}

template<typename dst_t, int qk, int qr, dequantize_kernel_t<dst_t> dq>
static void get_rows_cuda(const void * x, const int * y, dst_t * dst, const int nrows, const int ncols, cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num = (ncols/2 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(block_num, nrows, 1);
    k_get_rows<dst_t, qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(x, y, dst, ncols);
}

// TODO: move to context
static cublasHandle_t g_cublas_handle = nullptr;
static cudaStream_t g_cudaStream_main = nullptr;
static cudaEvent_t g_cudaEvent_main = nullptr;
static cudaStream_t g_cudaStreams[GGML_CUDA_MAX_SUBSTREAMS] = { };
static cudaEvent_t g_cudaEvents[GGML_CUDA_MAX_SUBSTREAMS] = { };
#define GGML_CUDA_MAX_DEVICES 16
static int g_compute_capabilities[GGML_CUDA_MAX_DEVICES];

static void ggml_init_cublas() {
    static bool initialized = false;

    if (!initialized) {
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        int64_t total_vram = 0;
        fprintf(stderr, "%s: found %d CUDA devices:\n", __func__, device_count);
        for (int id = 0; id < device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            fprintf(stderr, "  Device %d: %s (%.0f GB)\n", id, prop.name, prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
            total_vram += prop.totalGlobalMem;
            g_compute_capabilities[id] = 100*prop.major + 10*prop.minor;
        }

        // create main stream and event
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStream_main, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&g_cudaEvent_main, cudaEventDisableTiming));

        // create secondary streams and events
        for (int i = 0; i < GGML_CUDA_MAX_SUBSTREAMS; ++i) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams[i], cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&g_cudaEvents[i], cudaEventDisableTiming));
        }

        // create cublas handle
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
        //CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

        // configure logging to stdout
        //CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

        initialized = true;
    }
}

void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // The allocation error can be bypassed. A null ptr will assigned out of this function.
        // This can fixed the OOM error in WSL.
        cudaGetLastError();
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory: %s\n",
            size/1024.0/1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

void ggml_cuda_host_free(void * ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_add(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {

    const int64_t ne0 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    add_cuda((src0_t *)src0_d, (src1_t *) src1_d, (dst_t *) dst_d, ne0*i01_diff, stream);
    CUDA_CHECK(cudaGetLastError());

    UNUSED(src1);
    UNUSED(dst);
    UNUSED(i02);
    UNUSED(i1);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_mul(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    for (int64_t i01 = i01_low; i01 < i01_high; i01++) {
        const int64_t i11 = i1*ne11 + i01%ne11; // broadcast src1 across src0

        src0_t * src0_d_i01 = (src0_t *) src0_d + i01*ne00;
        src1_t * src1_d_i01 = (src1_t *) src1_d + i11*ne10;
        dst_t * dst_d_i01 = (dst_t *) dst_d + i01*ne00;

        // compute
        mul_cuda(src0_d_i01, src1_d_i01, dst_d_i01, ne00, ne10, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    UNUSED(dst);
    UNUSED(i02);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_silu(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    silu_cuda((src0_t *)src0_d, (dst_t *)dst_d, ne00*i01_diff, stream);
    CUDA_CHECK(cudaGetLastError());

    UNUSED(src1);
    UNUSED(src1_d);
    UNUSED(dst);
    UNUSED(i02);
    UNUSED(i1);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_rms_norm(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    rms_norm_cuda((src0_t *)src0_d, (dst_t *)dst_d, ne00, i01_diff, stream);
    CUDA_CHECK(cudaGetLastError());

    UNUSED(src1);
    UNUSED(src1_d);
    UNUSED(dst);
    UNUSED(i02);
    UNUSED(i1);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_dequantize_mul_mat_vec(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = i01_high - i01_low;

#ifdef GGML_CUDA_FORCE_DMMV
    const bool use_mul_mat_vec_q = false;
#else
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    const bool mul_mat_vec_q_implemented = src0->type == GGML_TYPE_Q4_0 ||
        src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 ||
        src0->type == GGML_TYPE_Q5_1 ||
        src0->type == GGML_TYPE_Q8_0;

    // The integer intrinsics used in mul_mat_vec_q are available with compute capability 6.
    // However, they have bad performance with Pascal cards.
    // Therefore, in a multi GPU setting decide at runtime which GPUs should use mul_mat_vec_q.
    const bool use_mul_mat_vec_q = g_compute_capabilities[id] >= 700 && mul_mat_vec_q_implemented;
#endif

    if (use_mul_mat_vec_q) {
        size_t as;
        void * src1_q8_1 = ggml_cuda_pool_malloc(ne00*sizeof(block_q8_1)/QK8_1, &as, stream);
        quantize_row_q8_1_cuda((src1_t *)src1_d, src1_q8_1, ne00, stream);

        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                mul_mat_vec_q4_0_q8_1_cuda(src0_d, src1_q8_1, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            case GGML_TYPE_Q4_1:
                mul_mat_vec_q4_1_q8_1_cuda(src0_d, src1_q8_1, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            case GGML_TYPE_Q5_0:
                mul_mat_vec_q5_0_q8_1_cuda(src0_d, src1_q8_1, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            case GGML_TYPE_Q5_1:
                mul_mat_vec_q5_1_q8_1_cuda(src0_d, src1_q8_1, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            case GGML_TYPE_Q8_0:
                mul_mat_vec_q8_0_q8_1_cuda(src0_d, src1_q8_1, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            default:
                GGML_ASSERT(false);
                break;
        }

        ggml_cuda_pool_free(src1_q8_1, as, stream);
    }
    else {
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                dequantize_mul_mat_vec_q4_0_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            case GGML_TYPE_Q4_1:
                dequantize_mul_mat_vec_q4_1_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            case GGML_TYPE_Q5_0:
                dequantize_mul_mat_vec_q5_0_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            case GGML_TYPE_Q5_1:
                dequantize_mul_mat_vec_q5_1_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            case GGML_TYPE_Q8_0:
                dequantize_mul_mat_vec_q8_0_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            /*
            case GGML_TYPE_Q2_K:
                dequantize_mul_mat_vec_q2_K_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q3_K:
                dequantize_mul_mat_vec_q3_K_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q4_K:
                dequantize_mul_mat_vec_q4_K_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, cudaStream_main);
                break;
            case GGML_TYPE_Q5_K:
                dequantize_mul_mat_vec_q5_K_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, cudaStream_main);
                break;
            */
            case GGML_TYPE_Q6_K:
                dequantize_mul_mat_vec_q6_K_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            case GGML_TYPE_F16:
                convert_mul_mat_vec_f16_cuda(src0_d, (src1_t *)src1_d, (dst_t *)dst_d, ne00, nrows, stream);
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
    }
    CUDA_CHECK(cudaGetLastError());

    UNUSED(src1);
    UNUSED(dst);
    UNUSED(i02);
    UNUSED(i1);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_rope(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {


    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    const int n_past = ((int32_t *) dst->params)[0];
    const int n_dims = ((int32_t *) dst->params)[1];
    const int mode   = ((int32_t *) dst->params)[2];
    //const int n_ctx  = ((int32_t *) dst->params)[3];
    GGML_ASSERT(mode == 0);

    const float theta_scale = powf(10000.0, -2.0f/n_dims);
    const float p = ((mode & 1) == 0 ? n_past + i02 : i02);

    // compute
    rope_cuda((src0_t *)src0_d, (dst_t *)dst_d, ne00, i01_diff, p, theta_scale, stream);
    CUDA_CHECK(cudaGetLastError());

    UNUSED(dst);
    UNUSED(src1);
    UNUSED(src1_d);
    UNUSED(i1);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_diag_mask_inf(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t i01_diff = i01_high - i01_low;

    const int n_past = ((int32_t *) dst->params)[0];

    // compute
    diag_mask_inf_cuda((src0_t *)src0_d, (dst_t *)dst_d, ne00, i01_diff, ne01, n_past, stream);
    CUDA_CHECK(cudaGetLastError());

    UNUSED(dst);
    UNUSED(src1);
    UNUSED(src1_d);
    UNUSED(i02);
    UNUSED(i1);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_soft_max(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    soft_max_cuda((src0_t *)src0_d, (dst_t *)dst_d, ne00, i01_diff, stream);
    CUDA_CHECK(cudaGetLastError());

    UNUSED(src1);
    UNUSED(src1_d);
    UNUSED(dst);
    UNUSED(i02);
    UNUSED(i1);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_scale(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {

    //const src1_t scale = ((src1_t *) src1->data)[0];

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    scale_cuda<src0_t, src1_t>((src0_t *)src0_d, (dst_t *)dst_d, (src1_t *)src1_d, ne00*i01_diff, stream);
    CUDA_CHECK(cudaGetLastError());

    UNUSED(src1);
    UNUSED(src1_d);
    UNUSED(dst);
    UNUSED(i02);
    UNUSED(i1);
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_get_rows(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int ncols = src0->ne[0];
    const int nrows = ggml_nelements(src1);

    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_cuda<dst_t, 1, 1, convert_fp16<dst_t>>(src0_d, (int *) src1_d, (dst_t *)dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda<dst_t, 1, 1, convert_fp32<dst_t>>(src0_d, (int *) src1_d, (dst_t *)dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda<dst_t, QK4_0, QR4_0, dequantize_q4_0<dst_t>>(src0_d, (int *) src1_d, (dst_t *)dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda<dst_t, QK4_1, QR4_1, dequantize_q4_1<dst_t>>(src0_d, (int *) src1_d, (dst_t *)dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda<dst_t, QK5_0, QR5_0, dequantize_q5_0<dst_t>>(src0_d, (int *) src1_d, (dst_t *)dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda<dst_t, QK5_1, QR5_1, dequantize_q5_1<dst_t>>(src0_d, (int *) src1_d, (dst_t *)dst_d, nrows, ncols, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda<dst_t, QK8_0, QR8_0, dequantize_q8_0<dst_t>>(src0_d, (int *) src1_d, (dst_t *)dst_d, nrows, ncols, stream);
            break;

        default:
            GGML_ASSERT(false);
            break;
    }
    CUDA_CHECK(cudaGetLastError());

    UNUSED(i02);
    UNUSED(i01_low);
    UNUSED(i01_high);
    UNUSED(i1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ggml_cuda_buffer {
    const char * name;

    void   * data;
    size_t   size;
    void   * device;
};

struct ggml_cuda_context {
    std::vector<ggml_cuda_buffer> buffers;
};

ggml_cuda_context * ggml_cuda_init() {
    ggml_init_cublas();

    ggml_cuda_context * ctx = new ggml_cuda_context;

    return ctx;
}

void ggml_cuda_free(ggml_cuda_context * ctx) {
    for (size_t n = 0; n < ctx->buffers.size(); ++n) {
        if (ctx->buffers[n].device != nullptr) {
            CUDA_CHECK(cudaFree(ctx->buffers[n].device));
        }
    }

    // this will free the global memory pool for all contexts
    ggml_cuda_pool_free_all();

    delete ctx;
}

static void * ggml_cuda_get_buffer(ggml_cuda_context * ctx, ggml_tensor * t) {
    return t->data;

    UNUSED(ctx);
}

static cudaError_t ggml_cuda_cpy_tensor_2d(ggml_cuda_context * ctx,
    void * dst, ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
    const char * src_ptr = (const char *) ggml_cuda_get_buffer(ctx, src);
    char * dst_ptr = (char *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    GGML_ASSERT(i1_low == 0);
    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, kind, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, kind, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, kind, stream);
            if (r != cudaSuccess) return r;
        }
        return cudaSuccess;
    }
}

static const ggml_type GGML_TYPE_NONE = GGML_TYPE_COUNT;

struct ggml_cuda_op_dispatch_t {
    ggml_cuda_op_t d[GGML_TYPE_COUNT][GGML_TYPE_COUNT+1][GGML_TYPE_COUNT] = { nullptr };
};

template<template <typename src0_t, typename src1_t, typename dst_t> class Op>
static ggml_cuda_op_dispatch_t gen_op_dispatch_table() {
    ggml_cuda_op_dispatch_t dispatch;

    dispatch.d[GGML_TYPE_F16][GGML_TYPE_NONE][GGML_TYPE_F16] = &Op<half, half, half>::op;
    //dispatch.d[GGML_TYPE_F16][GGML_TYPE_NONE][GGML_TYPE_F32] = &Op<half, half, float>::op;
    dispatch.d[GGML_TYPE_F16][GGML_TYPE_F16][GGML_TYPE_F16] = &Op<half, half, half>::op;
    dispatch.d[GGML_TYPE_F16][GGML_TYPE_F16][GGML_TYPE_F32] = &Op<half, half, float>::op;
    dispatch.d[GGML_TYPE_F16][GGML_TYPE_F32][GGML_TYPE_F16] = &Op<half, float, half>::op;
    dispatch.d[GGML_TYPE_F16][GGML_TYPE_F32][GGML_TYPE_F32] = &Op<half, float, float>::op;
    //dispatch.d[GGML_TYPE_F32][GGML_TYPE_NONE][GGML_TYPE_F16] = &Op<float, float, half>::op;
    dispatch.d[GGML_TYPE_F32][GGML_TYPE_NONE][GGML_TYPE_F32] = &Op<float, float, float>::op;
    //dispatch.d[GGML_TYPE_F32][GGML_TYPE_F16][GGML_TYPE_F16] = &Op<float, half, half>::op;
    dispatch.d[GGML_TYPE_F32][GGML_TYPE_F16][GGML_TYPE_F32] = &Op<float, half, float>::op;
    //dispatch.d[GGML_TYPE_F32][GGML_TYPE_F32][GGML_TYPE_F16] = &Op<float, float, half>::op;
    dispatch.d[GGML_TYPE_F32][GGML_TYPE_F32][GGML_TYPE_F32] = &Op<float, float, float>::op;

    return dispatch;
}

template<template <typename src0_t, typename src1_t, typename dst_t> class Op>
static ggml_cuda_op_t get_op_fn(ggml_type t0, ggml_type t1, ggml_type t2) {
    static const ggml_cuda_op_dispatch_t dispatch = gen_op_dispatch_table<Op>();

    if (dispatch.d[t0][t1][t2] == nullptr) {
        fprintf(stderr, "Unsupported type combination: %s %s %s\n",
                ggml_type_name(t0), ggml_type_name(t1), ggml_type_name(t2));
    }

    GGML_ASSERT(dispatch.d[t0][t1][t2] && "Unsupported type combination");
    return dispatch.d[t0][t1][t2];
}

template<template <typename src0_t, typename src1_t, typename dst_t> class Op>
static void ggml_cuda_op(ggml_cuda_context * ctx,
                    ggml_tensor * src0,
                    ggml_tensor * src1,
                    ggml_tensor * dst,
                    cudaStream_t main_stream,
                    bool flatten_rows) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t nrows0 = ggml_nrows(src0);

    const bool use_src1 = src1 != nullptr;
    const int64_t ne10 = use_src1 ? src1->ne[0] : 1;
    const int64_t ne11 = use_src1 ? src1->ne[1] : 1;
    const int64_t ne12 = use_src1 ? src1->ne[2] : 1;
    const int64_t ne13 = use_src1 ? src1->ne[3] : 1;

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    ggml_type t0 = src0->type;
    ggml_type t1 = use_src1 ? src1->type : GGML_TYPE_NONE;
    ggml_type t2 = dst->type;
    // HACK
    // get rows
    if (t1 == GGML_TYPE_I32) {
        t1 = t2;
    }
    // mul mat
    if (ggml_is_quantized(t0)) {
        t0 = t1;
    }

    ggml_cuda_op_t op = get_op_fn<Op>(t0, t1, t2);

    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    // strides for iteration over dims 3 and 2
    const int64_t num_iters = flatten_rows ? 1 : ne02 * ne03;
    const int64_t stride_mod = flatten_rows ? ne02 * ne03 : 1;
    const int64_t src0_stride = ne00 * ne01 * stride_mod;
    const int64_t src1_stride = ne10 * ne11 * stride_mod;
    const int64_t dst_stride = ne0 * ne1 * stride_mod;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t src1_ts = use_src1 ? ggml_type_size(src1->type) : 0;
    const size_t src1_bs = use_src1 ? ggml_blck_size(src1->type) : 1;
    const size_t dst_ts = ggml_type_size(dst->type);
    const size_t dst_bs = ggml_blck_size(dst->type);

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = use_src1 ? ggml_is_contiguous(src1) : true;

    void * src0_d = src0 ? ggml_cuda_get_buffer(ctx, src0) : nullptr;
    void * src1_d = src1 ? ggml_cuda_get_buffer(ctx, src1) : nullptr;
    void * dst_d  = dst  ? ggml_cuda_get_buffer(ctx, dst)  : nullptr;

    int64_t row_low = 0;
    int64_t row_high = nrows0;
    int64_t row_diff = row_high - row_low;

    size_t src0_as = 0;
    size_t src1_as = 0;
    if (!src0_is_contiguous) {
        src0_d = (float *) ggml_cuda_pool_malloc(row_diff*ne00 * src0_ts/src0_bs, &src0_as, main_stream);
    }

    if (!src1_is_contiguous) {
        src1_d = (float *) ggml_cuda_pool_malloc(num_iters*src1_stride * src1_ts/src1_bs, &src1_as, main_stream);
    }

    const int64_t i03_max = flatten_rows ? 1 : ne03;
    const int64_t i02_max = flatten_rows ? 1 : ne02;
    const int64_t rows_per_iter = flatten_rows ? nrows0 : ne01;
    const int64_t num_ops = i03_max * i02_max;

    if (num_ops > 1 && GGML_CUDA_MAX_SUBSTREAMS > 1) {
        // record an event on the stream to synchronize the sub-streams
        CUDA_CHECK(cudaEventRecord(g_cudaEvent_main, main_stream));
    }

    for (int64_t i03 = 0; i03 < i03_max; i03++) {
        const int64_t i13 = i03 % ne13;
        for (int64_t i02 = 0; i02 < i02_max; i02++) {
            const int64_t i12 = i02 % ne12;

            const int64_t i0 = i03*ne02 + i02;
            const int64_t i0_offset_low = row_low/rows_per_iter;
            //const int64_t i0_offset_high = row_high/rows_per_iter;

            int64_t i01_low = 0;
            int64_t i01_high = rows_per_iter;

            const int64_t i01_diff = i01_high - i01_low;
            if (i01_diff == 0) {
                continue;
            }
            const int64_t i11 = i13*ne12 + i12;

            cudaStream_t op_stream;
            if (num_ops > 1 && GGML_CUDA_MAX_SUBSTREAMS > 1) {
                op_stream = g_cudaStreams[i0 % GGML_CUDA_MAX_SUBSTREAMS];
                // wait for the main stream to finish, but only the first time per sub-stream
                if (i0 < GGML_CUDA_MAX_SUBSTREAMS) {
                    CUDA_CHECK(cudaStreamWaitEvent(op_stream, g_cudaEvent_main, 0));
                }
            } else {
                op_stream = main_stream;
            }
            // TODO: use different streams, record event, wait for all events on main stream at the end

            // for split tensors the data begins at i0 == i0_offset_low
            void * src0_d_i = (char *) src0_d + (i0 - i0_offset_low)*src0_stride*src0_ts/src0_bs;
            void * src1_d_i = (char *) src1_d + i11*src1_stride*src1_ts/src1_bs;
            void * dst_d_i  = (char *) dst_d + (i0 - i0_offset_low)*dst_stride*dst_ts/dst_bs;

            // copy src0, src1 to device if necessary
            // CUDA_CHECK(cudaEventRecord(cudaEvent_memcpy_src1, cudaStream_memcpy_src1));
            if (!src0_is_contiguous) {
                CUDA_CHECK(ggml_cuda_cpy_tensor_2d(ctx, src0_d_i, src0, i03, i02, i01_low, i01_high, op_stream));
            }
            if (!src1_is_contiguous) {
                CUDA_CHECK(ggml_cuda_cpy_tensor_2d(ctx, src1_d_i, src1, i03, i02, 0, ne11, op_stream));
            }

            op(src0, src1, dst,
                src0_d_i, src1_d_i, dst_d_i,
                i02, i01_low, i01_high, i11,
                op_stream);

            if (num_ops > 1 && GGML_CUDA_MAX_SUBSTREAMS > 1) {
                // record an event on the stream to synchronize with the main stream
                // only wait for the event if it is the last operation in this stream
                if (i0 >= (num_ops - GGML_CUDA_MAX_SUBSTREAMS)) {
                    CUDA_CHECK(cudaEventRecord(g_cudaEvents[i0 % GGML_CUDA_MAX_SUBSTREAMS], op_stream));
                }
            }
        }
    }

    if (num_ops > 1 && GGML_CUDA_MAX_SUBSTREAMS > 1) {
        // wait for all events on the main stream
        for (int64_t i0 = 0; i0 < std::min((int)num_ops, GGML_CUDA_MAX_SUBSTREAMS); i0++) {
            // wait on the main stream for the event
            CUDA_CHECK(cudaStreamWaitEvent(main_stream, g_cudaEvents[i0], 0));
        }
    }

    if (src1_as > 0) {
        ggml_cuda_pool_free(src1_d, src1_as, main_stream);
    }
    if (src0_as > 0) {
        ggml_cuda_pool_free(src0_d, src0_as, main_stream);
    }
}

static void ggml_cuda_cpy(ggml_cuda_context * ctx, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, cudaStream_t stream) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX);
    GGML_ASSERT(ggml_nbytes(src1) <= INT_MAX);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];

    cudaStream_t cudaStream_main = stream;

    void * d_src0 = src0 ? ggml_cuda_get_buffer(ctx, src0) : nullptr;
    void * d_src1 = src1 ? ggml_cuda_get_buffer(ctx, src1) : nullptr;

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_cuda<float, float>((char *) d_src0, (char *) d_src1, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, cudaStream_main);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_cuda<half, half>((char *) d_src0, (char *) d_src1, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, cudaStream_main);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_cuda<half, float>((char *) d_src0, (char *) d_src1, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, cudaStream_main);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_cuda<float, half>((char *) d_src0, (char *) d_src1, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, cudaStream_main);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_cuda<int32_t, int32_t>((char *) d_src0, (char *) d_src1, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, cudaStream_main);
    } else {
        GGML_ASSERT(false);
    }
    CUDA_CHECK(cudaGetLastError());

    UNUSED(dst);
}

static void ggml_cuda_mul_mat_vec_p021(ggml_cuda_context * ctx, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, cudaStream_t stream) {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    cudaStream_t cudaStream_main = stream;

    void * src0_d = src0 ? ggml_cuda_get_buffer(ctx, src0) : nullptr;
    void * src1_d = src1 ? ggml_cuda_get_buffer(ctx, src1) : nullptr;
    void * dst_d  = dst  ? ggml_cuda_get_buffer(ctx, dst)  : nullptr;

    if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        ggml_mul_mat_p021_cuda<half, half, half>((half *)src0_d, (half *)src1_d, (half *)dst_d, ne00, ne01, ne02, cudaStream_main);
    }
    else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        ggml_mul_mat_p021_cuda<half, float, float>((half *)src0_d, (float *)src1_d, (float *)dst_d, ne00, ne01, ne02, cudaStream_main);
    }
    else {
        GGML_ASSERT(false);
    }
}

static void ggml_cuda_mul_mat_vec_nc(ggml_cuda_context * ctx, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, cudaStream_t stream) {
    GGML_ASSERT(!ggml_is_contiguous(src0) && ggml_is_contiguous(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    cudaStream_t cudaStream_main = stream;

    void * src0_d = src0 ? ggml_cuda_get_buffer(ctx, src0) : nullptr;
    void * src1_d = src1 ? ggml_cuda_get_buffer(ctx, src1) : nullptr;
    void * dst_d  = dst  ? ggml_cuda_get_buffer(ctx, dst)  : nullptr;

    const int row_stride_x = nb01 / ggml_type_size(src0->type);
    const int channel_stride_x = nb02 / ggml_type_size(src0->type);

    if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        ggml_mul_mat_vec_nc_cuda<half, half, half>((half *)src0_d, (half *)src1_d, (half *)dst_d, ne00, ne01, row_stride_x, ne02, channel_stride_x, cudaStream_main);
    }
    else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        ggml_mul_mat_vec_nc_cuda<half, float, float>((half *)src0_d, (float *)src1_d, (float *)dst_d, ne00, ne01, row_stride_x, ne02, channel_stride_x, cudaStream_main);
    }
    else {
        GGML_ASSERT(false);
    }
}

static cudaDataType ggml_to_cuda_type(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16: return CUDA_R_16F;
        case GGML_TYPE_F32: return CUDA_R_32F;
        default: puts(ggml_type_name(t)); GGML_ASSERT(false);
    }
}

template<typename src0_t, typename src1_t, typename dst_t>
static void ggml_cuda_op_mul_mat_cublas(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    void * src0_d, void * src1_d, void * dst_d,
    int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int64_t ne0 = dst->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    const int ldc = ne0; //dst->backend == GGML_BACKEND_GPU && id == g_main_device ? ne0 : i01_diff;

    ggml_type ts0 = src0->type;
    ggml_type ts1 = src1->type;
    ggml_type td = dst->type;

    size_t src0_as = 0;
    cublasComputeType_t compute_type;

    if (ts0 == GGML_TYPE_F16 && ts1 == GGML_TYPE_F16 && td == GGML_TYPE_F16) {
        compute_type = CUBLAS_COMPUTE_16F;
    }
    else if (ts0 == GGML_TYPE_F32 && ts1 == GGML_TYPE_F32 && td == GGML_TYPE_F32) {
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
    else if (ts1 == GGML_TYPE_F32 && td == GGML_TYPE_F32) {
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;

        int ne = i01_diff * ne00;
        void * src0_f32 = ggml_cuda_pool_malloc(ne * sizeof(float), &src0_as, stream);

        const to_t_cuda_t<float> to_fp32_cuda = ggml_get_to_t_cuda<float>(src0->type);
        GGML_ASSERT(to_fp32_cuda);
        //printf("converting %s from %s\n", src0->name, ggml_type_name(src0->type));
        to_fp32_cuda(src0_d, (float *)src0_f32, ne, stream);
        CUDA_CHECK(cudaGetLastError());
        src0_d = src0_f32;
        ts0 = GGML_TYPE_F32;
    }
    else if (ts1 == GGML_TYPE_F16) {
        if (td == GGML_TYPE_F16) {
            compute_type = CUBLAS_COMPUTE_16F;
        }
        else if (td == GGML_TYPE_F32) {
            compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
        }
        else {
            GGML_ASSERT(false);
        }

        int ne = i01_diff * ne00;
        void * src0_f16 = ggml_cuda_pool_malloc(ne * sizeof(half), &src0_as, stream);

        const to_t_cuda_t<half> to_fp16_cuda = ggml_get_to_t_cuda<half>(src0->type);
        GGML_ASSERT(to_fp16_cuda);

        to_fp16_cuda(src0_d, (half *)src0_f16, ne, stream);
        CUDA_CHECK(cudaGetLastError());
        src0_d = src0_f16;
        ts0 = GGML_TYPE_F16;
    }
    else {
        fprintf(stderr, "cuBLAS: unsupported types: %s * %s -> %s\n",
            ggml_type_name(ts0), ggml_type_name(ts1), ggml_type_name(td));
        GGML_ASSERT(false);
    }

    half alpha_f16 = 1.0f;
    half beta_f16 = 0.0f;
    float alpha_f32 = 1.0f;
    float beta_f32 = 0.0f;
    const void * alpha;
    const void * beta;

    switch (compute_type) {
        case CUBLAS_COMPUTE_16F:
            alpha = &alpha_f16; beta = &beta_f16;
            break;
        case CUBLAS_COMPUTE_32F_FAST_TF32:
        case CUBLAS_COMPUTE_32F:
            alpha = &alpha_f32; beta = &beta_f32;
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    CUBLAS_CHECK(cublasSetStream(g_cublas_handle, stream));
    CUBLAS_CHECK(
        cublasGemmEx(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                i01_diff, ne11, ne10,
                alpha, src0_d, ggml_to_cuda_type(ts0), ne00,
                       src1_d, ggml_to_cuda_type(ts1), ne10,
                beta,  dst_d,  ggml_to_cuda_type(td), ldc,
                compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (src0_as) {
        ggml_cuda_pool_free(src0_d, src0_as, stream);
    }

    UNUSED(i02);
    UNUSED(i1);
}

#define DEFINE_GGML_CUDA_OP_S(op_name)                                                              \
    template<typename src0_t, typename src1_t, typename dst_t>                                      \
    struct ggml_cuda_op_ ## op_name ## _s {                                                         \
        static void op(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,       \
                       void * src0_d, void * src1_d, void * dst_d,                                  \
                       int64_t i02, int64_t i01_low, int64_t i01_high, int i1,                      \
                       cudaStream_t stream) {                                                       \
            ggml_cuda_op_ ## op_name<src0_t, src1_t, dst_t>(src0, src1, dst,                        \
                src0_d, src1_d, dst_d,                                                              \
                i02, i01_low, i01_high, i1,                                                         \
                stream);                                                                            \
        }                                                                                           \
    }

DEFINE_GGML_CUDA_OP_S(add);
DEFINE_GGML_CUDA_OP_S(mul);
DEFINE_GGML_CUDA_OP_S(scale);
DEFINE_GGML_CUDA_OP_S(mul_mat_cublas);
DEFINE_GGML_CUDA_OP_S(dequantize_mul_mat_vec);
DEFINE_GGML_CUDA_OP_S(silu);
DEFINE_GGML_CUDA_OP_S(soft_max);
DEFINE_GGML_CUDA_OP_S(diag_mask_inf);
DEFINE_GGML_CUDA_OP_S(rms_norm);
DEFINE_GGML_CUDA_OP_S(rope);
DEFINE_GGML_CUDA_OP_S(get_rows);

#undef DEFINE_GGML_CUDA_OP_S

static void ggml_cuda_mul_mat(ggml_cuda_context * ctx, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, cudaStream_t stream) {
    if (ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        ggml_cuda_mul_mat_vec_p021(ctx, src0, src1, dst, stream);
    } else if (!ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && src1->ne[1] == 1) {
        ggml_cuda_mul_mat_vec_nc(ctx, src0, src1, dst, stream);
    } else {
        if (src1->ne[1] == 1 && src0->ne[0] % GGML_CUDA_DMMV_X == 0 && src0->ne[1] % GGML_CUDA_DMMV_Y == 0) {
            ggml_cuda_op<ggml_cuda_op_dequantize_mul_mat_vec_s>(ctx, src0, src1, dst, stream, false);
        } else {
            ggml_cuda_op<ggml_cuda_op_mul_mat_cublas_s>(ctx, src0, src1, dst, stream, false);
        }
    }
}

static void ggml_cuda_exec_node(ggml_cuda_context * ctx, ggml_tensor * node, cudaStream_t stream) {
    ggml_tensor * src0 = node->src0;
    ggml_tensor * src1 = node->src1;
    ggml_tensor * dst  = node;

#if 0
    fprintf(stdout, "%s: %s %s %s %s (%s, %s, %s) %d\n",
                dst->name,
                ggml_op_name(dst->op),
                src0 ? ggml_type_name(src0->type) : "null",
                src1 ? ggml_type_name(src1->type) : "null",
                dst  ? ggml_type_name(dst->type)  : "null",
                src0 ? ggml_get_name(src0) : "null",
                src1 ? ggml_get_name(src1) : "null",
                dst  ? ggml_get_name(dst)  : "null",
                src1 ? ggml_is_contiguous(src1) : -1
            );
#endif
    switch ((int)dst->op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_NONE:
            {
                // noop
            } break;
        case GGML_OP_ADD:
            {
                ggml_cuda_op<ggml_cuda_op_add_s>(ctx, src0, src1, dst, stream, true);
            } break;
        case GGML_OP_MUL:
            {
                ggml_cuda_op<ggml_cuda_op_mul_s>(ctx, src0, src1, dst, stream, false); // TODO ggml_cuda_op needs modification for flatten
            } break;
        case GGML_OP_SCALE:
            {
                ggml_cuda_op<ggml_cuda_op_scale_s>(ctx, src0, src1, dst, stream, true);
            } break;
        case GGML_OP_SILU:
            {
                ggml_cuda_op<ggml_cuda_op_silu_s>(ctx, src0, src1, dst, stream, true);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_cuda_op<ggml_cuda_op_soft_max_s>(ctx, src0, src1, dst, stream, true);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_cuda_op<ggml_cuda_op_diag_mask_inf_s>(ctx, src0, src1, dst, stream, true);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_cuda_mul_mat(ctx, src0, src1, dst, stream);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_cuda_op<ggml_cuda_op_get_rows_s>(ctx, src0, src1, dst, stream, true);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_cuda_op<ggml_cuda_op_rms_norm_s>(ctx, src0, src1, dst, stream, true);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_cuda_op<ggml_cuda_op_rope_s>(ctx, src0, src1, dst, stream, false); // FIXME flatten changes results
            } break;
        case GGML_OP_CPY:
            {
                ggml_cuda_cpy(ctx, src0, src1, dst, stream);
            } break;
        default:
            fprintf(stderr, "%s: op = %8s not implemented\n", __func__, ggml_op_name(dst->op));
            GGML_ASSERT(false);
    }
}

static const int GGML_MAX_PARENTS = 2 + GGML_MAX_OPT;

static bool ggml_is_noop(ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// TODO: reduce number of streams and events
static void ggml_cuda_graph_exec_parallel(ggml_cuda_context * ctx, ggml_cgraph * gf, cudaStream_t mainStream) {
    // record an event for the nodes to add a dependency on
    cudaEvent_t mainEvent = g_cudaEvent_main;

    CUDA_CHECK(cudaEventRecord(mainEvent, mainStream));

    // TODO: move to context and free
    static std::vector<cudaStream_t> free_streams;
    static std::vector<cudaEvent_t> free_events;

    // TODO: preserve the order to allow reusing pool allocations
    free_streams.insert(free_streams.begin(), mainStream);

    std::unordered_set<cudaStream_t> node_streams;
    std::vector<cudaEvent_t> node_events;
    std::unordered_map<ggml_tensor *, cudaEvent_t> event_map;
    std::unordered_map<ggml_tensor *, cudaStream_t> stream_map;

    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];
        const bool is_noop = ggml_is_noop(node);

        // build a list of parents
        ggml_tensor * parents[GGML_MAX_PARENTS] = { node->src0, node->src1 };
        for (int j = 0; j < GGML_MAX_OPT; j++) {
            parents[j + 2] = node->opt[j];
        }

        // assign an stream for the node
        cudaStream_t stream = nullptr;

        // take a stream from a parent
        for (int j = 0; j < GGML_MAX_PARENTS; j++) {
            if (parents[j] && stream_map.count(parents[j]) && stream_map[parents[j]] != nullptr) {
                stream = stream_map[parents[j]];
                stream_map.erase(parents[j]);

                if (is_noop) {
                    // if this is a noop, we can use the parent's event
                    stream_map[node] = stream;
                    if (event_map.count(parents[j]) > 0) {
                        event_map[node] = event_map[parents[j]];
                    }
                }
                break;
            }
        }

        if (is_noop) {
            continue;
        }

        // otherwise, create a new stream
        if (!stream) {
            if (!free_streams.empty()) {
                stream = free_streams.back();
                free_streams.pop_back();
            }
            else {
                CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            }
        }

        // wait on parent streams
        bool waited = false;
        for (int j = 0; j < GGML_MAX_PARENTS; j++) {
            if (parents[j] && event_map.count(parents[j]) > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, event_map[parents[j]], 0));
                waited = true;
            }
        }

        // wait on the start event to introduce a dependency if no parents
        if (!waited) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, mainEvent, 0));
        }

        // execute the node
        ggml_cuda_exec_node(ctx, node, stream);

        // record an event for the node
        cudaEvent_t event;
        if (!free_events.empty()) {
            event = free_events.back();
            free_events.pop_back();
        }
        else {
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(event, stream));

        // save stream and event
        if (stream != mainStream) {
            node_streams.insert(stream);
        }
        node_events.push_back(event);
        event_map[node] = event;
        stream_map[node] = stream;
    }

    // wait for the group streams to finish
    for (auto & it : node_events) {
        CUDA_CHECK(cudaStreamWaitEvent(mainStream, it, 0));
    }

    //printf("used %d events and %d streams\n", (int)node_events.size(), (int)node_streams.size());

    // save streams and events for reuse
    free_streams.insert(free_streams.end(), node_streams.begin(), node_streams.end());
    free_events.insert(free_events.end(), node_events.begin(), node_events.end());
}

static void ggml_cuda_synchronize(struct ggml_cuda_context * ctx) {
    CUDA_CHECK(cudaStreamSynchronize(g_cudaStream_main));

    UNUSED(ctx);
}

static void ggml_cuda_cgraph_compute(ggml_cuda_context * ctx, ggml_cgraph * gf) {
    cudaStream_t stream = g_cudaStream_main;

    if (GGML_CUDA_SEQ_COMPUTE) {
        for (int i = 0; i < gf->n_nodes; ++i) {
            ggml_cuda_exec_node(ctx, gf->nodes[i], stream);
        }
    }
    else {
        ggml_cuda_graph_exec_parallel(ctx, gf, stream);
    }
}

// backend interface

struct ggml_backend_cuda_context {
    ggml_cuda_context * cuda_ctx = ggml_cuda_init();
};

static const char * ggml_backend_cuda_name(ggml_backend_context_t ctx) {
    return "CUDA";

    UNUSED(ctx);
}

static void ggml_backend_cuda_free_context(ggml_backend_context_t ctx) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)ctx;
    ggml_cuda_free(cuda_ctx->cuda_ctx);
    delete cuda_ctx;
}

struct cuda_backend_buffer {
    void * data;
    size_t offset;
    size_t size;
};

static const size_t TENSOR_ALIGNMENT = 128;

static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

static ggml_backend_buffer_t ggml_backend_cuda_alloc_buffer(ggml_backend_context_t ctx, size_t size) {
    cuda_backend_buffer * buffer = new cuda_backend_buffer;

    CUDA_CHECK(cudaMalloc(&buffer->data, size));
    buffer->offset = 0; // cudaMalloc returns aligned pointers
    buffer->size = size;

    return buffer;

    UNUSED(ctx);
}

static void ggml_backend_cuda_free_buffer(ggml_backend_context_t ctx, ggml_backend_buffer_t buffer) {
    cuda_backend_buffer * cuda_buffer = (cuda_backend_buffer *)buffer;
    CUDA_CHECK(cudaFree(cuda_buffer->data));
    delete cuda_buffer;

    UNUSED(ctx);
}

static void ggml_backend_cuda_reset_buffer(ggml_backend_context_t ctx, ggml_backend_buffer_t buffer) {
    cuda_backend_buffer * cuda_buffer = (cuda_backend_buffer *)buffer;
    cuda_buffer->offset = 0;

    UNUSED(ctx);
}

static void ggml_backend_cuda_alloc_tensor(ggml_backend_context_t ctx, ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    cuda_backend_buffer * cuda_buffer = (cuda_backend_buffer *)buffer;

    if (cuda_buffer->offset + ggml_nbytes(tensor) > cuda_buffer->size) {
        fprintf(stderr, "%s: not enough space in the CUDA buffer (needed %zu, available %zu)\n",
                __func__, ggml_nbytes(tensor), cuda_buffer->size - cuda_buffer->offset);
        GGML_ASSERT(false);
    }

    tensor->data = (char*)cuda_buffer->data + cuda_buffer->offset;
    cuda_buffer->offset = aligned_offset(cuda_buffer->data, cuda_buffer->offset + ggml_nbytes(tensor), TENSOR_ALIGNMENT);

    UNUSED(ctx);
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_context_t ctx, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    //ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)ctx;

    CUDA_CHECK(cudaMemcpyAsync((char*)tensor->data + offset, data, size, cudaMemcpyHostToDevice, g_cudaStream_main));

    UNUSED(ctx);
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_context_t ctx, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    //ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)ctx;

    CUDA_CHECK(cudaMemcpyAsync(data, (const char*)tensor->data + offset, size, cudaMemcpyDeviceToHost, g_cudaStream_main));

    UNUSED(ctx);
}

static void ggml_backend_cuda_synchronize(ggml_backend_context_t ctx) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)ctx;
    ggml_cuda_synchronize(cuda_ctx->cuda_ctx);
}

static ggml_graph_plan_t ggml_backend_cuda_graph_plan_create(ggml_backend_context_t ctx, ggml_cgraph * cgraph) {
    GGML_ASSERT(false);

    return nullptr;

    UNUSED(ctx);
    UNUSED(cgraph);
}

static void ggml_backend_cuda_graph_plan_free(ggml_backend_context_t ctx, ggml_graph_plan_t plan) {
    GGML_ASSERT(false);

    UNUSED(ctx);
    UNUSED(plan);
}

static void ggml_backend_cuda_graph_plan_compute(ggml_backend_context_t ctx, ggml_graph_plan_t plan) {
    GGML_ASSERT(false);

    UNUSED(ctx);
    UNUSED(plan);
}

static void ggml_backend_cuda_graph_compute(ggml_backend_context_t ctx, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)ctx;
    ggml_cuda_cgraph_compute(cuda_ctx->cuda_ctx, cgraph);
}

static ggml_backend_interface cuda_backend_interface = {
    /* .get_name            = */ ggml_backend_cuda_name,
    /* .free_context        = */ ggml_backend_cuda_free_context,
    /* .alloc_buffer        = */ ggml_backend_cuda_alloc_buffer,
    /* .free_buffer         = */ ggml_backend_cuda_free_buffer,
    /* .reset_buffer        = */ ggml_backend_cuda_reset_buffer,
    /* .alloc_tensor        = */ ggml_backend_cuda_alloc_tensor,
    /* .set_tensor_async    = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async    = */ ggml_backend_cuda_get_tensor_async,
    /* .synchronize         = */ ggml_backend_cuda_synchronize,
    /* .cpy_tensor_from     = */ nullptr,
    /* .cpy_tensor_to       = */ nullptr,
    /* .graph_plan_create   = */ ggml_backend_cuda_graph_plan_create,
    /* .graph_plan_free     = */ ggml_backend_cuda_graph_plan_free,
    /* .graph_plan_compute  = */ ggml_backend_cuda_graph_plan_compute,
    /* .graph_compute       = */ ggml_backend_cuda_graph_compute
};

ggml_backend ggml_backend_cuda_init(void) {
    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context;

    ggml_backend cuda_backend = {
        /* .interface = */ &cuda_backend_interface,
        /* .context   = */ ctx
    };
    return cuda_backend;
}
