#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <stdio.h>
#include <atomic>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "ggml-cuda.h"
#include "ggml.h"

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define CUDA_CHECK(err)                                                                 \
    do {                                                                                \
        cudaError_t err_ = (err);                                                       \
        if (err_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,   \
                cudaGetErrorString(err_));                                              \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#if CUDART_VERSION >= 12
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d: %s\n",                         \
                    err_, __FILE__, __LINE__, cublasGetStatusString(err_));             \
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
#endif // CUDART_VERSION >= 11

typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, float & v0, float & v1);
typedef void (*to_fp32_cuda_t)(const void * x, float * y, int k, cudaStream_t stream);
typedef void (*ggml_cuda_func_t)(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
typedef void (*ggml_cuda_op_t)(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i0_low, int64_t i0_high, int i1, cudaStream_t & cudaStream_main);

// QK = number of values after dequantization
// QR = QK / number of values before dequantization

#define QK4_0 32
#define QR4_0 2
typedef struct {
    half    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
#define QR4_1 2
typedef struct {
    half    d;              // delta
    half    m;              // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(ggml_fp16_t) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
#define QR5_0 2
typedef struct {
    half d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
#define QR5_1 2
typedef struct {
    half d;                 // delta
    half m;                 // min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
#define QR8_0 1
typedef struct {
    half    d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define WARP_SIZE 32

#define CUDA_MUL_BLOCK_SIZE 256

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_CUDA_DMMV_X
#define GGML_CUDA_DMMV_X 32
#endif
#ifndef GGML_CUDA_DMMV_Y
#define GGML_CUDA_DMMV_Y 1
#endif

static __global__ void mul_f32(const float * x, const float * y, float * dst, const int kx, const int ky) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= kx) {
        return;
    }
    dst[i] = x[i] * y[i%ky];
}

static __device__ void dequantize_q4_0(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const uint8_t vui = x[ib].qs[iqs];

    const int8_t vi0 = vui & 0xF;
    const int8_t vi1 = vui >> 4;

    v0 = (vi0 - 8)*d;
    v1 = (vi1 - 8)*d;
}

static __device__ void dequantize_q4_1(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float d = x[ib].d;
    const float m = x[ib].m;

    const uint8_t vui = x[ib].qs[iqs];

    const int8_t vi0 = vui & 0xF;
    const int8_t vi1 = vui >> 4;

    v0 = vi0*d + m;
    v1 = vi1*d + m;
}

static __device__ void dequantize_q5_0(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const uint8_t xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    const int32_t x0 = ((x[ib].qs[iqs] & 0xf) | xh_0) - 16;
    const int32_t x1 = ((x[ib].qs[iqs] >>  4) | xh_1) - 16;

    v0 = x0*d;
    v1 = x1*d;
}

static __device__ void dequantize_q5_1(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float d = x[ib].d;
    const float m = x[ib].m;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const uint8_t xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    const int32_t x0 = ((x[ib].qs[iqs] & 0xf) | xh_0);
    const int32_t x1 = ((x[ib].qs[iqs] >>  4) | xh_1);

    v0 = x0*d + m;
    v1 = x1*d + m;
}

static __device__ void dequantize_q8_0(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    const int8_t vi0 = x[ib].qs[iqs + 0];
    const int8_t vi1 = x[ib].qs[iqs + 1];

    v0 = vi0*d;
    v1 = vi1*d;
}

static __device__ void convert_f16(const void * vx, const int ib, const int iqs, float & v0, float & v1){
    const half * x = (const half *) vx;

    v0 = __half2float(x[ib + iqs + 0]);
    v1 = __half2float(x[ib + iqs + 1]);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void dequantize_block(const void * vx, float * y, const int k) {
    const int i = blockDim.x*blockIdx.x + 2*threadIdx.x;

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    float & v0 = y[iybs + iqs + 0];
    float & v1 = y[iybs + iqs + y_offset];
    dequantize_kernel(vx, ib, iqs, v0, v1);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void dequantize_mul_mat_vec(const void * vx, const float * y, float * dst, const int ncols) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

    float tmp = 0; // partial sum for thread in warp

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            float v0, v1;
            dequantize_kernel(vx, ib, iqs + j/qr, v0, v1);
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val

            // matrix multiplication
            tmp += v0 * y[iybs + iqs + j/qr + 0];
            tmp += v1 * y[iybs + iqs + j/qr + y_offset];
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
        }
    }

    // sum up partial sums and write back result
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void mul_f32_cuda(const float * x, const float * y, float * dst, const int kx, const int ky, cudaStream_t stream) {
    const int num_blocks = (kx + CUDA_MUL_BLOCK_SIZE - 1) / CUDA_MUL_BLOCK_SIZE;
    mul_f32<<<num_blocks, CUDA_MUL_BLOCK_SIZE, 0, stream>>>(x, y, dst, kx, ky);
}

static void dequantize_row_q4_0_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK4_0, QR4_0, dequantize_q4_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q4_1_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK4_1, QR4_1, dequantize_q4_1><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q5_0_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK5_0, QR5_0, dequantize_q5_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q5_1_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK5_1, QR5_1, dequantize_q5_1><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_row_q8_0_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<QK8_0, QR8_0, dequantize_q8_0><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_mul_mat_vec_q4_0_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q4_1_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q5_0_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q5_1_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void dequantize_mul_mat_vec_q8_0_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static void convert_fp16_to_fp32_cuda(const void * vx, float * y, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block<1, 1, convert_f16><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void convert_mul_mat_vec_f16_cuda(const void * vx, const float * y, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % GGML_CUDA_DMMV_X == 0);
    GGML_ASSERT(nrows % GGML_CUDA_DMMV_Y == 0);
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_DMMV_Y, 1);
    dequantize_mul_mat_vec<1, 1, convert_f16>
        <<<nrows/GGML_CUDA_DMMV_Y, block_dims, 0, stream>>>(vx, y, dst, ncols);
}

static to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
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
        case GGML_TYPE_F16:
            return convert_fp16_to_fp32_cuda;
        default:
            return nullptr;
    }
}

// buffer pool for cuda
#define MAX_CUDA_BUFFERS 256

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

struct cuda_buffer {
    void * ptr = nullptr;
    size_t size = 0;
};

static cuda_buffer g_cuda_buffer_pool[GGML_CUDA_MAX_DEVICES][MAX_CUDA_BUFFERS];
static std::atomic_flag g_cuda_pool_lock = ATOMIC_FLAG_INIT;

static void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[id][i];
        if (b.size >= size && b.ptr != nullptr) {
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
    }
    void * ptr;
    CUDA_CHECK(cudaMalloc((void **) &ptr, size));
    *actual_size = size;
    return ptr;
}

static void ggml_cuda_pool_free(void * ptr, size_t size) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[id][i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
    CUDA_CHECK(cudaFree(ptr));
}

#define GGML_CUDA_MAX_STREAMS 8 // Set this to 1 for reproducible matrix multiplication.
#define GGML_CUDA_MAX_EVENTS 64

static int g_device_count = -1;
static float g_tensor_split[GGML_CUDA_MAX_DEVICES] = {0};

static cublasHandle_t g_cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

static cudaStream_t g_cudaStreams_main[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { nullptr };

static cudaStream_t g_cudaStreams_memcpy_src1[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { nullptr };
static cudaEvent_t g_cudaEvents_memcpy_src1[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_EVENTS] = { nullptr };

void ggml_init_cublas() {
    static bool initialized = false;

    if (!initialized) {
        CUDA_CHECK(cudaGetDeviceCount(&g_device_count));
        GGML_ASSERT(g_device_count <= GGML_CUDA_MAX_DEVICES);
        int64_t total_vram = 0;
        fprintf(stderr, "%s: found %d CUDA devices:\n", __func__, g_device_count);
        for (int i = 0; i < g_device_count; ++i) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
            fprintf(stderr, "  %d. %s\n", i+1, prop.name);
            g_tensor_split[i] = total_vram;
            total_vram += prop.totalGlobalMem;
        }
        for (int i = 0; i < g_device_count; ++i) {
            g_tensor_split[i] /= total_vram;
        }

        for (int id = 0; id < g_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));

            // create streams
            for (int i = 0; i < GGML_CUDA_MAX_STREAMS; ++i) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams_main[id][i], cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams_memcpy_src1[id][i], cudaStreamNonBlocking));
            }
            // create events
            for (int i = 0; i < GGML_CUDA_MAX_EVENTS; ++i) {
                CUDA_CHECK(cudaEventCreateWithFlags(&g_cudaEvents_memcpy_src1[id][i], cudaEventDisableTiming));
            }

            // create cublas handle
            CUBLAS_CHECK(cublasCreate(&g_cublas_handles[id]));
            CUBLAS_CHECK(cublasSetMathMode(g_cublas_handles[id], CUBLAS_TF32_TENSOR_OP_MATH));
        }

        // configure logging to stdout
        // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

        initialized = true;
    }
}

void ggml_cuda_set_tensor_split(float * tensor_split) {
    bool all_zero = true;
    for (int i = 0; i < g_device_count; ++i) {
        if (tensor_split[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return;
    }
    float split_sum = 0.0f;
    for (int i = 0; i < g_device_count; ++i) {
        g_tensor_split[i] = split_sum;
        split_sum += tensor_split[i];
    }
    for (int i = 0; i < g_device_count; ++i) {
        g_tensor_split[i] /= split_sum;
    }
}

void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory: %s\n",
            size/1024.0/1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

void ggml_cuda_host_free(void * ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

static cudaError_t ggml_cuda_h2d_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    char * dst_char = (char *) dst;
    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const void * x = (const void *) ((const char *) src->data + i1_low*nb1 + i2*nb2 + i3*nb3);
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_char, x, i1_diff*nb1, cudaMemcpyHostToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_char, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyHostToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_char + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyHostToDevice, stream);
            if (r != cudaSuccess) return r;
        }
        return cudaSuccess;
    }
}

inline void ggml_cuda_op_mul(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i0_low, int64_t i0_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    for (int64_t i01 = i0_low; i01 < i0_high; i01++) {
        const int64_t i11 = i1*ne11 + i01%ne11; // broadcast src1 across src0

        float * src0_ddf_i01 = src0_ddf_i + i01*ne00;
        float * src1_ddf_i01 = src1_ddf_i + i11*ne10;
        float * dst_ddf_i01 = dst_ddf_i + i01*ne00;

        // compute
        mul_f32_cuda(src0_ddf_i01, src1_ddf_i01, dst_ddf_i01, ne00, ne10, cudaStream_main);
        CUDA_CHECK(cudaGetLastError());
    }

    (void) dst;
    (void) src0_ddq_i;
}

inline void ggml_cuda_op_dequantize_mul_mat_vec(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i0_low, int64_t i0_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddq_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = i0_high - i0_low;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            dequantize_mul_mat_vec_q4_0_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_mul_mat_vec_q8_0_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        case GGML_TYPE_F16:
            convert_mul_mat_vec_f16_cuda(src0_ddq_i, src1_ddf_i, dst_ddf_i, ne00, nrows, cudaStream_main);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
    CUDA_CHECK(cudaGetLastError());

    (void) src1;
    (void) dst;
    (void) src0_ddf_i;
    (void) i1;
}

inline void ggml_cuda_op_mul_mat_cublas(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i0_low, int64_t i0_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int64_t i0_diff = i0_high - i0_low;

    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    CUBLAS_CHECK(cublasSetStream(g_cublas_handles[id], cudaStream_main));
    CUBLAS_CHECK(
        cublasSgemm(g_cublas_handles[id], CUBLAS_OP_T, CUBLAS_OP_N,
                i0_diff, ne11, ne10,
                &alpha, src0_ddf_i, ne00,
                        src1_ddf_i, ne10,
                &beta,  dst_ddf_i,  i0_diff));

    (void) dst;
    (void) src0_ddq_i;
    (void) i1;
}

static void ggml_cuda_op(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                         ggml_cuda_op_t op, bool src0_needs_f32) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t nrows0 = ggml_nrows(src0);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    // strides for iteration over dims 3 and 2
    const int64_t src0_stride = ne00 * ne01;
    const int64_t src1_stride = ne10 * ne11;
    const int64_t dst_stride = ne0 * ne1;
    const int64_t num_iters = ne02 * ne03;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);

    struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    struct ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    struct ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;

    // indices of the devices on which the input data is stored
    int src0_id = src0_extra == nullptr ? -1 : src0_extra->i_device;
    int src1_id = src1_extra == nullptr ? -1 : src1_extra->i_device;

    const bool src0_on_device = src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT;
    const bool src0_is_f32 = src0->type == GGML_TYPE_F32;

    const bool src1_on_device = src1->backend == GGML_BACKEND_GPU || src1->backend == GGML_BACKEND_GPU_SPLIT;

    const bool dst_on_device = dst->backend == GGML_BACKEND_GPU || dst->backend == GGML_BACKEND_GPU_SPLIT;

    const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);

    // dd = data device
    char  * src0_ddq[GGML_CUDA_MAX_DEVICES] = {nullptr}; // quantized
    float * src0_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr}; // float
    float * src1_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr};
    float * dst_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr};

    // asq = actual size quantized, asf = actual size float
    size_t src0_asq[GGML_CUDA_MAX_DEVICES] = {0};
    size_t src0_asf[GGML_CUDA_MAX_DEVICES] = {0};
    size_t src1_asf[GGML_CUDA_MAX_DEVICES] = {0};
    size_t dst_asf[GGML_CUDA_MAX_DEVICES] = {0};

    for (int id = 0; id < g_device_count; ++id) {
        // if data is on one device (!= -1) but not this one, continue
        if (src0_id != -1 && src0_id != id) {
            continue;
        }
        if (src1_id != -1 && src1_id != id) {
            continue;
        }

        bool split = src0_id == -1 && src1_id == -1;
        int64_t row_low, row_high;
        if (split) {
            row_low = id == 0 ? 0 : nrows0*g_tensor_split[id];
            row_low -= row_low % GGML_CUDA_DMMV_Y;
            row_high = id == g_device_count - 1 ? nrows0 : nrows0*g_tensor_split[id + 1];
            row_high -= row_high % GGML_CUDA_DMMV_Y;
        } else {
            row_low = 0;
            row_high = ne01;
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t row_diff = row_high - row_low;

        cudaSetDevice(id);

        if (src0_on_device) {
            if (src0_is_f32) {
                src0_ddf[id] = (float *) src0_extra->data_device[id];
            } else {
                src0_ddq[id] = (char *) src0_extra->data_device[id];
            }
        } else {
            if (src0_is_f32) {
                src0_ddf[id] = (float *) ggml_cuda_pool_malloc(row_diff*ne00 * sizeof(float), &src0_asf[id]);
            } else {
                src0_ddq[id] = (char *) ggml_cuda_pool_malloc(row_diff*ne00 * src0_ts/src0_bs, &src0_asq[id]);
            }
        }

        if (src0_needs_f32 && !src0_is_f32) {
            src0_ddf[id] = (float *) ggml_cuda_pool_malloc(row_diff*ne00 * sizeof(float), &src0_asf[id]);
        }

        if (src1_on_device) {
            src1_ddf[id] = (float *) src1_extra->data_device[id];
        } else {
            src1_ddf[id] = (float *) ggml_cuda_pool_malloc(num_iters*src1_stride * sizeof(float), &src1_asf[id]);
        }
        if (dst_on_device) {
            dst_ddf[id] = (float *) dst_extra->data_device[id];
        } else {
            size_t size_dst_ddf = split ? row_diff*ne1 * sizeof(float) : num_iters*dst_stride * sizeof(float);
            dst_ddf[id] = (float *) ggml_cuda_pool_malloc(size_dst_ddf, &dst_asf[id]);
        }

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            const int64_t i13 = i03 % ne13;
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const int64_t i12 = i02 % ne12;

                const int64_t i0 = i03*ne02 + i02;
                const int64_t i0_offset_low = row_low/ne01;
                const int64_t i0_offset_high = row_high/ne01;

                int64_t i01_low = 0;
                int64_t i01_high = ne01;
                if (split) {
                    if (i0 < i0_offset_low || i0 > i0_offset_high) {
                        continue;
                    }
                    if (i0 == i0_offset_low) {
                        i01_low = row_low % ne01;
                    }
                    if (i0 == i0_offset_high) {
                        i01_high = row_high % ne01;
                    }
                }
                const int64_t i01_diff = i01_high - i01_low;
                if (i01_diff == 0) {
                    continue;
                }
                const int64_t i1 = i13*ne12 + i12;

                cudaStream_t cudaStream_main = g_cudaStreams_main[id][i0 % GGML_CUDA_MAX_STREAMS];
                cudaStream_t cudaStream_memcpy_src1 = g_cudaStreams_memcpy_src1[id][i0 % GGML_CUDA_MAX_STREAMS];
                cudaEvent_t  cudaEvent_memcpy_src1 = g_cudaEvents_memcpy_src1[id][i0 % GGML_CUDA_MAX_EVENTS];

                // for split tensors the data begins at i0 == i0_offset_low
                char  * src0_ddq_i = src0_ddq[id] + (i0 - i0_offset_low)*src0_stride*src0_ts/src0_bs;
                float * src0_ddf_i = src0_ddf[id] + (i0 - i0_offset_low)*src0_stride;
                float * src1_ddf_i = src1_ddf[id] + i1*src1_stride;
                float * dst_ddf_i = dst_ddf[id] + (i0 - i0_offset_low)*dst_stride;

                // for split tensors the data pointer needs to be rounded down
                // to the bin edge for i03, i02 bins beyond the first
                if (i0 - i0_offset_low > 0) {
                    src0_ddq_i -= (row_low % ne01)*ne00 * src0_ts/src0_bs;
                    src0_ddf_i -= (row_low % ne01)*ne00;
                }
                if (i0 - i0_offset_low > 0) {
                    dst_ddf_i -= (row_low % ne0)*ne1;
                }

                // copy src0, src1 to device if necessary
                if (!src1_on_device) {
                    CUDA_CHECK(ggml_cuda_h2d_tensor_2d(src1_ddf_i, src1, i03, i02, 0, ne11, cudaStream_memcpy_src1));
                }
                CUDA_CHECK(cudaEventRecord(cudaEvent_memcpy_src1, cudaStream_memcpy_src1));
                if (!src0_on_device) {
                    if (src0_is_f32) {
                        CUDA_CHECK(ggml_cuda_h2d_tensor_2d(src0_ddf_i, src0, i03, i02, i01_low, i01_high, cudaStream_main));
                    } else {
                        CUDA_CHECK(ggml_cuda_h2d_tensor_2d(src0_ddq_i, src0, i03, i02, i01_low, i01_high, cudaStream_main));
                    }
                }

                // convert src0 to f32 if it's necessary for the ggml_cuda_op
                if (src0_needs_f32 && !src0_is_f32) {
                    to_fp32_cuda(src0_ddq_i, src0_ddf_i, i01_diff*ne00, cudaStream_main);
                    CUDA_CHECK(cudaGetLastError());
                }

                // wait with main stream until src1 memcpy is done
                CUDA_CHECK(cudaStreamWaitEvent(cudaStream_main, cudaEvent_memcpy_src1, 0));

                // do the computation
                op(src0, src1, dst, src0_ddq_i, src0_ddf_i, src1_ddf_i, dst_ddf_i, i01_low, i01_high, i1, cudaStream_main);

                // copy dst to host if necessary
                if (!dst_on_device) {
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of cuBLAS matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        for (int64_t j = 0; j < ne1; ++j) {
                            float * dhf_dst_i = (float *) ((char *) dst->data + (j*ne0 + i01_low)*sizeof(float) + i02*nb2 + i03*nb3);
                            CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_ddf_i + j*i01_diff, i01_diff*sizeof(float),
                                                    cudaMemcpyDeviceToHost, cudaStream_main));
                        }
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_ddf_i, dst_stride*sizeof(float), cudaMemcpyDeviceToHost, cudaStream_main));
                    }
                }
            }
        }
    }

    // wait until each device is finished, then free their buffers
    for (int id = 0; id < g_device_count; ++id) {
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (src0_asq[id] > 0) {
            ggml_cuda_pool_free(src0_ddq[id], src0_asq[id]);
        }
        if (src0_asf[id] > 0) {
            ggml_cuda_pool_free(src0_ddf[id], src0_asf[id]);
        }
        if (src1_asf[id] > 0) {
            ggml_cuda_pool_free(src1_ddf[id], src1_asf[id]);
        }
        if (dst_asf[id] > 0) {
            ggml_cuda_pool_free(dst_ddf[id], dst_asf[id]);
        }
    }
}

bool ggml_cuda_can_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    GGML_ASSERT(src1->backend != GGML_BACKEND_GPU_SPLIT);
    (void) src0;
    (void) dst;
    return src1->backend == GGML_BACKEND_GPU;
}

void ggml_cuda_mul(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul, true);
}

bool ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    GGML_ASSERT(src0->backend != GGML_BACKEND_GPU);
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        ((ne0 >= 32 && ne1 >= 32 && ne10 >= 32) || src0->backend == GGML_BACKEND_GPU_SPLIT)) {
        return true;
    }

    return false;
}

bool ggml_cuda_mul_mat_use_f16(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * /* dst */) {
    size_t src0_sz = ggml_nbytes(src0);
    size_t src1_sz = ggml_nbytes(src1);

    // mul_mat_q: src0 is converted to fp32 on device
    size_t mul_mat_q_transfer = src0_sz + src1_sz;

    // mul_mat_f16: src1 is converted to fp16 on cpu
    size_t mul_mat_f16_transfer = src0_sz + sizeof(half) * ggml_nelements(src1);

    // choose the smaller one to transfer to the device
    // TODO: this is not always the best choice due to the overhead of converting to fp16
    return mul_mat_f16_transfer < mul_mat_q_transfer;
}

size_t ggml_cuda_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    if (ggml_cuda_mul_mat_use_f16(src0, src1, dst)) {
        return ggml_nelements(src1) * sizeof(ggml_fp16_t);
    }
    else {
        return 0;
    }
}

void ggml_cuda_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_can_mul_mat(src0, src1, dst));

    if (src0->type == GGML_TYPE_F32) {
        ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul_mat_cublas, true);
    } else if (ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F16) {
        if (src1->ne[1] == 1) {
            ggml_cuda_op(src0, src1, dst, ggml_cuda_op_dequantize_mul_mat_vec, false);
        } else {
            ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul_mat_cublas, true);
        }
    } else {
        GGML_ASSERT(false);
    }
}

static void ggml_cuda_mul_mat_f16(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, void * wdata, size_t /* wsize */) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;
    const int n_mm = ne03 * ne02;

    size_t x_size, y_size, d_size;
    half  * d_X =  (half *) ggml_cuda_pool_malloc(n_mm * sizeof(half) * x_ne, &x_size);
    half  * d_Y =  (half *) ggml_cuda_pool_malloc(n_mm * sizeof(half) * y_ne, &y_size);
    float * d_D = (float *) ggml_cuda_pool_malloc(n_mm * sizeof(float) * d_ne, &d_size);

    bool src1_cont_rows = nb10 == sizeof(float);
    bool src1_cont_cols = (size_t)nb11 == ne11*sizeof(float);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            int i = i03*ne02 + i02;
            cudaStream_t cudaStream = g_cudaStreams_main[0][i % GGML_CUDA_MAX_STREAMS];

            half  * c_X = d_X + i * x_ne;
            half  * c_Y = d_Y + i * y_ne;
            float * c_D = d_D + i * d_ne;

            // copy src0 to device
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(c_X, src0, i03, i02, 0, ne01, cudaStream));

            // convert src1 to fp16
            // TODO: use multiple threads
            ggml_fp16_t * const tmp = (ggml_fp16_t *) wdata + (ne11 * ne10) * (i03 * ne02 + i02);
            char * src1i = (char *) src1->data + i03*nb13 + i02*nb12;
            if (src1_cont_rows) {
                if (src1_cont_cols) {
                    ggml_fp32_to_fp16_row((float *) src1i, tmp, ne10*ne11);
                }
                else {
                    for (int64_t i01 = 0; i01 < ne11; i01++) {
                        ggml_fp32_to_fp16_row((float *) (src1i + i01*nb11), tmp + i01*ne10, ne10);
                    }
                }
            }
            else {
                for (int64_t i01 = 0; i01 < ne11; i01++) {
                    for (int64_t i00 = 0; i00 < ne10; i00++) {
                        // very slow due to no inlining
                        tmp[i01*ne10 + i00] = ggml_fp32_to_fp16(*(float *) (src1i + i01*nb11 + i00*nb10));
                    }
                }
            }

            // copy src1 to device
            CUDA_CHECK(cudaMemcpyAsync(c_Y, tmp, sizeof(half) * y_ne, cudaMemcpyHostToDevice, cudaStream));

            // compute
            CUBLAS_CHECK(cublasSetStream(g_cublas_handles[0], cudaStream));
            CUBLAS_CHECK(
                cublasGemmEx(g_cublas_handles[0], CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        &alpha, c_X, CUDA_R_16F, ne00,
                                c_Y, CUDA_R_16F, ne10,
                        &beta,  c_D, CUDA_R_32F, ne01,
                        CUBLAS_COMPUTE_32F_FAST_16F,
                        CUBLAS_GEMM_DEFAULT));

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            CUDA_CHECK(cudaMemcpyAsync(d, c_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, cudaStream));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    ggml_cuda_pool_free(d_X, x_size);
    ggml_cuda_pool_free(d_Y, y_size);
    ggml_cuda_pool_free(d_D, d_size);
}

void ggml_cuda_load_data(const char * fname, struct ggml_tensor * tensor, const size_t offset, int n_layer) {
    FILE * fp = fopen(fname, "rb");
    int nrows = ggml_nrows(tensor);
    const size_t nb1 = tensor->nb[1];
    ggml_backend backend = tensor->backend;
    struct ggml_tensor_extra_gpu * extra = (struct ggml_tensor_extra_gpu *) tensor->extra;

    for (int id = 0; id < g_device_count; ++id) {
        extra->data_device[id] = nullptr;

        int layer_low = id == 0 ? 0 : n_layer*g_tensor_split[id];
        int layer_high = id == g_device_count - 1 ? n_layer : n_layer*g_tensor_split[id + 1];
        if (backend == GGML_BACKEND_GPU && (extra->layer < layer_low || extra->layer >= layer_high)) {
            continue;
        }

        cudaSetDevice(id);

        int row_low, row_high;
        if (backend == GGML_BACKEND_GPU) {
            extra->i_device = id;

            row_low = 0;
            row_high = nrows;
        } else if (backend == GGML_BACKEND_GPU_SPLIT) {
            extra->i_device = -1;

            row_low = id == 0 ? 0 : nrows*g_tensor_split[id];
            row_low -= row_low % GGML_CUDA_DMMV_Y;
            row_high = id == g_device_count - 1 ? nrows : nrows*g_tensor_split[id + 1];
            row_high -= row_high % GGML_CUDA_DMMV_Y;
        } else {
            GGML_ASSERT(false);
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t nrows_split = row_high - row_low;

        const size_t offset_split = offset + row_low*nb1;
        const size_t size = ggml_nbytes_split(tensor, nrows_split);

        void * buf;
        CUDA_CHECK(cudaMalloc(&buf, size));
        void * buf_host = malloc(size);

#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset_split, SEEK_SET);
#else
        int ret = fseek(fp, (long) offset_split, SEEK_SET);
#endif
        GGML_ASSERT(ret == 0); // same

        size_t ret2 = fread(buf_host, size, 1, fp);
        if (ret2 != 1) {
            fprintf(stderr, "unexpectedly reached end of file");
            exit(1);
        }

        cudaMemcpy(buf, buf_host, size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        free(buf_host);
        extra->data_device[id] = buf;
    }

    tensor->extra = extra;
    fclose(fp);
}

void ggml_cuda_free_data(struct ggml_tensor * tensor) {
    if (tensor->backend != GGML_BACKEND_GPU && tensor->backend != GGML_BACKEND_GPU_SPLIT) {
        return;
    }

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int id = 0; id < g_device_count; ++id) {
        if (extra->data_device[id] == nullptr) {
            continue;
        }

        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaFree(extra->data_device[id]));
    }

    delete extra;
}

bool ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor){
    ggml_cuda_func_t func;

    switch (tensor->op) {
        case GGML_OP_MUL:
            if (!ggml_cuda_can_mul(tensor->src0, tensor->src1, tensor)) {
                return false;
            }
            func = ggml_cuda_mul;
            break;
        case GGML_OP_MUL_MAT:
            if (!ggml_cuda_can_mul_mat(tensor->src0, tensor->src1, tensor)) {
                return false;
            }

            // For prompt processing the multi GPU code is currently slower than the single GPU code that existed before.
            // To avoid a performance regression the old code is kept for now:
            if (g_device_count == 1 && tensor->src0->type == GGML_TYPE_F16 &&
                ggml_cuda_mul_mat_use_f16(tensor->src0, tensor->src1, tensor)) {

                if (params->ith != 0) {
                    return true;
                }
                if (params->type == GGML_TASK_COMPUTE) {
                    ggml_cuda_mul_mat_f16(tensor->src0, tensor->src1, tensor, params->wdata, params->wsize);
                    return true;
                }

                return false;
            }
            func = ggml_cuda_mul_mat;
            break;
        default:
            return false;
    }

    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }
    func(tensor->src0, tensor->src1, tensor);
    return true;
}
