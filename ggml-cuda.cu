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

#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);    \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

typedef void (*to_fp32_cuda_t)(const void * x, float * y, int k, cudaStream_t stream);

#define QK4_0 32
typedef struct {
    float   d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(float) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    float   d;              // delta
    float   m;              // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(float) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK4_2 16
typedef struct {
    half  d;                // delta
    uint8_t qs[QK4_2 / 2];  // nibbles / quants
} block_q4_2;
static_assert(sizeof(block_q4_2) == sizeof(ggml_fp16_t) + QK4_2 / 2, "wrong q4_2 block size/padding");

#define QK5_0 32
typedef struct {
    half d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
    half d;                 // delta
    half m;                 // min
    uint32_t qh;            // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
    float   d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(float) + QK8_0, "wrong q8_0 block size/padding");

static __global__ void dequantize_block_q4_0(const void * vx, float * y) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int i = blockIdx.x;

    const float d = x[i].d;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_0; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = (vi0 - 8)*d;
        const float v1 = (vi1 - 8)*d;

        y[i*QK4_0 + l + 0] = v0;
        y[i*QK4_0 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q4_1(const void * vx, float * y) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int i = blockIdx.x;

    const float d = x[i].d;
    const float m = x[i].m;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_1; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = vi0*d + m;
        const float v1 = vi1*d + m;

        y[i*QK4_1 + l + 0] = v0;
        y[i*QK4_1 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q4_2(const void * vx, float * y) {
    const block_q4_2 * x = (const block_q4_2 *) vx;

    const int i = blockIdx.x;

    const float d = x[i].d;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_2; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = (vi0 - 8)*d;
        const float v1 = (vi1 - 8)*d;

        y[i*QK4_2 + l + 0] = v0;
        y[i*QK4_2 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q5_0(const void * vx, float * y) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int i = blockIdx.x;

    const float d = x[i].d;

    const uint8_t * pp = x[i].qs;

    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

    for (int l = 0; l < QK5_0; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vh0 = ((qh & (1 << (l + 0))) >> (l + 0)) << 4;
        const int8_t vh1 = ((qh & (1 << (l + 1))) >> (l + 1)) << 4;

        const int8_t vi0 = ((vi & 0xf) | vh0);
        const int8_t vi1 = ((vi >>  4) | vh1);

        const float v0 = (vi0 - 16)*d;
        const float v1 = (vi1 - 16)*d;

        y[i*QK5_0 + l + 0] = v0;
        y[i*QK5_0 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q5_1(const void * vx, float * y) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int i = blockIdx.x;

    const float d = x[i].d;
    const float m = x[i].m;

    const uint8_t * pp = x[i].qs;

    const uint32_t qh = x[i].qh;

    for (int l = 0; l < QK5_1; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vh0 = ((qh & (1 << (l + 0))) >> (l + 0)) << 4;
        const int8_t vh1 = ((qh & (1 << (l + 1))) >> (l + 1)) << 4;

        const int8_t vi0 = (vi & 0xf) | vh0;
        const int8_t vi1 = (vi >>  4) | vh1;

        const float v0 = vi0*d + m;
        const float v1 = vi1*d + m;

        y[i*QK5_1 + l + 0] = v0;
        y[i*QK5_1 + l + 1] = v1;
    }
}

static __global__ void dequantize_block_q8_0(const void * vx, float * y) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int i = blockIdx.x;

    const float d = x[i].d;

    const int8_t * pp = x[i].qs;

    for (int l = 0; l < QK8_0; l++) {
        const int8_t vi = pp[l];

        y[i*QK8_0 + l] = vi*d;
    }
}

static void dequantize_row_q4_0_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK4_0;
    dequantize_block_q4_0<<<nb, 1, 0, stream>>>(vx, y);
}

static void dequantize_row_q4_1_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK4_1;
    dequantize_block_q4_1<<<nb, 1, 0, stream>>>(vx, y);
}

static void dequantize_row_q4_2_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK4_2;
    dequantize_block_q4_2<<<nb, 1, 0, stream>>>(vx, y);
}

static void dequantize_row_q5_0_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK5_0;
    dequantize_block_q5_0<<<nb, 1, 0, stream>>>(vx, y);
}

static void dequantize_row_q5_1_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK5_1;
    dequantize_block_q5_1<<<nb, 1, 0, stream>>>(vx, y);
}

static void dequantize_row_q8_0_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK8_0;
    dequantize_block_q8_0<<<nb, 1, 0, stream>>>(vx, y);
}

static __global__ void convert_fp16_to_fp32(const void * vx, float * y) {
    const half * x = (const half *) vx;

    const int i = blockIdx.x;

    y[i] = __half2float(x[i]);
}

static void convert_fp16_to_fp32_cuda(const void * x, float * y, int k, cudaStream_t stream) {
    convert_fp16_to_fp32<<<k, 1, 0, stream>>>(x, y);
}

static to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q4_2:
            return dequantize_row_q4_2_cuda;
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
#define MAX_CUDA_BUFFERS 16

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

static cuda_buffer g_cuda_buffer_pool[MAX_CUDA_BUFFERS];
static std::atomic_flag g_cuda_pool_lock = ATOMIC_FLAG_INIT;

static void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_cuda_pool_lock);

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[i];
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

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
    CUDA_CHECK(cudaFree(ptr));
}

static cublasHandle_t g_cublasH = nullptr;
static cudaStream_t g_cudaStream = nullptr;
static cudaStream_t g_cudaStream2 = nullptr;
static cudaEvent_t g_cudaEvent = nullptr;

void ggml_init_cublas() {
    if (g_cublasH == nullptr) {
        // create cublas handle, bind a stream
        CUBLAS_CHECK(cublasCreate(&g_cublasH));
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(g_cublasH, g_cudaStream));
        // enable tensor cores
        CUBLAS_CHECK(cublasSetMathMode(g_cublasH, CUBLAS_TENSOR_OP_MATH));

        // create additional stream and event for synchronization
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStream2, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&g_cudaEvent, cudaEventDisableTiming));

        // configure logging to stdout
        // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, NULL));
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

static cudaError_t ggml_cuda_h2d_tensor_2d(void * dst, const struct ggml_tensor * src, uint64_t i3, uint64_t i2, cudaStream_t stream) {
    const uint64_t ne0 = src->ne[0];
    const uint64_t ne1 = src->ne[1];
    const uint64_t nb0 = src->nb[0];
    const uint64_t nb1 = src->nb[1];
    const uint64_t nb2 = src->nb[2];
    const uint64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const size_t ts = ggml_type_size(type);
    const size_t bs = ggml_blck_size(type);

    const void * x = (const void *) ((const char *) src->data + i2*nb2 + i3*nb3);
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst, x, ne1*nb1, cudaMemcpyHostToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst, ts*ne0/bs, x, nb1, ts*ne0/bs, ne1, cudaMemcpyHostToDevice, stream);
    } else {
        for (uint64_t i1 = 0; i1 < ne1; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) ((char *) dst + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyHostToDevice, stream);
            if (r != cudaSuccess) return r;
        }
        return cudaSuccess;
    }
}

static void ggml_cuda_mul_mat_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;

    size_t x_size, y_size, d_size;
    float * d_X = (float *) ggml_cuda_pool_malloc(sizeof(float) * x_ne, &x_size);
    float * d_Y = (float *) ggml_cuda_pool_malloc(sizeof(float) * y_ne, &y_size);
    float * d_D = (float *) ggml_cuda_pool_malloc(sizeof(float) * d_ne, &d_size);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            // copy data to device
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(d_X, src0, i03, i02, g_cudaStream));
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(d_Y, src1, i03, i02, g_cudaStream));

            // compute
            CUBLAS_CHECK(
                cublasSgemm(g_cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        &alpha, d_X, ne00,
                                d_Y, ne10,
                        &beta,  d_D, ne01));

            // copy data to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            CUDA_CHECK(cudaMemcpyAsync(d, d_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, g_cudaStream));
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(g_cudaStream));
    ggml_cuda_pool_free(d_X, x_size);
    ggml_cuda_pool_free(d_Y, y_size);
    ggml_cuda_pool_free(d_D, d_size);
}

static void ggml_cuda_mul_mat_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];
    const ggml_type type = src0->type;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;

    size_t x_size, y_size, d_size, q_size;
    float * d_X = (float *) ggml_cuda_pool_malloc(sizeof(float) * x_ne, &x_size);
    float * d_Y = (float *) ggml_cuda_pool_malloc(sizeof(float) * y_ne, &y_size);
    float * d_D = (float *) ggml_cuda_pool_malloc(sizeof(float) * d_ne, &d_size);
    void  * d_Q = (void  *) ggml_cuda_pool_malloc(ggml_type_size(type) * x_ne / ggml_blck_size(type), &q_size);

    const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(type);
    GGML_ASSERT(to_fp32_cuda != NULL);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            // copy and convert to fp32 on device
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(d_Q, src0, i03, i02, g_cudaStream2));

            to_fp32_cuda(d_Q, d_X, x_ne, g_cudaStream2);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(g_cudaEvent, g_cudaStream2));

            // copy data to device
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(d_Y, src1, i03, i02, g_cudaStream));

            // wait for conversion
            CUDA_CHECK(cudaStreamWaitEvent(g_cudaStream, g_cudaEvent, 0));

            // compute
            CUBLAS_CHECK(
                cublasSgemm(g_cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        &alpha, d_X, ne00,
                                d_Y, ne10,
                        &beta,  d_D, ne01));

            // copy data to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            CUDA_CHECK(cudaMemcpyAsync(d, d_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, g_cudaStream));
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(g_cudaStream));
    ggml_cuda_pool_free(d_X, x_size);
    ggml_cuda_pool_free(d_Y, y_size);
    ggml_cuda_pool_free(d_D, d_size);
    ggml_cuda_pool_free(d_Q, q_size);
}

bool ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        (ne0 >= 32 && ne1 >= 32 && ne10 >= 32)) {

        return true;
    }

    return false;
}

void ggml_cuda_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_can_mul_mat(src0, src1, dst));

    const ggml_type type = src0->type;

    if (type == GGML_TYPE_F32) {
        ggml_cuda_mul_mat_f32(src0, src1, dst);
    }
    else if (type == GGML_TYPE_F16 || ggml_is_quantized(type)) {
        ggml_cuda_mul_mat_q(src0, src1, dst);
    }
    else {
        GGML_ASSERT(false);
    }
}
