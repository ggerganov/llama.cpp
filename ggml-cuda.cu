#include <stdint.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <atomic>
#include "ggml-cuda.h"

typedef uint16_t ggml_fp16_t;
static_assert(sizeof(__half) == sizeof(ggml_fp16_t), "wrong fp16 size");

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
    __half  d;              // delta
    uint8_t qs[QK4_2 / 2];  // nibbles / quants
} block_q4_2;
static_assert(sizeof(block_q4_2) == sizeof(ggml_fp16_t) + QK4_2 / 2, "wrong q4_2 block size/padding");

#define QK4_3 16
typedef struct {
    __half  d;              // delta
    __half  m;              // min
    uint8_t qs[QK4_3 / 2];  // nibbles / quants
} block_q4_3;
static_assert(sizeof(block_q4_3) == 2 * sizeof(ggml_fp16_t) + QK4_3 / 2, "wrong q4_3 block size/padding");

#define QK5_0 32
typedef struct {
    __half d;               // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
    __half d;               // delta
    __half m;               // min
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

static __global__ void dequantize_block_q4_3(const void * vx, float * y) {
    const block_q4_3 * x = (const block_q4_3 *) vx;

    const int i = blockIdx.x;

    const float d = x[i].d;
    const float m = x[i].m;

    const uint8_t * pp = x[i].qs;

    for (int l = 0; l < QK4_3; l += 2) {
        const uint8_t vi = pp[l/2];

        const int8_t vi0 = vi & 0xf;
        const int8_t vi1 = vi >> 4;

        const float v0 = vi0*d + m;
        const float v1 = vi1*d + m;

        y[i*QK4_3 + l + 0] = v0;
        y[i*QK4_3 + l + 1] = v1;
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

void dequantize_row_q4_0_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK4_0;
    dequantize_block_q4_0<<<nb, 1, 0, stream>>>(vx, y);
}

void dequantize_row_q4_1_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK4_1;
    dequantize_block_q4_1<<<nb, 1, 0, stream>>>(vx, y);
}

void dequantize_row_q4_2_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK4_2;
    dequantize_block_q4_2<<<nb, 1, 0, stream>>>(vx, y);
}

void dequantize_row_q4_3_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK4_3;
    dequantize_block_q4_3<<<nb, 1, 0, stream>>>(vx, y);
}

void dequantize_row_q5_0_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK5_0;
    dequantize_block_q5_0<<<nb, 1, 0, stream>>>(vx, y);
}

void dequantize_row_q5_1_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK5_1;
    dequantize_block_q5_1<<<nb, 1, 0, stream>>>(vx, y);
}

void dequantize_row_q8_0_cuda(const void * vx, float * y, int k, cudaStream_t stream) {
    const int nb = k / QK8_0;
    dequantize_block_q8_0<<<nb, 1, 0, stream>>>(vx, y);
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

void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size) {
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

void ggml_cuda_pool_free(void * ptr, size_t size) {
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

cublasHandle_t g_cublasH = NULL;
cudaStream_t g_cudaStream = NULL;

void ggml_init_cublas(void) {
    if (g_cublasH == NULL) {
        // create cublas handle, bind a stream
        CUBLAS_CHECK(cublasCreate(&g_cublasH));

        CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStream, cudaStreamNonBlocking));

        CUBLAS_CHECK(cublasSetStream(g_cublasH, g_cudaStream));

        // configure logging to stdout
        // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, NULL));
    }
}
