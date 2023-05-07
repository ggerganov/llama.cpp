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
    uint8_t qh[4];          // 5-th bit of quants
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

    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));

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

// TODO: optimize
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

#define GGML_CUDA_MAX_STREAMS 8
#define GGML_CUDA_MAX_EVENTS 64
static cublasHandle_t g_cublasH = nullptr;
static cudaStream_t g_cudaStreams[GGML_CUDA_MAX_STREAMS] = { nullptr };
static cudaStream_t g_cudaStreams2[GGML_CUDA_MAX_STREAMS] = { nullptr };
static cudaEvent_t g_cudaEvents[GGML_CUDA_MAX_EVENTS] = { nullptr };

void ggml_init_cublas() {
    if (g_cublasH == nullptr) {
        // create streams
        for (int i = 0; i < GGML_CUDA_MAX_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams[i], cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams2[i], cudaStreamNonBlocking));
        }
        // create events
        for (int i = 0; i < GGML_CUDA_MAX_EVENTS; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&g_cudaEvents[i], cudaEventDisableTiming));
        }

        // create cublas handle
        CUBLAS_CHECK(cublasCreate(&g_cublasH));
        CUBLAS_CHECK(cublasSetMathMode(g_cublasH, CUBLAS_TF32_TENSOR_OP_MATH));

        // configure logging to stdout
        // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));
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
    const int n_mm = ne03 * ne02;

    size_t x_size, y_size, d_size;
    float * d_X = (float *) ggml_cuda_pool_malloc(n_mm * sizeof(float) * x_ne, &x_size);
    float * d_Y = (float *) ggml_cuda_pool_malloc(n_mm * sizeof(float) * y_ne, &y_size);
    float * d_D = (float *) ggml_cuda_pool_malloc(n_mm * sizeof(float) * d_ne, &d_size);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            int i = i03*ne02 + i02;
            cudaStream_t cudaStream = g_cudaStreams[i % GGML_CUDA_MAX_STREAMS];

            float * c_X = d_X + i * x_ne;
            float * c_Y = d_Y + i * y_ne;
            float * c_D = d_D + i * d_ne;

            // copy data to device
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(c_X, src0, i03, i02, cudaStream));
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(c_Y, src1, i03, i02, cudaStream));

            // compute
            CUBLAS_CHECK(cublasSetStream(g_cublasH, cudaStream));
            CUBLAS_CHECK(
                cublasSgemm(g_cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        &alpha, c_X, ne00,
                                c_Y, ne10,
                        &beta,  c_D, ne01));

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
            cudaStream_t cudaStream = g_cudaStreams[i % GGML_CUDA_MAX_STREAMS];

            half  * c_X = d_X + i * x_ne;
            half  * c_Y = d_Y + i * y_ne;
            float * c_D = d_D + i * d_ne;

            // copy src0 to device
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(c_X, src0, i03, i02, cudaStream));

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
            CUBLAS_CHECK(cublasSetStream(g_cublasH, cudaStream));
            CUBLAS_CHECK(
                cublasGemmEx(g_cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
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

static void ggml_cuda_mul_mat_q_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
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
    const int n_mm = ne03 * ne02;
    const size_t q_sz = ggml_type_size(type) * x_ne / ggml_blck_size(type);

    size_t x_size, y_size, d_size, q_size;
    float * d_X = (float *) ggml_cuda_pool_malloc(n_mm * sizeof(float) * x_ne, &x_size);
    float * d_Y = (float *) ggml_cuda_pool_malloc(n_mm * sizeof(float) * y_ne, &y_size);
    float * d_D = (float *) ggml_cuda_pool_malloc(n_mm * sizeof(float) * d_ne, &d_size);
    char  * d_Q = (char  *) ggml_cuda_pool_malloc(n_mm * q_sz, &q_size);

    const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(type);
    GGML_ASSERT(to_fp32_cuda != nullptr);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            int i = i03*ne02 + i02;
            cudaStream_t cudaStream = g_cudaStreams[i % GGML_CUDA_MAX_STREAMS];
            cudaStream_t cudaStream2 = g_cudaStreams2[i % GGML_CUDA_MAX_STREAMS];
            cudaEvent_t  cudaEvent = g_cudaEvents[i % GGML_CUDA_MAX_EVENTS];

            float * c_X = d_X + i * x_ne;
            float * c_Y = d_Y + i * y_ne;
            float * c_D = d_D + i * d_ne;
            char  * c_Q = d_Q + i * q_sz;

            // copy src0 and convert to fp32 on device
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(c_Q, src0, i03, i02, cudaStream2));
            to_fp32_cuda(c_Q, c_X, x_ne, cudaStream2);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(cudaEvent, cudaStream2));

            // copy src1 to device
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(c_Y, src1, i03, i02, cudaStream));

            // wait for conversion
            CUDA_CHECK(cudaStreamWaitEvent(cudaStream, cudaEvent, 0));

            // compute
            CUBLAS_CHECK(cublasSetStream(g_cublasH, cudaStream));
            CUBLAS_CHECK(
                cublasSgemm(g_cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        &alpha, c_X, ne00,
                                c_Y, ne10,
                        &beta,  c_D, ne01));

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            CUDA_CHECK(cudaMemcpyAsync(d, c_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, cudaStream));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
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

void ggml_cuda_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, void * wdata, size_t wsize) {
    GGML_ASSERT(ggml_cuda_can_mul_mat(src0, src1, dst));

    if (src0->type == GGML_TYPE_F32) {
        ggml_cuda_mul_mat_f32(src0, src1, dst);
    }
    else if (src0->type == GGML_TYPE_F16) {
        if (ggml_cuda_mul_mat_use_f16(src0, src1, dst)) {
            ggml_cuda_mul_mat_f16(src0, src1, dst, wdata, wsize);
        }
        else {
            ggml_cuda_mul_mat_q_f32(src0, src1, dst);
        }
    }
    else if (ggml_is_quantized(src0->type)) {
        ggml_cuda_mul_mat_q_f32(src0, src1, dst);
    }
    else {
        GGML_ASSERT(false);
    }
}

size_t ggml_cuda_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    if (ggml_cuda_mul_mat_use_f16(src0, src1, dst)) {
        return ggml_nelements(src1) * sizeof(ggml_fp16_t);
    }
    else {
        return 0;
    }
}
