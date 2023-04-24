#include <stdint.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <atomic>
#include "ggml-cuda.h"

typedef uint16_t ggml_fp16_t;
static_assert(sizeof(__half) == sizeof(ggml_fp16_t), "wrong fp16 size");
 #define CUDA_MEM_DEBUG 0

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

#define MAX_CUDA_BUFFERS 512 // number of allocations the pool can hold
static const uint64_t MAX_CUDA_POOL_SIZE = static_cast<uint64_t>(1024) * 1024 * 1024 * 4;  // max memory to allocate for cuda buffers
static cuda_buffer g_cuda_buffer_pool[MAX_CUDA_BUFFERS];
static std::atomic_flag g_cuda_pool_lock = ATOMIC_FLAG_INIT;

static int g_cuda_free_buffer_indices[MAX_CUDA_BUFFERS]; // sorted list of free already allocated indices
static int g_cuda_free_buffer_count = 0;
static size_t g_cuda_pool_total_allocated = 0;

void cuda_pool_dump() {
    printf("========================================\n");
    printf("| Current CUDA Buffer Pool             |\n");
    printf("========================================\n");
    printf("| %-6s | %-12s | %-10s |\n", "Index", "Buffer Index", "Size (bytes)");
    printf("----------------------------------------\n");
    for (int i = 0; i < g_cuda_free_buffer_count; ++i) {
        int buffer_index = g_cuda_free_buffer_indices[i];
        printf("| %-6d | %-12d | %-10zu |\n",
               i, buffer_index, g_cuda_buffer_pool[buffer_index].size);
    }
    printf("========================================\n");
}

void cuda_pool_insert_idx(int index) {
    int left = 0;
    int right = g_cuda_free_buffer_count - 1;

    while (left <= right) {
        int mid = (left + right) / 2;
        if (g_cuda_buffer_pool[g_cuda_free_buffer_indices[mid]].size < g_cuda_buffer_pool[index].size) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    // Move elements to the right to make space for the new index
    for (int i = g_cuda_free_buffer_count; i > left; --i) {
        g_cuda_free_buffer_indices[i] = g_cuda_free_buffer_indices[i - 1];
    }

    g_cuda_free_buffer_indices[left] = index;
    g_cuda_free_buffer_count++;
    #if CUDA_MEM_DEBUG
    printf("INFO:: Inserted buffer at index %d with size %lld bytes\n", index, g_cuda_buffer_pool[index].size);
    cuda_pool_dump();
    #endif
}
// find the best fitting block in the pool and remove it from the indexed list
int cuda_pool_get_block(size_t size, size_t * actual_size) {
    int left = 0;
    int right = g_cuda_free_buffer_count - 1;
    int index = -1;

    while (left <= right) {
        int mid = (left + right) / 2;
        if (g_cuda_buffer_pool[g_cuda_free_buffer_indices[mid]].size >= size) {
            index = mid;
            if (mid > 0 && g_cuda_buffer_pool[g_cuda_free_buffer_indices[mid - 1]].size >= size) {
                right = mid - 1; // continue searching to the left for smaller fitting buffers
            } else {
                break; // found the smallest fitting buffer
            }
        } else {
            left = mid + 1;
        }
    }

    if (index != -1) {
        int buffer_index = g_cuda_free_buffer_indices[index];
        *actual_size = g_cuda_buffer_pool[buffer_index].size;

        // Remove the used index from the sorted array
        for (int i = index; i < g_cuda_free_buffer_count - 1; ++i) {
            g_cuda_free_buffer_indices[i] = g_cuda_free_buffer_indices[i + 1];
        }
        g_cuda_free_buffer_count--;
        #if CUDA_MEM_DEBUG
        printf("INFO:: Found buffer of size %lld bytes at index %d\n", *actual_size, buffer_index);
        cuda_pool_dump();
        #endif
        return buffer_index;
    }
    #if CUDA_MEM_DEBUG
    printf("INFO:: No buffer found for size %lld bytes\n", size);
    cuda_pool_dump();
    #endif
    return -1;
}


// mark all buffers as free
void ggml_cuda_pool_initialize() {
    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        g_cuda_buffer_pool[i].ptr = nullptr;
        g_cuda_buffer_pool[i].size = 0;
    }
    g_cuda_free_buffer_count = 0;
    g_cuda_pool_total_allocated = 0;
}
// Uses the existing pool of buffers to allocate memory efficienty or allocates a new buffer. Returns pointer
void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_cuda_pool_lock);

    int buffer_index = cuda_pool_get_block(size, actual_size);

    if (buffer_index != -1) {
        void * ptr = g_cuda_buffer_pool[buffer_index].ptr;
        g_cuda_buffer_pool[buffer_index].ptr = nullptr;
        g_cuda_buffer_pool[buffer_index].size = 0;
        g_cuda_pool_total_allocated -= *actual_size; 
        #if CUDA_MEM_DEBUG
        printf("INFO:: Allocated %lld bytes from buffer at index %d (total allocated: %lld bytes)\n", *actual_size, buffer_index, g_cuda_pool_total_allocated);
        cuda_pool_dump();
        #endif
        return ptr;
    }
    if (g_cuda_pool_total_allocated + size > MAX_CUDA_POOL_SIZE) {
        fprintf(stderr, "WARNING: CUDA pool is full, consider inceasing MAX_CUDA_POOL_SIZE. Trying to allocate anyway..\n");
    }
    void * ptr;
    CUDA_CHECK(cudaMalloc((void **) &ptr, size));
    *actual_size = size;
    g_cuda_pool_total_allocated += size;
    #if CUDA_MEM_DEBUG
    printf("INFO:: Allocated %lld bytes from cudaMalloc (total allocated: %lld bytes)\n", *actual_size, g_cuda_pool_total_allocated);
    cuda_pool_dump();
    #endif
    return ptr;
}

void ggml_cuda_pool_free(void * ptr, size_t size) {
    scoped_spin_lock lock(g_cuda_pool_lock);

    if (g_cuda_free_buffer_count < MAX_CUDA_BUFFERS) {
        int buffer_index = -1;
        for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
            if (g_cuda_buffer_pool[i].ptr == nullptr) {
                buffer_index = i;
                break;
            }
        }

        if (buffer_index != -1) {
            g_cuda_buffer_pool[buffer_index].ptr = ptr;
            g_cuda_buffer_pool[buffer_index].size = size;
            cuda_pool_insert_idx
        (buffer_index);
            g_cuda_pool_total_allocated += size; 
        }
    } else {
        fprintf(stderr, "WARNING: cuda buffer pool full, consider increasing MAX_CUDA_BUFFERS\n");
        CUDA_CHECK(cudaFree(ptr));
    }
}

cublasHandle_t g_cublasH = NULL;
cudaStream_t g_cudaStream = NULL;

void ggml_init_cublas(void) {
    if (g_cublasH == NULL) {
        ggml_cuda_pool_initialize();
        // create cublas handle, bind a stream
        CUBLAS_CHECK(cublasCreate(&g_cublasH));

        CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStream, cudaStreamNonBlocking));

        CUBLAS_CHECK(cublasSetStream(g_cublasH, g_cudaStream));

        // configure logging to stdout
        // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, NULL));
    }
}
