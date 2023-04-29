#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

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

extern cublasHandle_t g_cublasH;
extern cudaStream_t g_cudaStream;
extern cudaStream_t g_cudaStream2;
extern cudaEvent_t g_cudaEvent;

void   ggml_init_cublas(void);
void * ggml_cuda_host_malloc(size_t size);
void   ggml_cuda_host_free(void * ptr);

void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size);
void   ggml_cuda_pool_free(void * ptr, size_t size);

void dequantize_row_q4_0_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q4_1_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q4_2_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q5_0_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q5_1_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q8_0_cuda(const void * vx, float * y, int k, cudaStream_t stream);

cudaError_t ggml_cuda_h2d_tensor_2d(void * dst, const struct ggml_tensor * src, uint64_t i3, uint64_t i2, cudaStream_t stream);

typedef void (*dequantize_row_q_cuda_t)(const void * x, float * y, int k, cudaStream_t stream);
dequantize_row_q_cuda_t ggml_get_dequantize_row_q_cuda(enum ggml_type type);

#ifdef  __cplusplus
}
#endif
