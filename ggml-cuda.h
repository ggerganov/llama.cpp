#include <cublas_v2.h>
#include <cuda_runtime.h>

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
extern cudaStream_t   g_cudaStream;

void   ggml_init_cublas(void);
void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size);
void   ggml_cuda_pool_free(void * ptr, size_t size);

void dequantize_row_q4_0_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q4_1_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q4_2_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q4_3_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q5_0_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q5_1_cuda(const void * vx, float * y, int k, cudaStream_t stream);
void dequantize_row_q8_0_cuda(const void * vx, float * y, int k, cudaStream_t stream);

#ifdef  __cplusplus
}
#endif
