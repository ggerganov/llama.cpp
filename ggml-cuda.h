#if defined(GGML_USE_HIPBLAS)
#include "hipblas/hipblas.h"
#include "hip/hip_runtime.h"
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasCreate hipblasCreate
#define cublasGemmEx hipblasGemmEx
#define cublasHandle_t hipblasHandle_t
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasStatus_t hipblasStatus_t
#define CUDA_R_16F  HIPBLAS_R_16F
#define CUDA_R_32F  HIPBLAS_R_32F
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaStream_t hipStream_t
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaSuccess hipSuccess
#define GGML_USE_CUBLAS
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

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
void dequantize_row_q8_0_cuda(const void * vx, float * y, int k, cudaStream_t stream);

#ifdef  __cplusplus
}
#endif
