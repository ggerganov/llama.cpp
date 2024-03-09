#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

// Always success. To check if CUDA is actually loaded, use `ggml_cublas_loaded`.
GGML_API GGML_CALL void   ggml_init_cublas(void);

// Returns `true` if there are available CUDA devices and cublas loads successfully; otherwise, it returns `false`.
GGML_API GGML_CALL bool   ggml_cublas_loaded(void);

GGML_API GGML_CALL void * ggml_cuda_host_malloc(size_t size);
GGML_API GGML_CALL void   ggml_cuda_host_free(void * ptr);

GGML_API GGML_CALL bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
GGML_API GGML_CALL bool   ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

GGML_API GGML_CALL int    ggml_cuda_get_device_count(void);
GGML_API GGML_CALL void   ggml_cuda_get_device_description(int device, char * description, size_t description_size);

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device);

GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);
// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

#ifdef  __cplusplus
}
#endif
