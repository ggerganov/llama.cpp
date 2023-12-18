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
GGML_API void   ggml_init_cublas(void);

// Returns `true` if there are available CUDA devices and cublas loads successfully; otherwise, it returns `false`.
GGML_API bool   ggml_cublas_loaded(void);

GGML_API void * ggml_cuda_host_malloc(size_t size);
GGML_API void   ggml_cuda_host_free(void * ptr);

GGML_API bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
GGML_API void   ggml_cuda_set_tensor_split(const float * tensor_split);
GGML_API void   ggml_cuda_transform_tensor(void * data, struct ggml_tensor * tensor);
GGML_API void   ggml_cuda_alloc_tensor(struct ggml_tensor * tensor);
GGML_API void   ggml_cuda_free_data(struct ggml_tensor * tensor);
GGML_API void   ggml_cuda_cpy_1d(struct ggml_tensor * dst, const struct ggml_tensor * src);
GGML_API bool   debug_equal(short *a, short *b);
GGML_API void **ggml_cuda_get_data_pp(struct ggml_tensor * tensor);

GGML_API void   ggml_cuda_assign_buffers(struct ggml_tensor * tensor);
GGML_API void   ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor * tensor);
GGML_API void   ggml_cuda_assign_buffers_force_inplace(struct ggml_tensor * tensor);

GGML_API void   ggml_cuda_assign_buffers_no_alloc(struct ggml_tensor * tensor);
GGML_API void   ggml_cuda_assign_scratch_offset(struct ggml_tensor * tensor, size_t offset);
GGML_API void   ggml_cuda_copy_to_device(struct ggml_tensor * tensor);

GGML_API void   ggml_cuda_set_main_device(int main_device);
GGML_API void   ggml_cuda_set_mul_mat_q(bool mul_mat_q);
GGML_API void   ggml_cuda_set_scratch_size(size_t scratch_size);
GGML_API void   ggml_cuda_free_scratch(void);
GGML_API bool   ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

GGML_API int    ggml_cuda_get_device_count(void);
GGML_API void   ggml_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API size_t ggml_cuda_get_free_memory(int device);

GGML_API void   ggml_cuda_set_device_constants(float sparse_pred_threshold);

// backend API
GGML_API ggml_backend_t ggml_backend_cuda_init(void); // TODO: take a list of devices to use

#ifdef  __cplusplus
}
#endif
