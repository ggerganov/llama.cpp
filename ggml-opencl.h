#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_API void ggml_cl_init(void);

GGML_API void   ggml_cl_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
GGML_API bool   ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
GGML_API size_t ggml_cl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
GGML_API void   ggml_cl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

// GGML_API void * ggml_cl_host_malloc(size_t size);
// GGML_API void   ggml_cl_host_free(void * ptr);

GGML_API void ggml_cl_free_data(const struct ggml_tensor* tensor);

GGML_API void ggml_cl_transform_tensor(void * data, struct ggml_tensor * tensor);

// backend API

// GGML_API ggml_backend_t ggml_backend_opencl_init(void);

// GGML_API bool ggml_backend_is_opencl(ggml_backend_t backend);

GGML_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
// GGML_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

#ifdef  __cplusplus
}
#endif
