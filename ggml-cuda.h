#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_API void * ggml_cuda_host_malloc(size_t size);
GGML_API void   ggml_cuda_host_free(void * ptr);
GGML_API void   ggml_cuda_host_register(void * ptr, size_t size);
GGML_API void   ggml_cuda_host_unregister(void * ptr);

// backend API

GGML_API struct ggml_backend * ggml_backend_cuda_init();


#ifdef  __cplusplus
}
#endif
