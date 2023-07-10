#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void * ggml_cuda_host_malloc(size_t size);
void   ggml_cuda_host_free(void * ptr);

// backend API

struct ggml_backend ggml_backend_cuda_init();


#ifdef  __cplusplus
}
#endif
