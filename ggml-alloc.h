#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif


GGML_API struct ggml_allocator * ggml_allocator_new(void * data, size_t size, size_t alignment);
GGML_API struct ggml_allocator * ggml_allocator_new_measure(size_t alignment);
GGML_API void ggml_allocator_free(struct ggml_allocator * alloc);
GGML_API bool ggml_allocator_is_measure(struct ggml_allocator * alloc);
GGML_API void ggml_allocator_reset(struct ggml_allocator * alloc);
GGML_API void ggml_allocator_alloc_tensor(struct ggml_allocator * alloc, struct ggml_tensor * tensor);
GGML_API size_t ggml_allocator_alloc_graph_tensors(struct ggml_allocator * alloc, struct ggml_cgraph * graph);


#ifdef  __cplusplus
}
#endif
