#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct ggml_backend_buffer;

GGML_API struct ggml_allocr * ggml_allocr_new(void * data, size_t size, size_t alignment);
GGML_API struct ggml_allocr * ggml_allocr_new_measure(size_t alignment);
GGML_API struct ggml_allocr * ggml_allocr_new_from_buffer(struct ggml_backend_buffer * buffer);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
GGML_API void   ggml_allocr_set_parse_seq(struct ggml_allocr * alloc, const int * list, int n);

GGML_API void   ggml_allocr_free       (struct ggml_allocr * alloc);
GGML_API bool   ggml_allocr_is_measure (struct ggml_allocr * alloc);
GGML_API void   ggml_allocr_reset      (struct ggml_allocr * alloc);
GGML_API void   ggml_allocr_alloc      (struct ggml_allocr * alloc, struct ggml_tensor * tensor);
GGML_API size_t ggml_allocr_alloc_graph(struct ggml_allocr * alloc, struct ggml_cgraph * graph);
GGML_API size_t ggml_allocr_max_size   (struct ggml_allocr * alloc);

GGML_API size_t ggml_allocr_alloc_graph_n(
                    struct ggml_allocr * alloc,
                    struct ggml_cgraph ** graphs, int n_graphs,
                    struct ggml_tensor *** inputs, struct ggml_tensor *** outputs);

#ifdef  __cplusplus
}
#endif
