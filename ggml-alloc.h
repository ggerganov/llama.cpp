#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_buffer_type;

//
// Legacy API
//

typedef struct ggml_allocr * ggml_allocr_t;

// initialize allocator for use with CPU backend only
GGML_API ggml_allocr_t ggml_allocr_new(void * data, size_t size, size_t alignment);
GGML_API ggml_allocr_t ggml_allocr_new_measure(size_t alignment);

// initialize allocator for use with ggml-backend
GGML_API ggml_allocr_t ggml_allocr_new_from_buffer(struct ggml_backend_buffer * buffer);
GGML_API ggml_allocr_t ggml_allocr_new_from_backend(struct ggml_backend * backend, size_t size); // allocates an owned buffer
GGML_API ggml_allocr_t ggml_allocr_new_measure_from_backend(struct ggml_backend * backend);

GGML_API struct ggml_backend_buffer * ggml_allocr_get_buffer(ggml_allocr_t alloc);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
GGML_API void   ggml_allocr_set_parse_seq(ggml_allocr_t alloc, const int * list, int n);

GGML_API void   ggml_allocr_free       (ggml_allocr_t alloc);
GGML_API bool   ggml_allocr_is_measure (ggml_allocr_t alloc);
GGML_API void   ggml_allocr_reset      (ggml_allocr_t alloc);
GGML_API void   ggml_allocr_alloc      (ggml_allocr_t alloc, struct ggml_tensor * tensor);
GGML_API size_t ggml_allocr_max_size   (ggml_allocr_t alloc);

GGML_API size_t ggml_allocr_alloc_graph(ggml_allocr_t alloc, struct ggml_cgraph * graph);

//
// ggml-backend v2 API
//

// Separate tensor and graph allocator objects
// This is necessary for multi-backend allocation because the graph allocator needs to use multiple tensor allocators
// The original API is kept as a wrapper around the new API

// Tensor allocator
typedef struct ggml_tallocr * ggml_tallocr_t;

GGML_API ggml_tallocr_t ggml_tallocr_new(void * data, size_t size, size_t alignment);
GGML_API ggml_tallocr_t ggml_tallocr_new_measure(size_t alignment);
GGML_API ggml_tallocr_t ggml_tallocr_new_from_buffer(struct ggml_backend_buffer * buffer);
GGML_API ggml_tallocr_t ggml_tallocr_new_from_backend(struct ggml_backend * backend, size_t size); // allocates an owned buffer
GGML_API ggml_tallocr_t ggml_tallocr_new_measure_from_backend(struct ggml_backend * backend);

GGML_API struct ggml_backend_buffer * ggml_tallocr_get_buffer(ggml_tallocr_t talloc);

GGML_API void   ggml_tallocr_free       (ggml_tallocr_t talloc);
GGML_API bool   ggml_tallocr_is_measure (ggml_tallocr_t talloc);
GGML_API void   ggml_tallocr_reset      (ggml_tallocr_t talloc);
GGML_API void   ggml_tallocr_alloc      (ggml_tallocr_t talloc, struct ggml_tensor * tensor);
GGML_API size_t ggml_tallocr_max_size   (ggml_tallocr_t talloc);


// Graph allocator
typedef struct ggml_gallocr * ggml_gallocr_t;

GGML_API ggml_gallocr_t ggml_gallocr_new(void);
GGML_API void   ggml_gallocr_free(ggml_gallocr_t galloc);

GGML_API void   ggml_gallocr_set_parse_seq(ggml_gallocr_t galloc, const int * list, int n);
GGML_API size_t ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, ggml_tallocr_t talloc, struct ggml_cgraph * graph);

// Allocate tensors from the allocators given by the hash table
GGML_API void   ggml_gallocr_alloc_graph_n(
                    ggml_gallocr_t galloc,
                    struct ggml_cgraph * graph,
                    struct ggml_hash_set hash_set,
                    ggml_tallocr_t * hash_node_talloc);


// Utils
// Create a buffer and allocate all the tensors in a ggml_context
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, struct ggml_backend_buffer_type * buft);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, struct ggml_backend * backend);

#ifdef  __cplusplus
}
#endif
