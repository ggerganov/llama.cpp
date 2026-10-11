#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// name of the shared buffer type
#define GGML_DMABUF_NAME "DMA_BUF"

// buffer type backed by a dma-buf allocation
GGML_API ggml_backend_buffer_type_t ggml_backend_dmabuf_buffer_type(void);

// file descriptor of the dma-buf backing a buffer allocated with this buffer type, or -1
GGML_API int ggml_backend_dmabuf_buffer_fd(ggml_backend_buffer_t buffer);

// registered by consumers (e.g. a GPU backend) that import the dma-buf of a buffer.
// add() is called after a buffer is allocated, remove() before its dma-buf is released.
struct ggml_backend_dmabuf_importer {
    void (*add)   (void * user_data, void * ptr, size_t size, int fd);
    void (*remove)(void * user_data, int fd);
};

GGML_API void ggml_backend_dmabuf_register_importer  (const struct ggml_backend_dmabuf_importer * importer, void * user_data);
GGML_API void ggml_backend_dmabuf_unregister_importer(const struct ggml_backend_dmabuf_importer * importer, void * user_data);

#ifdef  __cplusplus
}
#endif
