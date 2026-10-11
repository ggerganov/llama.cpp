#pragma once

#include <cstddef>

// align size up to the allocation granularity of the dma-buf heap
size_t ggml_dmabuf_page_align(size_t size);

// allocate a dma-buf, returns a file descriptor or -1
// the source is selected with GGML_DMABUF_HEAP
int ggml_backend_dmabuf_alloc(size_t size);
