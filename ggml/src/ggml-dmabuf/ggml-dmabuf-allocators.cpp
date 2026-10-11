#include "ggml-dmabuf-allocators.h"
#include "ggml-impl.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <unistd.h>

size_t ggml_dmabuf_page_align(size_t size) {
    const long page = sysconf(_SC_PAGESIZE);
    const size_t align = page > 0 ? (size_t) page : 4096;
    return (size + align - 1) & ~(align - 1);
}

static int allocate_dmabuf_heap(const char * heap, size_t size) {
    char path[256];
    snprintf(path, sizeof(path), "/dev/dma_heap/%s", heap);

    int heap_fd = open(path, O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        GGML_LOG_ERROR("%s: failed to open %s\n", __func__, path);
        return -1;
    }

    struct dma_heap_allocation_data data = {
        /* .len        = */ size,
        /* .fd         = */ 0,
        /* .fd_flags   = */ O_RDWR | O_CLOEXEC,
        /* .heap_flags = */ 0,
    };

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) < 0) {
        GGML_LOG_ERROR("%s: DMA_HEAP_IOCTL_ALLOC failed for %zu bytes on %s\n", __func__, size, path);
        close(heap_fd);
        return -1;
    }
    close(heap_fd);

    const int fd = (int) data.fd;
    return fd;
}

int ggml_backend_dmabuf_alloc(size_t size) {
    const char * heap = getenv("GGML_DMABUF_HEAP");
    if (heap == nullptr) {
        heap = "system";
    }
    return allocate_dmabuf_heap(heap, size);
}
