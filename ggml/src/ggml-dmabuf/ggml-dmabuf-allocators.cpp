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

#ifdef GGML_DMABUF_AMDGPU
#include <drm/amdgpu_drm.h>
#endif

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

#ifdef GGML_DMABUF_AMDGPU
static int allocate_dmabuf_amdgpu(size_t size) {
    const char * device = getenv("GGML_DMABUF_AMDGPU_DEVICE");
    if (device == nullptr) {
        GGML_LOG_ERROR("%s: GGML_DMABUF_AMDGPU_DEVICE is not set\n", __func__);
        return -1;
    }

    const int drm_fd = open(device, O_RDWR | O_CLOEXEC);
    if (drm_fd < 0) {
        GGML_LOG_ERROR("%s: failed to open %s: %s\n", __func__, device, strerror(errno));
        return -1;
    }

    // default to GTT (system memory); set GGML_DMABUF_AMDGPU_VRAM=1 to allocate from VRAM
    const char * vram_env = getenv("GGML_DMABUF_AMDGPU_VRAM");
    const bool vram = vram_env != nullptr && atoi(vram_env) == 1;

    union drm_amdgpu_gem_create params = {};
    params.in.bo_size = size;
    params.in.alignment = ggml_dmabuf_page_align(1);
    params.in.domains = vram ? AMDGPU_GEM_DOMAIN_VRAM : AMDGPU_GEM_DOMAIN_GTT;
    params.in.domain_flags = AMDGPU_GEM_CREATE_EXPLICIT_SYNC;
    if (ioctl(drm_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &params) < 0) {
        GGML_LOG_ERROR("%s: DRM_IOCTL_AMDGPU_GEM_CREATE failed: %s\n", __func__, strerror(errno));
        close(drm_fd);
        return -1;
    }

    struct drm_prime_handle prime = {};
    prime.handle = params.out.handle;
    prime.flags = DRM_CLOEXEC | DRM_RDWR;
    if (ioctl(drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime) < 0) {
        GGML_LOG_ERROR("%s: DRM_IOCTL_PRIME_HANDLE_TO_FD failed: %s\n", __func__, strerror(errno));
        close(drm_fd);
        return -1;
    }
    close(drm_fd);
    return prime.fd;
}
#endif

int ggml_backend_dmabuf_alloc(size_t size) {
    const char * heap = getenv("GGML_DMABUF_HEAP");
    if (heap == nullptr) {
        heap = "system";
    }
    if (strcmp(heap, "amdgpu") == 0) {
#ifdef GGML_DMABUF_AMDGPU
        return allocate_dmabuf_amdgpu(size);
#else
        GGML_LOG_ERROR("%s: this build has no AMDGPU dma-buf allocator\n", __func__);
        return -1;
#endif
    }
    return allocate_dmabuf_heap(heap, size);
}
