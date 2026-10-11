#include "ggml-dmabuf.h"
#include "ggml-dmabuf-allocators.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstring>
#include <mutex>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

//
// importers - consumers that import the dma-buf of a buffer, notified on alloc/free
//

struct ggml_backend_dmabuf_importer_entry {
    const struct ggml_backend_dmabuf_importer * importer;
    void * user_data;
};

static std::mutex g_dmabuf_importer_mutex;
static std::vector<ggml_backend_dmabuf_importer_entry> g_dmabuf_importers;

static std::vector<ggml_backend_dmabuf_importer_entry> ggml_backend_dmabuf_importers_snapshot() {
    std::lock_guard<std::mutex> lock(g_dmabuf_importer_mutex);
    return g_dmabuf_importers;
}

void ggml_backend_dmabuf_register_importer(const struct ggml_backend_dmabuf_importer * importer, void * user_data) {
    std::lock_guard<std::mutex> lock(g_dmabuf_importer_mutex);
    g_dmabuf_importers.push_back({ importer, user_data });
}

void ggml_backend_dmabuf_unregister_importer(const struct ggml_backend_dmabuf_importer * importer, void * user_data) {
    std::lock_guard<std::mutex> lock(g_dmabuf_importer_mutex);
    for (auto it = g_dmabuf_importers.begin(); it != g_dmabuf_importers.end(); ++it) {
        if (it->importer == importer && it->user_data == user_data) {
            g_dmabuf_importers.erase(it);
            break;
        }
    }
}

//
// buffer
//

struct ggml_backend_dmabuf_buffer_context {
    int    fd;
    void * ptr;
};

static void ggml_backend_dmabuf_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_dmabuf_buffer_context * ctx = (ggml_backend_dmabuf_buffer_context *) buffer->context;
    for (const auto & entry : ggml_backend_dmabuf_importers_snapshot()) {
        entry.importer->remove(entry.user_data, ctx->fd);
    }
    munmap(ctx->ptr, buffer->size);
    close(ctx->fd);
    delete ctx;
}

static void * ggml_backend_dmabuf_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_dmabuf_buffer_context * ctx = (ggml_backend_dmabuf_buffer_context *) buffer->context;
    return ctx->ptr;
}

static void ggml_backend_dmabuf_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_dmabuf_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_dmabuf_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_dmabuf_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_dmabuf_buffer_context * ctx = (ggml_backend_dmabuf_buffer_context *) buffer->context;
    memset(ctx->ptr, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_dmabuf_buffer_i = {
    /* .free_buffer     = */ ggml_backend_dmabuf_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_dmabuf_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_dmabuf_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_dmabuf_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_dmabuf_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_dmabuf_buffer_clear,
    /* .reset           = */ NULL,
};

//
// buffer type
//

static const char * ggml_backend_dmabuf_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return GGML_DMABUF_NAME;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_dmabuf_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    static size_t alignment = 0;
    if (alignment == 0) {
        const long page = sysconf(_SC_PAGESIZE);
        alignment = page > 0 ? (size_t) page : 4096;
    }
    return alignment;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_dmabuf_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size = ggml_dmabuf_page_align(size == 0 ? 1 : size);

    const int fd = ggml_backend_dmabuf_alloc(size);
    if (fd < 0) {
        return nullptr;
    }

    void * ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        GGML_LOG_ERROR("%s: failed to mmap dma-buf of %zu bytes\n", __func__, size);
        close(fd);
        return nullptr;
    }

    ggml_backend_dmabuf_buffer_context * ctx = new ggml_backend_dmabuf_buffer_context {
        /* .fd  = */ fd,
        /* .ptr = */ ptr,
    };

    for (const auto & entry : ggml_backend_dmabuf_importers_snapshot()) {
        entry.importer->add(entry.user_data, ptr, size, fd);
    }

    return ggml_backend_buffer_init(buft, ggml_backend_dmabuf_buffer_i, ctx, size);
}

static bool ggml_backend_dmabuf_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static struct ggml_backend_buffer_type ggml_backend_dmabuf_buffer_type_static = {
    /* .iface   = */ {
        /* .get_name         = */ ggml_backend_dmabuf_buffer_type_get_name,
        /* .alloc_buffer     = */ ggml_backend_dmabuf_buffer_type_alloc_buffer,
        /* .alloc_buffer_n   = */ NULL,
        /* .get_alignment    = */ ggml_backend_dmabuf_buffer_type_get_alignment,
        /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
        /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
        /* .get_alloc_size_n = */ NULL, // defaults to per-tensor get_alloc_size
        /* .is_host          = */ ggml_backend_dmabuf_buffer_type_is_host,
    },
    /* .device   = */ NULL,
    /* .context  = */ NULL,
};

//
// buffer type / buffer introspection for other backends
//

ggml_backend_buffer_type_t ggml_backend_dmabuf_buffer_type(void) {
    return &ggml_backend_dmabuf_buffer_type_static;
}

int ggml_backend_dmabuf_buffer_fd(ggml_backend_buffer_t buffer) {
    if (buffer->buft != ggml_backend_dmabuf_buffer_type()) {
        return -1;
    }
    ggml_backend_dmabuf_buffer_context * ctx = (ggml_backend_dmabuf_buffer_context *) buffer->context;
    return ctx->fd;
}
