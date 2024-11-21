#include "ggml-amx.h"
#include "ggml-amx/common.h"
#include "ggml-amx/mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#if defined(__gnu_linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__AMX_INT8__)

// AMX buffer interface
static void ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)(buffer->context);
}

static void ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (qtype_has_amx_kernels(tensor->type)) {
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } else {
        memcpy((char *)tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(!qtype_has_amx_kernels(tensor->type));
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (qtype_has_amx_kernels(src->type)) {
            ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_backend_amx_get_alloc_size(dst));
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_amx_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_amx_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_amx_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_amx_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_amx_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_amx_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "AMX";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = aligned_alloc(TENSOR_ALIGNMENT, size);
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

static size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    return ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

static bool ggml_backend_amx_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_amx_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_amx_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_amx_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_amx_buffer_type_get_alloc_size,
            /* .is_host          = */ ggml_backend_amx_buffer_type_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_amx_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_amx;
}

// backend interface

static const char * ggml_backend_amx_name(ggml_backend_t backend) {
    return "AMX";

    GGML_UNUSED(backend);
}

static void ggml_backend_amx_free(ggml_backend_t backend) {
    ggml_backend_amx_context * ctx = (ggml_backend_amx_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_amx_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_amx_context * ctx = (ggml_backend_amx_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
        case GGML_OP_MUL_MAT:
            ggml_backend_amx_mul_mat(ctx, node);
            break;

        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;

        default:
            fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
            GGML_ASSERT(false);
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i ggml_backend_amx_i = {
    /* .get_name                = */ ggml_backend_amx_name,
    /* .free                    = */ ggml_backend_amx_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_amx_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_amx_guid() {
    static ggml_guid guid = { 0x13, 0xb8, 0xa4, 0xc4, 0xba, 0xfe, 0x51, 0x67, 0x87, 0x44, 0x55, 0x15, 0xb2, 0x35, 0x62, 0x3e };
    return &guid;
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

static bool ggml_amx_init() {
#if defined(__gnu_linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        return false;
    }
    return true;
#elif defined(_WIN32)
    return true;
#endif
}

ggml_backend_t ggml_backend_amx_init() {

    // invoke a Linux system call to request access to AMX features
    ggml_amx_init();

    // backend context
    ggml_backend_amx_context * ctx = new ggml_backend_amx_context;

    // ggml amx backend
    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_amx_guid(),
        /* .interface = */ ggml_backend_amx_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_amx_reg(), 0),
        /* .context   = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_amx(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_amx_guid());
}

void ggml_backend_amx_set_n_threads(ggml_backend_t backend_amx, int n_threads) {
    GGML_ASSERT(ggml_backend_is_amx(backend_amx));

    ggml_backend_amx_context * ctx = (ggml_backend_amx_context *)backend_amx->context;
    ctx->n_threads = n_threads;
}

// device interface

static const char * ggml_backend_amx_device_get_name(ggml_backend_dev_t dev) {
    return "AMX";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_amx_device_get_description(ggml_backend_dev_t dev) {
    return "Intel Advanced Matrix Extensions";

    GGML_UNUSED(dev);
}

static void ggml_backend_amx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_amx_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_amx_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_amx_device_get_name(dev);
    props->description = ggml_backend_amx_device_get_description(dev);
    props->type        = ggml_backend_amx_device_get_type(dev);
    ggml_backend_amx_device_get_memory(dev, &props->memory_free, &props->memory_total);

    // `buffer_from_host_ptr` is intended to be used in mmap, when memory layout unchanged
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_amx_device_init(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_amx_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_amx_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_amx_buffer_type();

    GGML_UNUSED(dev);
}

static bool ggml_backend_amx_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {

    // handle only 2d gemm for now
    auto is_contiguous_2d = [](const struct ggml_tensor * t) {
        return ggml_is_contiguous(t) && t->ne[3] == 1 && t->ne[2] == 1;
    };

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            const enum ggml_type type = src0->type;
            const int64_t ne0 = op->ne[0];

            // amx kernels enables for Q4_0, Q4_1, Q8_0, F16
            // Q4_K, Q5_K, Q6_K, IQ4_XS enabled for QK_K = 256
            bool has_amx_kernels = qtype_has_amx_kernels(type) || (type == GGML_TYPE_F16);

            bool can_use_amx =
                is_contiguous_2d(src0) &&       // src0 must be contiguous
                is_contiguous_2d(src1) &&       // src1 must be contiguous
                src1->type == GGML_TYPE_F32 &&  // src1 must be float32
                has_amx_kernels &&              // with amx kernel impls
                ne0 % (TILE_N * 2) == 0;        // out_features is 32x

            return can_use_amx;
        }
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_amx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_amx_buffer_type_get_name;

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_amx_device_i = {
    /* .get_name             = */ ggml_backend_amx_device_get_name,
    /* .get_description      = */ ggml_backend_amx_device_get_description,
    /* .get_memory           = */ ggml_backend_amx_device_get_memory,
    /* .get_type             = */ ggml_backend_amx_device_get_type,
    /* .get_props            = */ ggml_backend_amx_device_get_props,
    /* .init_backend         = */ ggml_backend_amx_device_init,
    /* .get_buffer_type      = */ ggml_backend_amx_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_amx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_amx_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_amx_reg_get_name(ggml_backend_reg_t reg) {
    return "AMX";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_amx_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_amx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_amx_device = {
        /* .iface   = */ ggml_backend_amx_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_amx_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_amx_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_amx_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_amx_reg_i = {
    /* .get_name         = */ ggml_backend_amx_reg_get_name,
    /* .get_device_count = */ ggml_backend_amx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_amx_reg_get_device,
    /* .get_proc_address = */ ggml_backend_amx_get_proc_address,
};

ggml_backend_reg_t ggml_backend_amx_reg(void) {
    static struct ggml_backend_reg ggml_backend_amx_reg = {
        /* .iface   = */ ggml_backend_amx_reg_i,
        /* .context = */ NULL,
    };

    return &ggml_backend_amx_reg;
}

#else // if defined(__AMX_INT8__)

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(void) {
    return nullptr;
}

bool ggml_backend_is_amx(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return false;
}

ggml_backend_t ggml_backend_amx_init(void) {
    fprintf(stderr, "GGML is not compiled with AMX support!\n");
    return nullptr;
}

void ggml_backend_amx_set_n_threads(ggml_backend_t backend_amx, int n_threads) {
    fprintf(stderr, "GGML is not compiled with AMX support!\n");

    GGML_UNUSED(backend_amx);
    GGML_UNUSED(n_threads);
}

ggml_backend_reg_t ggml_backend_amx_reg(void) {
    return nullptr;
}

#endif
