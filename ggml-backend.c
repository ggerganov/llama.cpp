#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAX(a, b) ((a) > (b) ? (a) : (b))


// backend buffer type

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, struct ggml_tensor * tensor) {
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        return buft->iface.get_alloc_size(buft, tensor);
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return buft->iface.supports_backend(buft, backend);
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t      buft,
        struct ggml_backend_buffer_i           iface,
               ggml_backend_buffer_context_t   context,
               size_t                          size) {
    ggml_backend_buffer_t buffer = malloc(sizeof(struct ggml_backend_buffer));

    GGML_ASSERT(iface.get_base != NULL);

    (*buffer) = (struct ggml_backend_buffer) {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
    };

    return buffer;
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    free(buffer);
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

void ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    }
}

size_t ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type(buffer), tensor);
}

ggml_backend_buffer_type_t ggml_backend_buffer_type(ggml_backend_buffer_t buffer) {
    return buffer->buft;
}

// backend

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    return backend->iface.get_default_buffer_type(backend);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    backend->iface.set_tensor_async(backend, tensor, data, offset, size);
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    backend->iface.get_tensor_async(backend, tensor, data, offset, size);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(tensor->buffer != NULL && "tensor buffer not set");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    tensor->buffer->iface.set_tensor(tensor->buffer, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(tensor->buffer != NULL && "tensor buffer not set");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    tensor->buffer->iface.get_tensor(tensor->buffer, tensor, data, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    backend->iface.graph_plan_free(backend, plan);
}

void ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    backend->iface.graph_plan_compute(backend, plan);

    // TODO: optional sync
    ggml_backend_synchronize(backend);
}

void ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    backend->iface.graph_compute(backend, cgraph);

    // TODO: optional sync
    ggml_backend_synchronize(backend);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return backend->iface.supports_op(backend, op);
}

// backend copy

static bool ggml_are_same_layout(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst) {
    //printf("src: %s ne: [%d %d %d %d] nb: [%d %d %d %d]\n", src->name, (int)src->ne[0], (int)src->ne[1], (int)src->ne[2], (int)src->ne[3], (int)src->nb[0], (int)src->nb[1], (int)src->nb[2], (int)src->nb[3]);
    //printf("dst: %s ne: [%d %d %d %d] nb: [%d %d %d %d]\n", dst->name, (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2], (int)dst->ne[3], (int)dst->nb[0], (int)dst->nb[1], (int)dst->nb[2], (int)dst->nb[3]);
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    // fprintf(stderr, "cpy tensor %s from %s to %s (%lu bytes)\n", src->name, ggml_backend_name(src->backend), ggml_backend_name(dst->backend), ggml_nbytes(src));

    if (src == dst) {
        return;
    }

    // TODO: allow backends to support copy to/from same backend

    if (dst->buffer->iface.cpy_tensor_from != NULL) {
        dst->buffer->iface.cpy_tensor_from(dst->buffer, src, dst);
    } else if (src->buffer->iface.cpy_tensor_to != NULL) {
        src->buffer->iface.cpy_tensor_to(src->buffer, src, dst);
    } else {
        // shouldn't be hit when copying from/to CPU
        #ifndef NDEBUG
        fprintf(stderr, "ggml_backend_tensor_copy: neither cpy_tensor_from nor cpy_tensor_to "
                        "are implemented for %s and %s, falling back to get/set\n", src->name, dst->name);
        #endif
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

// backend registry

#define GGML_MAX_BACKENDS_REG 16

struct ggml_backend_reg {
    char name[128];
    ggml_backend_init_fn init_fn;
    ggml_backend_buffer_type_t default_buffer_type;
    void * user_data;
};

static struct ggml_backend_reg ggml_backend_registry[GGML_MAX_BACKENDS_REG];
static size_t ggml_backend_registry_count = 0;

static ggml_backend_t ggml_backend_reg_cpu_init(const char * params, void * user_data);

static void ggml_backend_registry_init(void) {
    static bool initialized = false;

    if (initialized) {
        return;
    }

    initialized = true;

    ggml_backend_register("CPU", ggml_backend_reg_cpu_init, ggml_backend_cpu_buffer_type(), NULL);

    // add forward decls here to avoid including the backend headers
#ifdef GGML_USE_CUBLAS
    extern void ggml_backend_cuda_reg_devices(void);
    ggml_backend_cuda_reg_devices();
#endif

#ifdef GGML_USE_METAL
    extern ggml_backend_t ggml_backend_reg_metal_init(const char * params, void * user_data);
    extern ggml_backend_buffer_type_t ggml_backend_metal_buffer_type(void);
    ggml_backend_register("Metal", ggml_backend_reg_metal_init, ggml_backend_metal_buffer_type(), NULL);
#endif
}

void ggml_backend_register(const char * name, ggml_backend_init_fn init_fn, ggml_backend_buffer_type_t default_buffer_type, void * user_data) {
    GGML_ASSERT(ggml_backend_registry_count < GGML_MAX_BACKENDS_REG);

    int id = ggml_backend_registry_count;

    ggml_backend_registry[id] = (struct ggml_backend_reg) {
        /* .name                = */ {0},
        /* .fn                  = */ init_fn,
        /* .default_buffer_type = */ default_buffer_type,
        /* .user_data           = */ user_data,
    };

    snprintf(ggml_backend_registry[id].name, sizeof(ggml_backend_registry[id].name), "%s", name);

#ifndef NDEBUG
    fprintf(stderr, "%s: registered backend %s\n", __func__, name);
#endif

    ggml_backend_registry_count++;
}

size_t ggml_backend_reg_get_count(void) {
    ggml_backend_registry_init();

    return ggml_backend_registry_count;
}

size_t ggml_backend_reg_find_by_name(const char * name) {
    ggml_backend_registry_init();

    for (size_t i = 0; i < ggml_backend_registry_count; i++) {
        // TODO: case insensitive in a portable way
        if (strcmp(ggml_backend_registry[i].name, name) == 0) {
            return i;
        }
    }
    return SIZE_MAX;
}

// init from backend:params string
ggml_backend_t ggml_backend_reg_init_backend_from_str(const char * backend_str) {
    ggml_backend_registry_init();

    const char * params = strchr(backend_str, ':');
    char backend_name[128];
    if (params == NULL) {
        strcpy(backend_name, backend_str);
        params = "";
    } else {
        strncpy(backend_name, backend_str, params - backend_str);
        backend_name[params - backend_str] = '\0';
        params++;
    }

    size_t backend_i = ggml_backend_reg_find_by_name(backend_name);
    if (backend_i == SIZE_MAX) {
        fprintf(stderr, "%s: backend %s not found\n", __func__, backend_name);
        return NULL;
    }

    return ggml_backend_reg_init_backend(backend_i, params);
}

const char * ggml_backend_reg_get_name(size_t i) {
    ggml_backend_registry_init();

    GGML_ASSERT(i < ggml_backend_registry_count);
    return ggml_backend_registry[i].name;
}

ggml_backend_t ggml_backend_reg_init_backend(size_t i, const char * params) {
    ggml_backend_registry_init();

    GGML_ASSERT(i < ggml_backend_registry_count);
    return ggml_backend_registry[i].init_fn(params, ggml_backend_registry[i].user_data);
}

ggml_backend_buffer_type_t ggml_backend_reg_get_default_buffer_type(size_t i) {
    ggml_backend_registry_init();

    GGML_ASSERT(i < ggml_backend_registry_count);
    return ggml_backend_registry[i].default_buffer_type;
}

ggml_backend_buffer_t ggml_backend_reg_alloc_buffer(size_t i, size_t size) {
    ggml_backend_registry_init();

    GGML_ASSERT(i < ggml_backend_registry_count);
    return ggml_backend_buft_alloc_buffer(ggml_backend_registry[i].default_buffer_type, size);
}

// backend CPU

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)buffer->context;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_cpy_tensor_from(ggml_backend_buffer_t buffer, struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_cpy_tensor_to(ggml_backend_buffer_t buffer, struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));

    GGML_UNUSED(buffer);
}

static struct ggml_backend_buffer_i cpu_backend_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor_from = */ ggml_backend_cpu_buffer_cpy_tensor_from,
    /* .cpy_tensor_to   = */ ggml_backend_cpu_buffer_cpy_tensor_to,
};

// for buffers from ptr, free is not called
static struct ggml_backend_buffer_i cpu_backend_buffer_i_from_ptr = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor_from = */ ggml_backend_cpu_buffer_cpy_tensor_from,
    /* .cpy_tensor_to   = */ ggml_backend_cpu_buffer_cpy_tensor_to,
};

static const size_t TENSOR_ALIGNMENT = 64; // should be enough for AVX 512

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size += TENSOR_ALIGNMENT;   // malloc may return an address that is not aligned
    void * data = malloc(size); // TODO: maybe use GGML_ALIGNED_MALLOC?

    GGML_ASSERT(data != NULL && "failed to allocate buffer");

    return ggml_backend_buffer_init(buft, cpu_backend_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_cpu(backend);

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_cpu = {
        /* .iface = */ {
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .supports_backend = */ ggml_backend_cpu_buffer_type_supports_backend,
        },
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_cpu;
}

struct ggml_backend_cpu_context {
    int n_threads;
    void * work_data;
    size_t work_size;
};

static const char * ggml_backend_cpu_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    free(cpu_ctx->work_data);
    free(cpu_ctx);
    free(backend);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(backend);
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = malloc(sizeof(struct ggml_backend_plan_cpu));

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads);
    cpu_plan->cgraph = *cgraph;

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = malloc(cpu_plan->cplan.work_size);
    }

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    free(cpu_plan->cplan.work_data);
    free(cpu_plan);

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

static void ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads);

    if (cpu_ctx->work_size < cplan.work_size) {
        // TODO: may be faster to free and use malloc to avoid the copy
        cpu_ctx->work_data = realloc(cpu_ctx->work_data, cplan.work_size);
        cpu_ctx->work_size = cplan.work_size;
    }

    cplan.work_data = cpu_ctx->work_data;

    ggml_graph_compute(cgraph, &cplan);
}

static bool ggml_backend_cpu_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return true;

    GGML_UNUSED(backend);
    GGML_UNUSED(op);
}

static struct ggml_backend_i cpu_backend_i = {
    /* .get_name                = */ ggml_backend_cpu_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .get_default_buffer_type = */ ggml_backend_cpu_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_from_async   = */ NULL,
    /* .cpy_tensor_to_async     = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .supports_op             = */ ggml_backend_cpu_supports_op,
};

ggml_backend_t ggml_backend_cpu_init(void) {
    struct ggml_backend_cpu_context * ctx = malloc(sizeof(struct ggml_backend_cpu_context));

    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->work_data = NULL;
    ctx->work_size = 0;

    ggml_backend_t cpu_backend = malloc(sizeof(struct ggml_backend));

    *cpu_backend = (struct ggml_backend) {
        /* .interface = */ cpu_backend_i,
        /* .context   = */ ctx
    };
    return cpu_backend;
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend->iface.get_name == ggml_backend_cpu_name;
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_type(), cpu_backend_buffer_i_from_ptr, ptr, size);
}

static ggml_backend_t ggml_backend_reg_cpu_init(const char * params, void * user_data) {
    return ggml_backend_cpu_init();

    GGML_UNUSED(params);
    GGML_UNUSED(user_data);
}


// scheduler

#define GGML_MAX_BACKENDS 4
#define GGML_MAX_SPLITS 256
#define GGML_MAX_SPLIT_INPUTS 16

struct ggml_backend_sched_split {
    ggml_tallocr_t tallocr;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_MAX_SPLIT_INPUTS];
    int n_inputs;
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    int n_backends;
    ggml_backend_t backends[GGML_MAX_BACKENDS];
    ggml_tallocr_t  tallocs[GGML_MAX_BACKENDS];

    ggml_gallocr_t galloc;

    struct ggml_hash_set    hash_set;
    ggml_tallocr_t *        node_talloc;                     // [hash_set.size]
    struct ggml_tensor * (* node_copies)[GGML_MAX_BACKENDS]; // [hash_set.size][GGML_MAX_BACKENDS]

    struct ggml_cgraph * graph;
    struct ggml_backend_sched_split splits[GGML_MAX_SPLITS];
    int n_splits;

    struct ggml_context * ctx;

    // align context_buffer to GGML_MEM_ALIGN
    #ifdef _MSC_VER
    __declspec(align(GGML_MEM_ALIGN))
    #else
    __attribute__((aligned(GGML_MEM_ALIGN)))
    #endif
    char context_buffer[GGML_MAX_SPLITS*GGML_MAX_SPLIT_INPUTS*sizeof(struct ggml_tensor) + sizeof(struct ggml_cgraph)];
};

#define hash_id(node) ggml_hash_find_or_insert(sched->hash_set, node)
#define node_allocr(node) sched->node_talloc[hash_id(node)]

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// returns the priority of the backend, lower is better
static int sched_backend_prio(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return INT_MAX;
}

static int sched_allocr_prio(ggml_backend_sched_t sched, ggml_tallocr_t allocr) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->tallocs[i] == allocr) {
            return i;
        }
    }
    return INT_MAX;
}

static ggml_backend_t get_buffer_backend(ggml_backend_sched_t sched, ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return NULL;
    }
    // find highest prio backend that supports the buffer type
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_buft_supports_backend(buffer->buft, sched->backends[i])) {
            return sched->backends[i];
        }
    }
    GGML_ASSERT(false && "tensor buffer type not supported by any backend");
}

static ggml_backend_t get_allocr_backend(ggml_backend_sched_t sched, ggml_tallocr_t allocr) {
    if (allocr == NULL) {
        return NULL;
    }
    // find highest prio backend that supports the buffer type
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->tallocs[i] == allocr) {
            return sched->backends[i];
        }
    }
    GGML_UNREACHABLE();
}

#if 0
static char causes[GGML_DEFAULT_GRAPH_SIZE*8 + GGML_MAX_SPLITS*GGML_MAX_SPLIT_INPUTS][128]; // debug, remove
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static ggml_backend_t sched_backend_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    // if the dst tensor is already allocated in a buffer, we must assume that it is critical to keep it there
    // ie. kv cache updates
    // note that this doesn't allow fallback to CPU. need to add output tensors to the splits to copy the data back to the original backend.
    // dst
    ggml_backend_t cur_backend = get_buffer_backend(sched, node->buffer);
    if (cur_backend != NULL) {
        SET_CAUSE(node, "1.dst");
        return cur_backend;
    }

    // view_src
    if (node->view_src != NULL && get_buffer_backend(sched, node->view_src->buffer) != NULL) {
        SET_CAUSE(node, "1.vsrc");
        return get_buffer_backend(sched, node->view_src->buffer);
    }

    // src
    int cur_prio = INT_MAX;
    size_t cur_size = 0;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = node->src[i];
        if (src == NULL) {
            break;
        }
        ggml_backend_t src_backend = get_buffer_backend(sched, src->buffer);
        if (src_backend != NULL) {
            int src_prio = sched_backend_prio(sched, src_backend);
            size_t src_size = ggml_nbytes(src);
            if (src_prio < cur_prio && src_size >= cur_size) {
                cur_prio = src_prio;
                cur_size = src_size;
                cur_backend = src_backend;
                SET_CAUSE(node, "1.src%d", i);
            }
        }
    }
    return cur_backend;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        sprintf(buffer, "%zuM", size/1024/1024);
    } else {
        sprintf(buffer, "%zuK", size/1024);
    }
    return buffer;
}

static void sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = get_allocr_backend(sched, sched->splits[cur_split].tallocr);
            fprintf(stderr, "\n## SPLIT #%d: %s # %d inputs: ", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                fprintf(stderr, "[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            fprintf(stderr, "\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        ggml_tallocr_t node_allocr = node_allocr(node);
        ggml_backend_t node_backend = node_allocr ? get_allocr_backend(sched, node_allocr) : NULL; // FIXME:
        fprintf(stderr, "node #%3d (%10.10s): %20.20s (%4.4s) [%4.4s %8.8s]:", i, ggml_op_name(node->op), node->name,
            fmt_size(ggml_nbytes(node)), node_allocr ? ggml_backend_name(node_backend) : "NULL", GET_CAUSE(node));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            ggml_tallocr_t src_allocr = node_allocr(src);
            ggml_backend_t src_backend = src_allocr ? get_allocr_backend(sched, src_allocr) : NULL;
            fprintf(stderr, " %20.20s (%4.4s) [%4.4s %8.8s]", src->name,
                fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
        }
        fprintf(stderr, "\n");
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
// TODO: merge passes
static void sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset state
    size_t hash_size = sched->hash_set.size;
    memset(sched->hash_set.keys, 0, sizeof(sched->hash_set.keys[0]) * hash_size);
    memset(sched->node_talloc,   0, sizeof(sched->node_talloc[0])   * hash_size);
    memset(sched->node_copies,   0, sizeof(sched->node_copies[0])   * hash_size);
    sched->n_splits = 0;

    struct ggml_init_params params = {
        /* .mem_size =   */ sizeof(sched->context_buffer),
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    if (sched->ctx != NULL) {
        ggml_free(sched->ctx);
    }

    sched->ctx = ggml_init(params);

    // pass 1: assign backends to ops with allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        if (node_allocr(leaf) != NULL) {
            // do not overwrite user assignments
            continue;
        }
        ggml_backend_t leaf_backend = get_buffer_backend(sched, leaf->buffer);
        if (leaf_backend == NULL && leaf->view_src != NULL) {
            leaf_backend = get_buffer_backend(sched, leaf->view_src->buffer);
        }
        if (leaf_backend != NULL) {
            node_allocr(leaf) = ggml_backend_sched_get_tallocr(sched, leaf_backend);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (node_allocr(node) != NULL) {
            // do not overwrite user assignments
            continue;
        }
        ggml_backend_t node_backend = sched_backend_from_cur(sched, node);
        if (node_backend != NULL) {
            node_allocr(node) = ggml_backend_sched_get_tallocr(sched, node_backend);
        }
    }
    //printf("PASS 1 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);

    // pass 2: assign backends to ops from current assignments
    // TODO:
    //  - reuse sched_backend_from_cur
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        ggml_tallocr_t node_allocr = node_allocr(node);
        if (node_allocr == NULL) {
            int    cur_prio = INT_MAX;
            size_t cur_size = 0;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    break;
                }
                ggml_tallocr_t src_allocr = node_allocr(src);
                if (src_allocr != NULL) {
                    int    src_prio = sched_allocr_prio(sched, src_allocr);
                    size_t src_size = ggml_nbytes(src);
                    if (src_prio < cur_prio && src_size >= cur_size) {
                        cur_prio = src_prio;
                        cur_size = src_size;
                        node_allocr = src_allocr;
                        SET_CAUSE(node, "2.src%d", j);
                    }
                }
            }
            if (node_allocr != NULL) {
                node_allocr(node) = node_allocr;
            }
        }
    }
    //printf("PASS 2 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);

    // pass 3: assign backends to remaining src from dst (should only be leafs)
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        ggml_tallocr_t node_allocr = node_allocr(node);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            ggml_tallocr_t src_allocr = node_allocr(src);
            if (src_allocr == NULL) {
                node_allocr(src) = node_allocr;
            }
        }
    }
    //printf("PASS 3 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);

    // pass 4: split graph, find tensors that need to be copied
    // TODO:
    //  - when switching from a less preferred backend to a more preferred backend, check if it is possible to move the switch to an earlier point for the same cost
    // find first backend
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (node->view_src == NULL) {
            sched->splits[0].tallocr = node_allocr(node);
            break;
        }
    }
    sched->splits[0].i_start = 0;
    sched->splits[0].n_inputs = 0;
    memset(sched->splits[0].inputs, 0, sizeof(sched->splits[0].inputs)); //HACK
    ggml_tallocr_t cur_allocr = sched->splits[0].tallocr;
    size_t cur_backend_id = sched_allocr_prio(sched, cur_allocr);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        if (ggml_is_view_op(node->op)) {
            continue;
        }

        ggml_tallocr_t node_allocr = node_allocr(node);

        if (node_allocr != cur_allocr) {
            sched->splits[cur_split].i_end = i;
            cur_split++;
            GGML_ASSERT(cur_split < GGML_MAX_SPLITS);
            sched->splits[cur_split].tallocr = node_allocr;
            sched->splits[cur_split].i_start = i;
            sched->splits[cur_split].n_inputs = 0;
            memset(sched->splits[cur_split].inputs, 0, sizeof(sched->splits[cur_split].inputs)); //HACK
            cur_allocr = node_allocr;
            cur_backend_id = sched_allocr_prio(sched, cur_allocr);
        }

        // find inputs that are not on the same backend
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            ggml_tallocr_t src_allocr = node_allocr(src);
            if (src_allocr != node_allocr) {
                int n_inputs = sched->splits[cur_split].n_inputs++;
                GGML_ASSERT(n_inputs < GGML_MAX_SPLIT_INPUTS);
                sched->splits[cur_split].inputs[n_inputs] = (struct ggml_tensor *)src;

                // create copies
                size_t id = hash_id(src);
                if (sched->node_copies[id][cur_backend_id] == NULL) {
                    struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                    sched->node_copies[id][cur_backend_id] = tensor_copy;
                    node_allocr(tensor_copy) = cur_allocr;
                    ggml_backend_t backend = get_allocr_backend(sched, cur_allocr);
                    ggml_format_name(tensor_copy, "%s#%s", ggml_backend_name(backend), src->name);
                }
                node->src[j] = sched->node_copies[id][cur_backend_id];
            }
        }
    }
    sched->splits[cur_split].i_end = graph->n_nodes;
    sched->n_splits = cur_split + 1;

    //fprintf(stderr, "PASS 4 ASSIGNMENTS\n"); sched_print_assignments(sched, graph); fflush(stdout);

#if 1
    // sanity check: all sources should have the same backend as the node
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        ggml_tallocr_t node_allocr = node_allocr(node);
        if (node_allocr == NULL) {
            fprintf(stderr, "!!!!!!! %s has no backend\n", node->name);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                break;
            }
            ggml_tallocr_t src_allocr = node_allocr(src);
            if (src_allocr != node_allocr /* && src_backend != NULL */) { // ignore nulls for now
                fprintf(stderr, "!!!! %s has backend %s, src %d (%s) has backend %s\n",
                    node->name, node_allocr ? ggml_backend_name(get_allocr_backend(sched, node_allocr)) : "NULL",
                    j, src->name, src_allocr ? ggml_backend_name(get_allocr_backend(sched, src_allocr)) : "NULL");
            }
        }
    }
#endif

    // create copies of the graph for each split
    // FIXME: avoid this copy, pass split inputs to ggml_gallocr_alloc_graph_n in some other way
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(sched->ctx, graph->n_nodes + sched->n_splits*GGML_MAX_SPLIT_INPUTS, false);
    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * input = split->inputs[j];
            struct ggml_tensor * input_cpy = sched->node_copies[hash_id(input)][sched_allocr_prio(sched, split->tallocr)];
            input_cpy->src[0] = input;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }
    sched->graph = graph_copy;
}

static void sched_alloc_splits(ggml_backend_sched_t sched) {
    ggml_gallocr_alloc_graph_n(
        sched->galloc,
        sched->graph,
        sched->hash_set,
        sched->node_talloc);
}

static void sched_compute_splits(ggml_backend_sched_t sched) {
    uint64_t copy_us[GGML_MAX_BACKENDS] = {0};
    uint64_t compute_us[GGML_MAX_BACKENDS] = {0};

    struct ggml_backend_sched_split * splits = sched->splits;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &splits[i];
        ggml_backend_t split_backend = get_allocr_backend(sched, split->tallocr);
        int split_backend_id = sched_backend_prio(sched, split_backend);

        // copy the input tensors to the split backend
        uint64_t copy_start_us = ggml_time_us();
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * input = split->inputs[j];
            struct ggml_tensor * input_cpy = sched->node_copies[hash_id(input)][sched_backend_prio(sched, split_backend)];
            if (input->buffer == NULL) {
                if (input->view_src == NULL) {
                    fprintf(stderr, "input %s has no buffer and no view_src\n", input->name);
                    exit(1);
                }
                // FIXME: may need to use the sched buffer instead
                ggml_backend_view_init(input->view_src->buffer, input);
            }
            if (input_cpy->buffer == NULL) {
                fprintf(stderr, "input_cpy %s has no buffer\n", input_cpy->name);
                exit(1);
            }
            //GGML_ASSERT(input->buffer->backend != input_cpy->buffer->backend);
            //GGML_ASSERT(input_cpy->buffer->backend == split_backend);
            ggml_backend_tensor_copy(input, input_cpy);
        }
        // ggml_backend_synchronize(split_backend);
        int64_t copy_end_us = ggml_time_us();
        copy_us[split_backend_id] += copy_end_us - copy_start_us;

#if 0
        char split_filename[GGML_MAX_NAME];
        snprintf(split_filename, GGML_MAX_NAME, "split_%i_%s.dot", i, ggml_backend_name(split_backend));
        ggml_graph_dump_dot(split->graph, NULL, split_filename);
#endif

        uint64_t compute_start_us = ggml_time_us();
        ggml_backend_graph_compute(split_backend, &split->graph);
        // ggml_backend_synchronize(split_backend);
        uint64_t compute_end_us = ggml_time_us();
        compute_us[split_backend_id] += compute_end_us - compute_start_us;
    }

#if 0
    // per-backend timings
    fprintf(stderr, "sched_compute_splits times (%d splits):\n", sched->n_splits);
    for (int i = 0; i < sched->n_backends; i++) {
        if (copy_us[i] > 0 || compute_us[i] > 0) {
            fprintf(stderr, "\t%5.5s: %lu us copy, %lu us compute\n", ggml_backend_name(sched->backends[i]), copy_us[i], compute_us[i]);
        }
    }
#endif
}

static void sched_reset(ggml_backend_sched_t sched) {
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_tallocr_reset(sched->tallocs[i]);
    }
}

ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, int n_backends) {
    GGML_ASSERT(n_backends <= GGML_MAX_BACKENDS);

    struct ggml_backend_sched * sched = malloc(sizeof(struct ggml_backend_sched));
    memset(sched, 0, sizeof(struct ggml_backend_sched));

    sched->n_backends = n_backends;
    for (int i = 0; i < n_backends; i++) {
        sched->backends[i] = backends[i];
    }

    sched->galloc = ggml_gallocr_new();

    // init measure allocs for each backend
    for (int i = 0; i < n_backends; i++) {
        sched->tallocs[i] = ggml_tallocr_new_measure_from_backend(backends[i]);
    }

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_tallocr_free(sched->tallocs[i]);
    }
    ggml_gallocr_free(sched->galloc);
    free(sched->hash_set.keys);
    free(sched->node_talloc);
    free(sched->node_copies);
    free(sched);
}

void ggml_backend_sched_init_measure(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    // initialize hash tables
    size_t hash_size = measure_graph->visited_hash_table.size + GGML_MAX_SPLITS*GGML_MAX_SPLIT_INPUTS;
    sched->hash_set.size = hash_size;
    sched->hash_set.keys = malloc(sizeof(sched->hash_set.keys[0]) * hash_size);
    sched->node_talloc   = malloc(sizeof(sched->node_talloc[0])   * hash_size);
    sched->node_copies   = malloc(sizeof(sched->node_copies[0])   * hash_size);

    sched_split_graph(sched, measure_graph);
    sched_alloc_splits(sched);

    // allocate buffers and reset allocators
    for (int i = 0; i < sched->n_backends; i++) {
        size_t size = ggml_tallocr_max_size(sched->tallocs[i]);
        ggml_tallocr_free(sched->tallocs[i]);
        sched->tallocs[i] = ggml_tallocr_new_from_backend(sched->backends[i], size);
    }

    sched_reset(sched);
}

void ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched->hash_set.size >= graph->visited_hash_table.size + GGML_MAX_SPLITS*GGML_MAX_SPLIT_INPUTS);

    sched_split_graph(sched, graph);
    sched_alloc_splits(sched);
    sched_compute_splits(sched);
    sched_reset(sched);
}

ggml_tallocr_t ggml_backend_sched_get_tallocr(ggml_backend_sched_t sched, ggml_backend_t backend) {
    int backend_index = sched_backend_prio(sched, backend);
    return sched->tallocs[backend_index];
}

ggml_backend_buffer_t ggml_backend_sched_get_buffer(ggml_backend_sched_t sched, ggml_backend_t backend) {
    int backend_index = sched_backend_prio(sched, backend);
    return ggml_tallocr_get_buffer(sched->tallocs[backend_index]);
}

void ggml_backend_sched_set_node_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    int backend_index = sched_backend_prio(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    node_allocr(node) = sched->tallocs[backend_index];
}

// utils
void ggml_backend_view_init(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    tensor->backend = tensor->view_src->backend;
    ggml_backend_buffer_init_tensor(buffer, tensor);
}

void ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT((char *)addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
                (char *)ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(hash_set, src);
    if (id == GGML_HASHTABLE_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            break;
        }
        dst->src[i] = graph_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_init_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        ggml_backend_view_init(dst->view_src->buffer, dst);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            break;
        }
        graph_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    struct ggml_hash_set hash_set = {
        /* .size = */ graph->visited_hash_table.size,
        /* .keys = */ calloc(sizeof(hash_set.keys[0]) * graph->visited_hash_table.size, 1)
    };
    struct ggml_tensor ** node_copies = calloc(sizeof(node_copies[0]) * hash_set.size, 1);
    bool * node_init = calloc(sizeof(node_init[0]) * hash_set.size, 1);

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_init_tensor(hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    free(hash_set.keys);
    free(node_copies);
    free(node_init);

    return (struct ggml_backend_graph_copy) {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

void ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    for (int i = 0; i < g1->n_nodes; i++) {
        //printf("eval %d/%d\n", i, g1->n_nodes);
        struct ggml_tensor * t1 = g1->nodes[i];
        struct ggml_tensor * t2 = g2->nodes[i];

        assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

        struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
        struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

        ggml_backend_graph_compute(backend1, &g1v);
        ggml_backend_graph_compute(backend2, &g2v);

        if (ggml_is_view_op(t1->op)) {
            continue;
        }

        // compare results, calculate rms etc
        if (!callback(i, t1, t2, user_data)) {
            break;
        }
    }

    ggml_backend_graph_copy_free(copy);
}
