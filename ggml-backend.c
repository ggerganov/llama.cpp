#include "ggml-backend.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNUSED(x) (void)(x)

// allocator

static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

static inline size_t ggml_backend_buffer_get_alloc_size(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) {
    return alloc->interface.get_alloc_size(alloc, tensor);
}

static inline void ggml_backend_buffer_init_tensor(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) {
    alloc->interface.init_tensor(alloc, tensor);
}

void ggml_backend_buffer_free(struct ggml_backend_buffer * alloc) {
    alloc->interface.free_buffer(alloc);
    free(alloc);
}

// backend buffer allocator - simple - cannot free tensors, good for weights and small contexts

struct ggml_allocator_simple_context {
    void * data;
    size_t size;
    size_t offset;
    size_t alignment;
};

static void ggml_allocator_simple_free_buffer(struct ggml_backend_buffer * alloc) {
    struct ggml_allocator_simple_context * context = (struct ggml_allocator_simple_context *)alloc->context;
    free(context);
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))

static void ggml_allocator_simple_alloc_tensor(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) {
    struct ggml_allocator_simple_context * context = (struct ggml_allocator_simple_context *)alloc->context;

    size_t size = ggml_backend_buffer_get_alloc_size(alloc, tensor);

    if (!alloc->measure && context->offset + size > context->size) {
        fprintf(stderr, "%s: not enough space in the buffer (needed %zu, available %zu)\n",
                __func__, size, context->size - context->offset);
        GGML_ASSERT(!"not enough space in the buffer");
        return;
    }

    alloc->max_size = MAX(alloc->max_size, context->offset + size);
    tensor->data = (char*)context->data + context->offset;

    if (!alloc->measure) {
        if (alloc->interface.init_tensor) {
            ggml_backend_buffer_init_tensor(alloc, tensor);
        }
    }

    context->offset = aligned_offset(context->data, context->offset + size, context->alignment);
}

static void ggml_allocator_simple_free_tensor(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) {
    GGML_ASSERT(!"ggml_allocator_simple cannot free individual tensors");

    UNUSED(alloc);
    UNUSED(tensor);
}

static void ggml_allocator_simple_reset(struct ggml_backend_buffer * alloc) {
    struct ggml_allocator_simple_context * context = (struct ggml_allocator_simple_context *)alloc->context;
    context->offset = aligned_offset(context->data, 0, context->alignment);
}

size_t ggml_allocator_simple_get_alloc_size(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) {
    return ggml_nbytes(tensor);

    UNUSED(alloc);
}

static const struct ggml_backend_buffer_interface ggml_allocator_simple_interface = {
    /* .free_buffer    = */ ggml_allocator_simple_free_buffer,
    /* .alloc_tensor   = */ ggml_allocator_simple_alloc_tensor,
    /* .free_tensor    = */ ggml_allocator_simple_free_tensor,
    /* .reset          = */ ggml_allocator_simple_reset,
    /* .get_alloc_size = */ ggml_allocator_simple_get_alloc_size,
    /* .init_tensor    = */ NULL,
    /* .free_data      = */ NULL,
};

static struct ggml_backend_buffer * ggml_allocator_simple_init(void * data, size_t size, size_t alignment) {
    struct ggml_allocator_simple_context * ctx = malloc(sizeof(struct ggml_allocator_simple_context));
    ctx->data = data;
    ctx->size = size;
    ctx->offset = aligned_offset(data, 0, alignment);
    ctx->alignment = alignment;

    struct ggml_backend_buffer * allocator = malloc(sizeof(struct ggml_backend_buffer));
    *allocator = (struct ggml_backend_buffer){
        /* .interface    = */ ggml_allocator_simple_interface,
        /* .context      = */ ctx,
        /* .backend      = */ NULL,
        /* .backend_data = */ NULL,
        /* .measure      = */ false,
        /* .max_size     = */ 0,
    };
    return allocator;
}

//////////////////////////////////////////////////////////////

// backend buffer allocator - default - can free tensors

struct free_block {
    void * addr;
    size_t size;
};

#define MAX_FREE_BLOCKS 128

struct ggml_allocator_default_context {
    void * data;
    size_t size;
    size_t alignment;
    int n_free_blocks;
    struct free_block free_blocks[1024];
};

void ggml_allocator_default_free_buffer(struct ggml_backend_buffer * alloc) {
    struct ggml_allocator_default_context * allocator_ctx = (struct ggml_allocator_default_context *)alloc->context;
    free(allocator_ctx);
}

static const size_t MAX_SIZE_INIT = (1ULL<<40)-1;
void ggml_allocator_default_alloc_tensor(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) {
    struct ggml_allocator_default_context * allocator_ctx = (struct ggml_allocator_default_context *)alloc->context;

    /////
    if (alloc->measure && allocator_ctx->size != MAX_SIZE_INIT) {
        allocator_ctx->size = MAX_SIZE_INIT;
        //allocator_ctx->data = 0;
        allocator_ctx->free_blocks[0].size = MAX_SIZE_INIT;
        //allocator_ctx->free_blocks[0].addr = 0;
    }
    /////

    size_t size = ggml_backend_buffer_get_alloc_size(alloc, tensor);
    size = aligned_offset(NULL, size, allocator_ctx->alignment);

    // printf("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    size_t max_avail = 0;

    //fprintf(stderr, "%s: allocating %s - %zu bytes\n", __func__, tensor->name, size);

    // find the best fitting free block
    int best_fit_block = -1;
    size_t best_fit_size = SIZE_MAX;
    for (int i = 0; i < allocator_ctx->n_free_blocks; i++) {
        struct free_block * block = &allocator_ctx->free_blocks[i];
        max_avail = MAX(max_avail, block->size);
        if (block->size >= size && block->size <= best_fit_size) {
            best_fit_block = i;
            best_fit_size = block->size;
        }
    }

    // printf("block %d\n", best_fit_block);

    if (best_fit_block == -1) {
        fprintf(stderr, "%s: not enough space in the buffer (needed %zu, largest block available %zu)\n",
                __func__, size, max_avail);
        GGML_ASSERT(!"not enough space in the buffer");
        return;
    }
    struct free_block * block = &allocator_ctx->free_blocks[best_fit_block];
    void * addr = block->addr;
    block->addr = (char*)block->addr + size;
    block->size -= size;
    if (block->size == 0) {
        // remove block if empty
        allocator_ctx->n_free_blocks--;
        for (int j = best_fit_block; j < allocator_ctx->n_free_blocks; j++) {
            allocator_ctx->free_blocks[j] = allocator_ctx->free_blocks[j+1];
        }
    }

    alloc->max_size = MAX(alloc->max_size, (char*)addr - (char*)allocator_ctx->data + size);
    tensor->data = addr;

    if (!alloc->measure) {
        if (alloc->interface.init_tensor) {
            ggml_backend_buffer_init_tensor(alloc, tensor);
        }
    }
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
void ggml_allocator_default_free_tensor(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) {
    struct ggml_allocator_default_context * allocator_ctx = (struct ggml_allocator_default_context *)alloc->context;

    void * ptr = tensor->data;

    if (ptr < allocator_ctx->data || (char*)ptr >= (char*)allocator_ctx->data + alloc->max_size) {
        //fprintf(stderr, "%s: %s - tensor not in this buffer (%p - %p - %zu)\n", __func__, tensor->name, ptr, allocator_ctx->data, allocator_ctx->size);
        //GGML_ASSERT(!"trying to free a tensor that was not allocated by this allocator");
        return;
    }

    size_t size = ggml_backend_buffer_get_alloc_size(alloc, tensor);
    size = aligned_offset(NULL, size, allocator_ctx->alignment);
    //printf("%s: freeing %s (%zu bytes) - n_free_blocks = %d\n", __func__, tensor->name, size, allocator_ctx->n_free_blocks);

    // see if we can merge with an existing block
    for (int i = 0; i < allocator_ctx->n_free_blocks; i++) {
        struct free_block * block = &allocator_ctx->free_blocks[i];
        // check if ptr is at the end of the block
        if ((char*)block->addr + block->size == ptr) {
            block->size += size;
            // check if we can merge with the next block
            if (i < allocator_ctx->n_free_blocks - 1 && (char*)block->addr + block->size == allocator_ctx->free_blocks[i+1].addr) {
                block->size += allocator_ctx->free_blocks[i+1].size;
                allocator_ctx->n_free_blocks--;
                for (int j = i+1; j < allocator_ctx->n_free_blocks; j++) {
                    allocator_ctx->free_blocks[j] = allocator_ctx->free_blocks[j+1];
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if ((char*)ptr + size == block->addr) {
            block->addr = ptr;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0 && (char*)allocator_ctx->free_blocks[i-1].addr + allocator_ctx->free_blocks[i-1].size == block->addr) {
                allocator_ctx->free_blocks[i-1].size += block->size;
                allocator_ctx->n_free_blocks--;
                for (int j = i; j < allocator_ctx->n_free_blocks; j++) {
                    allocator_ctx->free_blocks[j] = allocator_ctx->free_blocks[j+1];
                }
            }
            return;
        }
    }
    // otherwise, add a new block
    if (allocator_ctx->n_free_blocks < MAX_FREE_BLOCKS) {
        // insert the new block in the correct position to keep the array sorted
        int insert_pos = 0;
        while (insert_pos < allocator_ctx->n_free_blocks && allocator_ctx->free_blocks[insert_pos].addr < ptr) {
            insert_pos++;
        }
        // shift all blocks from insert_pos onward to make room for the new block
        for (int i = allocator_ctx->n_free_blocks; i > insert_pos; i--) {
            allocator_ctx->free_blocks[i] = allocator_ctx->free_blocks[i-1];
        }
        // insert the new block
        allocator_ctx->free_blocks[insert_pos].addr = ptr;
        allocator_ctx->free_blocks[insert_pos].size = size;
        allocator_ctx->n_free_blocks++;
    }
    else {
        GGML_ASSERT(!"out of free blocks");
    }
}

static void ggml_allocator_default_reset(struct ggml_backend_buffer * alloc) {
    struct ggml_allocator_default_context * ctx = (struct ggml_allocator_default_context *)alloc->context;
    ctx->n_free_blocks = 1; // TODO
    size_t align_offset = aligned_offset(ctx->data, 0, ctx->alignment);
    ctx->free_blocks[0].addr = (char *)ctx->data + align_offset;
    ctx->free_blocks[0].size = ctx->size - align_offset;
}

static const struct ggml_backend_buffer_interface ggml_allocator_default_interface = {
    /* .free_buffer    = */ ggml_allocator_default_free_buffer,
    /* .alloc_tensor   = */ ggml_allocator_default_alloc_tensor,
    /* .free_tensor    = */ ggml_allocator_default_free_tensor,
    /* .reset          = */ ggml_allocator_default_reset,
    /* .get_alloc_size = */ ggml_allocator_simple_get_alloc_size,
    /* .init_tensor    = */ NULL,
    /* .free_data      = */ NULL,
};

struct ggml_backend_buffer * ggml_allocator_default_init(void * data, size_t size, size_t alignment) {
    struct ggml_allocator_default_context * ctx = malloc(sizeof(struct ggml_allocator_default_context) /* + n_free_blocks * sizeof(struct free_block) */);
    ctx->data = data;
    ctx->size = size;
    ctx->alignment = alignment;
    ctx->n_free_blocks = 1; // TODO
    size_t align_offset = aligned_offset(data, 0, alignment);
    ctx->free_blocks[0].addr = (char *)data + align_offset;
    ctx->free_blocks[0].size = size - align_offset;

    struct ggml_backend_buffer * allocator = malloc(sizeof(struct ggml_backend_buffer));
    *allocator = (struct ggml_backend_buffer){
        /* .interface    = */ ggml_allocator_default_interface,
        /* .context      = */ ctx,
        /* .backend      = */ NULL,
        /* .backend_data = */ NULL,
        /* .measure      = */ false,
        /* .max_size     = */ 0,
    };
    return allocator;
}

//struct ggml_backend_buffer * ggml_allocator_default_init(void * data, size_t size, size_t alignment) {
//    return ggml_allocator_simple_init(data, size, alignment);
//}

// buffer

struct ggml_buffer * ggml_buffer_alloc(struct ggml_backend * backend, size_t size, size_t max_tensors) {
    struct ggml_buffer * buffer = malloc(sizeof(struct ggml_buffer));
    buffer->mem_size = ggml_tensor_overhead() * max_tensors;
    buffer->mem_buffer = malloc(buffer->mem_size);
    size += 128 * max_tensors; // alignment overhead
    buffer->backend_buffer = backend->interface.alloc_buffer(backend, size);
    buffer->backend_buffer->backend = backend;
    return buffer;
}

struct ggml_buffer * ggml_buffer_measure_alloc(struct ggml_backend * backend, size_t max_tensors) {
    struct ggml_buffer * buffer = ggml_buffer_alloc(backend, 0, max_tensors);
    buffer->backend_buffer->measure = true;
    return buffer;
}

void ggml_buffer_free(struct ggml_buffer * buffer) {
    ggml_backend_buffer_free(buffer->backend_buffer);
    free(buffer->mem_buffer);
    free(buffer);
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

    // printf("cpy tensor %s from %s to %s (%lu bytes)\n", src->name, ggml_backend_name(src->backend), ggml_backend_name(dst->backend), ggml_nbytes(src));

    if (src == dst) {
        return;
    }

    if (dst->backend->interface.cpy_tensor_from != NULL) {
        dst->backend->interface.cpy_tensor_from(dst->backend->context, src, dst);
    } else if (src->backend->interface.cpy_tensor_to != NULL) {
        src->backend->interface.cpy_tensor_to(src->backend->context, src, dst);
    } else {
        // not ideal, but shouldn't be hit when copying from/to CPU
        // TODO: print a performance warning in debug builds
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

// backend CPU

struct ggml_backend_cpu_context {
    int n_threads;
    void * work_data;
    size_t work_size;
};

static const char * ggml_backend_cpu_name(struct ggml_backend * backend) {
    return "CPU";

    UNUSED(backend);
}

static void ggml_backend_cpu_free(struct ggml_backend * backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    free(cpu_ctx->work_data);
    free(cpu_ctx);
    free(backend);
}

static const size_t TENSOR_ALIGNMENT = 64; // should be enough for AVX 512

static void ggml_backend_cpu_free_buffer(struct ggml_backend_buffer * alloc) {
    free(alloc->backend_data);
}

static struct ggml_backend_buffer * ggml_backend_cpu_alloc_buffer(struct ggml_backend * backend, size_t size) {
    void * data = malloc(size);

    struct ggml_backend_buffer * buffer = ggml_allocator_default_init(data, size, TENSOR_ALIGNMENT);
    buffer->interface.free_data = ggml_backend_cpu_free_buffer;
    buffer->backend_data = data;

    return buffer;

    UNUSED(backend);
}

static void ggml_backend_cpu_set_tensor_async(struct ggml_backend * backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy((char *)tensor->data + offset, data, size);

    UNUSED(backend);
}

static void ggml_backend_cpu_get_tensor_async(struct ggml_backend * backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy(data, (const char *)tensor->data + offset, size);

    UNUSED(backend);
}

static void ggml_backend_cpu_synchronize(struct ggml_backend * backend) {
    UNUSED(backend);
}

static void ggml_backend_cpu_cpy_tensor_from(struct ggml_backend * backend, struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));

    UNUSED(backend);
}

static void ggml_backend_cpu_cpy_tensor_to(struct ggml_backend * backend, struct ggml_tensor * src, struct ggml_tensor * dst) {
    // for a backend such as CUDA that can queue async calls, it is ok to do this asynchronously, but it may not be the case for other backends
    ggml_backend_tensor_set_async(dst, src->data, 0, ggml_nbytes(src));

    UNUSED(backend);
}

struct ggml_backend_cpu_plan {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_graph_plan_t ggml_backend_cpu_graph_plan_create(struct ggml_backend * backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_cpu_plan * cpu_plan = malloc(sizeof(struct ggml_backend_cpu_plan));

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads);
    cpu_plan->cgraph = *cgraph;

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = malloc(cpu_plan->cplan.work_size);
    }

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(struct ggml_backend * backend, ggml_graph_plan_t plan) {
    struct ggml_backend_cpu_plan * cpu_plan = (struct ggml_backend_cpu_plan *)plan;

    free(cpu_plan->cplan.work_data);
    free(cpu_plan);

    UNUSED(backend);
}

static void ggml_backend_cpu_graph_plan_compute(struct ggml_backend * backend, ggml_graph_plan_t plan) {
    struct ggml_backend_cpu_plan * cpu_plan = (struct ggml_backend_cpu_plan *)plan;

    ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    UNUSED(backend);
}

static void ggml_backend_cpu_graph_compute(struct ggml_backend * backend, struct ggml_cgraph * cgraph) {
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

static struct ggml_backend_interface cpu_backend_interface = {
    /* .get_name            = */ ggml_backend_cpu_name,
    /* .free                = */ ggml_backend_cpu_free,
    /* .alloc_buffer        = */ ggml_backend_cpu_alloc_buffer,
    /* .set_tensor_async    = */ ggml_backend_cpu_set_tensor_async,
    /* .get_tensor_async    = */ ggml_backend_cpu_get_tensor_async,
    /* .synchronize         = */ ggml_backend_cpu_synchronize,
    /* .cpy_tensor_from     = */ ggml_backend_cpu_cpy_tensor_from,
    /* .cpy_tensor_to       = */ ggml_backend_cpu_cpy_tensor_to,
    /* .graph_plan_create   = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free     = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_compute  = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute       = */ ggml_backend_cpu_graph_compute
};

struct ggml_backend * ggml_backend_cpu_init(void) {
    struct ggml_backend_cpu_context * ctx = malloc(sizeof(struct ggml_backend_cpu_context));
    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->work_data = NULL;
    ctx->work_size = 0;

    struct ggml_backend * cpu_backend = malloc(sizeof(struct ggml_backend));

    *cpu_backend = (struct ggml_backend) {
        /* .interface = */ cpu_backend_interface,
        /* .context   = */ ctx
    };
    return cpu_backend;
}

void ggml_backend_cpu_set_n_threads(struct ggml_backend * backend_cpu, int n_threads) {
    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

// splits

struct ggml_graph_splits ggml_graph_split_init(void) {
    struct ggml_graph_splits splits = {0};
    return splits;
}

// TODO: this can be removed after allocating the graphs in a ggml_context
void ggml_graph_splits_free(struct ggml_graph_splits * splits) {
    for (int i = 0; i < splits->n_splits; i++) {
        if (splits->splits[i].graph) {
            free(splits->splits[i].graph);
        }
    }
}

void ggml_graph_splits_add_n_va(struct ggml_graph_splits * splits, struct ggml_tensor *** inputs, struct ggml_context * ctx, const char * fmt, va_list args) {
    GGML_ASSERT(splits->n_splits < GGML_MAX_SPLITS);

    struct ggml_graph_split * split = &splits->splits[splits->n_splits];


    if (splits->n_splits == 0) {
        // always add the first split
        int i = 0;
        while (inputs[i] != NULL) {
            GGML_ASSERT(i < GGML_MAX_SPLIT_INPUTS);
            split->src_inputs[i] = *inputs[i];
            split->dst_inputs[i] = *inputs[i];
            i++;
        }
        split->src_inputs[i] = NULL;
        split->dst_inputs[i] = NULL;
        split->ctx = ctx;
    }
    // check if the split is on the same context as the previous one
    else if (splits->n_splits > 0 && splits->splits[splits->n_splits - 1].ctx == ctx) {
        // add to the previous split
        char name[GGML_MAX_NAME - 2];
        int n = vsnprintf(name, sizeof(name), fmt, args);
        char new_name[GGML_MAX_NAME];
        snprintf(new_name, sizeof(new_name), "%.*s,%s", GGML_MAX_NAME - n - 2, splits->splits[splits->n_splits - 1].name, name);
        strcpy(splits->splits[splits->n_splits - 1].name, new_name);
        return;
    } else {
        // add a new split
        int i = 0;
        while (inputs[i] != NULL) {
            GGML_ASSERT(i < GGML_MAX_SPLIT_INPUTS);
            split->src_inputs[i] = *inputs[i];
            split->dst_inputs[i] = ggml_dup_tensor(ctx, *inputs[i]);
            ggml_format_name(split->dst_inputs[i], "%s (split output)", split->src_inputs[i]->name);
            // TODO: maybe support different layings in ggml_backend_cpy_tensor instead
            for (int j = 0; j < GGML_MAX_DIMS; j++) {
                split->dst_inputs[i]->nb[j] = split->src_inputs[i]->nb[j];
            }
            ggml_set_name(split->dst_inputs[i], ggml_get_name(*inputs[i]));
            *inputs[i] = split->dst_inputs[i];
            i++;
        }
        split->src_inputs[i] = NULL;
        split->dst_inputs[i] = NULL;
        split->ctx = ctx;
    }

    vsnprintf(split->name, GGML_MAX_NAME, fmt, args);
    split->graph = NULL;
    splits->n_splits++;
}

void ggml_graph_splits_add_n(struct ggml_graph_splits * splits, struct ggml_tensor *** input, struct ggml_context * ctx, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ggml_graph_splits_add_n_va(splits, input, ctx, fmt, args);
    va_end(args);
}

void ggml_graph_splits_add(struct ggml_graph_splits * splits, struct ggml_tensor ** input, struct ggml_context * ctx, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ggml_graph_splits_add_n_va(splits, (struct ggml_tensor**[2]){ input, NULL }, ctx, fmt, args);
    va_end(args);
}

void ggml_graph_splits_build_forward(struct ggml_graph_splits * splits, struct ggml_tensor * output) {
    struct ggml_tensor *last_outputs[2] = { output, NULL };
    struct ggml_tensor ** outputs;

    for (int i = 0; i < splits->n_splits; i++) {
        struct ggml_graph_split * split = &splits->splits[i];

        if (i < splits->n_splits - 1) {
            outputs = splits->splits[i + 1].src_inputs;
        } else {
            outputs = last_outputs;
        }

        // build the graph
        // TODO: allocate graphs in context
        split->graph = (struct ggml_cgraph *) malloc(sizeof(struct ggml_cgraph));
        memset(split->graph, 0, sizeof(struct ggml_cgraph));
        for (int j = 0; outputs[j] != NULL; j++) {
            ggml_build_forward_expand(split->graph, outputs[j]);
        }

        for (int j = 1; j < split->graph->n_nodes; j++) {
            if (split->graph->nodes[j]->backend != split->graph->nodes[0]->backend) {
                fprintf(stderr, "split %s: node %s has different backend (%s) than the first node (%s)\n",
                    split->name, split->graph->nodes[j]->name,
                    ggml_backend_name(split->graph->nodes[j]->backend),
                    ggml_backend_name(split->graph->nodes[0]->backend));
            }
        }
        for (int j = 1; j < split->graph->n_leafs; j++) {
            if (split->graph->leafs[j]->backend != split->graph->leafs[0]->backend) {
                fprintf(stderr, "split %s: leaf %s has different backend (%s) than the first leaf (%s)\n",
                    split->name, split->graph->leafs[j]->name,
                    ggml_backend_name(split->graph->leafs[j]->backend),
                    ggml_backend_name(split->graph->leafs[0]->backend));
            }
        }
    }

    // close graphs
    for (int i = 0; i < splits->n_splits; i++) {
        struct ggml_graph_split * split = &splits->splits[i];
        ggml_graph_close(split->graph);
    }
}

void ggml_graph_splits_compute(struct ggml_graph_splits * splits) {
    uint64_t copy_us = 0;
    uint64_t compute_cpu_us = 0;
    uint64_t compute_gpu_us = 0;
    int n_nodes = 0;
    for (int i = 0; i < splits->n_splits; i++) {
        struct ggml_graph_split * split = &splits->splits[i];

        //printf("computing split %i (%s) on backend %s (%i nodes)\n", i, split->name, ggml_backend_name(split->dst_inputs[0]->backend), split->graph->n_nodes);

        // copy the input tensor to the backend
        uint64_t copy_start_us = ggml_time_us();
        for (int j = 0; split->src_inputs[j] != NULL; j++) {
            //printf("\tcopying tensor %d (%s) (%s -> %s) (%lu bytes)\n", j, split->src_inputs[j]->name, ggml_backend_name(split->src_inputs[j]->backend), ggml_backend_name(split->dst_inputs[j]->backend), ggml_nbytes(split->src_inputs[j]));
            //printf("%p %p\n", split->src_inputs[j], split->dst_inputs[j]);
            ggml_backend_tensor_copy(split->src_inputs[j], split->dst_inputs[j]);
        }
        // ggml_backend_synchronize(split->dst_inputs[0]->backend);
        copy_us += ggml_time_us() - copy_start_us;

#if 0
        char split_filename[GGML_MAX_NAME];
        snprintf(split_filename, GGML_MAX_NAME, "split_%i.dot", i);
        ggml_graph_dump_dot(split->graph, NULL, split_filename);
#endif
        uint64_t start = ggml_time_us();
        ggml_backend_graph_compute(split->dst_inputs[0]->backend, split->graph);
        //ggml_backend_synchronize(split->dst_inputs[0]->backend);
        uint64_t end = ggml_time_us();
        if (strcmp(ggml_backend_name(split->dst_inputs[0]->backend), "CPU") == 0) {
            compute_cpu_us += end - start;
        } else {
            compute_gpu_us += end - start;
        }

        n_nodes += split->graph->n_nodes;
    }

    //printf("splits: %d, nodes: %d, copy: %.2fms, compute_cpu: %.2fms, compute_gpu: %.2fms\n", splits->n_splits, n_nodes, copy_us / 1000.0, compute_cpu_us / 1000.0, compute_gpu_us / 1000.0);
    //exit(0);
}

void ggml_graph_allocate_tensors(struct ggml_cgraph * graph, struct ggml_context * ctx) {
    ggml_graph_allocate_tensors_n(&graph, 1, ctx);
}

static bool ggml_is_view(struct ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_CPY;
}

struct ggml_tensor * view_parent(struct ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            return t->src[0];
        case GGML_OP_CPY:
            return t->src[1];
        default:
            return NULL;
    }
}

#if 0
void ggml_graph_allocate_tensors_n(struct ggml_cgraph ** graphs, int n_graphs, struct ggml_context * ctx) {
    struct ggml_buffer * buffer = ggml_get_buffer(ctx);
    for (int i = 0; i < n_graphs; i++) {
        struct ggml_cgraph * graph = graphs[i];
        for (int j = 0; j < graph->n_leafs; j++) {
            struct ggml_tensor * leaf = graph->leafs[j];
            GGML_ASSERT(leaf->backend == buffer->backend_buffer->backend);
            if (leaf->data == NULL) {
                //printf("allocating leaf %s\n", leaf->name);
                ggml_backend_buffer_tensor_alloc(buffer->backend_buffer, leaf);
            }
        }

        for (int j = 0; j < graph->n_nodes; j++) {
            struct ggml_tensor * node = graph->nodes[j];
            GGML_ASSERT(node->backend == buffer->backend_buffer->backend);
            if (node->data == NULL) {
                if (ggml_is_view(node)) {
                    size_t offset;
                    memcpy(&offset, node->op_params, sizeof(size_t));
                    switch(node->op) {
                        case GGML_OP_VIEW:
                            //printf("view %s (%s), offset %zu\n", node->name, ggml_op_name(node->op), offset);
                            node->data = (char *) node->src[0]->data + offset;
                            break;
                        case GGML_OP_RESHAPE:
                        case GGML_OP_TRANSPOSE:
                        case GGML_OP_PERMUTE:
                            node->data = node->src[0]->data;
                            break;
                        case GGML_OP_CPY:
                            node->data = node->src[1]->data;
                            break;
                        default:
                            GGML_ASSERT(!"unknown view op");
                            break;
                    }
                } else {
                    //printf("allocating tensor %s\n", node->name);
                    ggml_backend_buffer_tensor_alloc(buffer->backend_buffer, node);
                }
            }
        }
    }
    //printf("\n\n\n");
}
#else

void allocate_node(struct ggml_buffer * buffer, struct ggml_tensor * node) {
    if (node->data == NULL) {
        if (ggml_is_view(node)) {
            size_t offset;
            switch(node->op) {
                case GGML_OP_VIEW:
                    memcpy(&offset, node->op_params, sizeof(size_t));
                    node->data = (char *) node->src[0]->data + offset;
                    break;
                case GGML_OP_RESHAPE:
                case GGML_OP_TRANSPOSE:
                case GGML_OP_PERMUTE:
                    node->data = node->src[0]->data;
                    break;
                case GGML_OP_CPY:
                    node->data = node->src[1]->data;
                    break;
                default:
                    GGML_ASSERT(!"unknown view op");
                    break;
            }
        } else {
            //printf("allocating tensor %s\n", node->name);
            ggml_backend_buffer_tensor_alloc(buffer->backend_buffer, node);
        }
    }
}

void ggml_graph_allocate_tensors_n(struct ggml_cgraph ** graphs, int n_graphs, struct ggml_context * ctx) {
    struct ggml_buffer * buffer = ggml_get_buffer(ctx);

    // reset counters
    for (int g = 0; g < n_graphs; g++) {
        struct ggml_cgraph * gf = graphs[g];
        for (int i = 0; i < gf->n_nodes; i++) {
            struct ggml_tensor * node = gf->nodes[i];
            node->n_children = 0;
            node->n_views = 0;
        }
        for (int i = 0; i < gf->n_leafs; i++) {
            struct ggml_tensor * leaf = gf->leafs[i];
            leaf->n_children = 0;
            leaf->n_views = 0;
        }
    }

    // count number of children and views
    for (int g = 0; g < n_graphs; g++) {
        struct ggml_cgraph * gf = graphs[g];
        for (int i = 0; i < gf->n_nodes; i++) {
            struct ggml_tensor * node = gf->nodes[i];
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * parent = node->src[j];
                if (parent == NULL) {
                    break;
                }
                parent->n_children += 1;
                if (ggml_is_view(parent)) {
                    struct ggml_tensor * ancestor = parent;
                    do {
                        ancestor = view_parent(ancestor);
                    } while (ggml_is_view(ancestor));
                    ancestor->n_views += 1;
                }
            }
        }
    }

    // allocate tensors
    for (int g = 0; g < n_graphs; g++) {
        struct ggml_cgraph * gf = graphs[g];
        for (int i = 0; i < gf->n_nodes; i++) {
            struct ggml_tensor * node = gf->nodes[i];
            bool is_view = ggml_is_view(node);

            // allocate parents (leafs)
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * parent = node->src[j];
                if (parent == NULL) {
                    break;
                }
                allocate_node(buffer, parent);
            }

            // allocate node
            allocate_node(buffer, node);

            // update parents
            if (is_view) {
                struct ggml_tensor * ancestor = node;
                do {
                    ancestor = view_parent(ancestor);
                } while (ggml_is_view(ancestor));
                ancestor->n_views -= 1;
                if (ancestor->n_views == 0) {
                    ggml_backend_buffer_tensor_free(buffer->backend_buffer, ancestor);
                }
            } else {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * parent = node->src[j];
                    if (parent == NULL) {
                        break;
                    }
                    if (ggml_is_view(parent)) {
                        struct ggml_tensor * ancestor = parent;
                        do {
                            ancestor = view_parent(ancestor);
                        } while (ggml_is_view(ancestor));
                        ancestor->n_views -= 1;
                        if (ancestor->n_views == 0) {
                            ggml_backend_buffer_tensor_free(buffer->backend_buffer, ancestor);
                        }
                    }
                    else {
                        parent->n_children -= 1;
                        if (parent->n_children == 0) {
                            // free parent
                            ggml_backend_buffer_tensor_free(buffer->backend_buffer, parent);
                        }
                    }
                }
            }
        }
    }
}
#endif

void ggml_graph_splits_allocate_tensors(struct ggml_graph_splits * splits) {
    bool visited[GGML_MAX_SPLITS] = {false};
    for (int i = 0; i < splits->n_splits; i++) {
        if (!visited[i]) {
            struct ggml_graph_split * split = &splits->splits[i];
            struct ggml_context * ctx = split->ctx;
            struct ggml_cgraph * backend_graphs[GGML_MAX_SPLITS];
            int num_graphs = 0;
            for (int j = i; j < splits->n_splits; j++) {
                if (splits->splits[j].ctx == ctx) {
                    backend_graphs[num_graphs] = splits->splits[j].graph;
                    visited[j] = true;
                    num_graphs++;
                    // TODO: need to ensure that the output tensors are never freed
                    // maybe this can be done automatically in ggml_graph_allocate_tensors_n by assuming that n_childs == 0 => output tensor
                }
            }
            //printf("allocating tensors for %s [%d graphs/%d splits]\n", ggml_backend_name(ggml_get_buffer(ctx)->backend_buffer->backend), num_graphs, splits->n_splits);
            ggml_graph_allocate_tensors_n(backend_graphs, num_graphs, ctx);
        }
    }
    //printf("done allocating tensors\n");
}

