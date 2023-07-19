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

static inline size_t ggml_backend_buffer_get_alloc_size(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) { return alloc->interface.get_alloc_size(alloc, tensor); }
static inline void   ggml_backend_buffer_init_tensor(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) { alloc->interface.init_tensor(alloc, tensor); }


void ggml_backend_buffer_free(struct ggml_backend_buffer * alloc) {
    alloc->interface.free_buffer(alloc);
    free(alloc);
}

// backend buffer allocator - simple

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

static void ggml_allocator_simple_alloc_tensor(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) {
    struct ggml_allocator_simple_context * context = (struct ggml_allocator_simple_context *)alloc->context;
    size_t size = ggml_backend_buffer_get_alloc_size(alloc, tensor);
    if (context->offset + size > context->size) {
        fprintf(stderr, "%s: not enough space in the buffer (needed %zu, available %zu)\n",
                __func__, size, context->size - context->offset);
        GGML_ASSERT(!"not enough space in the buffer");
        return;
    }
    void * ptr = (char*)context->data + context->offset;
    context->offset = aligned_offset(context->data, context->offset + size, context->alignment);
    tensor->data = ptr;
    if (alloc->interface.init_tensor) {
        alloc->interface.init_tensor(alloc, tensor);
    }
}

static void ggml_allocator_simple_free_tensor(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) {
    GGML_ASSERT(!"ggml_simple_allocator cannot free individual tensors");

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

struct ggml_backend_buffer * ggml_allocator_simple_init(void * data, size_t size, size_t alignment) {
    struct ggml_allocator_simple_context * ctx = malloc(sizeof(struct ggml_allocator_simple_context));
    ctx->data = data;
    ctx->size = size;
    ctx->offset = aligned_offset(data, 0, alignment);
    ctx->alignment = alignment;

    struct ggml_backend_buffer * allocator = malloc(sizeof(struct ggml_backend_buffer));
    *allocator = (struct ggml_backend_buffer){
        /* .interface    = */ ggml_allocator_simple_interface,
        /* .context      = */ ctx,
        /* .backend_data = */ NULL,
    };
    return allocator;
}

// buffer

struct ggml_buffer * ggml_buffer_alloc(struct ggml_backend * backend, size_t size, size_t max_tensors) {
    struct ggml_buffer * buffer = malloc(sizeof(struct ggml_buffer));
    buffer->mem_size = ggml_tensor_overhead() * max_tensors;
    buffer->mem_buffer = malloc(buffer->mem_size);
    buffer->backend = backend;
    size += 128 * max_tensors; // alignment overhead
    buffer->backend_buffer = backend->interface.alloc_buffer(backend, size);
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

    struct ggml_backend_buffer * buffer = ggml_allocator_simple_init(data, size, TENSOR_ALIGNMENT);
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
        /* .interface     = */ cpu_backend_interface,
        /* .context       = */ ctx,
        /* .is_ram_shared = */ true,
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

    // check if the split is on the same backend as the previous one
    // FIXME: need to check all the inputs
    if ((*inputs[0])->backend == ggml_get_ctx_backend(ctx)) {
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
        } else {
            // add to the previous split
            char name[GGML_MAX_NAME - 2];
            int n = vsnprintf(name, sizeof(name), fmt, args);
            char new_name[GGML_MAX_NAME];
            snprintf(new_name, sizeof(new_name), "%.*s,%s", GGML_MAX_NAME - n - 2, splits->splits[splits->n_splits - 1].name, name);
            strcpy(splits->splits[splits->n_splits - 1].name, new_name);
            return;
        }
    } else {
        // add a new split
        int i = 0;
        while (inputs[i] != NULL) {
            GGML_ASSERT(i < GGML_MAX_SPLIT_INPUTS);
            split->src_inputs[i] = *inputs[i];
            split->dst_inputs[i] = ggml_dup_tensor(ctx, *inputs[i]);
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
            //printf("\tcopying tensor %d (%s) (%lu bytes)\n", j, split->src_inputs[j]->name, ggml_nbytes(split->src_inputs[j]));
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

#if 0
// default allocator
struct free_block {
    void * addr;
    size_t size;
};

struct ggml_backend_default_allocator_context {
    void * data;
    size_t alignment;
    int n_free_blocks;
    struct free_block free_blocks[];
};

void ggml_backend_default_allocator_free_context(ggml_allocator_context_t ctx) {
    struct ggml_backend_default_allocator_context * allocator_ctx = ctx;
    free(allocator_ctx);
}

ggml_allocator_context_t ggml_backend_default_allocator_context(void * data, size_t size, size_t alignment, int n_free_blocks) {
    struct ggml_backend_default_allocator_context * ctx = malloc(sizeof(struct ggml_backend_default_allocator_context) + n_free_blocks * sizeof(struct free_block));
    ctx->data = data;
    ctx->alignment = alignment;
    ctx->n_free_blocks = 1;
    size_t align_offset = align_offset(data, alignment);
    ctx->free_blocks[0].addr = (char *)data + align_offset;
    ctx->free_blocks[0].size = size - align_offset;
    return ctx;
}

void * ggml_backend_default_allocator_alloc(ggml_allocator_context_t ctx, size_t size) {
    struct ggml_backend_default_allocator_context * allocator_ctx = ctx;
    size = align_size(size, allocator_ctx->alignment);
    // find a free block
    for (int i = 0; i < allocator_ctx->n_free_blocks; i++) {
        struct free_block * block = &allocator_ctx->free_blocks[i];
        if (block->size >= size) {
            void * addr = block->addr;
            block->addr += size;
            block->size -= size;
            if (block->size == 0) {
                // remove block if empty
                allocator_ctx->n_free_blocks--;
                for (int j = i; j < allocator_ctx->n_free_blocks; j++) {
                    allocator_ctx->free_blocks[j] = allocator_ctx->free_blocks[j+1];
                }
            }
            return addr;
        }
    }
    return NULL;
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
void ggml_backend_default_allocator_free(ggml_allocator_context_t ctx, void * ptr, size_t size) {
    struct ggml_backend_default_allocator_context * allocator_ctx = ctx;
    size = align_size(size, allocator_ctx->alignment);
    // see if we can merge with an existing block
    for (int i = 0; i < allocator_ctx->n_free_blocks; i++) {
        struct free_block * block = &allocator_ctx->free_blocks[i];
        // check if ptr is at the end of the block
        if (block->addr + block->size == ptr) {
            block->size += size;
            // check if we can merge with the next block
            if (i < allocator_ctx->n_free_blocks - 1 && block->addr + block->size == allocator_ctx->free_blocks[i+1].addr) {
                block->size += allocator_ctx->free_blocks[i+1].size;
                allocator_ctx->n_free_blocks--;
                for (int j = i+1; j < allocator_ctx->n_free_blocks; j++) {
                    allocator_ctx->free_blocks[j] = allocator_ctx->free_blocks[j+1];
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if (ptr + size == block->addr) {
            block->addr = ptr;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0 && allocator_ctx->free_blocks[i-1].addr + allocator_ctx->free_blocks[i-1].size == block->addr) {
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

static bool ggml_is_view(struct ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}


NOTE: id can be n_leaf OR n_node instead, we can determine the type by checking if the node is a leaf or not

void allocate_graph(struct ggml_cgraph * gf, struct ggml_buffer * buffer) {
    int node_children_count[GGML_MAX_NODES*2];
    int node_view_count[GGML_MAX_NODES*2];
    memset(node_children_count, 0, sizeof(int) * (gf->n_nodes + gf->n_leafs));
    memset(node_view_count, 0, sizeof(int) * (gf->n_nodes + gf->n_leafs));

    // count number of children and views
    for (int i = 0; i < gf->n_nodes; i++) {
        struct ggml_tensor * node = gf->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                break;
            }
            // todo: ....
            node_children_count[parent->id] += 1;
            if (ggml_is_view(parent)) {
                struct ggml_tensor * ancestor = parent;
                do {
                    node_view_count[ancestor->id] += 1;
                    ancestor = ancestor->src[0];
                } while (ggml_is_view(ancestor));
            }
        }
    }

    // allocate tensors
    for (int i = 0; i < gf->n_nodes; i++) {
        struct ggml_tensor * node = gf->nodes[i];
        bool is_view = ggml_is_view(node);
        if (is_view) {
            // allocate view accordingly to the OP
            node->data = node->src[0]->data; // + offset
            struct ggml_tensor * ancestor = node->src[0];
            while (ggml_is_view(ancestor)) {
                ancestor = ancestor->src[0];
            }
            node_view_count[ancestor->id] -= 1;
        } else {
            if (node->data == NULL) {
                // allocate tensor
                // TODO: if last children and size == parent.size, then reuse parent tensor (auto in-place)
                // may need a list of ops that can be in-place
                ggml_backend_alloc_tensor(buffer, node);
            }
        }

        // update parents
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                break;
            }
            if (is_view) {
                node_view_count[parent->id] -= 1;
            }
            node_children_count[parent->id] -= 1;
            if (node_children_count[parent->id] == 0 && node_view_count[parent->id] == 0) {
                // free parent
                ggml_backend_free_tensor(buffer, parent);
            }
        }
    }
}

#endif
