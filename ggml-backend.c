#include "ggml-backend.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNUSED(x) (void)(x)

// backend buffer

struct ggml_buffer ggml_backend_alloc_buffer(struct ggml_backend * backend, size_t size, size_t max_tensors) {
    struct ggml_buffer buffer;
    buffer.mem_size = ggml_tensor_overhead() * max_tensors;
    buffer.mem_buffer = malloc(buffer.mem_size);
    buffer.backend = backend;
    // size += 128 * max_tensors; // alignment overhead
    buffer.backend_buffer = backend->interface->alloc_buffer(backend->context, size);
    return buffer;
}

void ggml_backend_free_buffer(struct ggml_buffer * buffer) {
    struct ggml_backend * backend = buffer->backend;
    backend->interface->free_buffer(backend->context, buffer->backend_buffer);
    free(buffer->mem_buffer);
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

void ggml_backend_cpy_tensor(struct ggml_tensor * dst, struct ggml_tensor * src) {
    //printf("src: %s ne: [%d %d %d %d] nb: [%d %d %d %d]\n", src->name, (int)src->ne[0], (int)src->ne[1], (int)src->ne[2], (int)src->ne[3], (int)src->nb[0], (int)src->nb[1], (int)src->nb[2], (int)src->nb[3]);
    //printf("dst: %s ne: [%d %d %d %d] nb: [%d %d %d %d]\n", dst->name, (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2], (int)dst->ne[3], (int)dst->nb[0], (int)dst->nb[1], (int)dst->nb[2], (int)dst->nb[3]);
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    // printf("cpy tensor %s from %s to %s (%lu bytes)\n", src->name, ggml_backend_name(src->backend), ggml_backend_name(dst->backend), ggml_nbytes(src));

    if (src == dst) {
        return;
    }

    if (dst->backend->interface->cpy_tensor_from != NULL) {
        dst->backend->interface->cpy_tensor_from(dst->backend->context, src, dst);
    } else if (src->backend->interface->cpy_tensor_to != NULL) {
        src->backend->interface->cpy_tensor_to(src->backend->context, src, dst);
    } else {
        // not ideal, but shouldn't be hit when copying from/to CPU
        // TODO: print a performance warning in debug builds
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_get_tensor(src, data, 0, nbytes);
        ggml_backend_set_tensor(dst, data, 0, nbytes);
        free(data);
    }
}

// backend CPU

struct ggml_backend_cpu_context {
    int n_threads;
    void * work_data;
    size_t work_size;
};

static const char * ggml_backend_cpu_name(ggml_backend_context_t ctx) {
    return "CPU";

    UNUSED(ctx);
}

static void ggml_backend_cpu_free_context(ggml_backend_context_t ctx) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)ctx;
    free(cpu_ctx->work_data);
    free(ctx);
}

struct cpu_backend_buffer {
    void * data;
    size_t offset;
    size_t size;
};

static const size_t TENSOR_ALIGNMENT = 64; // should be enough for AVX 512

static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

static ggml_backend_buffer_t ggml_backend_cpu_alloc_buffer(ggml_backend_context_t ctx, size_t size) {
    struct cpu_backend_buffer * buffer = malloc(sizeof(struct cpu_backend_buffer));
    buffer->data = malloc(size);
    buffer->offset = aligned_offset(buffer->data, 0, TENSOR_ALIGNMENT);
    buffer->size = size;
    return buffer;

    UNUSED(ctx);
}

static void ggml_backend_cpu_free_buffer(ggml_backend_context_t ctx, ggml_backend_buffer_t buffer) {
    struct cpu_backend_buffer * cpu_buffer = (struct cpu_backend_buffer *)buffer;
    free(cpu_buffer->data);
    free(cpu_buffer);

    UNUSED(ctx);
}

static void ggml_backend_cpu_reset_buffer(ggml_backend_context_t ctx, ggml_backend_buffer_t buffer) {
    struct cpu_backend_buffer * cpu_buffer = (struct cpu_backend_buffer *)buffer;
    cpu_buffer->offset = aligned_offset(cpu_buffer->data, 0, TENSOR_ALIGNMENT);

    UNUSED(ctx);
}

static void ggml_backend_cpu_alloc_tensor(ggml_backend_context_t ctx, ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    struct cpu_backend_buffer * cpu_buffer = (struct cpu_backend_buffer *)buffer;

    // TODO: make this error recoverable
    if (cpu_buffer->offset + ggml_nbytes(tensor) > cpu_buffer->size) {
        fprintf(stderr, "%s: not enough space in the buffer (needed %zu, available %zu)\n",
                __func__, ggml_nbytes(tensor), cpu_buffer->size - cpu_buffer->offset);
        GGML_ASSERT(false);
    }

    tensor->data = (char*)cpu_buffer->data + cpu_buffer->offset;
    cpu_buffer->offset = aligned_offset(cpu_buffer->data, cpu_buffer->offset + ggml_nbytes(tensor), TENSOR_ALIGNMENT);

    UNUSED(ctx);
}

static void ggml_backend_cpu_set_tensor_async(ggml_backend_context_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy((char *)tensor->data + offset, data, size);

    UNUSED(ctx);
}

static void ggml_backend_cpu_get_tensor_async(ggml_backend_context_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy(data, (const char *)tensor->data + offset, size);

    UNUSED(ctx);
}

static void ggml_backend_cpu_synchronize(ggml_backend_context_t ctx) {
    UNUSED(ctx);
}

static void ggml_backend_cpu_cpy_tensor_from(ggml_backend_context_t ctx, struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_get_tensor(src, dst->data, 0, ggml_nbytes(src));

    UNUSED(ctx);
}

static void ggml_backend_cpu_cpy_tensor_to(ggml_backend_context_t ctx, struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_set_tensor(dst, src->data, 0, ggml_nbytes(src));

    UNUSED(ctx);
}

struct ggml_backend_cpu_plan {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

static ggml_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_context_t ctx, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)ctx;

    struct ggml_backend_cpu_plan * cpu_plan = malloc(sizeof(struct ggml_backend_cpu_plan));

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads);
    cpu_plan->cgraph = *cgraph;

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = malloc(cpu_plan->cplan.work_size);
    }

    return cpu_plan;
}

static void ggml_backend_cpu_graph_plan_free(ggml_backend_context_t ctx, ggml_graph_plan_t plan) {
    struct ggml_backend_cpu_plan * cpu_plan = (struct ggml_backend_cpu_plan *)plan;

    free(cpu_plan->cplan.work_data);
    free(cpu_plan);

    UNUSED(ctx);
}

static void ggml_backend_cpu_graph_plan_compute(ggml_backend_context_t ctx, ggml_graph_plan_t plan) {
    struct ggml_backend_cpu_plan * cpu_plan = (struct ggml_backend_cpu_plan *)plan;

    ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    UNUSED(ctx);
}

static void ggml_backend_cpu_graph_compute(ggml_backend_context_t ctx, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)ctx;

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
    /* .free_context        = */ ggml_backend_cpu_free_context,
    /* .alloc_buffer        = */ ggml_backend_cpu_alloc_buffer,
    /* .free_buffer         = */ ggml_backend_cpu_free_buffer,
    /* .reset_buffer        = */ ggml_backend_cpu_reset_buffer,
    /* .alloc_tensor        = */ ggml_backend_cpu_alloc_tensor,
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

struct ggml_backend ggml_backend_cpu_init(void) {
    struct ggml_backend_cpu_context * ctx = malloc(sizeof(struct ggml_backend_cpu_context));
    ctx->n_threads = GGML_DEFAULT_N_THREADS;
    ctx->work_data = NULL;
    ctx->work_size = 0;

    struct ggml_backend cpu_backend = {
        /* .interface = */ &cpu_backend_interface,
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

    if ((*inputs[0])->backend == ggml_get_ctx_backend(ctx)) {
        if (splits->n_splits > 0) {
            char name[GGML_MAX_NAME - 1]; // silence -Wformat-truncation
            vsnprintf(name, sizeof(name), fmt, args);
            char new_name[GGML_MAX_NAME];
            snprintf(new_name, sizeof(new_name), "%s,%s", splits->splits[splits->n_splits - 1].name, name);
            strcpy(splits->splits[splits->n_splits - 1].name, new_name);
            return;
        }
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
        // *split->graph = ggml_build_forward_range(output, split->input);
        // *split->graph = ggml_build_forward(output);
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
            if (split->src_inputs[j] != split->dst_inputs[j]) {
                //printf("\tcopying tensor %d (%s) (%lu bytes)\n", j, split->src_inputs[j]->name, ggml_nbytes(split->src_inputs[j]));
                ggml_backend_cpy_tensor(split->dst_inputs[j], split->src_inputs[j]);
            }
        }
        ggml_backend_synchronize(split->dst_inputs[0]->backend);
        copy_us += ggml_time_us() - copy_start_us;

#if 0
        char split_filename[GGML_MAX_NAME];
        snprintf(split_filename, GGML_MAX_NAME, "split_%i.dot", i);
        ggml_graph_dump_dot(split->graph, NULL, split_filename);
#endif
        uint64_t start = ggml_time_us();
        ggml_backend_graph_compute(split->dst_inputs[0]->backend, split->graph);
        ggml_backend_synchronize(split->dst_inputs[0]->backend);
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
