#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

    typedef void * ggml_graph_plan_t;
    typedef void * ggml_backend_context_t;
    typedef void * ggml_backend_buffer_t;
    struct ggml_backend;

    // buffers have space for the tensor structs in host memory, and tensor data in backend-specific memory
    struct ggml_buffer {
        // host memory
        size_t mem_size;
        void * mem_buffer;

        // tensor data
        struct ggml_backend * backend;
        ggml_backend_buffer_t backend_buffer; // backend-specific data
    };

    struct ggml_backend_interface {
        const char * (*get_name)(ggml_backend_context_t ctx);

        void (*free_context)(ggml_backend_context_t ctx);

        // buffers
        ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_context_t ctx, size_t size);
        void                  (*free_buffer) (ggml_backend_context_t ctx, ggml_backend_buffer_t buffer);
        void                  (*reset_buffer)(ggml_backend_context_t ctx, ggml_backend_buffer_t buffer);
        void                  (*alloc_tensor)(ggml_backend_context_t ctx, ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);

        // TODO: pinned buffers for faster transfers between host and device

        // tensor data access
        // these functions can be asynchronous. helper functions are provided for synchronous access that automatically call synchronize
        void (*set_tensor_async)(ggml_backend_context_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(ggml_backend_context_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
        void (*synchronize)(ggml_backend_context_t ctx);

        // (optional) copy tensor between different backends, allow for single-copy tranfers
        void (*cpy_tensor_from)(ggml_backend_context_t ctx, struct ggml_tensor * src, struct ggml_tensor * dst);
        void (*cpy_tensor_to)  (ggml_backend_context_t ctx, struct ggml_tensor * src, struct ggml_tensor * dst);


        // compute graph with a plan
        ggml_graph_plan_t (*graph_plan_create) (ggml_backend_context_t ctx, struct ggml_cgraph * cgraph);
        void              (*graph_plan_free)   (ggml_backend_context_t ctx, ggml_graph_plan_t plan);
        void              (*graph_plan_compute)(ggml_backend_context_t ctx, ggml_graph_plan_t plan);

        // compute graph without a plan
        void              (*graph_compute)     (ggml_backend_context_t ctx, struct ggml_cgraph * cgraph);

        // check if a backend supports a given operation
        // this could be used to fallback automatically to the CPU backend if a backend doesn't support an operation
        // bool (*supports_op)(ggml_backend_context_t ctx, struct ggml_tensor * op);
    };

    struct ggml_backend {
        struct ggml_backend_interface * interface;
        ggml_backend_context_t context;
    };

    // backend helper functions
    static inline const char * ggml_backend_name(struct ggml_backend * backend) { return backend->interface->get_name(backend->context); }
    static inline void ggml_backend_free_context(struct ggml_backend * backend) { backend->interface->free_context(backend->context); }
    static inline void ggml_backend_set_tensor_async(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) { tensor->backend->interface->set_tensor_async(tensor->backend->context, tensor, data, offset, size); }
    static inline void ggml_backend_get_tensor_async(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) { tensor->backend->interface->get_tensor_async(tensor->backend->context, tensor, data, offset, size); }
    static inline void ggml_backend_set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) { tensor->backend->interface->set_tensor_async(tensor->backend->context, tensor, data, offset, size); tensor->backend->interface->synchronize(tensor->backend->context); }
    static inline void ggml_backend_get_tensor(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) { tensor->backend->interface->get_tensor_async(tensor->backend->context, tensor, data, offset, size); tensor->backend->interface->synchronize(tensor->backend->context); }
    static inline void ggml_backend_synchronize(struct ggml_backend * backend) { backend->interface->synchronize(backend->context); }
    static inline ggml_graph_plan_t ggml_backend_graph_plan_create(struct ggml_backend * backend, struct ggml_cgraph * cgraph) { return backend->interface->graph_plan_create(backend->context, cgraph); }
    static inline void ggml_backend_graph_plan_free(struct ggml_backend * backend, ggml_graph_plan_t plan) { backend->interface->graph_plan_free(backend->context, plan); }
    static inline void ggml_backend_graph_plan_compute(struct ggml_backend * backend, ggml_graph_plan_t plan) { backend->interface->graph_plan_compute(backend->context, plan); }
    static inline void ggml_backend_graph_compute(struct ggml_backend * backend, struct ggml_cgraph * cgraph) { backend->interface->graph_compute(backend->context, cgraph); }

    // buffer and tensor allocation
    GGML_API struct ggml_buffer ggml_backend_alloc_buffer(struct ggml_backend * backend, size_t size, size_t max_tensors);
    GGML_API void               ggml_backend_free_buffer(struct ggml_buffer * buffer);
    static inline void          ggml_backend_reset_buffer(struct ggml_buffer * buffer) { buffer->backend->interface->reset_buffer(buffer->backend->context, buffer->backend_buffer); }
    static inline void          ggml_backend_alloc_tensor(struct ggml_buffer * buffer, struct ggml_tensor * tensor) { buffer->backend->interface->alloc_tensor(buffer->backend->context, buffer->backend_buffer, tensor); }

    // tensor copy between different backends
    GGML_API void ggml_backend_cpy_tensor(struct ggml_tensor * dst, struct ggml_tensor * src);

    // CPU backend
    GGML_API struct ggml_backend ggml_backend_cpu_init(void);
    GGML_API void ggml_backend_cpu_set_n_threads(struct ggml_backend * backend_cpu, int n_threads);

    ///////////////////////////

    // graph splitting
    #define GGML_MAX_SPLITS 200
    #define GGML_MAX_SPLIT_INPUTS 4

    struct ggml_graph_split {
        char name[GGML_MAX_NAME];
        struct ggml_tensor * src_inputs[GGML_MAX_SPLIT_INPUTS + 1];
        struct ggml_tensor * dst_inputs[GGML_MAX_SPLIT_INPUTS + 1];
        struct ggml_cgraph * graph;
    };

    // TODO: this shouldn't be fixed size, allocate from ggml_context
    struct ggml_graph_splits {
        int n_splits;
        struct ggml_graph_split splits[GGML_MAX_SPLITS];
    };

    // TODO: allocate in ggml_context
    struct ggml_graph_splits ggml_graph_split_init(void);
    // this won't be needed once we can allocate graphs from a ggml_context
    GGML_API void ggml_graph_splits_free(struct ggml_graph_splits * splits);

    // add a split to the graph - single and multiple inputs versions
    GGML_API void ggml_graph_splits_add(struct ggml_graph_splits * splits, struct ggml_tensor ** input, struct ggml_context * ctx, const char * fmt, ...);
    GGML_API void ggml_graph_splits_add_n(struct ggml_graph_splits * splits, struct ggml_tensor *** inputs, struct ggml_context * ctx, const char * fmt, ...);

    // build graphs for all splits
    GGML_API void ggml_graph_splits_build_forward(struct ggml_graph_splits * splits, struct ggml_tensor * output);

    // compute
    GGML_API void ggml_graph_splits_compute(struct ggml_graph_splits * splits);

#ifdef  __cplusplus
}
#endif
