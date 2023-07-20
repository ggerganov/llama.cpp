#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif
    struct ggml_backend;


    // backend buffers
    typedef void * ggml_buffer_context_t;
    struct ggml_backend_buffer;

    struct ggml_backend_buffer_interface {
        // allocator functions
        void   (*free_buffer)   (struct ggml_backend_buffer * alloc);
        void   (*alloc_tensor)  (struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor);
        void   (*free_tensor)   (struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor);
        void   (*reset)         (struct ggml_backend_buffer * alloc);
        // functions overriden by the backend
        size_t (*get_alloc_size)(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor); // pre-allocation callback
        void   (*init_tensor)   (struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor); // post-allocation callback
        void   (*free_data)     (struct ggml_backend_buffer * alloc); // free backend-specific data // TODO: better name
    };

    struct ggml_backend_buffer {
        struct ggml_backend_buffer_interface interface;
        ggml_buffer_context_t context;
        void * backend_data;
    };

    // backend buffer helper functions
    GGML_API      void ggml_backend_buffer_free(struct ggml_backend_buffer * alloc);
    static inline void ggml_backend_buffer_tensor_alloc(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) { alloc->interface.alloc_tensor(alloc, tensor); }
    static inline void ggml_backend_buffer_free_tensor(struct ggml_backend_buffer * alloc, struct ggml_tensor * tensor) { alloc->interface.free_tensor(alloc, tensor); }
    static inline void ggml_backend_buffer_reset(struct ggml_backend_buffer * alloc) { alloc->interface.reset(alloc); }

    // default buffer allocators
    // simple buffer allocator: cannot free tensors, good for weights and small contexts
    // default buffer allocator: can free tensors, good for compute contexts
    GGML_API struct ggml_backend_buffer * ggml_allocator_simple_init(void * data, size_t size, size_t alignment);
    GGML_API struct ggml_backend_buffer * ggml_allocator_default_init(void * data, size_t size, size_t alignment, int max_free_blocks);

    // buffer

    // buffers have space for the tensor structs in host memory, and tensor data in backend-specific memory
    struct ggml_buffer {
        // host memory
        size_t mem_size;
        void * mem_buffer;

        // tensor data
        struct ggml_backend * backend;
        struct ggml_backend_buffer * backend_buffer;
    };

    GGML_API struct ggml_buffer * ggml_buffer_alloc(struct ggml_backend * backend, size_t size, size_t max_tensors);
    GGML_API void ggml_buffer_free(struct ggml_buffer * buffer);

    // backend
    typedef void * ggml_backend_context_t;
    typedef void * ggml_graph_plan_t;

    struct ggml_backend_interface {
        const char * (*get_name)(struct ggml_backend * backend);

        void (*free)(struct ggml_backend * backend);

        // buffer allocation
        struct ggml_backend_buffer * (*alloc_buffer)(struct ggml_backend * backend, size_t size);

        // tensor data access
        // these functions can be asynchronous. helper functions are provided for synchronous access that automatically call synchronize
        void (*set_tensor_async)(struct ggml_backend * backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(struct ggml_backend * backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
        void (*synchronize)     (struct ggml_backend * backend);

        // (optional) copy tensor between different backends, allow for single-copy tranfers
        void (*cpy_tensor_from)(struct ggml_backend * backend, struct ggml_tensor * src, struct ggml_tensor * dst);
        void (*cpy_tensor_to)  (struct ggml_backend * backend, struct ggml_tensor * src, struct ggml_tensor * dst);

        // compute graph with a plan
        ggml_graph_plan_t (*graph_plan_create) (struct ggml_backend * backend, struct ggml_cgraph * cgraph);
        void              (*graph_plan_free)   (struct ggml_backend * backend, ggml_graph_plan_t plan);
        void              (*graph_plan_compute)(struct ggml_backend * backend, ggml_graph_plan_t plan);

        // compute graph without a plan
        void              (*graph_compute)     (struct ggml_backend * backend, struct ggml_cgraph * cgraph);

        // check if a backend supports a given operation
        // this could be used to fallback automatically to the CPU backend if a backend doesn't support an operation
        // bool (*supports_op)(struct ggml_backend * backend, struct ggml_tensor * op);
    };

    struct ggml_backend {
        struct ggml_backend_interface interface;
        ggml_backend_context_t context;

        bool is_ram_shared;
    };

    // backend helper functions
    static inline const char * ggml_backend_name(struct ggml_backend * backend) { return backend->interface.get_name(backend); }
    static inline void ggml_backend_free(struct ggml_backend * backend) { backend->interface.free(backend); }
    static inline void ggml_backend_tensor_set_async(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) { tensor->backend->interface.set_tensor_async(tensor->backend, tensor, data, offset, size); }
    static inline void ggml_backend_tensor_get_async(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) { tensor->backend->interface.get_tensor_async(tensor->backend, tensor, data, offset, size); }
    static inline void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) { tensor->backend->interface.set_tensor_async(tensor->backend, tensor, data, offset, size); tensor->backend->interface.synchronize(tensor->backend); }
    static inline void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) { tensor->backend->interface.get_tensor_async(tensor->backend, tensor, data, offset, size); tensor->backend->interface.synchronize(tensor->backend); }
    static inline void ggml_backend_synchronize(struct ggml_backend * backend) { backend->interface.synchronize(backend); }
    static inline ggml_graph_plan_t ggml_backend_graph_plan_create(struct ggml_backend * backend, struct ggml_cgraph * cgraph) { return backend->interface.graph_plan_create(backend, cgraph); }
    static inline void ggml_backend_graph_plan_free(struct ggml_backend * backend, ggml_graph_plan_t plan) { backend->interface.graph_plan_free(backend, plan); }
    static inline void ggml_backend_graph_plan_compute(struct ggml_backend * backend, ggml_graph_plan_t plan) { backend->interface.graph_plan_compute(backend, plan); }
    static inline void ggml_backend_graph_compute(struct ggml_backend * backend, struct ggml_cgraph * cgraph) { backend->interface.graph_compute(backend, cgraph); }

    // tensor copy between different backends
    GGML_API void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);

    // CPU backend
    GGML_API struct ggml_backend * ggml_backend_cpu_init(void);
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
