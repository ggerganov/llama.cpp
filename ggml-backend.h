#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

    //
    // Backend buffer
    //

    struct ggml_backend_buffer;
    typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

    // backend buffer functions
    GGML_API void   ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
    GGML_API size_t ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);
    GGML_API void * ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
    GGML_API size_t ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
    GGML_API size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    GGML_API void   ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    GGML_API void   ggml_backend_buffer_free_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);

    //
    // Backend
    //

    struct ggml_backend;
    typedef struct ggml_backend * ggml_backend_t;
    typedef void * ggml_backend_graph_plan_t;

    GGML_API ggml_backend_t ggml_get_backend(const struct ggml_tensor * tensor);

    GGML_API const char * ggml_backend_name(ggml_backend_t backend);
    GGML_API void         ggml_backend_free(ggml_backend_t backend);

    GGML_API ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);

    GGML_API size_t ggml_backend_get_alignment(ggml_backend_t backend);

    GGML_API void ggml_backend_tensor_set_async(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get_async(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    GGML_API void ggml_backend_tensor_set(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    GGML_API void ggml_backend_synchronize(ggml_backend_t backend);

    GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create (ggml_backend_t backend, struct ggml_cgraph * cgraph);

    GGML_API void ggml_backend_graph_plan_free   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    GGML_API void ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    GGML_API void ggml_backend_graph_compute     (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API bool ggml_backend_supports_op       (ggml_backend_t backend, const struct ggml_tensor * op);

    // tensor copy between different backends
    GGML_API void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);

    //
    // CPU backend
    //

    GGML_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_API bool ggml_backend_is_cpu(ggml_backend_t backend);
    GGML_API void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads);

    // Create a backend buffer from an existing pointer
    GGML_API ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(ggml_backend_t backend_cpu, void * ptr, size_t size);


    //
    // Backend scheduler
    //

    // The backend scheduler allows for multiple backends to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
    /*
      Example usage:

        sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, num_backends);
        // sched is initialized with measure allocators and cannot be used until allocated with a measure graph

        // initialize buffers from a measure graph
        measure_graph = build_graph(sched); // use the allocr to allocate inputs as needed

        // in build_graph:
        build_graph(...) {
            // allocating tensors in a specific backend (optional, recommended: pre-allocate inputs in a different buffer)
            alloc_cpu = ggml_backend_sched_get_allocr(sched, backend_cpu);
            ggml_allocr_alloc(alloc_cpu, tensor);

            // manually assigning nodes to a backend (optional, shouldn't be needed in most cases)
            struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
            ggml_backend_sched_set_node_backend(sched, node, backend_gpu);
        }

        // allocate backend buffers from measure graph
        ggml_backend_sched_init_measure(sched, measure_graph);

        // the scheduler is now ready to compute graphs

        // compute
        graph = build_graph(sched);
        ggml_backend_sched_graph_compute(sched, graph);
    */

    struct ggml_backend_sched;
    typedef struct ggml_backend_sched * ggml_backend_sched_t;

    // Initialize a backend scheduler
    GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, int n_backends);

    GGML_API void ggml_backend_sched_free(ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    GGML_API void ggml_backend_sched_init_measure(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph);

    GGML_API ggml_tallocr_t        ggml_backend_sched_get_tallocr(ggml_backend_sched_t sched, ggml_backend_t backend);
    GGML_API ggml_backend_buffer_t ggml_backend_sched_get_buffer (ggml_backend_sched_t sched, ggml_backend_t backend);

    GGML_API void ggml_backend_sched_set_node_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);

    // Allocate a graph on the backend scheduler
    GGML_API void ggml_backend_sched_graph_compute(
            ggml_backend_sched_t sched,
            struct ggml_cgraph * graph);

#ifdef  __cplusplus
}
#endif
