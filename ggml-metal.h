// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, OpenCL, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include <stddef.h>
#include <stdbool.h>

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 16

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_context;

struct ggml_metal_context * ggml_metal_init(void);
void ggml_metal_free(struct ggml_metal_context * ctx);

// creates a mapping between a host memory buffer and a device memory buffer
// - make sure to map all buffers used in the graph before calling ggml_metal_graph_compute
// - the mapping is used during computation to determine the arguments of the compute kernels
// - you don't need to keep the host memory buffer allocated as it is never accessed by Metal
//
bool ggml_metal_add_buffer(
        struct ggml_metal_context * ctx,
                       const char * name,
                             void * data,
                           size_t   size);

// set data from host memory into the device
void ggml_metal_set_tensor(struct ggml_metal_context * ctx, struct ggml_tensor * t);

// get data from the device into host memory
void ggml_metal_get_tensor(struct ggml_metal_context * ctx, struct ggml_tensor * t);

// same as ggml_graph_compute but uses Metal
// creates gf->n_threads command buffers in parallel
void ggml_metal_graph_compute(struct ggml_metal_context * ctx, struct ggml_cgraph * gf);

#ifdef __cplusplus
}
#endif

