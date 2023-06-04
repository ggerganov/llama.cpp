#pragma once

#include <stddef.h>

#define GGML_METAL_MAX_BUFFERS 16

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_mtl_context;

struct ggml_mtl_context * ggml_mtl_init(void);

void ggml_mtl_free(struct ggml_mtl_context * ctx);

void ggml_mtl_add_buffer(
        struct ggml_mtl_context * ctx,
                     const char * name,
                           void * data,
                         size_t   size);

// set data from host memory into the device
void ggml_mtl_set_tensor(
        struct ggml_mtl_context * ctx,
             struct ggml_tensor * t);

// get data from the device into host memory
void ggml_mtl_get_tensor(
        struct ggml_mtl_context * ctx,
             struct ggml_tensor * t);

// return 0 on success
int ggml_mtl_graph_compute(
        struct ggml_mtl_context * ctx,
             struct ggml_cgraph * gf);

#ifdef __cplusplus
}
#endif

