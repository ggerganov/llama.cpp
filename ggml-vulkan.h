#pragma once

#include <cstddef>

#ifdef  __cplusplus
extern "C" {
#endif

struct ggml_kompute_context;


ggml_kompute_context * ggml_vk_init(void);
void ggml_metal_free(struct ggml_kompute_context * ctx);

// creates a mapping between a host memory buffer and a device memory buffer
// - make sure to map all buffers used in the graph before calling ggml_vk_graph_compute
// - the mapping is used during computation to determine the arguments of the compute kernels
// - you don't need to keep the host memory buffer allocated as it is never accessed by Vulkan
// - max_size specifies the maximum size of a tensor and is used to create shared views such
//   that it is guaranteed that the tensor will fit in at least one of the views
//
bool ggml_vk_add_buffer(
      struct ggml_kompute_context * ctx,
                       const char * name,
                             void * data,
                           size_t   size,
                           size_t   max_size);

void ggml_vk_set_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t);
void ggml_vk_get_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t);

void ggml_vk_dequantize_row_q4_0(const void * x, float * y, int k);
void ggml_vk_graph_compute(struct ggml_kompute_context * ctx, struct ggml_cgraph * cgraph);

#ifdef  __cplusplus
}
#endif
