#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"

GGML_API void ggml_vk_init(void);

GGML_API void ggml_vk_preallocate_buffers_graph(struct ggml_tensor * node);
GGML_API void ggml_vk_preallocate_buffers(void);
GGML_API void ggml_vk_build_graph(struct ggml_tensor * node, bool last_node);
GGML_API bool ggml_vk_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);
#ifdef GGML_VULKAN_CHECK_RESULTS
void ggml_vk_check_results_1(struct ggml_compute_params * params, struct ggml_tensor * tensor);
#endif
GGML_API void ggml_vk_graph_cleanup(void);

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_vk_init(void);

GGML_API GGML_CALL bool ggml_backend_is_vk(ggml_backend_t backend);

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(void);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

#ifdef  __cplusplus
}
#endif
