#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

GGML_API void ggml_vk_instance_init(void);
GGML_API void ggml_vk_init_cpu_assist(void);

GGML_API void ggml_vk_preallocate_buffers_graph_cpu_assist(struct ggml_tensor * node);
GGML_API void ggml_vk_preallocate_buffers_cpu_assist(void);
GGML_API void ggml_vk_build_graph_cpu_assist(struct ggml_tensor * node, bool last_node);
GGML_API bool ggml_vk_compute_forward_cpu_assist(struct ggml_compute_params * params, struct ggml_tensor * tensor);
#ifdef GGML_VULKAN_CHECK_RESULTS
void ggml_vk_check_results_1_cpu_assist(struct ggml_compute_params * params, struct ggml_tensor * tensor);
#endif
GGML_API void ggml_vk_graph_cleanup_cpu_assist(void);
GGML_API void ggml_vk_free_cpu_assist(void);

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_API GGML_CALL bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_API GGML_CALL int  ggml_backend_vk_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

#ifdef  __cplusplus
}
#endif
