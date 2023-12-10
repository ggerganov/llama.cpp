#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void ggml_vk_init(void);

void ggml_vk_preallocate_buffers_graph(struct ggml_tensor * node, struct ggml_cgraph * graph);
void ggml_vk_preallocate_buffers(void);
void ggml_vk_build_graph(struct ggml_tensor * node, struct ggml_cgraph * graph);
bool ggml_vk_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);
#ifdef GGML_VULKAN_CHECK_RESULTS
void ggml_vk_check_results_0(struct ggml_compute_params * params, struct ggml_tensor * tensor);
void ggml_vk_check_results_1(struct ggml_compute_params * params, struct ggml_tensor * tensor);
#endif
void ggml_vk_graph_cleanup(void);

void * ggml_vk_host_malloc(size_t size);
void   ggml_vk_host_free(void * ptr);

void ggml_vk_free_data(const struct ggml_tensor * tensor);

void ggml_vk_transform_tensor_temporary(void * data, struct ggml_tensor * tensor);
void ggml_vk_transform_tensor_static(void * data, struct ggml_tensor * tensor);
void ggml_vk_assign_buffer(struct ggml_tensor * tensor);
void ggml_vk_prepare_tensor(struct ggml_tensor * tensor);
void ggml_vk_cleanup(void);

bool ggml_vk_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);

#ifdef  __cplusplus
}
#endif
