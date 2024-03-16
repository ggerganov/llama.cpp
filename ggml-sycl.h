//
//  MIT license
//  Copyright (C) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_SYCL_MAX_DEVICES       16
#define GGML_SYCL_NAME "SYCL"

GGML_API void   ggml_init_sycl(void);
GGML_API bool   ggml_sycl_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);
GGML_API ggml_backend_t ggml_backend_sycl_init(int device);
GGML_API ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);
GGML_API ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void);
GGML_API void   ggml_backend_sycl_print_sycl_devices(void);
GGML_API GGML_CALL void   ggml_sycl_get_gpu_list(int *id_list, int max_len);
GGML_API GGML_CALL void   ggml_sycl_get_device_description(int device, char *description, size_t description_size);
GGML_API GGML_CALL int   ggml_backend_sycl_get_device_count();
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split);
GGML_API GGML_CALL void ggml_backend_sycl_get_device_memory(int device, size_t *free, size_t *total);
GGML_API GGML_CALL int ggml_backend_sycl_get_device_index(int device_id);

// TODO: these are temporary
//       ref: https://github.com/ggerganov/llama.cpp/pull/6022#issuecomment-1992615670
GGML_API GGML_CALL int ggml_backend_sycl_get_device_id(int device_index);
GGML_API GGML_CALL void ggml_backend_sycl_set_single_device_mode(int main_gpu_id);
GGML_API GGML_CALL void ggml_backend_sycl_set_mul_device_mode();
#ifdef  __cplusplus
}
#endif
