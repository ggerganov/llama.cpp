/*MIT license
  Copyright (C) 2024 Intel Corporation
  SPDX-License-Identifier: MIT
*/

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

#ifdef  __cplusplus
}
#endif
