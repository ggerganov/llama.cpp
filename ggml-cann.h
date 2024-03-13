#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#define GGML_CANN_NAME "CANN"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_CANN_MAX_DEVICES 16

GGML_API GGML_CALL void ggml_init_cann(void);

GGML_API GGML_CALL bool ggml_cann_compute_forward(
    struct ggml_compute_params* params, struct ggml_tensor* tensor);

GGML_API GGML_CALL uint32_t ggml_cann_get_device_count(void);
GGML_API GGML_CALL void ggml_cann_get_device_description(
    uint32_t device, char* description, size_t description_size);

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cann_init(uint32_t device);

GGML_API GGML_CALL bool ggml_backend_is_cann(ggml_backend_t backend);

GGML_API GGML_CALL ggml_backend_buffer_type_t
ggml_backend_cann_buffer_type(uint32_t device);

GGML_API GGML_CALL uint32_t ggml_backend_cann_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cann_get_device_description(
    uint32_t device, char* description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cann_get_device_memory(uint32_t device,
                                                            size_t* free,
                                                            size_t* total);

#ifdef __cplusplus
}
#endif
