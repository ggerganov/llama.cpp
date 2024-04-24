/*
 * MIT license
 * Copyright (C) 2024 GGML Authors
 * SPDX-License-Identifier: MIT
 *
 * this is implementation of ggml QNN(Qualcomm Nerual Network, aka AI Engine Direct) backend
 */
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif


#define GGML_QNN_NAME           "QNN"
#define GGML_QNN_MAX_DEVICES    3

//QNN cDSP and HTA backend would not be used currently, just focus on QNN CPU/GPU/HTP(aka DSP) backend currently
enum QNNBackend {
    QNN_CPU,
    QNN_GPU,
    QNN_HTP,
};

GGML_API int            ggml_backend_qnn_reg_devices();

/**
 *
 * @param device            0: QNN_CPU 1: QNN_GPU 2: QNN_HTP(aka DSP)
 * @param qnn_lib_path      qnn library path, such as "/data/data/com.ggml.llamacpp/" on Android which can got by JNI from Java layer
 * @return
 */
GGML_API ggml_backend_t ggml_backend_qnn_init(size_t dev_num, const char * qnn_lib_path);

GGML_API bool           ggml_backend_is_qnn(ggml_backend_t backend);

GGML_API void           ggml_backend_qnn_set_n_threads(ggml_backend_t backend, int n_threads);

GGML_API int            ggml_backend_qnn_get_device_count(void);
GGML_API void           ggml_backend_qnn_get_device_description(int device, char * description, size_t description_size);


GGML_API ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(size_t dev_num);


//temporary API, should be removed in the future
GGML_API bool           ggml_qnn_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);


#ifdef __cplusplus
}
#endif
