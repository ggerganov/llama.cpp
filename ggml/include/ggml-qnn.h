#pragma once

#include "ggml.h"

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_QNN_MAX_DEVICES 3

enum QNNBackend {
    QNN_BACKEND_CPU = 0,
    QNN_BACKEND_GPU,
    QNN_BACKEND_NPU,
    QNN_BACKEND_GGML, //"fake" QNN backend, used for compare performance between
                      // QNN and original GGML
};

GGML_API int ggml_backend_qnn_reg_devices(void);

/**
 *
 * @param device            0: QNN_BACKEND_CPU 1: QNN_BACKEND_GPU 2:
 * QNN_BACKEND_NPU
 * @param qnn_lib_path      qnn library path, such as "/data/local/tmp/" on
 * Android or specified in JNI layer
 * @return
 */
GGML_API ggml_backend_t ggml_backend_qnn_init(size_t dev_num, const char *qnn_lib_path);

GGML_API bool ggml_backend_is_qnn(ggml_backend_t backend);

GGML_API void ggml_backend_qnn_set_n_threads(ggml_backend_t backend, int thread_counts);

GGML_API int ggml_backend_qnn_get_device_count(void);

GGML_API void ggml_backend_qnn_get_device_description(size_t dev_num, char *description, size_t description_size);

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(size_t dev_num);

#ifdef __cplusplus
}
#endif
