
#pragma once

#include "ggml.h"

#include "ggml-backend.h"

#include "qnn.hpp"

struct ggml_backend_qnn_context {
    int device;
    int threads;
    char name[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];
    qnn::qnn_instance *instance;
    ggml_backend *backend;
    QNN_INTERFACE_VER_TYPE raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE raw_system_interface;
    qnn::qcom_socinfo socinfo;
};
