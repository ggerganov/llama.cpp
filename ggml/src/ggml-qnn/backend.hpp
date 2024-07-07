
#pragma once

#include <memory>
#include <unordered_map>

#include "ggml.h"

#include "ggml-backend.h"

#include "graph.hpp"
#include "qnn.hpp"

struct ggml_backend_qnn_context {
    int device;
    int threads;
    char name[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];
    std::shared_ptr<qnn::qnn_instance> instance;
    ggml_backend *backend;
    QNN_INTERFACE_VER_TYPE raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE raw_system_interface;
    qnn::qcom_socinfo socinfo;
    std::unordered_map<std::string, std::unique_ptr<qnn::ggml_qnn_graph_binary>> qnn_binary_graph_cache;
};
