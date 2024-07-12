
#pragma once

#include <memory>
#include <unordered_map>

#include "ggml.h"

#include "ggml-backend.h"

#include "graph.hpp"
#include "qnn.hpp"

namespace qnn {
typedef std::unordered_map<std::string, std::unique_ptr<qnn::ggml_qnn_graph_unary>> ggml_qnn_unary_graph_cache_t;
typedef std::unordered_map<std::string, std::unique_ptr<qnn::ggml_qnn_graph_binary>> ggml_qnn_binary_graph_cache_t;
} // namespace qnn

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
    qnn::ggml_qnn_unary_graph_cache_t qnn_unary_graph_cache;
    qnn::ggml_qnn_binary_graph_cache_t qnn_binary_graph_cache;
};
