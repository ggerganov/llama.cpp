
#pragma once

#include <memory>
#include <unordered_map>

#include "ggml.h"

#include "ggml-backend.h"

#include "graph.hpp"
#include "qnn-lib.hpp"

namespace qnn {
typedef std::unordered_map<std::string, std::unique_ptr<qnn::ggml_qnn_graph>> ggml_qnn_graph_cache_t;
} // namespace qnn

struct ggml_backend_qnn_context {
    int device;
    int threads;
    char name[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];
    ggml_backend *backend = nullptr;
    qnn::qcom_socinfo socinfo = {};
    std::shared_ptr<qnn::qnn_instance> instance;
    std::shared_ptr<qnn::qnn_interface> qnn_interface;
    qnn::ggml_qnn_graph_cache_t qnn_graph_cache;

    explicit ggml_backend_qnn_context(int device, int threads, const char *name, const char *lib) :
        device(device), threads(threads) {
        strncpy(this->name, name, GGML_MAX_NAME);
        strncpy(this->lib, lib, GGML_MAX_NAME);
    }
};
