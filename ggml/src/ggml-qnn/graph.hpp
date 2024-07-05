
#pragma once

#include <array>

#include "ggml-qnn.h"

#include "logger.hpp"
#include "qnn.hpp"

namespace qnn {

template <size_t _InputSize, size_t _OutputSize>
class ggml_qnn_graph {
public:
    typedef std::array<Qnn_Tensor_t, _InputSize> input_tensor_array_t;
    typedef std::array<Qnn_Tensor_t, _OutputSize> output_tensor_array_t;

    explicit ggml_qnn_graph(const std::string &graph_name, QNNBackend device, Qnn_ContextHandle_t qnn_context,
                            QNN_INTERFACE_VER_TYPE qnn_interface, size_t vtcm_size_in_mb) :
        _graph_name(graph_name), _device(device), _qnn_interface(qnn_interface) {
        QNN_LOG_INFO("graph name %s", graph_name.c_str());

        Qnn_ErrorHandle_t error = QNN_SUCCESS;
        Qnn_GraphHandle_t graph_handle = nullptr;
        if (device == QNN_BACKEND_NPU) {
            // TODO: fix graph config here for NPU
            QnnHtpGraph_CustomConfig_t hvx_config;
            hvx_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
            hvx_config.numHvxThreads = 8;
            QnnGraph_Config_t graph_hvx_config;
            graph_hvx_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graph_hvx_config.customConfig = &hvx_config;

            QnnHtpGraph_CustomConfig_t dlbc_config;
            dlbc_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
            dlbc_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC;
            dlbc_config.optimizationOption.floatValue = 1.0; // set to 0.0 to turn off DLBC
            QnnGraph_Config_t graph_dlbc_config;
            graph_dlbc_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graph_dlbc_config.customConfig = &dlbc_config;

            QnnHtpGraph_CustomConfig_t opt_config;
            opt_config.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
            opt_config.optimizationOption.floatValue = 1; // 1 / 3
            QnnGraph_Config_t graph_opt_config;
            graph_opt_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graph_opt_config.customConfig = &opt_config;

            QnnHtpGraph_CustomConfig_t vtcm_config;
            vtcm_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
            vtcm_config.vtcmSizeInMB = vtcm_size_in_mb;
            QnnGraph_Config_t graph_vtcm_config;
            graph_vtcm_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graph_vtcm_config.customConfig = &vtcm_config;

            const QnnGraph_Config_t *p_graphconfig[] = { &graph_hvx_config, &graph_dlbc_config, &graph_vtcm_config,
                                                         &graph_opt_config, nullptr };
            error = qnn_interface.graphCreate(qnn_context, graph_name.c_str(), p_graphconfig, &graph_handle);
        } else {
            error = qnn_interface.graphCreate(qnn_context, graph_name.c_str(), nullptr, &graph_handle);
        }

        if (error != QNN_SUCCESS) {
            QNN_LOG_INFO(
                "can't create qnn graph handle with graph name %s, "
                "error = %d\n",
                graph_name.c_str(), error);
            return;
        } else {
            QNN_LOG_INFO("create qnn graph handle with graph name %s ok\n", graph_name.c_str());
        }

        _graph_handle = graph_handle;
    }

    bool add_nodes(const std::string &op_name, const input_tensor_array_t &tensor_inputs,
                   const output_tensor_array_t &tensor_outputs) {
        if (!is_valid()) {
            QNN_LOG_ERROR("Invalid graph\n");
            return false;
        }

        _tensor_inputs = tensor_inputs;
        _tensor_outputs = tensor_outputs;

        Qnn_Param_t qnn_params[] = {};
        Qnn_OpConfig_t op_config = { .version = QNN_OPCONFIG_VERSION_1,
                                     .v1 = { _graph_name.c_str(), QNN_OP_PACKAGE_NAME_QTI_AISW, op_name.c_str(), 0,
                                             qnn_params, (uint32_t)_tensor_inputs.size(), _tensor_inputs.data(),
                                             (uint32_t)_tensor_outputs.size(), _tensor_outputs.data() } };
        auto error = _qnn_interface.graphAddNode(_graph_handle, op_config);
        if (error != QNN_SUCCESS) {
            QNN_LOG_ERROR("graphAddNode.error = %d\n", error);
            return false;
        }

        error = _qnn_interface.graphFinalize(_graph_handle, nullptr, nullptr);
        if (error != QNN_SUCCESS) {
            QNN_LOG_ERROR("graphFinalize.error = %d\n", error);
            return false;
        }

        return true;
    }

    bool execute() {
        auto error = _qnn_interface.graphExecute(_graph_handle, _tensor_inputs.data(), _tensor_inputs.size(),
                                                 _tensor_outputs.data(), _tensor_outputs.size(), nullptr, nullptr);
        if (_device == QNN_BACKEND_NPU) {
            if (error == QNN_COMMON_ERROR_SYSTEM_COMMUNICATION) {
                QNN_LOG_WARN("NPU crashed. SSR detected. Caused QNN graph execute error\n");
            }
        }

        if (error != QNN_SUCCESS) {
            QNN_LOG_INFO("error = %d\n", error);
            return false;
        }

        return true;
    }

    bool is_valid() const { return _graph_handle != nullptr; }

    Qnn_GraphHandle_t get_graph_handler() const { return _graph_handle; }

private:
    const std::string _graph_name;
    const QNNBackend _device;
    const QNN_INTERFACE_VER_TYPE _qnn_interface;
    Qnn_GraphHandle_t _graph_handle = nullptr;
    std::array<Qnn_Tensor_t, _InputSize> _tensor_inputs;
    std::array<Qnn_Tensor_t, _OutputSize> _tensor_outputs;

    ggml_qnn_graph(const ggml_qnn_graph &) = delete;
    void operator=(const ggml_qnn_graph &) = delete;
    ggml_qnn_graph(ggml_qnn_graph &&) = delete;
    void operator=(ggml_qnn_graph &&) = delete;
};

using ggml_qnn_graph_binary = ggml_qnn_graph<2, 1>;
using ggml_qnn_graph_unary = ggml_qnn_graph<1, 1>;

} // namespace qnn
