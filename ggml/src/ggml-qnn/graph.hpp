
#pragma once

#include <cstdio>
#include <memory>
#include <vector>

#include "ggml-qnn.h"

#include "logger.hpp"
#include "qnn-lib.hpp"
#include "tensor.hpp"

namespace qnn {

using ggml_tensor_array_t = std::vector<ggml_tensor *>;

class ggml_qnn_graph {
public:
    explicit ggml_qnn_graph(const std::string &graph_name, QNNBackend device,
                            std::shared_ptr<qnn_instance> qnn_instance, size_t vtcm_size_in_mb) :
        _graph_name(graph_name), _device(device), _qnn_instance(qnn_instance) {
        QNN_LOG_INFO("graph name %s", graph_name.c_str());

        auto qnn_interface = qnn_instance->get_qnn_interface();
        auto qnn_context = qnn_instance->get_qnn_context_handle();
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

            const QnnGraph_Config_t *graph_configs[] = { &graph_hvx_config, &graph_dlbc_config, &graph_vtcm_config,
                                                         &graph_opt_config, nullptr };
            error = qnn_interface->qnn_graph_create(qnn_context, graph_name.c_str(), graph_configs, &graph_handle);
        } else {
            error = qnn_interface->qnn_graph_create(qnn_context, graph_name.c_str(), nullptr, &graph_handle);
        }

        if (error != QNN_SUCCESS) {
            QNN_LOG_INFO(
                "can't create qnn graph handle with graph name %s, "
                "error = %d\n",
                graph_name.c_str(), error);
            return;
        }

        QNN_LOG_INFO("create qnn graph handle with graph name %s ok\n", graph_name.c_str());
        _graph_handle = graph_handle;
        _qnn_interface = qnn_interface;
    }

    ~ggml_qnn_graph() { QNN_LOG_DEBUG("graph name %s, destroy", _graph_name.c_str()); }

    bool build_graph(const std::string &op_name, const ggml_tensor_array_t &tensor_inputs,
                     const ggml_tensor_array_t &tensor_outputs) {
        if (!is_valid()) {
            QNN_LOG_ERROR("Invalid graph\n");
            return false;
        }

        QNN_LOG_DEBUG("graph name %s, build_graph start", _graph_name.c_str());
        _qnn_tensor_inputs.resize(tensor_inputs.size());
        _tensor_inputs.resize(tensor_inputs.size());
        for (size_t i = 0; i < tensor_inputs.size(); i++) {
            char buffer[GGML_MAX_NAME] = {};
            snprintf(buffer, GGML_MAX_NAME, "src%d", (int)i);
            auto qnn_tensor =
                std::make_shared<ggml_qnn_tensor>(std::string(buffer), _device, _graph_handle, _qnn_instance);
            auto *ggml_tensor = tensor_inputs[i];
            if (!qnn_tensor->bind_ggml_tensor(ggml_tensor, true)) {
                QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
                return false;
            }

            _qnn_tensor_inputs[i] = qnn_tensor->get_qnn_tensor();
            _tensor_inputs[i] = qnn_tensor;
        }

        _qnn_tensor_outputs.resize(tensor_outputs.size());
        _tensor_outputs.resize(tensor_outputs.size());
        for (size_t i = 0; i < tensor_outputs.size(); i++) {
            char buffer[GGML_MAX_NAME] = {};
            snprintf(buffer, GGML_MAX_NAME, "dst%d", (int)i);
            auto qnn_tensor =
                std::make_shared<ggml_qnn_tensor>(std::string(buffer), _device, _graph_handle, _qnn_instance);
            auto *ggml_tensor = tensor_outputs[i];
            if (!qnn_tensor->bind_ggml_tensor(ggml_tensor, false)) {
                QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
                return false;
            }

            _qnn_tensor_outputs[i] = qnn_tensor->get_qnn_tensor();
            _tensor_outputs[i] = qnn_tensor;
        }

        Qnn_OpConfig_t config = QNN_OPCONFIG_INIT;
        config.version = QNN_OPCONFIG_VERSION_1;
        auto &op_config = config.v1;
        op_config.name = _graph_name.c_str();
        op_config.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op_config.typeName = op_name.c_str();
        op_config.numOfParams = (uint32_t)_param_types.size();
        op_config.params = _param_types.data();
        op_config.numOfInputs = (uint32_t)_qnn_tensor_inputs.size();
        op_config.inputTensors = _qnn_tensor_inputs.data();
        op_config.numOfOutputs = (uint32_t)_qnn_tensor_outputs.size();
        op_config.outputTensors = _qnn_tensor_outputs.data();
        auto error = _qnn_interface->qnn_graph_add_node(_graph_handle, config);
        if (error != QNN_SUCCESS) {
            auto *error_str = get_qnn_error_string(error);
            if (error_str) {
                QNN_LOG_ERROR("qnn_graph_add_node.error: %s\n", error_str);
            } else {
                QNN_LOG_ERROR("qnn_graph_add_node.error: %d\n", error);
            }
            return false;
        }

        error = _qnn_interface->qnn_graph_finalize(_graph_handle, nullptr, nullptr);
        if (error != QNN_SUCCESS) {
            auto *error_str = get_qnn_error_string(error);
            if (error_str) {
                QNN_LOG_ERROR("qnn_graph_finalize.error: %s\n", error_str);
            } else {
                QNN_LOG_ERROR("qnn_graph_finalize.error: %d\n", error);
            }
            return false;
        }

        QNN_LOG_DEBUG("graph name %s, build_graph succeed", _graph_name.c_str());
        return true;
    }

    bool execute(const ggml_tensor_array_t &tensor_inputs, const ggml_tensor_array_t &tensor_outputs) {
        GGML_ASSERT(tensor_inputs.size() == _tensor_inputs.size());
        GGML_ASSERT(tensor_outputs.size() == _tensor_outputs.size());
        for (size_t i = 0; i < tensor_inputs.size(); i++) {
            auto *ggml_tensor = tensor_inputs[i];
            if (!_tensor_inputs[i]->bind_ggml_tensor(ggml_tensor, true)) {
                QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
                return false;
            }

            _qnn_tensor_inputs[i] = _tensor_inputs[i]->get_qnn_tensor();
        }

        for (size_t i = 0; i < tensor_outputs.size(); i++) {
            auto *ggml_tensor = tensor_outputs[i];
            if (!_tensor_outputs[i]->bind_ggml_tensor(ggml_tensor, false)) {
                QNN_LOG_ERROR("bind tensor %s failed\n", ggml_get_name(ggml_tensor));
                return false;
            }

            _qnn_tensor_outputs[i] = _tensor_outputs[i]->get_qnn_tensor();
        }

        auto error =
            _qnn_interface->qnn_graph_execute(_graph_handle, _qnn_tensor_inputs.data(), _qnn_tensor_inputs.size(),
                                              _qnn_tensor_outputs.data(), _qnn_tensor_outputs.size(), nullptr, nullptr);
        if (_device == QNN_BACKEND_NPU) {
            if (error == QNN_COMMON_ERROR_SYSTEM_COMMUNICATION) {
                QNN_LOG_WARN("NPU crashed. SSR detected. Caused QNN graph execute error\n");
            }
        }

        for (auto tensor : _tensor_inputs) {
            tensor->unbind_ggml_tensor();
        }

        for (auto tensor : _tensor_outputs) {
            tensor->unbind_ggml_tensor();
        }

        if (error != QNN_SUCCESS) {
            QNN_LOG_INFO("error = %d\n", error);
            return false;
        }

        return true;
    }

    bool is_valid() const { return _graph_handle != nullptr; }

    Qnn_GraphHandle_t get_graph_handler() const { return _graph_handle; }

    const std::string &get_name() const { return _graph_name; }

private:
    const std::string _graph_name;
    const QNNBackend _device;
    Qnn_GraphHandle_t _graph_handle = nullptr;
    std::shared_ptr<qnn_instance> _qnn_instance;
    std::shared_ptr<qnn_interface> _qnn_interface;
    std::vector<std::shared_ptr<qnn::ggml_qnn_tensor>> _tensor_inputs;
    std::vector<std::shared_ptr<qnn::ggml_qnn_tensor>> _tensor_outputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_inputs;
    std::vector<Qnn_Tensor_t> _qnn_tensor_outputs;
    std::vector<Qnn_Param_t> _param_types;

    DISABLE_COPY(ggml_qnn_graph);
    DISABLE_MOVE(ggml_qnn_graph);
};

} // namespace qnn
