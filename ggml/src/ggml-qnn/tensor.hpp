
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "ggml-qnn.h"

#include "QnnTensor.h"
#include "System/QnnSystemInterface.h"
#include "backend.hpp"
#include "graph.hpp"
#include "logger.hpp"
#include "qnn.hpp"
#include "utils.hpp"

namespace qnn {

class ggml_qnn_tensor {
public:
    static ggml_qnn_tensor *from_ggml_tensor(const ggml_tensor *tensor) {
        if (!tensor) {
            return nullptr;
        }

        return static_cast<ggml_qnn_tensor *>(tensor->extra);
    }

    explicit ggml_qnn_tensor(ggml_tensor *tensor, QNNBackend device, std::shared_ptr<qnn_instance> qnn_instance) :
        _tensor(tensor), _device(device), _qnn_instance(qnn_instance) {
        _tensor_name = ggml_get_name(tensor);
        if (_tensor_name.empty()) {
            static std::atomic_uint32_t unnamed_tensor_count = 0;
            char buffer[GGML_MAX_NAME] = {};
            snprintf(buffer, sizeof(buffer), "unnamed_%p", unnamed_tensor_count++);
            _tensor_name = buffer;
        }

        QNN_TENSOR_SET_NAME(_qnn_tensor, _tensor_name.c_str());
        _dimensions[0] = (uint32_t)tensor->ne[0];
        _dimensions[1] = (uint32_t)tensor->ne[1];
        _dimensions[2] = (uint32_t)tensor->ne[2];
        _dimensions[3] = (uint32_t)tensor->ne[3];
        QNN_TENSOR_SET_DIMENSIONS(_qnn_tensor, _dimensions);
        QNN_TENSOR_SET_TYPE(_qnn_tensor, device_tensortype_from_ggml_tensor(tensor));
        QNN_TENSOR_SET_DATA_FORMAT(_qnn_tensor, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
        QNN_TENSOR_SET_DATA_TYPE(_qnn_tensor, device_datatype_from_ggml_datatype(tensor->type));
        // TODO: set the quantizeParams base on the tensor type
        QNN_TENSOR_SET_RANK(_qnn_tensor, qnn::get_ggml_tensor_rank(tensor));

        const bool is_npu = device == QNN_BACKEND_NPU;
        if (is_npu) {
            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
            QNN_TENSOR_SET_MEM_HANDLE(_qnn_tensor, nullptr);
        } else {
            QNN_TENSOR_SET_MEM_TYPE(_qnn_tensor, QNN_TENSORMEMTYPE_RAW);
            Qnn_ClientBuffer_t client_buf = { tensor->data, get_ggml_tensor_data_size(tensor) };
            QNN_TENSOR_SET_CLIENT_BUF(_qnn_tensor, client_buf);
        }

        tensor->extra = this;
        QNN_LOG_DEBUG("create tensor %s with device %d", _tensor_name.c_str(), device);
    }

    template <size_t _InputSize, size_t _OutputSize>
    bool bind_to_graph(ggml_qnn_graph<_InputSize, _OutputSize> &graph) {
        if (!is_valid()) {
            QNN_LOG_WARN("tensor %s not valid", _tensor_name.c_str());
            return false;
        }

        if (_graph_handle) {
            if (_graph_handle != graph.get_graph_handler()) {
                QNN_LOG_WARN("tensor %s has been bound to another graph", _tensor_name.c_str());
                return false;
            } else {
                QNN_LOG_INFO("tensor %s already bound to same graph %s", _tensor_name.c_str(),
                             graph.get_name().c_str());
                return true;
            }
        }

        Qnn_Tensor_t tensor = _qnn_tensor;
        if (!graph.create_graph_tensor(tensor)) {
            QNN_LOG_WARN("create graph tensor failed, tensor %s", _tensor_name.c_str());
            return false;
        }

        if (!alloc_rpc_mem()) {
            QNN_LOG_WARN("alloc rpc mem failed, tensor %s", _tensor_name.c_str());
            return false;
        }

        QNN_TENSOR_SET_ID(_qnn_tensor, QNN_TENSOR_GET_ID(tensor));
        _graph_handle = graph.get_graph_handler();

        QNN_LOG_DEBUG("bind tensor %s to graph %s", _tensor_name.c_str(), graph.get_name().c_str());
        return true;
    }

    bool write_to_qnn_tensor() {
        if (!is_valid()) {
            QNN_LOG_WARN("tensor %s not valid", _tensor_name.c_str());
            return false;
        }

        auto tensor_type = QNN_TENSOR_GET_TYPE(_qnn_tensor);
        if (tensor_type != QNN_TENSOR_TYPE_APP_WRITE && tensor_type != QNN_TENSOR_TYPE_APP_READWRITE) {
            QNN_LOG_WARN("tensor %s not writable", _tensor_name.c_str());
            return false;
        }

        if (should_use_mem_handle()) {
            uint8_t *qnn_buffer = static_cast<uint8_t *>(
                _qnn_instance->get_rpcmem_from_memhandle(QNN_TENSOR_GET_MEM_HANDLE(_qnn_tensor)));
            if (qnn_buffer) {
                memcpy(qnn_buffer, _tensor->data, ggml_nbytes(_tensor));
            } else {
                QNN_LOG_WARN("can't find rpcmem from qnn mem handle\n");
                return false;
            }
        }

        // For CPU and GPU, the data is already in the tensor.
        return true;
    }

    bool read_from_qnn_tensor() {
        if (!is_valid()) {
            QNN_LOG_WARN("tensor %s not valid", _tensor_name.c_str());
            return false;
        }

        auto tensor_type = QNN_TENSOR_GET_TYPE(_qnn_tensor);
        if (tensor_type != QNN_TENSOR_TYPE_APP_READ && tensor_type != QNN_TENSOR_TYPE_APP_READWRITE) {
            QNN_LOG_WARN("tensor %s not readable", _tensor_name.c_str());
            return false;
        }

        if (should_use_mem_handle()) {
            uint8_t *qnn_buffer = static_cast<uint8_t *>(
                _qnn_instance->get_rpcmem_from_memhandle(QNN_TENSOR_GET_MEM_HANDLE(_qnn_tensor)));
            if (qnn_buffer) {
                memcpy(_tensor->data, qnn_buffer, ggml_nbytes(_tensor));
            } else {
                QNN_LOG_WARN("can't find rpcmem from qnn mem handle\n");
                return false;
            }
        }

        // For CPU and GPU, the data is already in the tensor.
        return true;
    }

    bool is_valid() const { return _tensor; }
    const Qnn_Tensor_t &get_qnn_tensor() const { return _qnn_tensor; }

private:
    bool alloc_rpc_mem() {
        if (!should_use_mem_handle()) {
            return true;
        }

        uint8_t *qnn_buffer =
            static_cast<uint8_t *>(_qnn_instance->alloc_rpcmem(ggml_nbytes(_tensor), alignof(void *)));
        if (!qnn_buffer) {
            QNN_LOG_WARN("alloc rpc mem failure, %s\n", strerror(errno));
            QNN_LOG_DEBUG("tensor name %s", _tensor_name.c_str());
            return false;
        }

        QNN_LOG_INFO("tensor %s: alloc rpcmem(%p) successfully\n", _tensor_name.c_str(), qnn_buffer);
        
        auto error = _qnn_instance->register_rpcmem(qnn_buffer, &_qnn_tensor);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("register rpc mem failure, %d\n", (int)error);
            QNN_LOG_DEBUG("tensor name %s", _tensor_name.c_str());
            return false;
        }

        return true;
    }

    bool should_use_mem_handle() const { return _device == QNN_BACKEND_NPU; }

    const ggml_tensor *_tensor;
    QNNBackend _device;
    std::shared_ptr<qnn_instance> _qnn_instance;
    Qnn_Tensor_t _qnn_tensor = QNN_TENSOR_INIT;
    uint32_t _dimensions[4] = {};
    std::string _tensor_name;
    Qnn_GraphHandle_t _graph_handle = nullptr;

    ggml_qnn_tensor(const ggml_qnn_tensor &) = delete;
    void operator=(const ggml_qnn_tensor &) = delete;
    ggml_qnn_tensor(ggml_qnn_tensor &&) = delete;
    void operator=(ggml_qnn_tensor &&) = delete;
};

} // namespace qnn
