
#pragma once

#include "ggml-qnn.h"

#include "QnnTensor.h"
#include "System/QnnSystemInterface.h"
#include "backend.hpp"
#include "qnn.hpp"

namespace qnn {

template <Qnn_TensorType_t _tensorType>
class ggml_qnn_tensor_readwrite {
public:
    explicit ggml_qnn_tensor_readwrite(const ggml_tensor *tensor, Qnn_GraphHandle_t graph_handle,
                                       ggml_backend_qnn_context *ctx) :
        _tensor(tensor), _qnn_tensor(reinterpret_cast<Qnn_Tensor_t *>(tensor->extra)), _context(ctx) {
        _old_dimensions = QNN_VER_PTR(*_qnn_tensor)->dimensions;
        const auto qnn_data_type = datatype_from_ggml_datatype(tensor->type);
        const bool is_npu = ctx->device == QNN_BACKEND_NPU;
        QNN_VER_PTR(*_qnn_tensor)->type = _tensorType;
        if (is_npu) {
            QNN_VER_PTR(*_qnn_tensor)->memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            QNN_VER_PTR(*_qnn_tensor)->clientBuf = { .data = nullptr, .dataSize = 0 };
        }

        auto err = ctx->raw_interface.tensorCreateGraphTensor(graph_handle, _qnn_tensor);
        if (err != QNN_SUCCESS) {
            QNN_LOG_INFO("error = %d\n", err);
            QNN_LOG_DEBUG("tensor%p name %s", _qnn_tensor, QNN_TENSOR_GET_NAME(*_qnn_tensor));
            _context = nullptr;
            return;
        }

        _dimensions[0] = (uint32_t)tensor->ne[0];
        _dimensions[1] = (uint32_t)tensor->ne[1];
        _dimensions[2] = (uint32_t)tensor->ne[2];
        _dimensions[3] = (uint32_t)tensor->ne[3];
        QNN_VER_PTR(*_qnn_tensor)->dimensions = _dimensions;
        QNN_VER_PTR(*_qnn_tensor)->rank = qnn::get_ggml_tensor_rank(tensor);
        QNN_VER_PTR(*_qnn_tensor)->dataType = qnn_data_type;

        if (is_npu) {
            auto *instance = ctx->instance;
            uint8_t *qnn_buffer = static_cast<uint8_t *>(instance->alloc_rpcmem(ggml_nbytes(tensor), alignof(void *)));
            if (!qnn_buffer) {
                QNN_LOG_WARN("alloc rpcmem failure, %s\n", strerror(errno));
                QNN_LOG_DEBUG("tensor%p name %s", _qnn_tensor, QNN_TENSOR_GET_NAME(*_qnn_tensor));
                _context = nullptr;
                // No free for _qnn_tensor, because it's not registered.
                return;
            } else {
                QNN_LOG_INFO("alloc rpcmem successfully\n");
            }

            instance->register_rpcmem(qnn_buffer, _qnn_tensor);
            if (_tensorType == QNN_TENSOR_TYPE_APP_WRITE || _tensorType == QNN_TENSOR_TYPE_APP_READWRITE) {
                memcpy(qnn_buffer, tensor->data, ggml_nbytes(tensor));
            }
        } else {
            QNN_VER_PTR(*_qnn_tensor)->clientBuf = { tensor->data, get_ggml_tensor_data_size(tensor) };
        }
    }

    explicit ggml_qnn_tensor_readwrite(const ggml_tensor *tensor, Qnn_Tensor_t *qnn_tensor,
                                       ggml_backend_qnn_context *ctx) :
        _tensor(tensor), _qnn_tensor(qnn_tensor), _context(ctx) {
        _old_dimensions = QNN_VER_PTR(*_qnn_tensor)->dimensions;
        const auto qnn_data_type = qnn::datatype_from_ggml_datatype(tensor->type);
        const bool is_npu = ctx->device == QNN_BACKEND_NPU;

        _dimensions[0] = (uint32_t)tensor->ne[0];
        _dimensions[1] = (uint32_t)tensor->ne[1];
        _dimensions[2] = (uint32_t)tensor->ne[2];
        _dimensions[3] = (uint32_t)tensor->ne[3];
        QNN_VER_PTR(*_qnn_tensor)->dimensions = _dimensions;
        QNN_VER_PTR(*_qnn_tensor)->rank = get_ggml_tensor_rank(tensor);
        QNN_VER_PTR(*_qnn_tensor)->dataType = qnn_data_type;

        if (is_npu) {
            uint8_t *qnn_buffer =
                static_cast<uint8_t *>(ctx->instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*_qnn_tensor)->memHandle));
            if (qnn_buffer) {
                memcpy(qnn_buffer, tensor->data, ggml_nbytes(tensor));
            } else {
                QNN_LOG_WARN("can't find rpcmem from qnn mem handle\n");
                QNN_LOG_DEBUG("tensor%p name %s", _qnn_tensor, QNN_TENSOR_GET_NAME(*_qnn_tensor));
                _context = nullptr;
                return;
            }
        } else {
            QNN_VER_PTR(*_qnn_tensor)->clientBuf = { tensor->data, get_ggml_tensor_data_size(tensor) };
        }
    }

    ~ggml_qnn_tensor_readwrite() {
        if ((_tensorType == QNN_TENSOR_TYPE_APP_READWRITE || _tensorType == QNN_TENSOR_TYPE_APP_READ) && _context &&
            _context->device == QNN_BACKEND_NPU) {
            uint8_t *qnn_buffer = static_cast<uint8_t *>(
                _context->instance->get_rpcmem_from_memhandle(QNN_VER_PTR(*_qnn_tensor)->memHandle));
            memcpy(_tensor->data, qnn_buffer, ggml_nbytes(_tensor));
        }

        QNN_VER_PTR(*_qnn_tensor)->dimensions = _old_dimensions;
    }

    bool is_valid() const { return _context; }
    Qnn_Tensor_t *get_qnn_tensor() const { return _qnn_tensor; }

private:
    const ggml_tensor *_tensor;
    Qnn_Tensor_t *_qnn_tensor;
    ggml_backend_qnn_context *_context;
    uint32_t *_old_dimensions;
    uint32_t _dimensions[4] = {};

    ggml_qnn_tensor_readwrite(const ggml_qnn_tensor_readwrite &) = delete;
    void operator=(const ggml_qnn_tensor_readwrite &) = delete;
    ggml_qnn_tensor_readwrite(ggml_qnn_tensor_readwrite &&) = delete;
    void operator=(ggml_qnn_tensor_readwrite &&) = delete;
};

using ggml_qnn_tensor_output = ggml_qnn_tensor_readwrite<QNN_TENSOR_TYPE_APP_READ>;
using ggml_qnn_tensor_input = ggml_qnn_tensor_readwrite<QNN_TENSOR_TYPE_APP_WRITE>;

} // namespace qnn
