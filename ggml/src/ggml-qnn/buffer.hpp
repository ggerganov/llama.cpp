#pragma once

#include <cstdint>

#include "logger.hpp"
#include "qnn-lib.hpp"

namespace qnn {
class ggml_qnn_rpc_buffer {
public:
    ggml_qnn_rpc_buffer(std::shared_ptr<qnn_instance> qnn_instance, size_t size, uint32_t rank, uint32_t *dimensions,
                        Qnn_DataType_t data_type) :
        _qnn_instance(qnn_instance), _size(size) {

        _qnn_rpc_buffer = static_cast<uint8_t *>(qnn_instance->alloc_rpcmem(size, alignof(void *)));
        _qnn_rpc_mem_handle = qnn_instance->register_rpcmem(_qnn_rpc_buffer, rank, dimensions, data_type);
        if (!_qnn_rpc_buffer || !_qnn_rpc_mem_handle) {
            QNN_LOG_WARN("register rpc mem failure\n");
            // let the destructor free the buffer
            return;
        }

        QNN_LOG_DEBUG("alloc rpcmem(%p) successfully, size %d\n", _qnn_rpc_buffer, (int)size);
    }
    ~ggml_qnn_rpc_buffer() {
        if (_qnn_instance) {
            if (_qnn_rpc_mem_handle) {
                _qnn_instance->unregister_rpcmem(_qnn_rpc_mem_handle);
            }

            if (_qnn_rpc_buffer) {
                _qnn_instance->free_rpcmem(_qnn_rpc_buffer);
            }
        }
    }

    bool is_valid() const { return _qnn_rpc_buffer && _qnn_rpc_mem_handle; }

    uint8_t *get_buffer() const { return _qnn_rpc_buffer; }
    size_t get_size() const { return _size; }
    Qnn_MemHandle_t get_mem_handle() const { return _qnn_rpc_mem_handle; }

private:
    std::shared_ptr<qnn_instance> _qnn_instance;
    size_t _size = 0;
    uint8_t *_qnn_rpc_buffer = nullptr;
    Qnn_MemHandle_t _qnn_rpc_mem_handle = nullptr;

    ggml_qnn_rpc_buffer(const ggml_qnn_rpc_buffer &) = delete;
    void operator=(const ggml_qnn_rpc_buffer &) = delete;
    ggml_qnn_rpc_buffer(ggml_qnn_rpc_buffer &&) = delete;
    void operator=(ggml_qnn_rpc_buffer &&) = delete;
};

} // namespace qnn
