
#include "qnn-lib.hpp"

namespace qnn {

qnn_system_interface::qnn_system_interface(const QnnSystemInterface_t &qnn_sys_interface, dl_handler_t lib_handle) :
    _qnn_sys_interface(qnn_sys_interface), _lib_handle(lib_handle) {
    qnn_system_context_create(&_qnn_system_handle);
    if (_qnn_system_handle) {
        QNN_LOG_INFO("initialize qnn system successfully\n");
    } else {
        QNN_LOG_WARN("can not create QNN system contenxt\n");
    }
}

qnn_system_interface::~qnn_system_interface() {
    if (_qnn_system_handle) {
        if (qnn_system_context_free(_qnn_system_handle) != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN system context\n");
        }
    } else {
        QNN_LOG_WARN("system handle is null\n");
    }

    if (_lib_handle) {
        int dlclose_error = dl_unload(_lib_handle);
        if (dlclose_error != 0) {
            QNN_LOG_WARN("failed to close QnnSystem library, error %s\n", dl_error());
        }
    } else {
        QNN_LOG_WARN("system lib handle is null\n");
    }
}

} // namespace qnn
