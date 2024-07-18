#include "ggml-qnn.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ggml-backend-impl.h"

#include "ggml-qnn/backend-ops.hpp"
#include "ggml-qnn/backend.hpp"
#include "ggml-qnn/logger.hpp"
#include "ggml-qnn/tensor.hpp"
#include "ggml-qnn/utils.hpp"

// =================================================================================================
//
//  self-defined macro / data structure
//
// =================================================================================================
#ifdef NDEBUG
#define ENABLE_QNNBACKEND_PERF 0 // enable/disable op's perf info
#else
#define ENABLE_QNNBACKEND_PERF 1 // enable/disable op's perf info
#endif

#define QNN_BACKEND_NAME "qnn"

// according to the QNN SDK Reference Guide,
// CPU - Choose a non-quantized model.Quantized models are currently incompatible with the CPU backend
// GPU - Choose a non-quantized model.Quantized models are currently incompatible with the GPU backend
// HTP - Choose a quantized model. Quantized models are required when running on the HTP backend
// DSP - Choose a quantized model. Quantized models are required when running on the DSP backend
// HTA - Choose a quantized model. Quantized models are required when running on the HTA backend
//
// only focus on Qualcomm CPU/GPU/NPU backend in this implementation of QNN backend for ggml currently,
// CPU: Qualcomm Kryo CPU
// GPU: Qualcomm Adreno GPU
// NPU: Qualcomm NPU: aka HTP(Hexagon Tensor Processor), ~= cDSP(Compute DSP) +
//                    HMX(Hexagon Matrix eXtensions)/HTA(Hexagon Tensor Accelerator)

static struct ggml_backend_qnn_context g_qnn_mgr[GGML_QNN_MAX_DEVICES] = {
    ggml_backend_qnn_context(QNN_BACKEND_CPU, 1, "qnn-cpu", "libQnnCpu.so"), /* QNN_BACKEND_CPU */
    ggml_backend_qnn_context(QNN_BACKEND_GPU, 1, "qnn-gpu", "libQnnGpu.so"), /* QNN_BACKEND_GPU */
    ggml_backend_qnn_context(QNN_BACKEND_NPU, 1, "qnn-npu", "libQnnHtp.so"), /* QNN_BACKEND_NPU */
};

class ggml_backend_qnn_buffer_context {
public:
    ggml_backend_qnn_buffer_context(QNNBackend device, std::shared_ptr<qnn::qnn_instance> instance, size_t size) :
        _device(device), _instance(instance), _name(QNN_BACKEND_NAME + std::to_string(device)) {

        size_t size_page = sysconf(_SC_PAGESIZE);

        // TODO: for qnn npu, a better way here is to reuse the buffer allocated by qnn rpc, will save an extra copy
        _buffer = qnn::align_alloc(size_page, size);

        if (!_buffer) {
            QNN_LOG_WARN("failed to allocate %.2f MiB\n", float(size / (1 << 20)));
            return;
        }

        _buffer_size = size;
    }

    ~ggml_backend_qnn_buffer_context() {
        _tensors.clear();

        // the free will do nothing if the _buffer is nullptr
        qnn::align_free(_buffer);
    }

    bool is_valid() const { return _buffer != nullptr; }

    bool init_tensor(ggml_tensor *tensor) {
        auto qnn_tensor = std::make_unique<qnn::ggml_qnn_tensor>(tensor, _device, _instance);
        if (!qnn_tensor->is_valid()) {
            QNN_LOG_WARN("Create ggml_qnn_tensor failed");
            return false;
        }

        _tensors.push_back(std::move(qnn_tensor));
        return true;
    }

    void *get_buffer() { return _buffer; }
    size_t get_buffer_size() { return _buffer_size; }

private:
    QNNBackend _device;
    std::shared_ptr<qnn::qnn_instance> _instance;
    std::string _name;
    std::list<std::unique_ptr<qnn::ggml_qnn_tensor>> _tensors;
    void *_buffer = nullptr;
    size_t _buffer_size = 0;
};

struct ggml_backend_qnn_buffer_type_context {
    size_t device;
    std::string name;
};

// =================================================================================================
//
//  QNN backend internal helper functions
//
// =================================================================================================

// =================================================================================================
//
//  implementation of QNN backend for GGML
//
// =================================================================================================
static bool ggml_qnn_compute_forward(ggml_backend_qnn_context *ctx, struct ggml_tensor *tensor) {
    size_t unary_op_idx = tensor->op;
    if (tensor->op == GGML_OP_UNARY) {
        unary_op_idx = qnn::kGgmlUnaryOpStart + ggml_get_unary_op(tensor);
    }

    auto unary_op = qnn::ggml_qnn_unary_op_array()[unary_op_idx];
    if (unary_op) {
        return unary_op(ctx, tensor->src[0], tensor);
    }

    auto binary_op = qnn::ggml_qnn_binary_op_array()[tensor->op];
    if (binary_op) {
        return binary_op(ctx, tensor->src[0], tensor->src[1], tensor);
    }

    QNN_LOG_WARN("unsupported op %d", tensor->op);
    return false;
}

static const char *ggml_backend_qnn_buffer_get_name(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return "QNN";
}

GGML_CALL static bool ggml_backend_buffer_is_qnn(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_qnn_buffer_get_name;
}

GGML_CALL static void ggml_backend_qnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context *ctx = (ggml_backend_qnn_buffer_context *)buffer->context;

    delete ctx;
}

GGML_CALL static void *ggml_backend_qnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context *ctx = (ggml_backend_qnn_buffer_context *)buffer->context;

    return ctx->get_buffer();
}

GGML_CALL static void ggml_backend_qnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor) {
    ggml_backend_qnn_buffer_context *ctx = (ggml_backend_qnn_buffer_context *)buffer->context;

    if (!ctx->init_tensor(tensor)) {
        QNN_LOG_WARN("init ggml_qnn_tensor failed");
        return;
    }
}

GGML_CALL static void ggml_backend_qnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor,
                                                         const void *data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);

    memcpy((char *)tensor->data + offset, data, size);
}

GGML_CALL static void ggml_backend_qnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor *tensor,
                                                         void *data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *)tensor->data + offset, size);
}

GGML_CALL static bool ggml_backend_qnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor *src,
                                                         struct ggml_tensor *dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

GGML_CALL static void ggml_backend_qnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_qnn_buffer_context *ctx = (ggml_backend_qnn_buffer_context *)buffer->context;

    memset(ctx->get_buffer(), value, ctx->get_buffer_size());
}

static ggml_backend_buffer_i ggml_backend_qnn_buffer_interface = {
    /* .get_name        = */ ggml_backend_qnn_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_qnn_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_qnn_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_qnn_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_qnn_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_qnn_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_qnn_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_qnn_buffer_clear,
    /* .reset           = */ nullptr,
};

GGML_CALL static const char *ggml_backend_qnn_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "QNN";
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                 size_t size) {
    ggml_backend_qnn_buffer_type_context *buft_ctx = (ggml_backend_qnn_buffer_type_context *)buft->context;
    ggml_backend_qnn_buffer_context *ctx =
        new ggml_backend_qnn_buffer_context((QNNBackend)buft_ctx->device, g_qnn_mgr[buft_ctx->device].instance, size);
    if (!ctx->is_valid()) {
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_qnn_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_qnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}

// TODO: this value is an experimental value, works fine with whisper/llm/minicpm-v inference on Android
GGML_CALL static size_t ggml_backend_qnn_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);

    return (96 * 1024 * 1024);
}

GGML_CALL static bool ggml_backend_qnn_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

GGML_CALL static const char *ggml_backend_qnn_name(ggml_backend_t backend) {
    ggml_backend_qnn_context *ctx = (ggml_backend_qnn_context *)backend->context;
    return g_qnn_mgr[ctx->device].name;
}

GGML_CALL static void ggml_backend_qnn_free(ggml_backend_t backend) {
    QNN_LOG_INFO("enter %s", __func__);
    ggml_backend_qnn_context *ctx = (ggml_backend_qnn_context *)backend->context;
    QNN_LOG_INFO("idx %d, name:%s", ctx->device, g_qnn_mgr[ctx->device].name);

    auto instance = g_qnn_mgr[ctx->device].instance;
    if (instance) {
        ctx->qnn_unary_graph_cache.clear();
        for (const auto &graph_item : ctx->qnn_binary_graph_cache) {
            QNN_LOG_INFO("graph type:%s", graph_item.first.c_str());
        }

        ctx->qnn_binary_graph_cache.clear();

        instance->qnn_finalize();
        g_qnn_mgr[ctx->device].instance.reset();
    }

    if (g_qnn_mgr[ctx->device].backend != nullptr) {
        delete backend;
        g_qnn_mgr[ctx->device].backend = nullptr;
    }
    QNN_LOG_INFO("leave %s", __func__);
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_qnn_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_qnn_context *ctx = (ggml_backend_qnn_context *)backend->context;

    return ggml_backend_qnn_buffer_type(ctx->device);
}

GGML_CALL static ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, ggml_cgraph *cgraph) {
    enum ggml_status result = GGML_STATUS_SUCCESS;
    ggml_backend_qnn_context *ctx = (ggml_backend_qnn_context *)backend->context;
    GGML_UNUSED(ctx);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
            node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        bool ok = ggml_qnn_compute_forward(ctx, node);
        if (!ok) {
            QNN_LOG_DEBUG("error: op not supported %s (%s)\n", node->name, ggml_op_name(node->op));
        }
    }

    return result;
}

GGML_CALL static bool ggml_backend_qnn_supports_op(ggml_backend_t backend, const ggml_tensor *op) {
    GGML_UNUSED(backend);

    if (op->op == GGML_OP_NONE) {
        return true;
    }

    if (op->op == GGML_OP_UNARY) {
        if (!qnn::ggml_qnn_unary_op_array()[qnn::kGgmlUnaryOpStart + ggml_get_unary_op(op)]) {
            QNN_LOG_DEBUG("unsupported unary op %d", ggml_get_unary_op(op));
            return false;
        }

        if (!op->src[0]) {
            QNN_LOG_DEBUG("src0 is nullptr");
            return false;
        }
    } else {
        if (!qnn::ggml_qnn_unary_op_array()[op->op] && !qnn::ggml_qnn_binary_op_array()[op->op]) {
            QNN_LOG_DEBUG("unsupported op %d", op->op);
            return false;
        }

        if (!op->src[0] || !op->src[1]) {
            QNN_LOG_DEBUG("src0 or src1 is nullptr");
            return false;
        }
    }

    switch (op->src[0]->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_I8:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
            break;
        default:
            QNN_LOG_DEBUG("unsupported src0 type %d", op->src[0]->type);
            return false;
    }

    return true;
}

GGML_CALL static bool ggml_backend_qnn_offload_op(ggml_backend_t backend, const ggml_tensor *op) {
    GGML_UNUSED(backend);

    return op->ne[0] > 1 && op->ne[1] > 1;
}

static ggml_backend_i ggml_backend_qnn_interface = {
    /* .get_name                = */ ggml_backend_qnn_name,
    /* .free                    = */ ggml_backend_qnn_free,
    /* .get_default_buffer_type = */ ggml_backend_qnn_get_default_buffer_type,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_qnn_graph_compute,
    /* .supports_op             = */ ggml_backend_qnn_supports_op,
    /* .supports_buft           = */ nullptr,
    /* .offload_op              = */ ggml_backend_qnn_offload_op,
    /* .event_new               = */ nullptr,
    /* .event_free              = */ nullptr,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .event_synchronize       = */ nullptr,
};

static ggml_guid_t ggml_backend_qnn_guid() {
    static ggml_guid guid = { 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
                              0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09 };
    return &guid;
}

static ggml_backend_t ggml_backend_qnn_reg_init(const char *params, void *user_data) {
    if (nullptr == params) {
        // QNN library path
        // can be hardcoded to "/data/local/tmp/" for Android command line application
        // or specified in JNI layer for Android APK
        params = "/data/local/tmp/";
    }
    ggml_backend_t qnn_backend = ggml_backend_qnn_init((int)(intptr_t)user_data, params);

    return qnn_backend;
}

bool ggml_backend_is_qnn(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_qnn_guid());
}

void ggml_backend_qnn_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_qnn(backend));

    auto *ctx = (ggml_backend_qnn_context *)backend->context;
    ctx->threads = n_threads;
}

int ggml_backend_qnn_get_device_count() { return GGML_QNN_MAX_DEVICES; }

void ggml_backend_qnn_get_device_description(size_t dev_num, char *description, size_t description_size) {
    if (nullptr == description || 0 == description_size) {
        QNN_LOG_WARN("invalid param");
        return;
    }

    if (dev_num >= GGML_QNN_MAX_DEVICES) {
        QNN_LOG_WARN("invalid param");
        return;
    }

    snprintf(description, description_size, "%s", g_qnn_mgr[dev_num].name);
}

ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(size_t device) {
    if (device >= GGML_QNN_MAX_DEVICES) {
        QNN_LOG_DEBUG(
            "ggml_backend_qnn_buffer_type error: device_index:%d is "
            "out of range [0, %d]\n",
            device, GGML_QNN_MAX_DEVICES - 1);
        return nullptr;
    }

    static ggml_backend_qnn_buffer_type_context ggml_backend_qnn_buffer_type_contexts[GGML_QNN_MAX_DEVICES];
    static ggml_backend_buffer_type ggml_backend_qnn_buffer_types[GGML_QNN_MAX_DEVICES];
    static bool ggml_backend_qnn_buffer_type_initialized = false;
    if (!ggml_backend_qnn_buffer_type_initialized) {
        for (size_t i = 0; i < GGML_QNN_MAX_DEVICES; i++) {
            auto &context = ggml_backend_qnn_buffer_type_contexts[i];
            context = { i, std::string(QNN_BACKEND_NAME) + std::to_string(i) };
            ggml_backend_qnn_buffer_types[i] = {
                /* .iface   = */ { /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
                                   /* .alloc_buffer     = */ ggml_backend_qnn_buffer_type_alloc_buffer,
                                   /* .get_alignment    = */ ggml_backend_qnn_buffer_type_get_alignment,
                                   /* .get_max_size     = */ ggml_backend_qnn_buffer_type_get_max_size,
                                   /* .get_alloc_size   = */ nullptr, // defaults to ggml_nbytes
                                   /* .is_host          = */ ggml_backend_qnn_buffer_is_host },
                /* .context = */ &context,
            };
        }
        ggml_backend_qnn_buffer_type_initialized = true;
    }

    return &ggml_backend_qnn_buffer_types[device];
}

/**
 *
 * @param device            0: QNN_BACKEND_CPU 1: QNN_BACKEND_GPU 2: QNN_BACKEND_NPU
 * @param qnn_lib_path      qnn library path, such as "/data/local/tmp/" on Android or specified in JNI layer
 * @return
 */
ggml_backend_t ggml_backend_qnn_init(size_t device, const char *qnn_lib_path) {
    int result = 0;

    if (nullptr == qnn_lib_path) {
        QNN_LOG_ERROR("invalid qnn lib path\n");
        return nullptr;
    }

    QNN_LOG_DEBUG("device %d", device);
    QNN_LOG_DEBUG("qnn_lib_path %s", qnn_lib_path);
    if (device >= GGML_QNN_MAX_DEVICES) {
        QNN_LOG_ERROR("invalid device %d", device);
        return nullptr;
    }

    std::string path = qnn_lib_path;
    if (QNN_BACKEND_NPU == device) {
        if (0 == setenv("LD_LIBRARY_PATH",
                        (path + ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/"
                                "dsp:/vendor/dsp/images")
                            .c_str(),
                        1)) {
            QNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            QNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
        if (0 == setenv("ADSP_LIBRARY_PATH",
                        (path + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/"
                                "rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp")
                            .c_str(),
                        1)) {
            QNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            QNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
    } else {
        if (0 == setenv("LD_LIBRARY_PATH", path.c_str(), 1)) {
            QNN_LOG_INFO("%s backend setenv successfully\n", qnn::get_backend_name(device));
        } else {
            QNN_LOG_ERROR("%s backend setenv failure\n", qnn::get_backend_name(device));
        }
    }

    auto instance = std::make_shared<qnn::qnn_instance>(qnn_lib_path, g_qnn_mgr[device].lib, "");
    result = instance->qnn_init(nullptr);
    if (result != 0) {
        QNN_LOG_WARN("init qnn subsystem failed with qnn backend %s, pls check why\n", qnn::get_backend_name(device));
        return nullptr;
    }
    auto qnn_interface = instance->get_qnn_interface();
    if (!qnn_interface) {
        QNN_LOG_WARN("qnn subsystem failure\n");
        return nullptr;
    }

    std::string device_name = qnn::get_backend_name(device);
    QNN_LOG_INFO("qnn device name %s", device_name.c_str());
    auto &qnn_device = g_qnn_mgr[device];
    qnn_device.instance = instance;
    qnn_device.qnn_interface = qnn_interface;
    qnn_device.socinfo = instance->get_soc_info();

    ggml_backend_t qnn_backend = new ggml_backend{ /* .guid      = */ ggml_backend_qnn_guid(),
                                                   /* .iface     = */ ggml_backend_qnn_interface,
                                                   /* .context   = */ &g_qnn_mgr[device] };
    g_qnn_mgr[device].backend = qnn_backend;

    return qnn_backend;
}

extern "C" GGML_CALL void ggml_backend_qnn_reg_devices();

GGML_CALL void ggml_backend_qnn_reg_devices() {
    for (size_t idx = 0; idx < GGML_QNN_MAX_DEVICES; idx++) {
        char name[GGML_MAX_NAME];
        ggml_backend_qnn_get_device_description(idx, name, GGML_MAX_NAME);
        ggml_backend_register(name, ggml_backend_qnn_reg_init, ggml_backend_qnn_buffer_type(idx),
                              (void *)(intptr_t)idx);
    }
}
