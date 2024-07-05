#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#include <vector>
#include <thread>
#include <mutex>
#include <set>
#include <tuple>
#include <queue>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <memory>
#include <regex>
#include <random>
#include <functional>
#include <condition_variable>
#include <cassert>
#include <unordered_set>
#include <utility>

#include "ggml-qnn.h"

#include "ggml-backend-impl.h"

#include "ggml-qnn/logger.hpp"
#include "ggml-qnn/utils.hpp"
#include "ggml-qnn/tensor.hpp"
#include "ggml-qnn/backend.hpp"
#include "ggml-qnn/backend-ops.hpp"

// =================================================================================================
//
//  forward declaration
//
// =================================================================================================
static int free_qnn_tensor(Qnn_Tensor_t & tensor);

// =================================================================================================
//
//  self-defined macro / data structure
//
// =================================================================================================
#ifdef NDEBUG
#define ENABLE_QNNBACKEND_PERF      0     // enable/disable op's perf info
#else
#define ENABLE_QNNBACKEND_PERF      1     // enable/disable op's perf info
#endif

#define QNN_BACKEND_NAME            "qnn"

static struct qnn::qcom_socinfo g_qnn_soc_info_table[] = {
        /* Qualcomm SnapDragon 8 Gen 1 */
        [qnn::SM8450] = {
                .soc_model         = qnn::SM8450,
                .htp_arch          = qnn::V69,
                .vtcm_size_in_mb   = 8},

        /* Qualcomm SnapDragon 8 Gen 1+ */
        [qnn::SM8475] = {
                .soc_model         = qnn::SM8475,
                .htp_arch          = qnn::V69,
                .vtcm_size_in_mb   = 8},

        /* Qualcomm SnapDragon 8 Gen 2 */
        [qnn::SM8550] = {
                .soc_model         = qnn::SM8550,
                .htp_arch          = qnn::V73,
                .vtcm_size_in_mb   = 8},

        /* Qualcomm SnapDragon 8 Gen 3 */
        [qnn::SM8650] = {
                .soc_model         = qnn::SM8650,
                .htp_arch          = qnn::V75,
                .vtcm_size_in_mb   = 8},

};

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
    [QNN_BACKEND_CPU] = {.device               = 0,
                         .threads              = 1,
                         .name                 = "qnn-cpu",
                         .lib                  = "libQnnCpu.so",
                         .instance             = nullptr,
                         .backend              = nullptr,
                         .raw_interface        = {},
                         .raw_system_interface = {},
                         .socinfo              = {}},

    [QNN_BACKEND_GPU] = {.device               = 1,
                         .threads              = 1,
                         .name                 = "qnn-gpu",
                         .lib                  = "libQnnGpu.so",
                         .instance             = nullptr,
                         .backend              = nullptr,
                         .raw_interface        = {},
                         .raw_system_interface = {},
                         .socinfo              = {}},

    [QNN_BACKEND_NPU] = {.device               = 2,
                         .threads              = 1,
                         .name                 = "qnn-npu",
                         .lib                  = "libQnnHtp.so",
                         .instance             = nullptr,
                         .backend              = nullptr,
                         .raw_interface        = {},
                         .raw_system_interface = {},
                         .socinfo              = {}},
};

struct ggml_backend_qnn_buffer_context {
    ggml_backend_qnn_buffer_context(size_t device)
            : device(device)
            , name(QNN_BACKEND_NAME + std::to_string(device)) {}

    ~ggml_backend_qnn_buffer_context() {
        if (buffer) {
            free(buffer);
        }

        for (auto * sub_buffer : sub_buffers) {
            free(sub_buffer);
        }

        for (auto * qnn_tensor : qnn_tensors) {
            free_qnn_tensor(*qnn_tensor);
            free(qnn_tensor);
        }

        sub_buffers.clear();
        qnn_tensors.clear();
    }
    void * buffer = nullptr;

    struct ggml_backend_qnn_context * backend_ctx = nullptr;

    size_t                      buffer_size = 0;
    std::vector<void *>         sub_buffers;
    std::vector<Qnn_Tensor_t *> qnn_tensors;
    size_t                      device;
    std::string                 name;
};

struct ggml_backend_qnn_buffer_type_context {
    size_t      device;
    std::string name;
};

// =================================================================================================
//
//  QNN backend internal helper functions
//
// =================================================================================================
static size_t memscpy(void * dst, size_t dst_size, const void * src, size_t copy_size) {
    if (!dst || !src || !dst_size || !copy_size) return 0;

    size_t min_size = dst_size < copy_size ? dst_size : copy_size;

    memcpy(dst, src, min_size);

    return min_size;
}

static int deep_copy_qnn_tensors(Qnn_Tensor_t & src, Qnn_Tensor_t & dst) {
    int err = 0;
    VALIDATE_TENSOR_VERSION(src, err);

    dst.version = src.version;
    QNN_TENSOR_SET_NAME(
        dst, ::strndup(QNN_TENSOR_GET_NAME(src),std::string(QNN_TENSOR_GET_NAME(src)).size()));
    if (nullptr == QNN_TENSOR_GET_NAME(dst)) {
        return 1;
    }
    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
    QNN_TENSOR_SET_MEM_TYPE(dst, QNN_TENSOR_GET_MEM_TYPE(src));

    if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_RAW) {
        Qnn_ClientBuffer_t client_buf = {nullptr, 0};
        QNN_TENSOR_SET_CLIENT_BUF(dst, client_buf);
    } else if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_MEMHANDLE) {
        QNN_TENSOR_SET_MEM_HANDLE(dst, nullptr);
    } else {
        return 1;
    }

    Qnn_QuantizeParams_t       src_qparam = QNN_TENSOR_GET_QUANT_PARAMS(src);
    Qnn_QuantizationEncoding_t encoding   = src_qparam.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        Qnn_QuantizeParams_t   src_qparam_cpy = src_qparam;
        Qnn_AxisScaleOffset_t & axis_scale_offset = src_qparam_cpy.axisScaleOffsetEncoding;
        Qnn_ScaleOffset_t ** scaleOffset = & axis_scale_offset.scaleOffset;
        size_t              scaleOffsetSize = axis_scale_offset.numScaleOffsets * sizeof(Qnn_ScaleOffset_t);
        *scaleOffset = (Qnn_ScaleOffset_t *) malloc(scaleOffsetSize);
        memscpy(*scaleOffset, scaleOffsetSize,
                src_qparam.axisScaleOffsetEncoding.scaleOffset,
                scaleOffsetSize);
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam_cpy);
    } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
        Qnn_QuantizeParams_t     src_qparam_cpy = src_qparam;
        Qnn_BwAxisScaleOffset_t & bwaxis_scale_offset = src_qparam_cpy.bwAxisScaleOffsetEncoding;
        size_t    scaleSize = bwaxis_scale_offset.numElements * sizeof(float);
        float **  scales    = &bwaxis_scale_offset.scales;
        int32_t ** offsets  = &bwaxis_scale_offset.offsets;
        *scales             = (float *) malloc(scaleSize);
        memscpy(*scales, scaleSize, src_qparam.bwAxisScaleOffsetEncoding.scales,
                scaleSize);

        if (bwaxis_scale_offset.offsets != nullptr) {
            size_t offsetSize = bwaxis_scale_offset.numElements * sizeof(int32_t);
            *offsets = (int32_t *) malloc(offsetSize);
            memscpy(*offsets, offsetSize,
                    src_qparam.bwAxisScaleOffsetEncoding.offsets, offsetSize);
        }
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam_cpy);
    } else {
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam);
    }

    uint32_t rank = QNN_TENSOR_GET_RANK(src);
    QNN_TENSOR_SET_RANK(dst, rank);
    size_t    dim_size    = rank * sizeof(uint32_t);
    uint32_t * dimensions = (uint32_t *) malloc(dim_size);
    if (dimensions == nullptr) {
        QNN_LOG_WARN("deep_copy_qnn_tensors() allocation error while copying "
                     "tensor %s\n",
                     QNN_TENSOR_GET_NAME(src));
        return 1;
    }
    memscpy(dimensions, dim_size, QNN_TENSOR_GET_DIMENSIONS(src), dim_size);
    QNN_TENSOR_SET_DIMENSIONS(dst, dimensions);

    return err;
}

static int free_qnn_tensor(Qnn_Tensor_t & tensor) {
    int err = 0;
    VALIDATE_TENSOR_VERSION(tensor, err);

    free((void *) QNN_TENSOR_GET_NAME(tensor));
    free(QNN_TENSOR_GET_DIMENSIONS(tensor));

    return err;
}

// =================================================================================================
//
//  implementation of QNN backend for GGML
//
// =================================================================================================
static bool ggml_qnn_can_handle_op(ggml_backend_qnn_context * ctx,
                                   const struct ggml_tensor * tensor,
                                   bool b_dump_tensor_info) {
    if (ggml_is_empty(tensor) || !qnn::ggml_qnn_op_array()[tensor->op]) {
        return false;
    }

    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    if (nullptr == src0 || nullptr == src1) {
        return false;
    }

    const auto ne00 = src0->ne[0];
    const auto ne01 = src0->ne[1];
    const auto ne10 = src1->ne[0];
    const auto ne11 = src1->ne[1];
    // make qnn_get_ggml_tensor_rank and QNN SDK happy
    if (ne00 <= 1 || ne01 <= 1 || ne10 <= 1 || ne11 <= 1) {
        return false;
    }

    // TODO: support other GGML OPs using QNN API
    // a GENERAL approach could fix this problem in a standalone PR of refine ggml backend
    // subsystem for hybrid inference between CPU&GPU / CPU&NPU easily(less the 100 LoC and no
    // side-effect to the existing codes) for ANY ggml backends which the backend's
    // ggml_backend_xxx_buffer_is_host return true. this approach could be found at:
    // https://github.com/ggerganov/llama.cpp/pull/7641
    bool supported_op = false;
    supported_op = (tensor->op == GGML_OP_ADD);
    supported_op = ((tensor->op == GGML_OP_ADD) || (tensor->op == GGML_OP_MUL_MAT));
    if (!supported_op) {
        return false;
    }

    //TODO: support other quantized data type
    if (ggml_is_quantized(src0->type)) {
        if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_0) {
            return false;
        }
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        if (ne00 <= 32 || ne01 <= 32 || ne10 <= 32 || ne11 <= 32) {
            //comment it for make UT of mul_mat with QNN RPC happy
            //return false;
        }
    }

    return true;
}

bool ggml_qnn_compute_forward(ggml_backend_qnn_context * ctx, struct ggml_tensor * tensor) {
    auto func = qnn::ggml_qnn_op_array()[tensor->op];
    if (!func) {
        QNN_LOG_WARN("unsupported op %d", tensor->op);
        return false;
    }

    func(ctx, tensor->src[0], tensor->src[1], tensor);
    return true;
}

static const char * ggml_backend_qnn_buffer_get_name(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return "QNN";
}

GGML_CALL static bool ggml_backend_buffer_is_qnn(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_qnn_buffer_get_name;
}

GGML_CALL static void ggml_backend_qnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *) buffer->context;

    delete ctx;
}

GGML_CALL static void * ggml_backend_qnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *) buffer->context;

    return ctx->buffer;
}

GGML_CALL static void ggml_backend_qnn_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                        ggml_tensor * tensor) {
    Qnn_ErrorHandle_t               error = QNN_SUCCESS;
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *) buffer->context;

    static int idx                        = 0;
    char       tensor_name[GGML_MAX_NAME] = {0};
    snprintf(tensor_name, GGML_MAX_NAME, "tensor_%04d", idx++);

    uint32_t dimensions[] = {(uint32_t) tensor->ne[0], (uint32_t) tensor->ne[1],
                             (uint32_t) tensor->ne[2],
                             (uint32_t) tensor->ne[3]};
    Qnn_DataType_t qnn_data_type =
        qnn::datatype_from_ggml_datatype(tensor->type);
    Qnn_TensorType_t qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;

    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
    } else if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
    }
    Qnn_Tensor_t qnn_tensor = QNN_TENSOR_INIT;

    Qnn_TensorMemType_t qnn_mem_type = QNN_TENSORMEMTYPE_RAW;
    if (ctx->device == QNN_BACKEND_GPU) {
        qnn_mem_type = QNN_TENSORMEMTYPE_MEMHANDLE;
    }

    qnn_tensor = {
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {.id         = 0,
                    .name       = tensor_name,
                    .type       = qnn_tensor_type,
                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                    .dataType   = qnn_data_type,
                    .quantizeParams =
                            {QNN_DEFINITION_UNDEFINED,
                             QNN_QUANTIZATION_ENCODING_UNDEFINED,
                             {.scaleOffsetEncoding = {.scale  = 0.0000000000000000f,
                                     .offset = 0}}},
                    .rank       = qnn::get_ggml_tensor_rank(tensor),
                    .dimensions = dimensions,
                    .memType    = qnn_mem_type,
                    {.clientBuf = {.data = nullptr, .dataSize = 0}}}}};

    Qnn_Tensor_t * p_qnn_tensor =
        (Qnn_Tensor_t *)calloc(1, sizeof(Qnn_Tensor_t));
    if (nullptr == p_qnn_tensor) {
        QNN_LOG_WARN("calloc failed");
        return;
    }
    error = deep_copy_qnn_tensors(qnn_tensor, *p_qnn_tensor);
    if (error != QNN_SUCCESS) {
        free(p_qnn_tensor);
        QNN_LOG_WARN("init tensor failed");
        return;
    }
    tensor->extra = p_qnn_tensor;
    ctx->qnn_tensors.push_back(p_qnn_tensor);
}

GGML_CALL static void ggml_backend_qnn_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                       ggml_tensor * tensor, const void * data,
                                       size_t offset, size_t size) {
    GGML_UNUSED(buffer);

    memcpy((char *) tensor->data + offset, data, size);
}

GGML_CALL static void ggml_backend_qnn_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                       const ggml_tensor * tensor, void * data,
                                       size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

GGML_CALL static bool ggml_backend_qnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                       const struct ggml_tensor * src,
                                       struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

GGML_CALL static void ggml_backend_qnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *) buffer->context;

    memset(ctx->buffer, value, ctx->buffer_size);
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

GGML_CALL static const char * ggml_backend_qnn_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "QNN";
}

static void * ggml_qnn_host_malloc(size_t n) {
    void * data = nullptr;
    int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        QNN_LOG_WARN("%s: error: posix_memalign failed\n", __func__);
        return nullptr;
    }

    return data;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_qnn_buffer_type_context * buft_ctx = (ggml_backend_qnn_buffer_type_context *)buft->context;
    ggml_backend_qnn_buffer_context * ctx = new ggml_backend_qnn_buffer_context(buft_ctx->device);

    size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    // TODO:use pre-allocated buffer in internal memory pool
    ctx->buffer      = ggml_qnn_host_malloc(size_aligned);
    ctx->buffer_size = size_aligned;

    ctx->backend_ctx = &g_qnn_mgr[buft_ctx->device];

    if (nullptr == ctx->buffer) {
        QNN_LOG_WARN("%s: failed to allocate %.2f MiB\n", __func__, size / (1 << 20));
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_qnn_buffer_interface,ctx, size);
}

GGML_CALL static size_t ggml_backend_qnn_buffer_type_get_alignment(
    ggml_backend_buffer_type_t buft) {
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

GGML_CALL static const char * ggml_backend_qnn_name(ggml_backend_t backend) {
    return "QNN";
}

GGML_CALL static void ggml_backend_qnn_free(ggml_backend_t backend) {
    QNN_LOG_INFO("enter %s", __func__);
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;
    QNN_LOG_INFO("idx %d, name:%s", ctx->device, g_qnn_mgr[ctx->device].name);

    auto *instance = g_qnn_mgr[ctx->device].instance;
    if (instance != nullptr) {
        for (const auto &graph_item: ctx->qnn_graph_map) {
            QNN_LOG_INFO("graph type:%s", graph_item.first.c_str());
        }

        ctx->qnn_graph_map.clear();

        instance->qnn_finalize();
        delete instance;
        g_qnn_mgr[ctx->device].instance = nullptr;
    }

    if (g_qnn_mgr[ctx->device].backend != nullptr) {
        delete backend;
        g_qnn_mgr[ctx->device].backend = nullptr;
    }
    QNN_LOG_INFO("leave %s", __func__);
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_qnn_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;

    return ggml_backend_qnn_buffer_type(ctx->device);
}

GGML_CALL static ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    enum ggml_status          result = GGML_STATUS_SUCCESS;
    ggml_backend_qnn_context * ctx   = (ggml_backend_qnn_context *) backend->context;
    GGML_UNUSED(ctx);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        bool ok = ggml_qnn_compute_forward(ctx, node);
        if (!ok) {
            QNN_LOG_DEBUG("error: op not supported %s (%s)\n", node->name, ggml_op_name(node->op));
        }
    }

    return result;
}

GGML_CALL static bool ggml_backend_qnn_supports_op(ggml_backend_t backend,
                                                   const ggml_tensor * op) {
    ggml_backend_qnn_context *ctx = (ggml_backend_qnn_context *) backend->context;

    return (ggml_qnn_can_handle_op(ctx, op, false));
}

GGML_CALL static bool ggml_backend_qnn_offload_op(ggml_backend_t backend,const ggml_tensor * tensor) {
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;

    return ggml_qnn_can_handle_op(ctx, tensor, false);
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
    static ggml_guid guid = {
            0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
            0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09
            };
    return &guid;
}

static ggml_backend_t ggml_backend_qnn_reg_init(const char * params, void * user_data) {
    if (nullptr == params) {
        // QNN library path
        // can be hardcoded to "/data/local/tmp/" for Android command line application
        // or specified in JNI layer for Android APK
        params = "/data/local/tmp/";
    }
    ggml_backend_t qnn_backend = ggml_backend_qnn_init((int) (intptr_t) user_data, params);

    return qnn_backend;
}

bool ggml_backend_is_qnn(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_qnn_guid());
}

void ggml_backend_qnn_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_qnn(backend));

    auto * ctx = (ggml_backend_qnn_context *) backend->context;
    ctx->threads = n_threads;
}

const char * ggml_backend_qnn_get_name(ggml_backend_t backend) {
    return backend->iface.get_name(backend);
}

int ggml_backend_qnn_get_device_count() {
    return GGML_QNN_MAX_DEVICES;
}

void ggml_backend_qnn_get_device_description(size_t dev_num, char * description, size_t description_size) {
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
        QNN_LOG_DEBUG("ggml_backend_qnn_buffer_type error: device_index:%d is "
                      "out of range [0, %d]\n",
                      device, GGML_QNN_MAX_DEVICES - 1);
        return nullptr;
    }

    static ggml_backend_qnn_buffer_type_context ggml_backend_qnn_buffer_type_contexts[GGML_QNN_MAX_DEVICES];
    static ggml_backend_buffer_type ggml_backend_qnn_buffer_types[GGML_QNN_MAX_DEVICES];
    static bool ggml_backend_qnn_buffer_type_initialized = false;
    if (!ggml_backend_qnn_buffer_type_initialized) {
        for (size_t i = 0; i < GGML_QNN_MAX_DEVICES; i++) {
            auto & context = ggml_backend_qnn_buffer_type_contexts[i];
            context = { i, std::string(QNN_BACKEND_NAME) + std::to_string(i) };
            ggml_backend_qnn_buffer_types[i] = {
                /* .iface   = */ {
                    /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
                    /* .alloc_buffer     = */ ggml_backend_qnn_buffer_type_alloc_buffer,
                    /* .get_alignment    = */ ggml_backend_qnn_buffer_type_get_alignment,
                    /* .get_max_size     = */ ggml_backend_qnn_buffer_type_get_max_size,
                    /* .get_alloc_size   = */ nullptr, // defaults to ggml_nbytes
                    /* .is_host          = */ ggml_backend_qnn_buffer_is_host
                    },
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
ggml_backend_t ggml_backend_qnn_init(size_t device, const char * qnn_lib_path) {
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
                        (path +
                         ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/"
                         "rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp")
                            .c_str(),
                        1)) {
            QNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            QNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
    } else {
        if (0 == setenv("LD_LIBRARY_PATH", path.c_str(), 1)) {
            QNN_LOG_INFO("%s backend setenv successfully\n",
                         qnn::get_backend_name(device));
        } else {
            QNN_LOG_ERROR("%s backend setenv failure\n",
                          qnn::get_backend_name(device));
        }
    }

    auto *instance = new qnn::qnn_instance(qnn_lib_path, g_qnn_mgr[device].lib, "");
    result = instance->qnn_init(nullptr);
    if (0 != result) {
        QNN_LOG_WARN(
            "init qnn subsystem failed with qnn backend %s, pls check why\n",
            qnn::get_backend_name(device));
        delete instance;
        return nullptr;
    }
    auto qnn_interface = instance->get_qnn_interface();
    if (!qnn_interface.is_loaded()) {
        QNN_LOG_WARN("qnn subsystem failure\n");
        delete instance;
        return nullptr;
    }

    std::string device_name = qnn::get_backend_name(device);
    QNN_LOG_INFO("qnn device name %s", device_name.c_str());
    g_qnn_mgr[device].instance             = instance;
    g_qnn_mgr[device].raw_interface        = instance->get_qnn_raw_interface();
    g_qnn_mgr[device].raw_system_interface = instance->get_qnn_raw_system_interface();
    g_qnn_mgr[device].socinfo              = instance->get_soc_info();

    ggml_backend_t qnn_backend =
        new ggml_backend{/* .guid      = */ ggml_backend_qnn_guid(),
                         /* .iface     = */ ggml_backend_qnn_interface,
                         /* .context   = */ &g_qnn_mgr[device]};
    g_qnn_mgr[device].backend = qnn_backend;

    return qnn_backend;
}

extern "C" GGML_CALL int ggml_backend_qnn_reg_devices(void);

GGML_CALL int ggml_backend_qnn_reg_devices() {
    for (size_t idx = 0; idx < GGML_QNN_MAX_DEVICES; idx++) {
        char name[GGML_MAX_NAME];
        ggml_backend_qnn_get_device_description(idx, name, GGML_MAX_NAME);
        ggml_backend_register(name, ggml_backend_qnn_reg_init,
                              ggml_backend_qnn_buffer_type(idx),
                              (void *) (intptr_t) idx);
    }

    return GGML_QNN_MAX_DEVICES;
}
