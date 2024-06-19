#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
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
#include <unordered_map>
#include <condition_variable>
#include <cassert>
#include <unordered_set>
#include <utility>

#include "ggml-qnn.h"

#include "ggml-backend-impl.h"

#include "ggml-qnn/logger.hpp"
#include "ggml-qnn/utils.hpp"
#include "ggml-qnn/backend.hpp"
#include "ggml-qnn/tensor.hpp"

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

typedef void (*ggml_qnn_func_t)(ggml_backend_qnn_context * ctx,
                                const ggml_tensor * src0,
                                const ggml_tensor * src1,
                                ggml_tensor * dst);

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
// TODO: only support GGML_OP_ADD/GGML_OP_MUL/GGML_OP_MUL_MAT
static const char * qnn_opname_from_ggmlop(enum ggml_op ggmlop) {
    switch (ggmlop) {
        case GGML_OP_ADD:
            return QNN_OP_ELEMENT_WISE_ADD;
        case GGML_OP_MUL:
            return QNN_OP_ELEMENT_WISE_MULTIPLY;
        case GGML_OP_MUL_MAT:
            return QNN_OP_MAT_MUL;
        default:
            break;
    }
    return nullptr;
}

static bool qnn_is_valid_params(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                            const ggml_tensor * src1, ggml_tensor * dst) {
    if ((nullptr == ctx) || (nullptr == src0) || (nullptr == src1) || (nullptr == dst)) {
        QNN_LOG_WARN("invalid params\n");
        return false;
    }

    qnn::qnn_instance *instance = nullptr;
    Qnn_Tensor_t      *tensor_0 = nullptr;
    Qnn_Tensor_t      *tensor_1 = nullptr;
    Qnn_Tensor_t      *tensor_2 = nullptr;
    tensor_0 = (Qnn_Tensor_t *) src0->extra;
    tensor_1 = (Qnn_Tensor_t *) src1->extra;
    tensor_2 = (Qnn_Tensor_t *) dst->extra;
    instance = ctx->instance;
    if ((nullptr == instance) || (nullptr == tensor_0) || (nullptr == tensor_1) || (nullptr == tensor_2)) {
        QNN_LOG_WARN("invalid params\n");
        return false;
    }

    return true;
}

#ifndef NDEBUG
#define CHECK_PARAMS(ctx, src0, src1, dst)                          \
    do {                                                            \
        if (!qnn_is_valid_params((ctx), (src0), (src1), (dst))) {   \
            return;                                                 \
        }                                                           \
    } while (0)

#else
#define CHECK_PARAMS(ctx, src0, src1, dst)
#endif

#if ENABLE_QNNBACKEND_PERF
class qnn_perf {
public:
    qnn_perf(const std::string & perf_name) : _perf_name(std::move(perf_name)) {};
    qnn_perf() = delete;
    qnn_perf(const qnn_perf & ) = delete;
    qnn_perf & operator= (const qnn_perf & ) = delete;

    void start() {
        _begin_time = ggml_time_us();
    }

    void info() {
        _end_time = ggml_time_us();
        _duration = (_end_time - _begin_time);
        QNN_LOG_INFO("duration of %s : %lld microseconds\n", _perf_name.c_str(), _duration);
    }

private:
    int64_t _begin_time = 0LL;
    int64_t _end_time   = 0LL;
    int64_t _duration   = 0LL;
    std::string _perf_name;
};
#else
class qnn_perf {
public:
    qnn_perf(const std::string & perf_name) {}
    qnn_perf() = delete;
    qnn_perf(const qnn_perf & ) = delete;
    qnn_perf & operator= (const qnn_perf & ) = delete;

    void start() {}
    void info() {}
};
#endif

#define VALIDATE(value, status)                                                \
    do {                                                                       \
        status = value;                                                        \
        if (status != QNN_SUCCESS) {                                           \
            QNN_LOG_WARN("%s expected QNN_SUCCESS\n", #value);                 \
            return status;                                                     \
        }                                                                      \
    } while (0)

#define QNN_TENSOR_GET_ID(tensor)                   get_qnn_tensorid(tensor)
#define QNN_TENSOR_GET_NAME(tensor)                 get_qnn_tensorname(tensor)
#define QNN_TENSOR_GET_TYPE(tensor)                 get_qnn_tensortype(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor)          get_qnn_tensor_dataformat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor)            get_qnn_tensor_datatype(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor)         get_qnn_tensor_quantparams(tensor)
#define QNN_TENSOR_GET_RANK(tensor)                 get_qnn_tensor_rank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor)           get_qnn_tensor_dimensions(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor)             get_qnn_tensor_memtype(tensor)

#define QNN_TENSOR_SET_ID(tensor, value)            set_qnn_tensor_id(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value)          set_qnn_tensor_name(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value)          set_qnn_tensor_type(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value)   set_qnn_tensor_dataformat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value)     set_qnn_tensor_datatype(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value)  set_qnn_tensor_quantparams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value)          set_qnn_tensor_rank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value)    set_qnn_tensor_dimensions(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value)      set_qnn_tensor_memtype(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value)    set_qnn_tensor_clientbuf(tensor, value)
#define QNN_TENSOR_SET_MEM_HANDLE(tensor, value)    set_qnn_tensor_memhandle(tensor, value)
#define VALIDATE_TENSOR_VERSION(tensor, err)        VALIDATE(validate_tensor_version(tensor), err)

static inline int validate_tensor_version(Qnn_Tensor_t tensor) {
    if (tensor.version != QNN_TENSOR_VERSION_1) {
        QNN_LOG_WARN(
            "validate_tensor_version() tensor %s, got unsupported version %d\n",
            tensor.v1.name, tensor.version);
        return 1;
    }
    return 0;
}

static inline uint32_t get_qnn_tensorid(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.id;
    }

    return 0u;
}

static inline const char * get_qnn_tensorname(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.name;
    }
    return nullptr;
}

static inline Qnn_TensorType_t get_qnn_tensortype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.type;
    }
    return QNN_TENSOR_TYPE_UNDEFINED;
}

static inline Qnn_TensorDataFormat_t
    get_qnn_tensor_dataformat(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataFormat;
    }
    return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}

static inline Qnn_DataType_t
    get_qnn_tensor_datatype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataType;
    }
    return QNN_DATATYPE_UNDEFINED;
}

static inline Qnn_QuantizeParams_t
    get_qnn_tensor_quantparams(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.quantizeParams;
    }
    return QNN_QUANTIZE_PARAMS_INIT;
}

static inline uint32_t get_qnn_tensor_rank(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.rank;
    }
    return 0u;
}

static inline uint32_t * get_qnn_tensor_dimensions(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dimensions;
    }
    return nullptr;
}

static inline Qnn_TensorMemType_t get_qnn_tensor_memtype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.memType;
    }
    return QNN_TENSORMEMTYPE_UNDEFINED;
}

static inline void set_qnn_tensor_id(Qnn_Tensor_t & tensor, uint32_t id) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.id = id;
    }
}

static inline void set_qnn_tensor_name(Qnn_Tensor_t & tensor, const char * name) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.name = name;
    }
}

static inline void set_qnn_tensor_type(Qnn_Tensor_t & tensor, Qnn_TensorType_t type) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.type = type;
    }
}

static inline void set_qnn_tensor_dataformat(Qnn_Tensor_t & tensor, Qnn_TensorDataFormat_t format) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dataFormat = format;
    }
}

static inline void set_qnn_tensor_datatype(Qnn_Tensor_t & tensor, Qnn_DataType_t dataType) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dataType = dataType;
    }
}

static inline void set_qnn_tensor_quantparams(Qnn_Tensor_t & tensor, Qnn_QuantizeParams_t params) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.quantizeParams = params;
    }
}

static inline void set_qnn_tensor_rank(Qnn_Tensor_t & tensor, uint32_t rank) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.rank = rank;
    }
}

static inline void set_qnn_tensor_dimensions(Qnn_Tensor_t & tensor, uint32_t * dims) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dimensions = dims;
    }
}

static inline void set_qnn_tensor_memtype(Qnn_Tensor_t & tensor, Qnn_TensorMemType_t mem_type) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memType = mem_type;
    }
}

static inline void set_qnn_tensor_clientbuf(Qnn_Tensor_t & tensor, Qnn_ClientBuffer_t client_buf) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.clientBuf = client_buf;
    }
}

static inline void set_qnn_tensor_memhandle(Qnn_Tensor_t & tensor, Qnn_MemHandle_t handle) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memHandle = handle;
    }
}

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
    if (ggml_is_empty(tensor) || tensor->op == GGML_OP_RESHAPE ||
        tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_VIEW ||
        tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_NONE) {
        return false;
    }

    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    if (nullptr == src0 || nullptr == src1) {
        return false;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
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


//TODO: this function can be removed later because there are duplicated codes with ggml_qnn_mul_mat
//      keep it for illustrate how to implement a specified GGMPL OP using QNN API + QNN RPC
static void ggml_qnn_add(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                         const ggml_tensor * src1, ggml_tensor * dst) {
    Qnn_ErrorHandle_t  error             = QNN_SUCCESS;
    bool               graph_initialized = false;
    qnn::qnn_instance *instance          = nullptr;
    std::string        graph_name        = "ggml_op_qnn_add";
    Qnn_GraphHandle_t  graph_handle      = nullptr;
    Qnn_Param_t        qnn_params[]      = {};
    enum ggml_op       ggmlop            = GGML_OP_ADD;

    CHECK_PARAMS(ctx, src0, src1, dst);
    instance = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface = ctx->raw_interface;

    qnn_perf perf("ggml_qnn_add");
    perf.start();

    std::string map_entry = std::string(ggml_op_name(ggmlop));
    if (instance->_qnn_graph_map.find(map_entry) !=
        instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle      = std::get<0>(graph_item);
    }

    if (!graph_initialized) {
        graph_name = graph_name + "_" + std::to_string(ctx->threads) +
                     "_" + src0->name + "_" + src1->name;
        QNN_LOG_INFO("graph name %s", graph_name.c_str());
        if (ctx->device == QNN_BACKEND_NPU) {
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
            opt_config.optimizationOption.floatValue = 1;    // 1 / 3
            QnnGraph_Config_t graph_opt_config;
            graph_opt_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graph_opt_config.customConfig = &opt_config;

            QnnHtpGraph_CustomConfig_t vtcm_config;
            vtcm_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
            vtcm_config.vtcmSizeInMB = ctx->socinfo.vtcm_size_in_mb;
            QnnGraph_Config_t graph_vtcm_config;
            graph_vtcm_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graph_vtcm_config.customConfig = &vtcm_config;

            const QnnGraph_Config_t * p_graphconfig[] = {&graph_hvx_config,
                                                         &graph_dlbc_config,
                                                         &graph_vtcm_config,
                                                         &graph_opt_config,
                                                         NULL};
            error = qnn_raw_interface.graphCreate(
                    instance->get_qnn_context_handle(), graph_name.c_str(), p_graphconfig,
                    &graph_handle);
        } else {
            error = qnn_raw_interface.graphCreate(
                    instance->get_qnn_context_handle(), graph_name.c_str(), nullptr,
                    &graph_handle);
        }

        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("can't create qnn graph handle with graph name %s, "
                         "error = %d\n",
                         graph_name.c_str(), error);
            goto failure;
        } else {
            QNN_LOG_INFO("create qnn graph handle with graph name %s ok\n", graph_name.c_str());
        }

        qnn::ggml_qnn_tensor_input tensor_input0(src0, graph_handle, ctx);
        if (!tensor_input0.is_valid()) {
            goto failure;
        }
        qnn::ggml_qnn_tensor_input tensor_input1(src1, graph_handle, ctx);
        if (!tensor_input1.is_valid()) {
            QNN_LOG_INFO("error = %d\n", error);
            goto failure;
        }
        qnn::ggml_qnn_tensor_output tensor_output(dst, graph_handle, ctx);
        if (!tensor_output.is_valid()) {
            goto failure;
        }

        Qnn_Tensor_t   tensor_inputs[]  = {*tensor_input0.get_qnn_tensor(), *tensor_input1.get_qnn_tensor()};
        Qnn_Tensor_t   tensor_outputs[] = {*tensor_output.get_qnn_tensor()};
        Qnn_OpConfig_t op_config        = {
            (Qnn_OpConfigVersion_t) 1,
            .v1 = {"ggml_op_add",
                   QNN_OP_PACKAGE_NAME_QTI_AISW,
                   QNN_OP_ELEMENT_WISE_ADD,
                   0, qnn_params,
                   2, tensor_inputs,
                   1,tensor_outputs}
        };
        error = qnn_raw_interface.graphAddNode(graph_handle, op_config);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
            goto failure;
        }
        error = qnn_raw_interface.graphFinalize(graph_handle,
                                                nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
            goto failure;
        }
        error = qnn_raw_interface.graphExecute(graph_handle,
                                           tensor_inputs, 2,
                                           tensor_outputs, 1,
                                           nullptr, nullptr);
        if (ctx->device == QNN_BACKEND_NPU) {
            if (QNN_COMMON_ERROR_SYSTEM_COMMUNICATION == error) {
                QNN_LOG_WARN("NPU crashed. SSR detected. Caused QNN graph execute error\n");
            }
        }
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
            goto failure;
        }

        auto graph_item = std::make_tuple(graph_handle, 
                                          tensor_input0.get_qnn_tensor(), 
                                          tensor_input1.get_qnn_tensor(), 
                                          tensor_output.get_qnn_tensor());
        instance->_qnn_graph_map[map_entry] = graph_item;
    } else {
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        qnn::ggml_qnn_tensor_input tensor_input0(src0, std::get<1>(graph_item), ctx);
        qnn::ggml_qnn_tensor_input tensor_input1(src1, std::get<2>(graph_item), ctx);
        qnn::ggml_qnn_tensor_output tensor_output(dst, std::get<3>(graph_item), ctx);

        Qnn_Tensor_t tensor_inputs[]  = {*tensor_input0.get_qnn_tensor(), *tensor_input1.get_qnn_tensor()};
        Qnn_Tensor_t tensor_outputs[] = {*tensor_output.get_qnn_tensor()};
        error = qnn_raw_interface.graphExecute(graph_handle,
                                           tensor_inputs,2,
                                           tensor_outputs,1,
                                           nullptr, nullptr);
        if (ctx->device == QNN_BACKEND_NPU) {
            if (QNN_COMMON_ERROR_SYSTEM_COMMUNICATION == error) {
                QNN_LOG_WARN("NPU crashed. SSR detected. Caused QNN graph execute error\n");
            }
        }
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
            goto failure;
        }
    }

failure:
    if (QNN_SUCCESS != error) {
        QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64
                              " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
                      src0->name, src0->type, ggml_type_name(src0->type),
                      src0->ne[0], src0->ne[1], src0->ne[2], src0->nb[0],
                      src0->nb[1], src0->nb[2]);
        QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64
                              " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
                      src1->name, src1->type, ggml_type_name(src1->type),
                      src1->ne[0], src1->ne[1], src1->ne[2], src1->nb[0],
                      src1->nb[1], src1->nb[2]);
        QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64
                              " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
                      dst->name, dst->type, ggml_type_name(dst->type),
                      dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0],
                      dst->nb[1], dst->nb[2]);
    }

    perf.info();
}

/*
 * ggml_qnn_mul_mat was re-added as a standalone function because
 * the following comments came from https://github.com/ggerganov/llama.cpp/pull/1632
 * MUL_MAT take most of the compute time (about 95%).
 * So to speed up llama, we have to focus on MUL_MAT.
 *
 * We have three kinds of MUL_MAT to compute:
 * mul_mat_f32:     both src0 and src1 are F32.
 * mul_mat_f16_f32: src0 is F16 and src1 is F32.
 * mul_mat_q_f32:   src0 is quantized (Q4_0, Q4_1, ...), and src1 is F32.
 */
static void ggml_qnn_mul_mat(ggml_backend_qnn_context * ctx,
                             const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst) {
    Qnn_ErrorHandle_t  error              = QNN_SUCCESS;
    bool               graph_initialized  = false;
    qnn::qnn_instance *instance           = nullptr;
    std::string        graph_name         = "ggml_op_qnn_mul_mat";
    Qnn_GraphHandle_t  graph_handle       = nullptr;
    Qnn_Param_t        qnn_params[]             = {};
    enum ggml_op       ggmlop             = GGML_OP_MUL_MAT;

    CHECK_PARAMS(ctx, src0, src1, dst);
    instance = ctx->instance;
    QNN_INTERFACE_VER_TYPE qnn_raw_interface = ctx->raw_interface;

    qnn_perf perf("ggml_qnn_mul_mat");
    perf.start();

    std::string map_entry = std::string(ggml_op_name(ggmlop));
    if (instance->_qnn_graph_map.find(map_entry) !=
        instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle      = std::get<0>(graph_item);
    }

    //TODO: for scenarios of quantized data in src0
    //      pass-1: dequantize src0 to FP32
    //      pass-2: dq-src0 * src1
    //      the performance gains is worth although there is performance loss in pass-1

    if (!graph_initialized) {
        graph_name = graph_name + "_" + std::to_string(ctx->threads) +
                     "_" + src0->name + "_" + src1->name;
        QNN_LOG_INFO("graph name %s", graph_name.c_str());
        if (ctx->device == QNN_BACKEND_NPU) {
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
            opt_config.optimizationOption.floatValue = 1; //1 / 3
            QnnGraph_Config_t graph_opt_config;
            graph_opt_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graph_opt_config.customConfig = &opt_config;

            QnnHtpGraph_CustomConfig_t vtcm_config;
            vtcm_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
            vtcm_config.vtcmSizeInMB = ctx->socinfo.vtcm_size_in_mb;
            QnnGraph_Config_t graph_vtcm_config;
            graph_vtcm_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graph_vtcm_config.customConfig = &vtcm_config;

            const QnnGraph_Config_t * p_graphconfig[] = {&graph_hvx_config,
                                                         &graph_dlbc_config,
                                                         &graph_vtcm_config,
                                                         &graph_opt_config,
                                                         NULL};
            error = qnn_raw_interface.graphCreate(
                    instance->get_qnn_context_handle(), graph_name.c_str(), p_graphconfig,
                    &graph_handle);
        } else {
            error = qnn_raw_interface.graphCreate(
                    instance->get_qnn_context_handle(), graph_name.c_str(), nullptr,
                    &graph_handle);
        }
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("can't create qnn graph handle with graph name %s, "
                         "error = %d\n",
                         graph_name.c_str(), error);
            goto failure;
        }

        qnn::ggml_qnn_tensor_input tensor_input0(src0, graph_handle, ctx);
        if (!tensor_input0.is_valid()) {
            goto failure;
        }
        qnn::ggml_qnn_tensor_input tensor_input1(src1, graph_handle, ctx);
        if (!tensor_input1.is_valid()) {
            goto failure;
        }
        qnn::ggml_qnn_tensor_output tensor_output(dst, graph_handle, ctx);
        if (!tensor_output.is_valid()) {
            goto failure;
        }

        Qnn_Tensor_t   tensor_inputs[]  = {*tensor_input0.get_qnn_tensor(), *tensor_input1.get_qnn_tensor()};
        Qnn_Tensor_t   tensor_outputs[] = {*tensor_output.get_qnn_tensor()};
        Qnn_OpConfig_t op_config = {
                (Qnn_OpConfigVersion_t) 1,
                .v1 = {"ggml_op_mul_mat",
                       QNN_OP_PACKAGE_NAME_QTI_AISW,
                       QNN_OP_MAT_MUL,
                       0, qnn_params,
                       2, tensor_inputs,
                       1, tensor_outputs}
        };
        error = qnn_raw_interface.graphAddNode(graph_handle, op_config);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
            goto failure;
        }
        error = qnn_raw_interface.graphFinalize(graph_handle,
                                                nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
            goto failure;
        }
        error = qnn_raw_interface.graphExecute(graph_handle,
                                           tensor_inputs, 2,
                                           tensor_outputs, 1,
                                           nullptr, nullptr);
        if (ctx->device == QNN_BACKEND_NPU) {
            if (QNN_COMMON_ERROR_SYSTEM_COMMUNICATION == error) {
                QNN_LOG_WARN("NPU crashed. SSR detected. Caused QNN graph execute error\n");
            }
        }
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
            goto failure;
        }

        auto graph_item = std::make_tuple(graph_handle, 
                                          tensor_input0.get_qnn_tensor(), 
                                          tensor_input1.get_qnn_tensor(), 
                                          tensor_output.get_qnn_tensor());
        instance->_qnn_graph_map[map_entry] = graph_item;
    } else {
        auto & graph_item= instance->_qnn_graph_map[map_entry];
        qnn::ggml_qnn_tensor_input tensor_input0(src0, std::get<1>(graph_item), ctx);
        qnn::ggml_qnn_tensor_input tensor_input1(src1, std::get<2>(graph_item), ctx);
        qnn::ggml_qnn_tensor_output tensor_output(dst, std::get<3>(graph_item), ctx);

        Qnn_Tensor_t tensor_inputs[]  = {*tensor_input0.get_qnn_tensor(), *tensor_input1.get_qnn_tensor()};
        Qnn_Tensor_t tensor_outputs[] = {*tensor_output.get_qnn_tensor()};
        error = qnn_raw_interface.graphExecute(graph_handle,
                                           tensor_inputs, 2,
                                           tensor_outputs, 1,
                                           nullptr, nullptr);
        if (ctx->device == QNN_BACKEND_NPU) {
            if (QNN_COMMON_ERROR_SYSTEM_COMMUNICATION == error) {
                QNN_LOG_WARN("NPU crashed. SSR detected. Caused QNN graph execute error\n");
            }
        }
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
            goto failure;
        }
    }

failure:
    if (QNN_SUCCESS != error) {
        QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64
                              " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
                      src0->name, src0->type, ggml_type_name(src0->type),
                      src0->ne[0], src0->ne[1], src0->ne[2], src0->nb[0],
                      src0->nb[1], src0->nb[2]);
        QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64
                              " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
                      src1->name, src1->type, ggml_type_name(src1->type),
                      src1->ne[0], src1->ne[1], src1->ne[2], src1->nb[0],
                      src1->nb[1], src1->nb[2]);
        QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64
                              " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
                      dst->name, dst->type, ggml_type_name(dst->type), dst->ne[0],
                      dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2]);
    }

    perf.info();
}

static void ggml_qnn_repeat(ggml_backend_qnn_context * ctx,
                            const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst) {
}

static void ggml_qnn_get_rows(ggml_backend_qnn_context * ctx,
                              const ggml_tensor * src0, const ggml_tensor * src1,
                              ggml_tensor * dst) {
}

static void ggml_qnn_acc(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                         const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_div(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                         const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_gelu(ggml_backend_qnn_context * ctx,
                          const ggml_tensor * src0, const ggml_tensor * src1,
                          ggml_tensor * dst) {
}

static void ggml_qnn_silu(ggml_backend_qnn_context * ctx,
                          const ggml_tensor * src0, const ggml_tensor * src1,
                          ggml_tensor * dst) {
}

static void ggml_qnn_gelu_quick(ggml_backend_qnn_context * ctx,
                                const ggml_tensor * src0,
                                const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_tanh(ggml_backend_qnn_context * ctx,
                          const ggml_tensor * src0, const ggml_tensor * src1,
                          ggml_tensor * dst) {
}

static void ggml_qnn_relu(ggml_backend_qnn_context * ctx,
                          const ggml_tensor * src0, const ggml_tensor * src1,
                          ggml_tensor * dst) {
}

static void ggml_qnn_hardsigmoid(ggml_backend_qnn_context * ctx,
                                 const ggml_tensor * src0,
                                 const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_hardswish(ggml_backend_qnn_context * ctx,
                               const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst) {
}

static void ggml_qnn_leaky_relu(ggml_backend_qnn_context * ctx,
                                const ggml_tensor * src0,
                                const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_sqr(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                         const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_norm(ggml_backend_qnn_context * ctx,
                          const ggml_tensor * src0, const ggml_tensor * src1,
                          ggml_tensor * dst) {
}

static void ggml_qnn_group_norm(ggml_backend_qnn_context * ctx,
                                const ggml_tensor * src0,
                                const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_concat(ggml_backend_qnn_context * ctx,
                            const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst) {
}

static void ggml_qnn_upscale(ggml_backend_qnn_context * ctx,
                             const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst) {
}

static void ggml_qnn_pad(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                         const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_rms_norm(ggml_backend_qnn_context * ctx,
                              const ggml_tensor * src0, const ggml_tensor * src1,
                              ggml_tensor * dst) {
}

static void ggml_qnn_cpy(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                         const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_dup(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                         const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_qnn_cpy(ctx, src0, dst, nullptr);
    (void) src1;
}

static void ggml_qnn_mul_mat_id(ggml_backend_qnn_context * ctx,
                                const ggml_tensor * src0,
                                const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_scale(ggml_backend_qnn_context * ctx,
                           const ggml_tensor * src0, const ggml_tensor * src1,
                           ggml_tensor * dst) {
}

static void ggml_qnn_clamp(ggml_backend_qnn_context * ctx,
                           const ggml_tensor * src0, const ggml_tensor * src1,
                           ggml_tensor * dst) {
}

static void ggml_qnn_diag_mask_inf(ggml_backend_qnn_context * ctx,
                                   const ggml_tensor * src0,
                                   const ggml_tensor * src1, ggml_tensor * dst) {
}

static void ggml_qnn_soft_max(ggml_backend_qnn_context * ctx,
                              const ggml_tensor * src0, const ggml_tensor * src1,
                              ggml_tensor * dst) {
}

static void ggml_qnn_rope(ggml_backend_qnn_context * ctx,
                          const ggml_tensor * src0, const ggml_tensor * src1,
                          ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
}

static void ggml_qnn_pool2d(ggml_backend_qnn_context * ctx,
                            const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst) {
}

static void ggml_qnn_im2col(ggml_backend_qnn_context * ctx,
                            const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst) {
}

static void ggml_qnn_sum_rows(ggml_backend_qnn_context * ctx,
                              const ggml_tensor * src0, const ggml_tensor * src1,
                              ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
}

static void ggml_qnn_argsort(ggml_backend_qnn_context * ctx,
                             const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
}

static void ggml_qnn_nop(ggml_backend_qnn_context * ctx, const ggml_tensor * src0,
                         const ggml_tensor * src1, ggml_tensor * dst) {
    (void)src0;
    (void)src1;
    (void)dst;
}

bool ggml_qnn_compute_forward(ggml_backend_qnn_context * ctx,
                              struct ggml_compute_params * params,
                              struct ggml_tensor * tensor) {
    ggml_qnn_func_t        func        = nullptr;

    switch (tensor->op) {
    case GGML_OP_ADD:
        func = ggml_qnn_add;
        break;
    case GGML_OP_MUL_MAT:
        func = ggml_qnn_mul_mat;
        break;
    case GGML_OP_REPEAT:
        func = ggml_qnn_repeat;
        break;
    case GGML_OP_GET_ROWS:
        func = ggml_qnn_get_rows;
        break;
    case GGML_OP_DUP:
        func = ggml_qnn_dup;
        break;
    case GGML_OP_ACC:
        func = ggml_qnn_acc;
        break;
    case GGML_OP_DIV:
        func = ggml_qnn_div;
        break;
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(tensor)) {
        case GGML_UNARY_OP_GELU:
            func = ggml_qnn_gelu;
            break;
        case GGML_UNARY_OP_SILU:
            func = ggml_qnn_silu;
            break;
        case GGML_UNARY_OP_GELU_QUICK:
            func = ggml_qnn_gelu_quick;
            break;
        case GGML_UNARY_OP_TANH:
            func = ggml_qnn_tanh;
            break;
        case GGML_UNARY_OP_RELU:
            func = ggml_qnn_relu;
            break;
        case GGML_UNARY_OP_HARDSIGMOID:
            func = ggml_qnn_hardsigmoid;
            break;
        case GGML_UNARY_OP_HARDSWISH:
            func = ggml_qnn_hardswish;
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_NORM:
        func = ggml_qnn_norm;
        break;
    case GGML_OP_GROUP_NORM:
        func = ggml_qnn_group_norm;
        break;
    case GGML_OP_CONCAT:
        func = ggml_qnn_concat;
        break;
    case GGML_OP_UPSCALE:
        func = ggml_qnn_upscale;
        break;
    case GGML_OP_PAD:
        func = ggml_qnn_pad;
        break;
    case GGML_OP_LEAKY_RELU:
        func = ggml_qnn_leaky_relu;
        break;
    case GGML_OP_RMS_NORM:
        func = ggml_qnn_rms_norm;
        break;
    case GGML_OP_MUL_MAT_ID:
        func = ggml_qnn_mul_mat_id;
        break;
    case GGML_OP_SCALE:
        func = ggml_qnn_scale;
        break;
    case GGML_OP_SQR:
        func = ggml_qnn_sqr;
        break;
    case GGML_OP_CLAMP:
        func = ggml_qnn_clamp;
        break;
    case GGML_OP_CPY:
        func = ggml_qnn_cpy;
        break;
    case GGML_OP_CONT:
        func = ggml_qnn_dup;
        break;
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        func = ggml_qnn_nop;
        break;
    case GGML_OP_DIAG_MASK_INF:
        func = ggml_qnn_diag_mask_inf;
        break;
    case GGML_OP_SOFT_MAX:
        func = ggml_qnn_soft_max;
        break;
    case GGML_OP_ROPE:
        func = ggml_qnn_rope;
        break;
    case GGML_OP_IM2COL:
        func = ggml_qnn_im2col;
        break;
    case GGML_OP_POOL_2D:
        func = ggml_qnn_pool2d;
        break;
    case GGML_OP_SUM_ROWS:
        func = ggml_qnn_sum_rows;
        break;
    case GGML_OP_ARGSORT:
        func = ggml_qnn_argsort;
        break;
    default:
        return false;
    }

    if (nullptr != func) {
        func(ctx, tensor->src[0], tensor->src[1], tensor);
    }

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

GGML_CALL static bool ggml_backend_qnn_buffer_type_supports_backend(
    ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    GGML_UNUSED(buft);

    return ggml_backend_is_qnn(backend) || ggml_backend_is_cpu(backend);
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
        // TODO: this should be done inside the destructor
        std::map<std::string,
                 std::tuple<Qnn_GraphHandle_t, Qnn_Tensor_t *, Qnn_Tensor_t *,
                            Qnn_Tensor_t *>>::iterator graph_it;
        for (graph_it = instance->_qnn_graph_map.begin();
             graph_it != instance->_qnn_graph_map.end(); graph_it++) {
            auto & graph_item   = graph_it->second;
            Qnn_GraphHandle_t & graph_handle = std::get<0>(graph_item);
            GGML_UNUSED(graph_handle);
            QNN_LOG_INFO("graph type:%s", graph_it->first.c_str());
        }
        instance->_qnn_graph_map.clear();

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

    ggml_compute_params params = {};
    params.type                = GGML_TASK_TYPE_COMPUTE;
    params.ith                 = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        bool ok = ggml_qnn_compute_forward(ctx, &params, node);
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

    return ggml_qnn_compute_forward(ctx, nullptr, (ggml_tensor *) tensor);
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
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_qnn_graph_compute,
    /* .supports_op             = */ ggml_backend_qnn_supports_op,
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
                    /* .supports_backend = */ ggml_backend_qnn_buffer_type_supports_backend,
                    /* .is_host          = */ ggml_backend_qnn_buffer_is_host
                    },
                /* .context = */ & context,
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
