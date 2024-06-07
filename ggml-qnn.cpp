#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
#include <stdatomic.h>

#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnBackend.h"
#include "QnnGraph.h"
#include "QnnProperty.h"
#include "QnnTensor.h"
#include "QnnInterface.h"
#include "Saver/QnnSaver.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"

#include "ggml-qnn.h"

#include "ggml-backend-impl.h"

#if (defined __ANDROID__) || (defined ANDROID)
#include <android/log.h>
#endif


// =================================================================================================
//
//  forward/external/helper declaration
//
// =================================================================================================
class qnn_instance;


static void ggml_qnn_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...);


// =================================================================================================
//
//  self-defined macro / data structure
//
// =================================================================================================
#define RPCMEM_DEFAULT_FLAGS                            1
#define RPCMEM_HEAP_ID_SYSTEM                           25

#define GGML_QNN_LOGBUF_LEN                             4096

#define GGML_QNN_DEBUG                                  1  //for troubleshooting QNN backend

#define QNN_LOG_ERROR(...) ggml_qnn_log_internal(GGML_LOG_LEVEL_DEBUG,  __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define QNN_LOG_WARN(...)  ggml_qnn_log_internal(GGML_LOG_LEVEL_DEBUG , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define QNN_LOG_INFO(...)  ggml_qnn_log_internal(GGML_LOG_LEVEL_DEBUG , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#if GGML_QNN_DEBUG
#define QNN_LOG_DEBUG(...) ggml_qnn_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define QNN_LOG_DEBUG(...)
#endif

#define QNN_VER_PTR(x)                                  (&((x).v1))
#define GGML_QNN_NAME                                   "qnn"

#define VALIDATE(value, status)                         \
  do {                                                  \
    status = value;                                     \
    if (status != QNN_SUCCESS) {                        \
      QNN_LOG_WARN("%s expected QNN_SUCCESS\n", #value);       \
      return status;                                    \
    }                                                   \
  } while (0)

#define VALIDATE_TENSOR_VERSION(tensor, err)            VALIDATE(validate_tensor_version(tensor), err)

#define QNN_TENSOR_GET_ID(tensor)                       get_qnn_tensorid(tensor)
#define QNN_TENSOR_GET_NAME(tensor)                     get_qnn_tensorname(tensor)
#define QNN_TENSOR_GET_TYPE(tensor)                     get_qnn_tensortype(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor)              get_qnn_tensor_dataformat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor)                get_qnn_tensor_datatype(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor)             get_qnn_tensor_quantparams(tensor)
#define QNN_TENSOR_GET_RANK(tensor)                     get_qnn_tensor_rank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor)               get_qnn_tensor_dimensions(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor)                 get_qnn_tensor_memtype(tensor)

#define QNN_TENSOR_SET_ID(tensor, value)                set_qnn_tensor_id(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value)              set_qnn_tensor_name(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value)              set_qnn_tensor_type(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value)       set_qnn_tensor_dataformat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value)         set_qnn_tensor_datatype(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value)      set_qnn_tensor_quantparams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value)              set_qnn_tensor_rank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value)        set_qnn_tensor_dimensions(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value)          set_qnn_tensor_memtype(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value)        set_qnn_tensor_clientbuf(tensor, value)
#define QNN_TENSOR_SET_MEM_HANDLE(tensor, value)        set_qnn_tensor_memhandle(tensor, value)

using pfn_rpc_mem_init                                  = void (*)(void);
using pfn_rpc_mem_deinit                                = void (*)(void);
using pfn_rpc_mem_alloc                                 = void *(*)(int, uint32_t, int);
using pfn_rpc_mem_free                                  = void (*)(void *);
using pfn_rpc_mem_to_fd                                 = int (*)(void *);

using _pfn_QnnSaver_initialize                          = decltype(QnnSaver_initialize);
using _pfn_QnnInterface_getProviders                    = decltype(QnnInterface_getProviders);
using _pfn_QnnSystemInterface_getProviders              = decltype(QnnSystemInterface_getProviders);



enum class ggml_qnn_profile_level {
    profile_off     = 0,
    profile_basic   = 1,
    profile_detail  = 2
};

struct ggml_backend_qnn_context {
    int device;
    int threads;
    char name[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];
    qnn_instance * instance;
    struct ggml_backend * backend;
    QNN_INTERFACE_VER_TYPE raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE raw_system_interface;
} ;

typedef void (* ggml_qnn_func_t)(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

typedef void (* ggml_qnn_func_common_t)(ggml_backend_qnn_context * ctx, const ggml_op ggml_op, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// =================================================================================================
//
//  static global variables
//
// =================================================================================================
//static ggml_backend_t g_qnn_backend = nullptr;

//according to the QNN SDK Reference Guide,
//CPU - Choose a non-quantized model. Quantized models are currently incompatible with the CPU backend
//GPU - Choose a non-quantized model. Quantized models are currently incompatible with the GPU backend
//HTP - Choose a quantized model. Quantized models are required when running on the HTP backend
//DSP - Choose a quantized model. Quantized models are required when running on the DSP backend
//HTA - Choose a quantized model. Quantized models are required when running on the HTA backend
//
//only focus on Qualcomm CPU/GPU/NPU backend in this implementation of QNN backend for ggml currently
//Qualcomm CPU: Qualcomm Kryo   CPU
//Qualcomm GPU: Qualcomm Adreno GPU
//Qualcomm NPU: aka HTP(Hexagon Tensor Processor), ~= cDSP(Compute DSP) + HMX(Hexagon Matrix eXtensions)/HTA(Hexagon Tensor Accelerator)

static struct ggml_backend_qnn_context g_qnn_mgr[GGML_QNN_MAX_DEVICES] = {
        [QNN_BACKEND_CPU]   = {.device = 0, .threads = 1, .name =   "qnn-cpu", .lib = "libQnnCpu.so", .instance = nullptr, .backend = nullptr, .raw_interface = {}, .raw_system_interface = {}},
        [QNN_BACKEND_GPU]   = {.device = 1, .threads = 1, .name =   "qnn-gpu", .lib = "libQnnGpu.so", .instance = nullptr, .backend = nullptr, .raw_interface = {}, .raw_system_interface = {}},
        [QNN_BACKEND_NPU]   = {.device = 2, .threads = 1, .name =   "qnn-npu", .lib = "libQnnHtp.so", .instance = nullptr, .backend = nullptr, .raw_interface = {}, .raw_system_interface = {}},
};

// =================================================================================================
//
//  QNN helper functions and other internal helper functions
//
// =================================================================================================
static inline int validate_tensor_version(Qnn_Tensor_t tensor) {
    if (tensor.version != QNN_TENSOR_VERSION_1) {
        QNN_LOG_WARN("validate_tensor_version() tensor %s, got unsupported version %d\n",
              tensor.v1.name,
              tensor.version);
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


static inline Qnn_TensorDataFormat_t get_qnn_tensor_dataformat(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataFormat;
    }
    return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}


static inline Qnn_DataType_t get_qnn_tensor_datatype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataType;
    }
    return QNN_DATATYPE_UNDEFINED;
}


static inline Qnn_QuantizeParams_t get_qnn_tensor_quantparams(const Qnn_Tensor_t & tensor) {
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


static inline void set_qnn_tensor_memtype(Qnn_Tensor_t & tensor, Qnn_TensorMemType_t memType) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memType = memType;
    }
}


static inline void set_qnn_tensor_clientbuf(Qnn_Tensor_t & tensor, Qnn_ClientBuffer_t clientBuf) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.clientBuf = clientBuf;
    }
}


static inline void set_qnn_tensor_memhandle(Qnn_Tensor_t & tensor, Qnn_MemHandle_t handle) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memHandle = handle;
    }
}


static size_t memscpy(void * dst, size_t dstSize, const void * src, size_t copySize) {
    if (!dst || !src || !dstSize || !copySize)
        return 0;

    size_t minSize = dstSize < copySize ? dstSize : copySize;

    memcpy(dst, src, minSize);

    return minSize;
}


static char * ggml_qnn_strndup(const char * source, size_t maxlen) {
    return ::strndup(source, maxlen);
}


static int deep_copy_qnn_tensors(Qnn_Tensor_t & src, Qnn_Tensor_t & dst) {
    int err = 0;
    VALIDATE_TENSOR_VERSION(src, err);

    dst.version = src.version;
    QNN_TENSOR_SET_NAME(
            dst, ggml_qnn_strndup(QNN_TENSOR_GET_NAME(src), std::string(QNN_TENSOR_GET_NAME(src)).size()));
    if (QNN_TENSOR_GET_NAME(dst) == nullptr) {
        return 1;
    }
    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
    QNN_TENSOR_SET_MEM_TYPE(dst, QNN_TENSOR_GET_MEM_TYPE(src));

    // Only metadata (i.e. non-static data) is copied from source to destination. The union still
    // must be initialized so that the clientBuf/memHandle do not contain garbage data
    if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_RAW) {
        Qnn_ClientBuffer_t client_buf = {nullptr, 0};
        QNN_TENSOR_SET_CLIENT_BUF(dst, client_buf);
    } else if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_MEMHANDLE) {
        QNN_TENSOR_SET_MEM_HANDLE(dst, nullptr);
    } else {
        return 1;
    }

    Qnn_QuantizeParams_t src_qparam      = QNN_TENSOR_GET_QUANT_PARAMS(src);
    Qnn_QuantizationEncoding_t encoding = src_qparam.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        // need to allocate and copy memory for scaleOffset as it is a pointer array
        Qnn_QuantizeParams_t src_qparam_cpy      = src_qparam;
        Qnn_AxisScaleOffset_t &axis_scale_offset = src_qparam_cpy.axisScaleOffsetEncoding;
        Qnn_ScaleOffset_t **scaleOffset          = &axis_scale_offset.scaleOffset;
        size_t scaleOffsetSize = axis_scale_offset.numScaleOffsets * sizeof(Qnn_ScaleOffset_t);
        *scaleOffset           = (Qnn_ScaleOffset_t *)malloc(scaleOffsetSize);
        memscpy(*scaleOffset,
                scaleOffsetSize,
                src_qparam.axisScaleOffsetEncoding.scaleOffset,
                scaleOffsetSize);
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam_cpy);
    } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
        // need to allocate and copy memory for scaleOffset as it is a pointer array
        Qnn_QuantizeParams_t src_qparam_cpy          = src_qparam;
        Qnn_BwAxisScaleOffset_t &bwaxis_scale_offset = src_qparam_cpy.bwAxisScaleOffsetEncoding;
        size_t scaleSize                           = bwaxis_scale_offset.numElements * sizeof(float);
        float **scales                             = &bwaxis_scale_offset.scales;
        int32_t **offsets                          = &bwaxis_scale_offset.offsets;
        *scales                                    = (float *)malloc(scaleSize);
        memscpy(*scales, scaleSize, src_qparam.bwAxisScaleOffsetEncoding.scales, scaleSize);

        // only copy offsets if present, nullptr implies all offsets are 0
        if (bwaxis_scale_offset.offsets != nullptr) {
            size_t offsetSize = bwaxis_scale_offset.numElements * sizeof(int32_t);
            *offsets          = (int32_t *)malloc(offsetSize);
            memscpy(*offsets, offsetSize, src_qparam.bwAxisScaleOffsetEncoding.offsets, offsetSize);
        }
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam_cpy);
    } else {
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam);
    }

    // allocate and copy memory for all the pointer members
    uint32_t rank = QNN_TENSOR_GET_RANK(src);
    QNN_TENSOR_SET_RANK(dst, rank);
    size_t dim_size       = rank * sizeof(uint32_t);
    uint32_t * dimensions = (uint32_t *)malloc(dim_size);
    if (dimensions == nullptr) {
        QNN_LOG_WARN("deep_copy_qnn_tensors() allocation error while copying tensor %s\n", QNN_TENSOR_GET_NAME(src));
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


static uint32_t ggml_get_tensor_rank(const ggml_tensor * tensor) {
    uint32_t rank = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if ((0 != tensor->ne[i]) && (1 != tensor->ne[i])) {
            rank++;
        }
    }
    return rank;
}


//TODO: mapping more ggml data type to QNN data type
//ref:explanation of k-quants, https://github.com/ggerganov/llama.cpp/pull/1684
static Qnn_DataType_t qnn_datatype_from_ggml_datatype(enum ggml_type ggmltype) {
    switch (ggmltype) {
        case GGML_TYPE_F16:
            return QNN_DATATYPE_FLOAT_16;
        case GGML_TYPE_F32:
            return QNN_DATATYPE_FLOAT_32;
        case GGML_TYPE_I8:
            return QNN_DATATYPE_INT_8;
        case GGML_TYPE_Q8_0:
            return QNN_DATATYPE_SFIXED_POINT_8;
        default:
            break;

    }
    return QNN_DATATYPE_UNDEFINED;
}


//TODO: only support GGML_OP_ADD/GGML_OP_MUL/GGML_OP_MUL_MAT
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


static uint32_t ggml_get_tensor_data_size(const ggml_tensor * tensor) {
    /*
    size_t data_size = ggml_row_size(tensor->type, tensor->ne[0]);
    size_t n_dims = ggml_get_tensor_rank(tensor);
    for (int i = 1; i < n_dims; i++) {
        data_size *= tensor->ne[i];
    }

    return data_size;
    */
    return ggml_nbytes(tensor);
}


template<typename Fn>
Fn load_qnn_functionpointers(void * handle, const char * function_name) {
    return reinterpret_cast<Fn>(dlsym(handle, function_name));
}


static const char * get_qnn_backend_name(int n_backend_type) {
    switch (n_backend_type) {
        case 0:
            return "QNN-CPU";
        case 1:
            return "QNN-GPU";
        case 2:
            return "QNN-NPU";
        case 3:
            return "ggml";      //"fake" QNN backend, used for compare performance between QNN backend and original GGML

        default:
            return "unknown";
    }
}


static intptr_t align_to(size_t alignment, intptr_t offset) {
    return offset % alignment == 0 ? offset
                                   : offset +
                                     (static_cast<intptr_t>(alignment) -
                                      offset % static_cast<intptr_t>(alignment));
}


static void ggml_qnn_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...) {
    static std::mutex ggml_qnn_log_internal_mutex;
    static char s_ggml_qnn_log_internal_buf[GGML_QNN_LOGBUF_LEN];

    {
        std::lock_guard<std::mutex> lock(ggml_qnn_log_internal_mutex);
        va_list args;
        va_start(args, format);
        int len_prefix = snprintf(s_ggml_qnn_log_internal_buf, GGML_QNN_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_ggml_qnn_log_internal_buf + len_prefix, GGML_QNN_LOGBUF_LEN - len_prefix, format, args);
        if (len < (GGML_QNN_LOGBUF_LEN - len_prefix)) {
#if (defined __ANDROID__) || (defined ANDROID)
            //for Android APK
            __android_log_print(level, "ggml-qnn", "%s\n", s_ggml_qnn_log_internal_buf);
#endif
            //for Android command line application or WoA
            printf("%s\n", s_ggml_qnn_log_internal_buf);
        }
        va_end(args);
    }
}


// =================================================================================================
//
//  wrapper class of Qualcomm QNN(Qualcomm Neural Network, aka Qualcomm AI Engine Direct) SDK
//
// =================================================================================================
class qnn_interface {

#define DEFINE_SHIM_FUNCTION_INTERFACE(F, pointer_name)           \
  template <typename... Args>                                     \
  inline auto qnn_##F(Args... args) const {                       \
    return (_qnn_interface->QNN_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                             \
  }


#define DEFINE_SHIM_FUNCTION_SYS_INTERFACE(F, pointer_name)                  \
  template <typename... Args>                                                \
  inline auto qnn_##F(Args... args) const {                                  \
    return (_qnn_sys_interface->QNN_SYSTEM_INTERFACE_VER_NAME.pointer_name)( \
        std::forward<Args>(args)...);                                        \
  }

    friend class qnn_instance;

public:
    qnn_interface() = default;

    // QnnBackend
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_create, backendCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_free, backendFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_register_op_package, backendRegisterOpPackage);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_validate_op_config, backendValidateOpConfig);

    DEFINE_SHIM_FUNCTION_INTERFACE(backend_get_api_version, backendGetApiVersion);

    // QnnDevice
    DEFINE_SHIM_FUNCTION_INTERFACE(device_create, deviceCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_free, deviceFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_infrastructure, deviceGetInfrastructure);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_platform_info, deviceGetPlatformInfo);

    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_info, deviceGetInfo);

    // QnnContext
    DEFINE_SHIM_FUNCTION_INTERFACE(context_create, contextCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary_size, contextGetBinarySize);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary, contextGetBinary);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_create_from_binary, contextCreateFromBinary);

    DEFINE_SHIM_FUNCTION_INTERFACE(context_free, contextFree);

    // QnnGraph
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_create, graphCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_add_node, graphAddNode);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_finalize, graphFinalize);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_execute, graphExecute);

    DEFINE_SHIM_FUNCTION_INTERFACE(graph_retrieve, graphRetrieve);

    // QnnLog
    DEFINE_SHIM_FUNCTION_INTERFACE(log_create, logCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(log_free, logFree);

    DEFINE_SHIM_FUNCTION_INTERFACE(log_set_log_level, logSetLogLevel);

    // QnnProfile
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_create, profileCreate);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_events, profileGetEvents);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_sub_events, profileGetSubEvents);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_event_data, profileGetEventData);

    DEFINE_SHIM_FUNCTION_INTERFACE(profile_free, profileFree);

    // QnnMem
    DEFINE_SHIM_FUNCTION_INTERFACE(mem_register, memRegister);

    DEFINE_SHIM_FUNCTION_INTERFACE(mem_de_register, memDeRegister);

    // QnnProperty
    DEFINE_SHIM_FUNCTION_INTERFACE(property_has_capability, propertyHasCapability);

    // QnnTensor
    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_context_tensor, tensorCreateContextTensor);

    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_graph_tensor, tensorCreateGraphTensor);

    // QnnSystem
    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_create, systemContextCreate);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_get_binary_info, systemContextGetBinaryInfo);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_free, systemContextFree);

    void set_qnn_interface(const QnnInterface_t * qnn_interface) {
        _qnn_interface = qnn_interface;
    }

    void set_qnn_system_interface(const QnnSystemInterface_t * qnn_sys_interface) {
        _qnn_sys_interface = qnn_sys_interface;
    }

    uint32_t get_backend_id() const {
        return _qnn_interface->backendId;
    }

    bool is_loaded() const {
        return ((_qnn_sys_interface != nullptr) && (_qnn_interface != nullptr));
    }

private:
    const QnnInterface_t *_qnn_interface = nullptr;

    const QnnSystemInterface_t *_qnn_sys_interface = nullptr;
};



// =================================================================================================
//
//  wrapper class of Qualcomm QNN(Qualcomm Neural Network, aka Qualcomm AI Engine Direct) SDK
//
//  and
//
//  resource management of QNN resources for GGML's QNN backend
// =================================================================================================
class qnn_instance {
public:
    using BackendIdType = decltype(QnnInterface_t{}.backendId);

    explicit qnn_instance(const std::string & lib_path, const std::string & backend_name,
                                const std::string & model_name) :
            _lib_path(std::move(lib_path)),
            _backend_name(std::move(backend_name)),
            _model_name(std::move(model_name)) {};

    ~qnn_instance() {
    }

    int qnn_init(const QnnSaver_Config_t ** saver_config);

    int qnn_finalize();

    const qnn_interface &get_qnn_interface() {
        if (!_qnn_interface.is_loaded()) {
            QNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_interface;
    }


    const QNN_INTERFACE_VER_TYPE &get_qnn_raw_interface() {
        if (!_qnn_interface.is_loaded()) {
            QNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_raw_interface;
    }

    const QNN_SYSTEM_INTERFACE_VER_TYPE &get_qnn_raw_system_interface() {
        if (!_qnn_interface.is_loaded()) {
            QNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_raw_system_interface;
    }

    const Qnn_LogHandle_t get_qnn_log_handle() { return _qnn_log_handle; }

    const Qnn_ProfileHandle_t get_qnn_profile_handle() { return _qnn_profile_handle; }

    const Qnn_DeviceHandle_t get_qnn_device_handle() { return _qnn_device_handle; }

    const Qnn_BackendHandle_t get_qnn_backend_handle() { return _qnn_backend_handle; }

    const Qnn_ContextHandle_t get_qnn_context_handle() { return _qnn_context_handle; }

    const QnnSystemContext_Handle_t get_qnn_system_handle() { return _qnn_system_handle; }

    const Qnn_GraphHandle_t get_qnn_graph_handle() { return _qnn_graph_handle; }


    int init_qnn_graph(const char * graph_name,
                       bool debug,
                       uint8_t do_node_validation = 1,
                       const QnnGraph_Config_t ** graph_configs = nullptr
    );

    int finalize_qnn_graph();

    int init_htp_perfinfra() {
        QnnDevice_Infrastructure_t device_infra = nullptr;
        int error = _qnn_raw_interface.deviceGetInfrastructure(&device_infra);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to get qnn device infra\n");
            return 1;
        }

        QnnHtpDevice_Infrastructure_t *htp_infra = static_cast<QnnHtpDevice_Infrastructure_t *>(device_infra);
        QnnHtpDevice_PerfInfrastructure_t *htp_perfinfra = &htp_infra->perfInfra;
        uint32_t power_configid = 1;
        uint32_t device_id = 0;
        uint32_t core_id = 0;
        htp_perfinfra->createPowerConfigId(device_id, core_id, &power_configid);
        _qnn_htp_perfinfra = htp_perfinfra;
        _qnn_power_configid = power_configid;

        return 0;
    }


    int set_rpc_polling() {
        if (_qnn_rpc_pollingtime > 0) {
            QnnHtpPerfInfrastructure_PowerConfig_t rpc_pollingTime;
            memset(&rpc_pollingTime, 0, sizeof(rpc_pollingTime));
            rpc_pollingTime.option =
                    QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
            rpc_pollingTime.rpcPollingTimeConfig = _qnn_rpc_pollingtime;
            const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&rpc_pollingTime, nullptr};
            if (_qnn_htp_perfinfra) {
                _qnn_htp_perfinfra->setPowerConfig(_qnn_power_configid, powerConfigs);
            }
        }
        return 0;
    }


    int set_high_performance_mode() {
        if (nullptr == _qnn_htp_perfinfra) {
            QNN_LOG_DEBUG("perf intra is null\n");
            return 1;
        }

        QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
        memset(&powerConfig, 0, sizeof(powerConfig));
        powerConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        powerConfig.dcvsV3Config.dcvsEnable = 0;
        powerConfig.dcvsV3Config.setDcvsEnable = 1;
        powerConfig.dcvsV3Config.contextId = _qnn_power_configid;
        powerConfig.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        powerConfig.dcvsV3Config.setSleepLatency = 1; // True to consider Latency parameter otherwise False
        powerConfig.dcvsV3Config.setBusParams = 1; // True to consider Bus parameter otherwise False
        powerConfig.dcvsV3Config.setCoreParams = 1; // True to consider Core parameter otherwise False
        powerConfig.dcvsV3Config.sleepDisable = 0; // True to consider sleep/LPM modes, False to enable
        powerConfig.dcvsV3Config.setSleepDisable = 0; // True to consider sleep disable/enable parameter otherwise False
        // set Sleep latency parameter
        uint32_t latencyValue = 40;
        powerConfig.dcvsV3Config.sleepLatency = latencyValue; // range 40-2000 micro sec
        // set Bus Clock Parameters (refer QnnHtpPerfInfrastructure_VoltageCorner_t enum)
        powerConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        powerConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        // set Core Clock Parameters (refer QnnHtpPerfInfrastructure_VoltageCorner_t enum)
        powerConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        powerConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        // set power config with different performance parameters
        const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&powerConfig, nullptr};

        _qnn_htp_perfinfra->setPowerConfig(_qnn_power_configid, powerConfigs);

        return 0;
    }

    std::string &get_qnn_graph_name() { return _graph_name; }

    bool is_rpcmem_initialized() {
        return _rpcmem_initialized;
    }

    void set_rpcmem_initialized(bool initialized) {
        _rpcmem_initialized = initialized;
    }

    int32_t rpcmem_to_fd(void * buf);

    int register_rpcmem(void * p_data, Qnn_Tensor_t * p_tensor);

    void unregister_rpcmem();

    void *alloc_rpcmem(size_t bytes, size_t alignment);

    void free_rpcmem(void * buf);

    bool is_rpcmem_allocated(void * buf);

    bool is_rpcmem_registered(Qnn_MemHandle_t handle) {
        return _qnn_mem_set.count(handle) != 0U;
    }

public:
    std::map<std::string, std::tuple<Qnn_GraphHandle_t, Qnn_Tensor_t *, Qnn_Tensor_t *, Qnn_Tensor_t *>> _qnn_graph_map;

private:
    int load_system();

    int unload_system();

    int load_backend(std::string &lib_path, const QnnSaver_Config_t ** saver_config);

    int unload_backend();

    void set_qnn_raw_interface(QNN_INTERFACE_VER_TYPE & raw_interface) {
        _qnn_raw_interface = raw_interface;
    }

    void set_qnn_raw_system_interface(QNN_SYSTEM_INTERFACE_VER_TYPE &raw_interface) {
        _qnn_raw_system_interface = raw_interface;
    }

private:
    static constexpr const int _required_num_providers = 1;

private:
    std::string _lib_path;
    std::string _backend_name;
    std::string _model_name;                         // prebuilt QNN model name, not used in currently
    BackendIdType _backend_id;

    bool _debug_tensor                      = false; // flag to indicate if requested graph is to be run in debug mode
    bool _do_node_validations               = true;  // flag to indicate whether all add_node calls need to be validated
    QnnLog_Level_t _qnn_log_level           = QNN_LOG_LEVEL_DEBUG;

    ggml_qnn_profile_level _profile_level   = ggml_qnn_profile_level::profile_detail;

    qnn_interface _qnn_interface;

    void *_system_lib_handle = nullptr;

    Qnn_GraphHandle_t _qnn_graph_handle = nullptr;

    Qnn_LogHandle_t _qnn_log_handle = nullptr;

    Qnn_ProfileHandle_t _qnn_profile_handle = nullptr;

    Qnn_DeviceHandle_t _qnn_device_handle = nullptr;

    Qnn_BackendHandle_t _qnn_backend_handle = nullptr;

    Qnn_ContextHandle_t _qnn_context_handle = nullptr;

    QnnSystemContext_Handle_t _qnn_system_handle = nullptr;

    QnnHtpDevice_PerfInfrastructure_t *_qnn_htp_perfinfra = nullptr;
    uint32_t _qnn_power_configid = 1;
    uint32_t _qnn_rpc_pollingtime = 9999; // 0-10000 us for high performing

    QNN_INTERFACE_VER_TYPE _qnn_raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE _qnn_raw_system_interface;

    std::unordered_set<Qnn_MemHandle_t> _qnn_mem_set;

    static std::mutex _init_mutex;
    static std::unordered_map<BackendIdType, void *> _loaded_lib_handle;
    static std::unordered_map<std::string, BackendIdType> _lib_path_to_backend_id;
    static std::unordered_map<BackendIdType, const QnnInterface_t *> _loaded_backend;

    void *_rpc_lib_handle = nullptr;
    std::atomic_bool _rpcmem_initialized{false};
    pfn_rpc_mem_alloc _pfn_rpc_mem_alloc;
    pfn_rpc_mem_free _pfn_rpc_mem_free;
    pfn_rpc_mem_to_fd _pfn_rpc_mem_to_fd;
    pfn_rpc_mem_init  _pfn_rpc_mem_init;
    pfn_rpc_mem_deinit _pfn_rpc_mem_deinit;
    std::unordered_map<void *, void *> _rpcmem_store_map;


    std::string _graph_name;
};



// =================================================================================================
//
//  implementation of wrapper class
//
// =================================================================================================
std::mutex qnn_instance::_init_mutex;

std::unordered_map<qnn_instance::BackendIdType, void *> qnn_instance::_loaded_lib_handle;

std::unordered_map<std::string, qnn_instance::BackendIdType> qnn_instance::_lib_path_to_backend_id;

std::unordered_map<qnn_instance::BackendIdType, const QnnInterface_t *> qnn_instance::_loaded_backend;


void * qnn_instance::alloc_rpcmem(size_t bytes, size_t alignment) {
    if (!_rpcmem_initialized) {
        QNN_LOG_WARN("rpc memory not initialized\n");
        return nullptr;
    }

    auto allocate_bytes = static_cast<int32_t>(bytes + alignment);
    void *buf = _pfn_rpc_mem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, allocate_bytes);
    if (buf == nullptr) {
        QNN_LOG_WARN("failed to allocate rpc memory\n");
        return nullptr;
    }

    auto aligned_buf = reinterpret_cast<void *>(align_to(alignment,
                                                         reinterpret_cast<intptr_t>(buf)));
    bool status = _rpcmem_store_map.insert(std::pair<void *, void *>(aligned_buf, buf)).second;
    if (!status) {
        QNN_LOG_WARN("failed to allocate rpc memory\n");
        _pfn_rpc_mem_free(buf);
    }

    return aligned_buf;
}


void qnn_instance::free_rpcmem(void * buf) {
    if (!_rpcmem_initialized) {
        QNN_LOG_WARN("rpc memory not initialized\n");
    } else if (0 == _rpcmem_store_map.count(buf)) {
        QNN_LOG_WARN("no allocated tensor\n");
    } else {
        _pfn_rpc_mem_free(_rpcmem_store_map[buf]);
        _rpcmem_store_map.erase(buf);
    }
}


int32_t qnn_instance::rpcmem_to_fd(void * buf) {
    int32_t mem_fd = -1;
    if (!is_rpcmem_initialized()) {
        QNN_LOG_WARN("rpc memory not initialized\n");
    } else {
        mem_fd = _pfn_rpc_mem_to_fd(buf);
    }

    return mem_fd;
}


int qnn_instance::register_rpcmem(void * p_data, Qnn_Tensor_t * p_tensor) {
    if (nullptr == p_data || (nullptr == p_tensor)) {
        QNN_LOG_WARN("invalid param\n");
        return 1;
    }

    if (!is_rpcmem_initialized()) {
        QNN_LOG_WARN("rpc memory not initialized\n");
        return 2;
    }

    if (is_rpcmem_allocated(p_data)) {
        QNN_LOG_WARN("rpc memory already allocated\n");
        //return 3;
    }
    if (is_rpcmem_registered((QNN_VER_PTR(*p_tensor)->memHandle))) {
        QNN_LOG_WARN("tensor %s has been registered shared memory\n", (QNN_VER_PTR(*p_tensor)->name));
        return 4;
    }

    int32_t mem_fd = rpcmem_to_fd(p_data);
    if (-1 == mem_fd) {
        QNN_LOG_WARN("failed to get file descriptor\n");
        return 5;
    }
    QNN_LOG_DEBUG("mem_fd %d\n", mem_fd);
    Qnn_MemDescriptor_t descriptor = {
            {QNN_VER_PTR(*p_tensor)->rank, QNN_VER_PTR(*p_tensor)->dimensions, nullptr},
            QNN_VER_PTR(*p_tensor)->dataType,
            QNN_MEM_TYPE_ION,
            {{mem_fd}}};
    Qnn_MemHandle_t handle = nullptr;
    int error = QNN_SUCCESS;
    error = _qnn_interface.qnn_mem_register(
            _qnn_context_handle,
            &descriptor,
            /*numDescriptors=*/1,
            &handle);
    if (error != QNN_SUCCESS) {
        QNN_LOG_WARN("failed to register shared memory, error %d, %s\n", QNN_GET_ERROR_CODE(error),
              strerror(error));
        return 6;
    } else {
        QNN_LOG_INFO("tensor %s successfully register shared memory\n", (QNN_VER_PTR(*p_tensor)->name));
    }
    QNN_VER_PTR(*p_tensor)->memHandle = handle;
    _qnn_mem_set.insert(handle);

    return 0;
}


void qnn_instance::unregister_rpcmem() {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    if (_qnn_mem_set.empty()) {
        QNN_LOG_WARN("no rpcmem registered\n");
    }

    for (auto &mem_handle : _qnn_mem_set) {
        error = _qnn_interface.qnn_mem_de_register(&mem_handle, 1);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to unregister shared memory, error %d\n", QNN_GET_ERROR_CODE(error));
        }
    }
    _qnn_mem_set.clear();
}


bool qnn_instance::is_rpcmem_allocated(void * buf) {
    return _rpcmem_store_map.count(buf) != 0U;
}


int qnn_instance::load_backend(std::string & lib_path, const QnnSaver_Config_t ** saver_config) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    QNN_LOG_DEBUG("lib_path:%s\n", lib_path.c_str());

    void *lib_handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (nullptr == lib_handle) {
        QNN_LOG_WARN("can not open QNN library %s, with error: %s", lib_path.c_str(), dlerror());
        return 1;
    }

    // load get_provider function
    auto get_providers = load_qnn_functionpointers<_pfn_QnnInterface_getProviders *>(lib_handle,
                                                                                     "QnnInterface_getProviders");
    if (nullptr == get_providers) {
        QNN_LOG_WARN("can not load symbol QnnInterface_getProviders : %s", dlerror());
        return 2;
    }

    // get QnnInterface Providers
    std::uint32_t num_providers = 0;
    const QnnInterface_t **provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        QNN_LOG_WARN("failed to get providers, error %d", QNN_GET_ERROR_CODE(error));
        return 3;
    }
    QNN_LOG_DEBUG("num_providers=%d\n", num_providers);
    if (num_providers != _required_num_providers) {
        QNN_LOG_WARN("providers is %d instead of required %d", num_providers, _required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        QNN_LOG_WARN("failed to get qnn interface providers\n");
        return 5;
    }
    bool found_valid_interface = false;
    QNN_INTERFACE_VER_TYPE qnn_interface;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_API_VERSION_MAJOR == provider_list[idx]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <= provider_list[idx]->apiVersion.coreApiVersion.minor) {
            found_valid_interface = true;
            qnn_interface = provider_list[idx]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }

    if (!found_valid_interface) {
        QNN_LOG_WARN("unable to find a valid qnn interface\n");
        return 6;
    } else {
        QNN_LOG_INFO("find a valid qnn interface\n");
    }
    set_qnn_raw_interface(qnn_interface);

    BackendIdType backend_id = provider_list[0]->backendId;
    _lib_path_to_backend_id[lib_path] = backend_id;
    if (_loaded_backend.count(backend_id) > 0) {
        QNN_LOG_WARN("lib_path %s is loaded, but backend %d already exists\n",
              lib_path.c_str(), backend_id);
    }
    _loaded_backend[backend_id] = provider_list[0];
    if (_loaded_lib_handle.count(backend_id) > 0) {
        QNN_LOG_WARN("closing %p\n", _loaded_lib_handle[backend_id]);
        int dlclose_error = dlclose(_loaded_lib_handle[backend_id]);
        if (dlclose_error != 0) {
            QNN_LOG_WARN("fail to close %p with error %s\n", _loaded_lib_handle[backend_id], dlerror());
        }
    }
    _loaded_lib_handle[backend_id] = lib_handle;
    _backend_id = backend_id;

    return 0;
}


int qnn_instance::unload_backend() {
    int dlclose_error = 0;
    for (auto &it : _loaded_lib_handle) {
        dlclose_error = dlclose(it.second);
        if (dlclose_error != 0) {
            QNN_LOG_WARN("failed to close QNN backend %d, error %s\n", it.first, dlerror());
        }
    }

    _loaded_lib_handle.clear();
    _lib_path_to_backend_id.clear();
    _loaded_backend.clear();

    return 0;
}


int qnn_instance::load_system() {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    std::string system_lib_path = _lib_path + "libQnnSystem.so";
    QNN_LOG_DEBUG("system_lib_path:%s\n", system_lib_path.c_str());

    _system_lib_handle = dlopen(system_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (nullptr == _system_lib_handle) {
        QNN_LOG_WARN("can not open QNN library %s, error: %s\n", system_lib_path.c_str(), dlerror());
        return 1;
    }

    auto * get_providers = reinterpret_cast<_pfn_QnnSystemInterface_getProviders *>(dlsym(
            _system_lib_handle, "QnnSystemInterface_getProviders"));
    if (nullptr == get_providers) {
        QNN_LOG_WARN("can not load QNN symbol QnnSystemInterface_getProviders: %s\n", dlerror());
        return 2;
    }

    uint32_t num_providers = 0;
    const QnnSystemInterface_t ** provider_list = nullptr;
    error = get_providers(&provider_list, &num_providers);
    if (error != QNN_SUCCESS) {
        QNN_LOG_WARN("failed to get providers, error %d\n", QNN_GET_ERROR_CODE(error));
        return 3;
    }

    if (num_providers != _required_num_providers) {
        QNN_LOG_WARN("providers is %d instead of required %d\n", num_providers, _required_num_providers);
        return 4;
    }

    if (nullptr == provider_list) {
        QNN_LOG_WARN("can not get providers\n");
        return 5;
    }

    QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface;
    bool found_valid_system_interface = false;
    for (size_t idx = 0; idx < num_providers; idx++) {
        if (QNN_SYSTEM_API_VERSION_MAJOR ==
            provider_list[idx]->systemApiVersion.major &&
            QNN_SYSTEM_API_VERSION_MINOR <=
            provider_list[idx]->systemApiVersion.minor) {
            found_valid_system_interface = true;
            qnn_system_interface = provider_list[idx]->QNN_SYSTEM_INTERFACE_VER_NAME;
            break;
        }
    }
    if (!found_valid_system_interface) {
        QNN_LOG_WARN("unable to find a valid qnn system interface\n");
        return 6;
    } else {
        QNN_LOG_INFO("find a valid qnn system interface\n");
    }
    set_qnn_raw_system_interface(qnn_system_interface);

    _qnn_interface.set_qnn_system_interface(provider_list[0]);

    _qnn_interface.qnn_system_context_create(&_qnn_system_handle);
    if (nullptr == _qnn_system_handle) {
        QNN_LOG_WARN("can not create QNN system contenxt\n");
    } else {
        QNN_LOG_INFO("initialize qnn system successfully\n");
    }

    return 0;
}


int qnn_instance::unload_system() {
    int result = 0;

    if (nullptr == _system_lib_handle) {
        QNN_LOG_DEBUG("system lib handle is null\n");
        return 1;
    }

    if (nullptr != _qnn_system_handle) {
        result = _qnn_interface.qnn_system_context_free(_qnn_system_handle);
        if (result != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN system context\n");
        }
        _qnn_system_handle = nullptr;
    }

    int dlclose_error = dlclose(_system_lib_handle);
    if (dlclose_error != 0) {
        QNN_LOG_WARN("failed to close QnnSystem library, error %s\n", dlerror());
        return 2;
    }

    _system_lib_handle = nullptr;

    return result;
}


static void ggml_qnn_logcallback(const char * fmt,
                                 QnnLog_Level_t level,
                                 uint64_t timestamp,
                                 va_list argp) {

    static std::mutex log_mutex;
    static unsigned char s_ggml_qnn_logbuf[GGML_QNN_LOGBUF_LEN];

    const char * log_level_desc = "";
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:
            log_level_desc = " ERROR ";
            break;
        case QNN_LOG_LEVEL_WARN:
            log_level_desc = "WARNING";
            break;
        case QNN_LOG_LEVEL_INFO:
            log_level_desc = "  INFO ";
            break;
        case QNN_LOG_LEVEL_DEBUG:
            log_level_desc = " DEBUG ";
            break;
        case QNN_LOG_LEVEL_VERBOSE:
            log_level_desc = "VERBOSE";
            break;
        case QNN_LOG_LEVEL_MAX:
            log_level_desc = "UNKNOWN";
            break;
    }

    double ms = (double) timestamp / 1000000.0;
#if GGML_QNN_DEBUG
    {
        std::lock_guard<std::mutex> lock(log_mutex);

        memset(s_ggml_qnn_logbuf, 0, GGML_QNN_LOGBUF_LEN);
        vsnprintf(reinterpret_cast<char *const>(s_ggml_qnn_logbuf), GGML_QNN_LOGBUF_LEN, fmt, argp);
        QNN_LOG_DEBUG("%8.1fms [%-7s] %s\n", ms, log_level_desc, s_ggml_qnn_logbuf);
    }
#endif
}


int qnn_instance::qnn_init(const QnnSaver_Config_t ** saver_config) {
    BackendIdType backend_id = QNN_BACKEND_ID_NULL;
    QNN_LOG_DEBUG("enter qni_init\n");

    const std::lock_guard<std::mutex> lock(_init_mutex);

    if (0 != load_system()) {
        QNN_LOG_WARN("can not load QNN system lib, pls check why?\n");
        return 1;
    } else {
        QNN_LOG_DEBUG("load QNN system lib successfully\n");
    }

    std::string bakend_lib_path = _lib_path + _backend_name;
    if (0 == _lib_path_to_backend_id.count(bakend_lib_path)) {
        int is_load_ok = load_backend(bakend_lib_path, saver_config);
        if (0 != is_load_ok) {
            QNN_LOG_WARN("failed to load QNN backend\n");
            return 2;
        }
    }

    backend_id = _lib_path_to_backend_id[bakend_lib_path];
    if (0 == _loaded_backend.count(backend_id) ||
        0 == _loaded_lib_handle.count(backend_id)) {
        QNN_LOG_WARN("library %s is loaded but loaded backend count=%zu, loaded lib_handle count=%zu\n",
              bakend_lib_path.c_str(),
              _loaded_backend.count(backend_id),
              _loaded_lib_handle.count(backend_id));
        return 3;
    }

    _qnn_interface.set_qnn_interface(_loaded_backend[backend_id]);

    _qnn_interface.qnn_log_create(ggml_qnn_logcallback, _qnn_log_level, &_qnn_log_handle);
    if (nullptr == _qnn_log_handle) {
        QNN_LOG_WARN("why failed to initialize qnn log\n"); //NPU backend not work on Qualcomm SoC based low-end phone
        return 4;
    } else {
        QNN_LOG_DEBUG("initialize qnn log successfully\n");
    }

    std::vector<const QnnBackend_Config_t *> temp_backend_config;
    _qnn_interface.qnn_backend_create(_qnn_log_handle, temp_backend_config.empty() ? nullptr
                                                                                   : temp_backend_config.data(),
                                      &_qnn_backend_handle);
    if (nullptr == _qnn_backend_handle) {
        QNN_LOG_WARN("why failed to initialize qnn backend\n");
        return 5;
    } else {
        QNN_LOG_DEBUG("initialize qnn backend successfully\n");
    }

    if (nullptr != _qnn_raw_interface.propertyHasCapability) {
        auto qnnStatus = _qnn_raw_interface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus) {
            QNN_LOG_WARN("device property is not supported\n");
        }
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
            QNN_LOG_WARN("device property is not known to backend\n");
        }
    }

    auto qnnStatus = _qnn_raw_interface.deviceCreate(_qnn_log_handle, nullptr, &_qnn_device_handle);
    if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
        QNN_LOG_WARN("failed to create QNN device\n");
    } else {
        QNN_LOG_INFO("create device successfully\n");
    }

    if (ggml_qnn_profile_level::profile_off != _profile_level) {
        QNN_LOG_INFO("profiling turned on; level = %d", _profile_level);
        if (ggml_qnn_profile_level::profile_basic == _profile_level) {
            QNN_LOG_INFO("basic profiling requested. creating Qnn Profile object\n");
            if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
                    _qnn_backend_handle, QNN_PROFILE_LEVEL_BASIC, &_qnn_profile_handle)) {
                QNN_LOG_WARN("unable to create profile handle in the backend\n");
                return 6;
            } else {
                QNN_LOG_DEBUG("initialize qnn profile successfully\n");
            }
        } else if (ggml_qnn_profile_level::profile_detail == _profile_level) {
            QNN_LOG_INFO("detailed profiling requested. Creating Qnn Profile object\n");
            if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
                    _qnn_backend_handle, QNN_PROFILE_LEVEL_DETAILED, &_qnn_profile_handle)) {
                QNN_LOG_WARN("unable to create profile handle in the backend\n");
                return 7;
            } else {
                QNN_LOG_DEBUG("initialize qnn profile successfully\n");
            }
        }
    }

    _rpc_lib_handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (nullptr == _rpc_lib_handle) {
        QNN_LOG_WARN("failed to load qualcomm's rpc lib, error:%s\n", dlerror());
        return 8;
    } else {
        QNN_LOG_DEBUG("load rpcmem lib successfully\n");
        set_rpcmem_initialized(true);
    }
    _pfn_rpc_mem_init   = reinterpret_cast<pfn_rpc_mem_init>(dlsym(_rpc_lib_handle, "rpcmem_init"));
    _pfn_rpc_mem_deinit = reinterpret_cast<pfn_rpc_mem_deinit>(dlsym(_rpc_lib_handle, "rpcmem_deinit"));
    _pfn_rpc_mem_alloc  = reinterpret_cast<pfn_rpc_mem_alloc>(dlsym(_rpc_lib_handle,"rpcmem_alloc"));
    _pfn_rpc_mem_free   = reinterpret_cast<pfn_rpc_mem_free>(dlsym(_rpc_lib_handle, "rpcmem_free"));
    _pfn_rpc_mem_to_fd  = reinterpret_cast<pfn_rpc_mem_to_fd>(dlsym(_rpc_lib_handle,"rpcmem_to_fd"));
    if (nullptr == _pfn_rpc_mem_alloc || nullptr == _pfn_rpc_mem_free
        || nullptr == _pfn_rpc_mem_to_fd) {
        QNN_LOG_WARN("unable to access symbols in QNN RPC lib. dlerror(): %s", dlerror());
        dlclose(_rpc_lib_handle);
        return 9;
    }

    if (nullptr != _pfn_rpc_mem_init) // make Qualcomm's SoC based low-end phone happy
        _pfn_rpc_mem_init();

    std::vector<const QnnContext_Config_t *> temp_context_config;
    _qnn_interface.qnn_context_create(_qnn_backend_handle, _qnn_device_handle,
                                      temp_context_config.empty() ? nullptr
                                                                  : temp_context_config.data(),
                                      &_qnn_context_handle);
    if (nullptr == _qnn_context_handle) {
        QNN_LOG_WARN("why failed to initialize qnn context\n");
        return 10;
    } else {
        QNN_LOG_DEBUG("initialize qnn context successfully\n");
    }

    QNN_LOG_DEBUG("leave qni_init\n");

    return 0;
}


int qnn_instance::qnn_finalize() {
    int ret_status = 0;
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    if (nullptr != _pfn_rpc_mem_deinit) // make Qualcomm's SoC based low-end phone happy
        _pfn_rpc_mem_deinit();

    if (dlclose(_rpc_lib_handle) != 0) {
        QNN_LOG_WARN("failed to unload qualcomm's rpc lib, error:%s\n", dlerror());
    } else {
        QNN_LOG_DEBUG("succeed to close rpcmem lib\n");
    }

    if (nullptr != _qnn_context_handle) {
        error = _qnn_interface.qnn_context_free(_qnn_context_handle, _qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN context_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_context_handle = nullptr;
    }

    if (nullptr != _qnn_profile_handle) {
        error = _qnn_interface.qnn_profile_free(_qnn_profile_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN profile_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_profile_handle = nullptr;
    }

    if (nullptr != _qnn_device_handle) {
        error = _qnn_interface.qnn_device_free(_qnn_device_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN device_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));

        }
        _qnn_device_handle = nullptr;
    }

    if (nullptr != _qnn_backend_handle) {
        error = _qnn_interface.qnn_backend_free(_qnn_backend_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN backend_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_backend_handle = nullptr;

    }

    if (nullptr != _qnn_log_handle) {
        error = _qnn_interface.qnn_log_free(_qnn_log_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to free QNN log_handle: ID %u, error %d\n",
                  _qnn_interface.get_backend_id(), QNN_GET_ERROR_CODE(error));
        }
        _qnn_log_handle = nullptr;
    }

    unload_backend();

    unload_system();

    return ret_status;
}


int qnn_instance::init_qnn_graph(const char * graph_name, bool debug, uint8_t do_node_validation,
                                   const QnnGraph_Config_t ** graph_configs) {
    int result = 0;

    if (nullptr == graph_name) {
        QNN_LOG_WARN("graph name is null\n");
        return 1;
    }

    if (!_graph_name.empty()) {
        QNN_LOG_WARN("qnn model for graph %s already initialized\n", graph_name);
        return 2;
    }

    if (!do_node_validation) {
        QNN_LOG_WARN("node validation disabled, backend will not perform op validation prior to adding node\n");
    }

    _graph_name = graph_name;
    _debug_tensor = debug;
    _do_node_validations = do_node_validation;

    result = _qnn_raw_interface.graphCreate(_qnn_context_handle, graph_name, graph_configs,
                                            &_qnn_graph_handle);
    if (result != QNN_GRAPH_NO_ERROR || nullptr == _qnn_graph_handle) {
        QNN_LOG_WARN("failed to create graph in qnn context\n");
        return 3;
    } else {
        QNN_LOG_INFO("succeed to create graph %s, %p\n", graph_name, _qnn_graph_handle);
    }

    return 0;
}


int qnn_instance::finalize_qnn_graph() {
    if (nullptr != _qnn_graph_handle) {
        if (_qnn_raw_interface.graphFinalize(_qnn_graph_handle, _qnn_profile_handle, nullptr) !=
            QNN_GRAPH_NO_ERROR) {
            QNN_LOG_WARN("finalizing graph failure\n");
            //return 1;
        }
    } else {
        QNN_LOG_DEBUG("qnn graph handle is null\n");
    }

    return 0;
}



// =================================================================================================
//
//  implementation of GGML's QNN backend
//
// =================================================================================================
static bool ggml_qnn_can_handle_op(const struct ggml_tensor * tensor, bool b_dump_tensor_info) {
    if (nullptr == tensor)
        return false;
    if (b_dump_tensor_info) {
        QNN_LOG_DEBUG("op name:%s, tensor type:%s", ggml_op_name(tensor->op),
                      ggml_type_name(tensor->type));
    }
    //only support the following 3 OPs currently and ensure tensor->src[0] and tensor->src[1] is not nullptr
    bool supported_op = ((tensor->op == GGML_OP_ADD) || (tensor->op == GGML_OP_MUL) || (tensor->op == GGML_OP_MUL_MAT));
    if (!supported_op) {
        return false;
    }

    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int64_t ne0  = tensor->ne[0];
    const int64_t ne1  = tensor->ne[1];

    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);

    if (b_dump_tensor_info) {
        QNN_LOG_DEBUG("src0 type:%s", ggml_type_name(tensor->src[0]->type));
        QNN_LOG_DEBUG("src1 type:%s", ggml_type_name(tensor->src[1]->type));

        if (tensor->op == GGML_OP_MUL_MAT) {
                QNN_LOG_DEBUG("GGML_OP_MUL_MAT");
                QNN_LOG_DEBUG(
                        "src0 %15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
                        src0->name,
                        src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
                        src0->nb[0], src0->nb[1], src0->nb[2]);
                QNN_LOG_DEBUG(
                        "src1 %15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
                        src1->name,
                        src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
                        src1->nb[0], src1->nb[1], src1->nb[2]);
                QNN_LOG_DEBUG(
                        "     %15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
                        tensor->name,
                        tensor->type, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2],
                        tensor->nb[0],
                        tensor->nb[1], tensor->nb[2]);

        }
    }

    if (ggml_is_empty(tensor) || tensor->op == GGML_OP_RESHAPE || tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_NONE) {
        return false;
    }

    //make ggml_get_tensor_rank and QNN SDK happy
    if ((ne00 <= 1 || ne01 <= 1 || ne10 <= 1 || ne11 <= 1)) {
        return false;
    }

    // GPU/NPU inference will slower then CPU inference when tensor->ne[1] < min batch size
    if (tensor->ne[1] < 32) {
        return false;
    }

    int qtype = src0->type;
    return (qtype == GGML_TYPE_F32 || qtype == GGML_TYPE_F16 || qtype == GGML_TYPE_Q8_0)
               && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);

}


static void ggml_qnn_add(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    bool graph_initialized                      = false;
    int64_t n_begin_time                        = 0LL;
    int64_t n_end_time                          = 0LL;
    int64_t n_duration                          = 0LL;

    qnn_instance * instance                     = nullptr;

    std::string graph_name                      = "ggml_op_qnn_add";
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * tensor_0                     = nullptr;
    Qnn_Tensor_t * tensor_1                     = nullptr;
    Qnn_Tensor_t * tensor_2                     = nullptr;
    Qnn_Param_t qnn_params[]                    = {};
    enum ggml_op ggmlop                         = GGML_OP_ADD;
    Qnn_DataType_t src0_qnn_type                = QNN_DATATYPE_FLOAT_32;
    Qnn_DataType_t src1_qnn_type                = QNN_DATATYPE_FLOAT_32;
    Qnn_DataType_t dst_qnn_type                 = QNN_DATATYPE_FLOAT_32;

    if ((nullptr == src0) || (nullptr == src1) || (nullptr == dst)) {
        QNN_LOG_WARN("pls check why GGML tensor is null");
        return;
    }
    tensor_0                                    = (Qnn_Tensor_t *)src0->extra;
    tensor_1                                    = (Qnn_Tensor_t *)src1->extra;
    tensor_2                                    = (Qnn_Tensor_t *)dst->extra;
    if ((nullptr == tensor_0) || (nullptr == tensor_1) || (nullptr == tensor_2)) {
        QNN_LOG_WARN("pls check why QNN tensor is null");
        return;
    }
    if (nullptr == ctx) {
        QNN_LOG_WARN("pls check why backend ctx is null");
        return;
    }
    instance                                    = ctx->instance;
    if (nullptr == instance) {
        QNN_LOG_WARN("pls check why qnn instance is null");
        return;
    }
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;

    n_begin_time                                = ggml_time_us();

    QNN_LOG_DEBUG("call %s\n", __func__);
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src0->name,
          src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
          src0->nb[0], src0->nb[1], src0->nb[2]);
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src1->name,
          src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
          src1->nb[0], src1->nb[1], src1->nb[2]);
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          dst->name,
          dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0],
          dst->nb[1], dst->nb[2]);
    QNN_LOG_DEBUG("%d, %d, %d, %d", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    QNN_LOG_DEBUG("tensor0 name %s", QNN_TENSOR_GET_NAME(*tensor_0));
    QNN_LOG_DEBUG("tensor1 name %s", QNN_TENSOR_GET_NAME(*tensor_1));
    QNN_LOG_DEBUG("tensor2 name %s", QNN_TENSOR_GET_NAME(*tensor_2));

    QNN_VER_PTR(*tensor_0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_2)->type = QNN_TENSOR_TYPE_APP_READ;

    src0_qnn_type                = qnn_datatype_from_ggml_datatype(src0->type);
    src1_qnn_type                = qnn_datatype_from_ggml_datatype(src1->type);
    dst_qnn_type                 = qnn_datatype_from_ggml_datatype(dst->type);

    uint32_t dimensions_input_0[] = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                         (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
    uint32_t dimensions_input_1[] = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                         (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
    uint32_t dimensions_output[]  = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                         (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};

    std::string map_entry         = std::string(ggml_op_name(ggmlop));
    if (instance->_qnn_graph_map.find(map_entry) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
    }

    uint32_t * tensor_0_dimensions = QNN_VER_PTR(*tensor_0)->dimensions;
    uint32_t * tensor_1_dimensions = QNN_VER_PTR(*tensor_1)->dimensions;
    uint32_t * tensor_2_dimensions = QNN_VER_PTR(*tensor_2)->dimensions;

    if (!graph_initialized) {
        graph_name = graph_name + "_" + std::to_string(ctx->threads) + src0->name + "_" + src1->name;
        QNN_LOG_DEBUG("graph name %s", graph_name.c_str());
        //QnnGraph_Config_t graph_config;
        //graph_config.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
        //graph_config.customConfig = strdup(graph_name.c_str());
        //const QnnGraph_Config_t  * p_graph_config = &graph_config;
        error = qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(), graph_name.c_str(), nullptr, &graph_handle);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        error = qnn_raw_interface.tensorCreateGraphTensor(graph_handle, tensor_0);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.tensorCreateGraphTensor(graph_handle, tensor_1);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.tensorCreateGraphTensor(graph_handle, tensor_2);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }

        QNN_VER_PTR(*tensor_0)->clientBuf = {src0->data, ggml_get_tensor_data_size(src0)};
        QNN_VER_PTR(*tensor_1)->clientBuf = {src1->data, ggml_get_tensor_data_size(src1)};
        QNN_VER_PTR(*tensor_2)->clientBuf = {dst->data, ggml_get_tensor_data_size(dst)};

        QNN_VER_PTR(*tensor_0)->dimensions = dimensions_input_0;
        QNN_VER_PTR(*tensor_0)->rank = ggml_get_tensor_rank(src0);
        QNN_VER_PTR(*tensor_0)->dataType = src0_qnn_type;
        QNN_VER_PTR(*tensor_1)->dimensions = dimensions_input_1;
        QNN_VER_PTR(*tensor_1)->rank = ggml_get_tensor_rank(src1);
        QNN_VER_PTR(*tensor_1)->dataType = src1_qnn_type;
        QNN_VER_PTR(*tensor_2)->dimensions = dimensions_output;
        QNN_VER_PTR(*tensor_2)->rank = ggml_get_tensor_rank(dst);
        QNN_VER_PTR(*tensor_2)->dataType = dst_qnn_type;

        Qnn_Tensor_t tensor_inputs[] = {
                *tensor_0,
                *tensor_1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *tensor_2
        };
        Qnn_OpConfig_t op_config = {
                (Qnn_OpConfigVersion_t) 1, .v1 = {
                        "ggml_op_add",
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        QNN_OP_ELEMENT_WISE_ADD,
                        0,
                        qnn_params,
                        2,
                        tensor_inputs,
                        1,
                        tensor_outputs
                }
        };
        error = qnn_raw_interface.graphAddNode(graph_handle, op_config);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.graphExecute(graph_handle, tensor_inputs, 2, tensor_outputs, 1, nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        auto  graph_item = std::make_tuple(graph_handle, tensor_0, tensor_1, tensor_2);
        instance->_qnn_graph_map[map_entry] = graph_item;
    } else {
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
        tensor_0     = std::get<1>(graph_item);
        tensor_1     = std::get<2>(graph_item);
        tensor_2     = std::get<3>(graph_item);

        //QNN_LOG_DEBUG("%d, %d, %d, %d", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
        uint32_t dimensions_input_0[] = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                         (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
        uint32_t dimensions_input_1[] = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                         (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
        uint32_t dimensions_output[]  = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                         (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};

        QNN_VER_PTR(*tensor_0)->clientBuf = {src0->data, ggml_get_tensor_data_size(src0)};
        QNN_VER_PTR(*tensor_1)->clientBuf = {src1->data, ggml_get_tensor_data_size(src1)};
        QNN_VER_PTR(*tensor_2)->clientBuf = {dst->data, ggml_get_tensor_data_size(dst)};

        QNN_VER_PTR(*tensor_0)->dimensions = dimensions_input_0;
        QNN_VER_PTR(*tensor_0)->rank = ggml_get_tensor_rank(src0);
        QNN_VER_PTR(*tensor_0)->dataType = src0_qnn_type;
        QNN_VER_PTR(*tensor_1)->dimensions = dimensions_input_1;
        QNN_VER_PTR(*tensor_1)->rank = ggml_get_tensor_rank(src1);
        QNN_VER_PTR(*tensor_1)->dataType = src1_qnn_type;
        QNN_VER_PTR(*tensor_2)->dimensions = dimensions_output;
        QNN_VER_PTR(*tensor_2)->rank = ggml_get_tensor_rank(dst);
        QNN_VER_PTR(*tensor_2)->dataType = dst_qnn_type;

        Qnn_Tensor_t tensor_inputs[] = {
                *tensor_0,
                *tensor_1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *tensor_2
        };
        error = qnn_raw_interface.graphExecute(graph_handle, tensor_inputs, 2, tensor_outputs, 1, nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
    }
    QNN_VER_PTR(*tensor_0)->dimensions = tensor_0_dimensions;
    QNN_VER_PTR(*tensor_1)->dimensions = tensor_1_dimensions;
    QNN_VER_PTR(*tensor_2)->dimensions = tensor_2_dimensions;
    n_end_time = ggml_time_us();
    n_duration = (n_end_time - n_begin_time) / 1000;
    QNN_LOG_DEBUG("duration of ggml_qnn_add : %lld milliseconds\n", n_duration);
}



/*
 * ggml_qnn_mul_mat was re-added as a standalone function because
 * the following comments came from https://github.com/ggerganov/llama.cpp/pull/1632
 * MUL_MAT take most of the compute time (about 95%). So to speed up llama, we have to focus on MUL_MAT.
 * We have three kinds of MUL_MAT to compute:
 * mul_mat_f32: both src0 and src1 are F32.
 * mul_mat_f16_f32: src0 is F16 and src1 is F32.
 * mul_mat_q_f32: src0 is quantized (Q4_0, Q4_1, ...), and src1 is F32.
*/
static void ggml_qnn_mul_mat(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    bool graph_initialized                      = false;
    int64_t n_begin_time                        = 0LL;
    int64_t n_end_time                          = 0LL;
    int64_t n_duration                          = 0LL;

    qnn_instance * instance                     = nullptr;

    std::string graph_name                      = "ggml_op_qnn_mul_mat";
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * tensor_0                     = nullptr;
    Qnn_Tensor_t * tensor_1                     = nullptr;
    Qnn_Tensor_t * tensor_2                     = nullptr;

    Qnn_Param_t qnn_params[]                    = {};

    enum ggml_op ggmlop                         = GGML_OP_ADD;
    Qnn_DataType_t src0_qnn_type                = QNN_DATATYPE_FLOAT_32;
    Qnn_DataType_t src1_qnn_type                = QNN_DATATYPE_FLOAT_32;
    Qnn_DataType_t dst_qnn_type                 = QNN_DATATYPE_FLOAT_32;

    if ((nullptr == src0) || (nullptr == src1) || (nullptr == dst)) {
        QNN_LOG_WARN("pls check why GGML tensor is null");
        return;
    }
    tensor_0                                    = (Qnn_Tensor_t *)src0->extra;
    tensor_1                                    = (Qnn_Tensor_t *)src1->extra;
    tensor_2                                    = (Qnn_Tensor_t *)dst->extra;
    if ((nullptr == tensor_0) || (nullptr == tensor_1) || (nullptr == tensor_2)) {
        QNN_LOG_WARN("pls check why QNN tensor is null");
        return;
    }
    if (nullptr == ctx) {
        QNN_LOG_WARN("pls check why backend ctx is null");
        return;
    }
    instance                                    = ctx->instance;
    if (nullptr == instance) {
        QNN_LOG_WARN("pls check why qnn instance is null");
        return;
    }
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;

    n_begin_time                                = ggml_time_us();
    QNN_LOG_DEBUG("call %s\n", __func__);
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src0->name,
          src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
          src0->nb[0], src0->nb[1], src0->nb[2]);
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src1->name,
          src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
          src1->nb[0], src1->nb[1], src1->nb[2]);
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          dst->name,
          dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0],
          dst->nb[1], dst->nb[2]);
    QNN_LOG_DEBUG("%d, %d, %d, %d", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    QNN_LOG_DEBUG("tensor0 name %s", QNN_TENSOR_GET_NAME(*tensor_0));
    QNN_LOG_DEBUG("tensor1 name %s", QNN_TENSOR_GET_NAME(*tensor_1));
    QNN_LOG_DEBUG("tensor2 name %s", QNN_TENSOR_GET_NAME(*tensor_2));

    QNN_VER_PTR(*tensor_0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_2)->type = QNN_TENSOR_TYPE_APP_READ;

    src0_qnn_type                = qnn_datatype_from_ggml_datatype(src0->type);
    src1_qnn_type                = qnn_datatype_from_ggml_datatype(src1->type);
    dst_qnn_type                 = qnn_datatype_from_ggml_datatype(dst->type);

    uint32_t dimensions_input_0[] = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                         (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
    uint32_t dimensions_input_1[] = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                         (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
    uint32_t dimensions_output[]  = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                         (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};

    std::string map_entry        = std::string(ggml_op_name(ggmlop));
    if (instance->_qnn_graph_map.find(map_entry) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
    }

    uint32_t * tensor_0_dimensions = QNN_VER_PTR(*tensor_0)->dimensions;
    uint32_t * tensor_1_dimensions = QNN_VER_PTR(*tensor_1)->dimensions;
    uint32_t * tensor_2_dimensions = QNN_VER_PTR(*tensor_2)->dimensions;

    if (!graph_initialized) {
        graph_name = graph_name + "_" + std::to_string(ctx->threads) + src0->name + "_" + src1->name;
        QNN_LOG_DEBUG("graph name %s", graph_name.c_str());
        error = qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(), graph_name.c_str(), nullptr, &graph_handle);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("can't create qnn graph handle with graph name %s, error = %d\n", graph_name.c_str(), error);
            return;
        }
        error = qnn_raw_interface.tensorCreateGraphTensor(graph_handle, tensor_0);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.tensorCreateGraphTensor(graph_handle, tensor_1);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.tensorCreateGraphTensor(graph_handle, tensor_2);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }

        QNN_VER_PTR(*tensor_0)->clientBuf = {src0->data, ggml_get_tensor_data_size(src0)};
        QNN_VER_PTR(*tensor_1)->clientBuf = {src1->data, ggml_get_tensor_data_size(src1)};
        QNN_VER_PTR(*tensor_2)->clientBuf = {dst->data, ggml_get_tensor_data_size(dst)};

        QNN_VER_PTR(*tensor_0)->dimensions = dimensions_input_0;
        QNN_VER_PTR(*tensor_0)->rank = ggml_get_tensor_rank(src0);
        QNN_VER_PTR(*tensor_0)->dataType = src0_qnn_type;
        QNN_VER_PTR(*tensor_1)->dimensions = dimensions_input_1;
        QNN_VER_PTR(*tensor_1)->rank = ggml_get_tensor_rank(src1);
        QNN_VER_PTR(*tensor_1)->dataType = src1_qnn_type;
        QNN_VER_PTR(*tensor_2)->dimensions = dimensions_output;
        QNN_VER_PTR(*tensor_2)->rank = ggml_get_tensor_rank(dst);
        QNN_VER_PTR(*tensor_2)->dataType = dst_qnn_type;

        Qnn_Tensor_t tensor_inputs[] = {
                *tensor_0,
                *tensor_1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *tensor_2
        };
        Qnn_OpConfig_t op_config = {
                (Qnn_OpConfigVersion_t) 1, .v1 = {
                        "ggml_op_mul_mat",
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        QNN_OP_MAT_MUL,
                        0,
                        qnn_params,
                        2,
                        tensor_inputs,
                        1,
                        tensor_outputs
                }
        };
        error = qnn_raw_interface.graphAddNode(graph_handle, op_config);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.graphExecute(graph_handle, tensor_inputs, 2, tensor_outputs, 1, nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        auto  graph_item = std::make_tuple(graph_handle, tensor_0, tensor_1, tensor_2);
        instance->_qnn_graph_map[map_entry] = graph_item;
    } else {
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
        tensor_0     = std::get<1>(graph_item);
        tensor_1     = std::get<2>(graph_item);
        tensor_2     = std::get<3>(graph_item);

        QNN_LOG_DEBUG("%d, %d, %d, %d", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
        uint32_t dimensions_input_0[] = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                         (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
        uint32_t dimensions_input_1[] = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                         (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
        uint32_t dimensions_output[]  = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                         (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};
        QNN_VER_PTR(*tensor_0)->dimensions = dimensions_input_0;
        QNN_VER_PTR(*tensor_0)->rank = ggml_get_tensor_rank(src0);
        QNN_VER_PTR(*tensor_0)->dataType = src0_qnn_type;
        QNN_VER_PTR(*tensor_1)->dimensions = dimensions_input_1;
        QNN_VER_PTR(*tensor_1)->rank = ggml_get_tensor_rank(src1);
        QNN_VER_PTR(*tensor_1)->dataType = src1_qnn_type;
        QNN_VER_PTR(*tensor_2)->dimensions = dimensions_output;
        QNN_VER_PTR(*tensor_2)->rank = ggml_get_tensor_rank(dst);
        QNN_VER_PTR(*tensor_2)->dataType = dst_qnn_type;

        QNN_VER_PTR(*tensor_0)->clientBuf = {src0->data, ggml_get_tensor_data_size(src0)};
        QNN_VER_PTR(*tensor_1)->clientBuf = {src1->data, ggml_get_tensor_data_size(src1)};
        QNN_VER_PTR(*tensor_2)->clientBuf = {dst->data, ggml_get_tensor_data_size(dst)};

        Qnn_Tensor_t tensor_inputs[] = {
                *tensor_0,
                *tensor_1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *tensor_2
        };
        error = qnn_raw_interface.graphExecute(graph_handle, tensor_inputs, 2, tensor_outputs, 1, nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
    }
    QNN_VER_PTR(*tensor_0)->dimensions = tensor_0_dimensions;
    QNN_VER_PTR(*tensor_1)->dimensions = tensor_1_dimensions;
    QNN_VER_PTR(*tensor_2)->dimensions = tensor_2_dimensions;
    n_end_time = ggml_time_us();
    n_duration = (n_end_time - n_begin_time) / 1000;
    QNN_LOG_DEBUG("duration of ggml_qnn_mul_mat : %lld milliseconds\n", n_duration);
    QNN_LOG_DEBUG("call %s done\n", __func__);
}


//common function for GGML OPs using QNN API
static void ggml_qnn_hanlde_op(ggml_backend_qnn_context * ctx, const enum ggml_op ggmlop, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    bool graph_initialized                      = false;
    int64_t n_begin_time                        = 0LL;
    int64_t n_end_time                          = 0LL;
    int64_t n_duration                          = 0LL;

    qnn_instance * instance                     = nullptr;

    std::string qnn_graph_name                  = "ggml_qnn_graph";
    std::string qnn_op_config_name              = "ggml_qnn_op_config";
    const char * qnn_op_name                    = nullptr;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * tensor_0                     = nullptr;
    Qnn_Tensor_t * tensor_1                     = nullptr;
    Qnn_Tensor_t * tensor_2                     = nullptr;

    Qnn_Param_t qnn_params[]                    = {};

    Qnn_DataType_t src0_qnn_type                = QNN_DATATYPE_FLOAT_32;
    Qnn_DataType_t src1_qnn_type                = QNN_DATATYPE_FLOAT_32;
    Qnn_DataType_t dst_qnn_type                 = QNN_DATATYPE_FLOAT_32;

    if ((nullptr == src0) || (nullptr == src1) || (nullptr == dst)) {
        QNN_LOG_WARN("pls check why GGML tensor is null");
        return;
    }
    tensor_0                                    = (Qnn_Tensor_t *)src0->extra;
    tensor_1                                    = (Qnn_Tensor_t *)src1->extra;
    tensor_2                                    = (Qnn_Tensor_t *)dst->extra;
    if ((nullptr == tensor_0) || (nullptr == tensor_1) || (nullptr == tensor_2)) {
        QNN_LOG_WARN("pls check why QNN tensor is null");
        return;
    }
    if (nullptr == ctx) {
        QNN_LOG_WARN("pls check why backend ctx is null");
        return;
    }
    instance                                    = ctx->instance;
    if (nullptr == instance) {
        QNN_LOG_WARN("pls check why qnn instance is null");
        return;
    }
    QNN_INTERFACE_VER_TYPE qnn_raw_interface    = ctx->raw_interface;

    src0_qnn_type                               = qnn_datatype_from_ggml_datatype(src0->type);
    src1_qnn_type                               = qnn_datatype_from_ggml_datatype(src1->type);
    dst_qnn_type                                = qnn_datatype_from_ggml_datatype(dst->type);
    qnn_op_name                                 = qnn_opname_from_ggmlop(ggmlop);
    if (nullptr == qnn_op_name) {
        QNN_LOG_WARN("pls check why can not get QNN OP name with ggml op %d(%s)", ggmlop, ggml_op_name(ggmlop));
        return;
    }

    n_begin_time                                = ggml_time_us();

    QNN_LOG_DEBUG("call %s\n", __func__);
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src0->name,
          src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
          src0->nb[0], src0->nb[1], src0->nb[2]);
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src1->name,
          src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
          src1->nb[0], src1->nb[1], src1->nb[2]);
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          dst->name,
          dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0],
          dst->nb[1], dst->nb[2]);
    QNN_LOG_DEBUG("%d, %d, %d, %d", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    QNN_LOG_DEBUG("tensor0 name %s", QNN_TENSOR_GET_NAME(*tensor_0));
    QNN_LOG_DEBUG("tensor1 name %s", QNN_TENSOR_GET_NAME(*tensor_1));
    QNN_LOG_DEBUG("tensor2 name %s", QNN_TENSOR_GET_NAME(*tensor_2));

    QNN_VER_PTR(*tensor_0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_2)->type = QNN_TENSOR_TYPE_APP_READ;
    uint32_t dimensions_input_0[] = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                         (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
    uint32_t dimensions_input_1[] = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                         (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
    uint32_t dimensions_output[]  = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                         (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};


    std::string map_entry           = std::string(ggml_op_name(ggmlop));
    if (instance->_qnn_graph_map.find(map_entry) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
    }

    uint32_t * tensor_0_dimensions = QNN_VER_PTR(*tensor_0)->dimensions;
    uint32_t * tensor_1_dimensions = QNN_VER_PTR(*tensor_1)->dimensions;
    uint32_t * tensor_2_dimensions = QNN_VER_PTR(*tensor_2)->dimensions;

    if (!graph_initialized) {
        qnn_graph_name = qnn_graph_name + "_" + ggml_op_name(ggmlop) + std::to_string(ctx->threads) + src0->name + "_" + src1->name;
        qnn_op_config_name = qnn_op_config_name + "_" + ggml_op_name(ggmlop) + std::to_string(ctx->threads) + src0->name + "_" + src1->name;
        QNN_LOG_DEBUG("qnn graph name %s", qnn_graph_name.c_str());
        QNN_LOG_DEBUG("qnn op_config name %s", qnn_op_config_name.c_str());
        error = qnn_raw_interface.graphCreate(instance->get_qnn_context_handle(), qnn_graph_name.c_str(), nullptr, &graph_handle);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("can't create qnn graph handle with ggml op %s, graph name %s, error = %d\n", ggml_op_name(ggmlop), qnn_graph_name.c_str(), error);
            return;
        }

        error = qnn_raw_interface.tensorCreateGraphTensor(graph_handle, tensor_0);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.tensorCreateGraphTensor(graph_handle, tensor_1);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.tensorCreateGraphTensor(graph_handle, tensor_2);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }

        QNN_VER_PTR(*tensor_0)->clientBuf = {src0->data, ggml_get_tensor_data_size(src0)};
        QNN_VER_PTR(*tensor_1)->clientBuf = {src1->data, ggml_get_tensor_data_size(src1)};
        QNN_VER_PTR(*tensor_2)->clientBuf = {dst->data, ggml_get_tensor_data_size(dst)};

        QNN_VER_PTR(*tensor_0)->dimensions = dimensions_input_0;
        QNN_VER_PTR(*tensor_0)->rank = ggml_get_tensor_rank(src0);
        QNN_VER_PTR(*tensor_0)->dataType = src0_qnn_type;
        QNN_VER_PTR(*tensor_1)->dimensions = dimensions_input_1;
        QNN_VER_PTR(*tensor_1)->rank = ggml_get_tensor_rank(src1);
        QNN_VER_PTR(*tensor_1)->dataType = src1_qnn_type;
        QNN_VER_PTR(*tensor_2)->dimensions = dimensions_output;
        QNN_VER_PTR(*tensor_2)->rank = ggml_get_tensor_rank(dst);
        QNN_VER_PTR(*tensor_2)->dataType = dst_qnn_type;

        Qnn_Tensor_t tensor_inputs[] = {
                *tensor_0,
                *tensor_1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *tensor_2
        };
        Qnn_OpConfig_t op_config = {
                (Qnn_OpConfigVersion_t) 1, .v1 = {
                        qnn_op_config_name.c_str(),
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        qnn_op_name,
                        0,
                        qnn_params,
                        2,
                        tensor_inputs,
                        1,
                        tensor_outputs
                }
        };
        error = qnn_raw_interface.graphAddNode(graph_handle, op_config);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.graphFinalize(graph_handle, nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        error = qnn_raw_interface.graphExecute(graph_handle, tensor_inputs, 2, tensor_outputs, 1, nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
        auto  graph_item = std::make_tuple(graph_handle, tensor_0, tensor_1, tensor_2);
        instance->_qnn_graph_map[map_entry] = graph_item;
    } else {
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
        tensor_0     = std::get<1>(graph_item);
        tensor_1     = std::get<2>(graph_item);
        tensor_2     = std::get<3>(graph_item);

        QNN_LOG_DEBUG("%d, %d, %d, %d", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
        uint32_t dimensions_input_0[] = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                         (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
        uint32_t dimensions_input_1[] = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                         (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
        uint32_t dimensions_output[]  = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                         (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};
        QNN_VER_PTR(*tensor_0)->dimensions = dimensions_input_0;
        QNN_VER_PTR(*tensor_0)->rank = ggml_get_tensor_rank(src0);
        QNN_VER_PTR(*tensor_0)->dataType = src0_qnn_type;
        QNN_VER_PTR(*tensor_1)->dimensions = dimensions_input_1;
        QNN_VER_PTR(*tensor_1)->rank = ggml_get_tensor_rank(src1);
        QNN_VER_PTR(*tensor_1)->dataType = src1_qnn_type;
        QNN_VER_PTR(*tensor_2)->dimensions = dimensions_output;
        QNN_VER_PTR(*tensor_2)->rank = ggml_get_tensor_rank(dst);
        QNN_VER_PTR(*tensor_2)->dataType = dst_qnn_type;

        QNN_VER_PTR(*tensor_0)->clientBuf = {src0->data, ggml_get_tensor_data_size(src0)};
        QNN_VER_PTR(*tensor_1)->clientBuf = {src1->data, ggml_get_tensor_data_size(src1)};
        QNN_VER_PTR(*tensor_2)->clientBuf = {dst->data, ggml_get_tensor_data_size(dst)};

        Qnn_Tensor_t tensor_inputs[] = {
                *tensor_0,
                *tensor_1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *tensor_2
        };
        error = qnn_raw_interface.graphExecute(graph_handle, tensor_inputs, 2, tensor_outputs, 1, nullptr, nullptr);
        if (QNN_SUCCESS != error) {
            QNN_LOG_INFO("error = %d\n", error);
        }
    }
    QNN_VER_PTR(*tensor_0)->dimensions = tensor_0_dimensions;
    QNN_VER_PTR(*tensor_1)->dimensions = tensor_1_dimensions;
    QNN_VER_PTR(*tensor_2)->dimensions = tensor_2_dimensions;
    n_end_time = ggml_time_us();
    n_duration = (n_end_time - n_begin_time) / 1000;
    QNN_LOG_DEBUG("duration of ggml_qnn_%s : %lld milliseconds\n", ggml_op_name(ggmlop), n_duration);
    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_repeat(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_get_rows(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_acc(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_div(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_gelu(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_silu(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_gelu_quick(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_tanh(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_relu(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_hardsigmoid(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_hardswish(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_leaky_relu(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_sqr(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_norm(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_group_norm(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_concat(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_upscale(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_pad(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_rms_norm(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_cpy(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_dup(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_qnn_cpy(ctx, src0, dst, nullptr);
    (void) src1;
}


static void ggml_qnn_mul_mat_id(ggml_backend_qnn_context * ctx,
                                const ggml_tensor * src0,
                                const ggml_tensor * src1,
                                ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_scale(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_clamp(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_diag_mask_inf(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_soft_max(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_rope(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);

}


static void ggml_qnn_pool2d(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_im2col(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_sum_rows(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_argsort(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_nop(ggml_backend_qnn_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    (void) src0;
    (void) src1;
    (void) dst;
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


bool ggml_qnn_compute_forward(ggml_backend_qnn_context * ctx, struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    ggml_qnn_func_t func                = nullptr;
    ggml_qnn_func_common_t  func_common = nullptr;

    switch (tensor->op) {
        case GGML_OP_ADD:
            func = ggml_qnn_add;
            break;

        case GGML_OP_MUL:
            func_common = ggml_qnn_hanlde_op;
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

    if (nullptr != func)
        func(ctx, tensor->src[0], tensor->src[1], tensor);

    if (nullptr != func_common)
        func_common(ctx, tensor->op, tensor->src[0], tensor->src[1], tensor);

    return true;
}


struct ggml_backend_qnn_buffer_context {
    ggml_backend_qnn_buffer_context(size_t device) :
            device(device),
            name(GGML_QNN_NAME + std::to_string(device)) {
    }

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
    void * buffer       = nullptr;

    struct ggml_backend_qnn_context * backend_ctx = nullptr;

    size_t buffer_size  = 0;
    std::vector<void *> sub_buffers;
    std::vector<Qnn_Tensor_t *> qnn_tensors;
    size_t device;
    std::string name;
};


struct ggml_backend_qnn_buffer_type_context {
    size_t device;
    std::string name;
};


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


GGML_CALL static void ggml_backend_qnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;

    static int idx = 0;
    char tensor_name[GGML_MAX_NAME] = { 0 };
    snprintf(tensor_name, GGML_MAX_NAME, "tensor_%04d", idx++);

    uint32_t dimensions[]           = {(uint32_t) tensor->ne[0], (uint32_t) tensor->ne[1], (uint32_t) tensor->ne[2], (uint32_t) tensor->ne[3]};
    Qnn_DataType_t  qnn_data_type   = qnn_datatype_from_ggml_datatype(tensor->type);
    Qnn_TensorType_t qnn_tensor_type= QNN_TENSOR_TYPE_APP_WRITE;

    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
    } else if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
    }
    Qnn_Tensor_t  qnn_tensor = {
            .version= QNN_TENSOR_VERSION_1,
            {.v1= {
                    .id = 0,
                    .name = tensor_name,
                    .type = qnn_tensor_type,
                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                    .dataType = qnn_data_type,
                    .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                      QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                      {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                    .rank = ggml_get_tensor_rank(tensor),
                    .dimensions = dimensions,
                    .memType = QNN_TENSORMEMTYPE_RAW,
                    {.clientBuf = {.data = nullptr,
                            .dataSize = 0}}}}
    };
    Qnn_Tensor_t  * p_qnn_tensor = (Qnn_Tensor_t *)calloc(1, sizeof(Qnn_Tensor_t));
    if (nullptr == p_qnn_tensor) {
        QNN_LOG_WARN("calloc failed");
        return;
    }
    error = deep_copy_qnn_tensors(qnn_tensor, *p_qnn_tensor);
    if (error != QNN_SUCCESS) {
        free(p_qnn_tensor);
        QNN_LOG_DEBUG("init tensor failed");
        return;
    }
    tensor->extra = p_qnn_tensor;
    ctx->qnn_tensors.push_back(p_qnn_tensor);
}


GGML_CALL static void ggml_backend_qnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);

    memcpy((char *)tensor->data + offset, data, size);
}


GGML_CALL static void ggml_backend_qnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *)tensor->data + offset, size);
}


GGML_CALL static bool ggml_backend_qnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
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
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        QNN_LOG_WARN("%s: error: posix_memalign failed\n", __func__);
        return nullptr;
    }

    return data;
}


GGML_CALL static ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_qnn_buffer_type_context * buft_ctx = (ggml_backend_qnn_buffer_type_context *)buft->context;
    ggml_backend_qnn_buffer_context * ctx = new ggml_backend_qnn_buffer_context(buft_ctx->device);

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    //TODO:use pre-allocated buffer in internal memory pool
    ctx->buffer = ggml_qnn_host_malloc(size_aligned);
    ctx->buffer_size = size_aligned;

    ctx->backend_ctx = &g_qnn_mgr[buft_ctx->device];

    if (nullptr == ctx->buffer) {
        QNN_LOG_WARN("%s: failed to allocate %.2f MiB\n", __func__, size / (1 << 20));
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_qnn_buffer_interface, ctx, size);
}


GGML_CALL static size_t ggml_backend_qnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}


//TODO: this value is an experimental value, works fine with whisper/llm/minicpm-v inference on Android
GGML_CALL static size_t ggml_backend_qnn_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);

    return (96 * 1024 * 1024);
}


GGML_CALL static bool ggml_backend_qnn_buffer_type_supports_backend(ggml_backend_buffer_type_t buft,
                                                          ggml_backend_t backend) {
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
    QNN_LOG_INFO("enter %s", __func__ );
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;
    QNN_LOG_DEBUG("idx %d, name:%s", ctx->device, g_qnn_mgr[ctx->device].name);

    qnn_instance * instance = (qnn_instance*)g_qnn_mgr[ctx->device].instance;
    if (instance != nullptr) {
        std::map<std::string, std::tuple<Qnn_GraphHandle_t, Qnn_Tensor_t *, Qnn_Tensor_t *, Qnn_Tensor_t *>>::iterator graph_it;
        for (graph_it = instance->_qnn_graph_map.begin(); graph_it != instance->_qnn_graph_map.end(); graph_it++) {
            auto & graph_item = graph_it->second;
            Qnn_GraphHandle_t & graph_handle = std::get<0>(graph_item);
            GGML_UNUSED(graph_handle);
            QNN_LOG_DEBUG("graph type:%s", graph_it->first.c_str());
        }
        instance->_qnn_graph_map.clear();

        instance->qnn_finalize();
        delete instance;
        g_qnn_mgr[ctx->device].instance = nullptr;
    }

    if (g_qnn_mgr[ctx->device].backend      != nullptr) {
        delete backend;
        g_qnn_mgr[ctx->device].backend = nullptr;
    }
    QNN_LOG_INFO("leave %s", __func__ );
}


GGML_CALL static ggml_backend_buffer_type_t ggml_backend_qnn_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;

    return ggml_backend_qnn_buffer_type(ctx->device);
}


GGML_CALL static ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    enum ggml_status result         = GGML_STATUS_SUCCESS;
    ggml_backend_qnn_context * ctx  = (ggml_backend_qnn_context *) backend->context;
    GGML_UNUSED(ctx);

    ggml_compute_params params = {};
    params.type = GGML_TASK_TYPE_COMPUTE;
    params.ith = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        bool ok = ggml_qnn_compute_forward(ctx, &params, node);
        if (!ok) {
            QNN_LOG_DEBUG("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
    }

    return result;
}


GGML_CALL static bool ggml_backend_qnn_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    GGML_UNUSED(backend);

    return (ggml_qnn_can_handle_op(op, true));
}


//note: this function be used with proposal/refined ggml backend subsystem in this PR:
// https://github.com/ggerganov/llama.cpp/pull/7641
// new ggml backend(only using system memory: ggml_backend_xxx_buffer_is_host return true)
// can following this style for mixed inference between CPU&GPU / CPU&NPU very easily
GGML_CALL static bool ggml_backend_qnn_offload_op(ggml_backend_t backend, const ggml_tensor * tensor) {
    ggml_backend_qnn_context * ctx  = (ggml_backend_qnn_context *) backend->context;

    return ggml_qnn_compute_forward(ctx, nullptr, (ggml_tensor*)tensor);
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
    static ggml_guid guid = {0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81, 0x92, 0xa3, 0xb4, 0xc5,
                             0xd6, 0xe7, 0xf8, 0x09};
    return &guid;
}


static ggml_backend_t ggml_backend_qnn_reg_init(const char * params, void * user_data) {
    if (nullptr == params) {
        //QNN library path
        //can be hardcoded to "/data/local/tmp/" for Android command line application
        //or specified in JNI layer for Android APK
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

    struct ggml_backend_qnn_context * ctx = (struct ggml_backend_qnn_context *)backend->context;
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
    QNN_LOG_DEBUG("description:%s", description);
}


ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(size_t device) {
    if (device >= GGML_QNN_MAX_DEVICES) {
        QNN_LOG_DEBUG("ggml_backend_qnn_buffer_type error: device_index:%d is out of range [0, %d]\n",
               device, GGML_QNN_MAX_DEVICES - 1);
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_qnn_buffer_types[GGML_QNN_MAX_DEVICES];

    static bool ggml_backend_qnn_buffer_type_initialized = false;

    if (!ggml_backend_qnn_buffer_type_initialized) {
        for (int i = 0; i < GGML_QNN_MAX_DEVICES; i++) {
            ggml_backend_qnn_buffer_types[i] = {
                /* .iface   = */ {
                    /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
                    /* .alloc_buffer     = */ ggml_backend_qnn_buffer_type_alloc_buffer,
                    /* .get_alignment    = */ ggml_backend_qnn_buffer_type_get_alignment,
                    /* .get_max_size     = */ ggml_backend_qnn_buffer_type_get_max_size,
                    /* .get_alloc_size   = */ nullptr,// defaults to ggml_nbytes
                    /* .supports_backend = */ ggml_backend_qnn_buffer_type_supports_backend,
                    /* .is_host          = */ ggml_backend_qnn_buffer_is_host
                },
                /* .context = */ new ggml_backend_qnn_buffer_type_context { device, GGML_QNN_NAME + std::to_string(device) },
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
                        (path +
                         ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images").c_str(),
                        1)) {
            QNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            QNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
        if (0 == setenv("ADSP_LIBRARY_PATH",
                        (path +
                         ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp").c_str(),
                        1)) {
            QNN_LOG_INFO("QNN NPU backend setenv successfully");
        } else {
            QNN_LOG_ERROR("QNN NPU backend setenv failure");
        }
    } else {
        if (0 == setenv("LD_LIBRARY_PATH",
                        path.c_str(),
                        1)) {
            QNN_LOG_INFO("%s backend setenv successfully\n", get_qnn_backend_name(device));
        } else {
            QNN_LOG_ERROR("%s backend setenv failure\n", get_qnn_backend_name(device));
        }
    }

    qnn_instance * instance = nullptr;
    instance = new qnn_instance(qnn_lib_path, g_qnn_mgr[device].lib, "");
    result = instance->qnn_init(nullptr);
    if (0 != result) {
        QNN_LOG_WARN("init qnn subsystem failed with qnn backend %s, pls check why\n", get_qnn_backend_name(device));
        delete instance;
        return nullptr;
    }
    qnn_interface qnn_interface                             = instance->get_qnn_interface();
    if (!qnn_interface.is_loaded()) {
        QNN_LOG_WARN("qnn subsystem failure\n");
        delete instance;
        return nullptr;
    }

    std::string device_name = get_qnn_backend_name(device);
    QNN_LOG_INFO("qnn device name %s", device_name.c_str());
    instance->init_qnn_graph(device_name.c_str(), false);
    g_qnn_mgr[device].instance                  = instance;
    g_qnn_mgr[device].raw_interface             = instance->get_qnn_raw_interface();
    g_qnn_mgr[device].raw_system_interface      = instance->get_qnn_raw_system_interface();

    ggml_backend_t qnn_backend = new ggml_backend{
                /* .guid      = */ ggml_backend_qnn_guid(),
                /* .iface     = */ ggml_backend_qnn_interface,
                /* .context   = */ &g_qnn_mgr[device]
    };
    g_qnn_mgr[device].backend   = qnn_backend;

    return qnn_backend;
}


extern "C" GGML_CALL int ggml_backend_qnn_reg_devices(void);

GGML_CALL int ggml_backend_qnn_reg_devices() {
    for (size_t idx = 0; idx < GGML_QNN_MAX_DEVICES; idx++) {
        char name[GGML_MAX_NAME];
        ggml_backend_qnn_get_device_description(idx, name, GGML_MAX_NAME);
        ggml_backend_register(name, ggml_backend_qnn_reg_init, ggml_backend_qnn_buffer_type(idx),
                              (void *) (intptr_t)idx);
    }

    return GGML_QNN_MAX_DEVICES;
}
