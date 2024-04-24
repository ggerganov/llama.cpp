/*
 * MIT license
 * Copyright (C) 2024 GGML Authors
 * SPDX-License-Identifier: MIT
 *
 * this is implementation of ggml QNN(Qualcomm Neural Network, aka AI Engine Direct) backend
 *
 * status:
 *
 * 1. core implementation(data path works fine as expected with whisper.cpp using QNN CPU/GPU backend on Qualcomm's SoC based low-end phone
 *
 * 2. core implementation(data path works fine as expected with whisper.cpp using QNN HTP(aka DSP) backend on Qualcomm's soC based high-end phone
 *
 * 3. core implementation(data path works fine as expected with llama.cpp using QNN CPU/GPU/HTP(aka DSP) backend on Qualcomm's soC based high-end phone
 *
 * 4. GGML_OP_MUL_MAT & GGML_OP_MUL & GGML_OP_ADD using QNN API has been completed
 *
 * todo:
 *
 * 1. lack of implementation of other GGML-OPs using QNN API
 *
 * 2. only support FP32 / FP16 and the input and output tensors must be of the same data type
 *
 * 3. QNN's RPC feature(which useful for QNN HTP(aka DSP) backend) not used
 *
 * 4. multi QNN backend(CPU/GPU/DSP) simultaneously not support
 *
 * 5. multithreading not work with QNN GPU/HTP(aka DSP) backend
 *
 */
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


// =================================================================================================
//
//  forward/external/helper declaration
//
// =================================================================================================
class qnn_instance;

//TODO: should be removed because this is a workaround method during development stage
extern "C" void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

#if (defined __ANDROID__) || (defined ANDROID) //Qualcomm's QNN could running on Windows over ARM(aka WoA)
extern "C" int __android_log_print(int prio, const char * tag, const char * fmt, ...)
__attribute__((__format__(printf, 3, 4)));
#endif

static void ggml_qnn_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...);



// =================================================================================================
//
//  self-defined macro / data structure
//
// =================================================================================================
#define RPCMEM_DEFAULT_FLAGS                            1
#define RPCMEM_HEAP_ID_SYSTEM                           25

#define GGML_DUMP_TENSOR(tensor)                        ggml_tensor_dump(tensor, #tensor)

#define GGML_QNN_LOGBUF_LEN                             4096
#define GGML_QNN_MAX_BUFFERS                            128
#define MATRIX_ROW_PADDING                              512

#define BUF_MAJOR_MASK                                  0xFF000000
#define BUF_CONTROL_BASE                                0xEE000000

#define GGML_QNN_DEBUG                                  1

#define QNN_LOG_ERROR(...) ggml_qnn_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define QNN_LOG_WARN(...)  ggml_qnn_log_internal(GGML_LOG_LEVEL_DEBUG , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define QNN_LOG_INFO(...)  ggml_qnn_log_internal(GGML_LOG_LEVEL_DEBUG , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#if GGML_QNN_DEBUG
#define QNN_LOG_DEBUG(...) ggml_qnn_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define QNN_LOG_DEBUG(...)
#endif


#define VALIDATE(value, status)                         \
  do {                                                  \
    status = value;                                     \
    if (status != QNN_SUCCESS) {                        \
      QNN_LOG_WARN("%s expected QNN_SUCCESS\n", #value);       \
      return status;                                    \
    }                                                   \
  } while (0)

#define VALIDATE_TENSOR_VERSION(tensor, err)            VALIDATE(validate_tensor_version(tensor), err)

#define VALIDATE_OP_CONFIG_VERSION(op, err)             VALIDATE(validate_opconfig_version(op), err)

#define QNN_VER_PTR(x)                                  (&((x).v1))
#define QNN_OP_CFG_VALID(opConfig)                      ((opConfig).version == QNN_OPCONFIG_VERSION_1)

#define QNN_OP_CFG_GET_NAME(opConfig)                   get_qnn_oponfig_name(opConfig)
#define QNN_OP_CFG_GET_PACKAGE_NAME(opConfig)           get_qnn_opconfig_packagename(opConfig)
#define QNN_OP_CFG_GET_TYPE_NAME(opConfig)              get_qnn_opconfig_typename(opConfig)
#define QNN_OP_CFG_GET_NUM_PARAMS(opConfig)             get_qnn_opconfig_numparams(opConfig)
#define QNN_OP_CFG_GET_PARAMS(opConfig)                 get_qnn_opconfig_params(opConfig)
#define QNN_OP_CFG_GET_NUM_INPUTS(opConfig)             get_qnn_opconfig_numinputs(opConfig)
#define QNN_OP_CFG_GET_INPUTS(opConfig)                 get_qnn_opconfig_inputs(opConfig)
#define QNN_OP_CFG_GET_NUM_OUTPUTS(opConfig)            get_qnn_opconfig_numoutputs(opConfig)
#define QNN_OP_CFG_GET_OUTPUTS(opConfig)                get_qnn_opconfig_outputs(opConfig)

#define QNN_OP_CFG_SET_NAME(opConfig, value)            set_qnn_opconfig_name(opConfig, value)
#define QNN_OP_CFG_SET_PACKAGE_NAME(opConfig, value)    set_qnn_opconfig_packagename(opConfig, value)
#define QNN_OP_CFG_SET_TYPE_NAME(opConfig, value)       set_qnn_opconfig_typename(opConfig, value)

#define QNN_OP_CFG_SET_PARAMS(opConfig, numOfParams, params) \
  set_qnn_opconfig_params(opConfig, numOfParams, params)

#define QNN_OP_CFG_SET_INPUTS(opConfig, numOfInputs, inputTensors) \
  set_qnn_opconfig_inputs(opConfig, numOfInputs, inputTensors)

#define QNN_OP_CFG_SET_OUTPUTS(opConfig, numOfOutputs, outputTensors) \
  set_qnn_opconfig_outputs(opConfig, numOfOutputs, outputTensors)

#define QNN_TENSOR_GET_ID(tensor)                       get_qnn_tensorid(tensor)
#define QNN_TENSOR_GET_NAME(tensor)                     get_qnn_tensorname(tensor)
#define QNN_TENSOR_GET_TYPE(tensor)                     get_qnn_tensortype(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor)              get_qnn_tensor_dataformat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor)                get_qnn_tensor_datatype(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor)             get_qnn_tensor_quantparams(tensor)
#define QNN_TENSOR_GET_RANK(tensor)                     get_qnn_tensor_rank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor)               get_qnn_tensor_dimensions(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor)                 get_qnn_tensor_memtype(tensor)
#define QNN_TENSOR_GET_CLIENT_BUF(tensor)               get_qnn_tensor_clientbuf(tensor)
#define QNN_TENSOR_GET_MEM_HANDLE(tensor)               get_qnn_tensor_memhandle(tensor)

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



typedef struct qnn_buf_s qnn_buf_t;
typedef struct qnn_buf_s qnn_buf_buffer_t;
typedef struct buf_element_s buf_element_t;
typedef void (*ggml_qnn_func_t)(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
typedef void (*ggml_qnn_func_common_t)(const ggml_op ggmlop, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

enum class ggml_qnn_profile_level {
    profile_off = 0,
    profile_basic = 1,
    profile_detail = 2
};


struct buf_element_s {
    buf_element_t        * next;

    unsigned char        * mem;
    unsigned char        * content;   /* start of raw content in mem  */

    uint32_t              size ;      /* size of content  */
    int32_t               max_size;   /* size of pre-allocated memory pointed to by mem   */
    uint32_t              type;
    void (*free_buffer) (buf_element_t * buf);
    void                 * source;   /* CPU, GPU, DSP, ... */
    int                   id;
} ;


struct qnn_buf_s {
    buf_element_t  * first, * last;

    size_t          qnn_buf_size;
    uint32_t        qnn_buf_data_size;
    void            * qnn_buf_empty_cb_data;
    const           char * name;

    pthread_mutex_t mutex;
    pthread_cond_t  not_empty;

    void (*put) (qnn_buf_t * fifo, buf_element_t * buf);

    buf_element_t *(*get) (qnn_buf_t * fifo);

    void (*clear) (qnn_buf_t * fifo) ;

    int (*size) (qnn_buf_t * fifo);

    int (*num_free) (qnn_buf_t * fifo);

    uint32_t (*data_size) (qnn_buf_t * fifo);

    void (*destroy) (qnn_buf_t * fifo);

    buf_element_t * (*buffer_alloc) (qnn_buf_t * self);

    buf_element_t * (*buffer_try_alloc) (qnn_buf_t * self);

    buf_element_t   * buffer_pool_top;
    pthread_mutex_t  buffer_pool_mutex;
    pthread_cond_t   buffer_pool_cond_not_empty;
    int              buffer_pool_num_free;
    int              buffer_pool_capacity;
    int              buffer_pool_buf_size;
    void            * buffer_pool_base; /* used to free mem pool */
} ;


struct ggml_backend_qnn_context {
    int device;
    int threads;
    char name[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];
    qnn_instance * instance;
    qnn_buf_t * buffer_pool;
    struct ggml_backend * backend;
    QNN_INTERFACE_VER_TYPE raw_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE raw_system_interface;
} ;


// =================================================================================================
//
//  static global variables
//
// =================================================================================================
//TODO: should be removed for support multi QNN backend simultaneously
static ggml_backend_t g_qnn_backend = nullptr;

//TODO: should be removed for support multi QNN backend simultaneously
static int g_current_device        = 3; // 3 is the default ggml backend

static bool GGML_OP_HAS_INIT    [GGML_OP_COUNT] = { 0 };
static bool GGML_OP_HAS_FINALIZE[GGML_OP_COUNT] = { 0 };
static void ggml_setup_op_has_task_pass(void) {
    {   // INIT
        bool * p = GGML_OP_HAS_INIT;

        p[GGML_OP_ACC                    ] = true;
        p[GGML_OP_MUL_MAT                ] = true;
        p[GGML_OP_MUL_MAT_ID             ] = true;
        p[GGML_OP_OUT_PROD               ] = true;
        p[GGML_OP_SET                    ] = true;
        p[GGML_OP_GET_ROWS_BACK          ] = true;
        p[GGML_OP_DIAG_MASK_INF          ] = true;
        p[GGML_OP_DIAG_MASK_ZERO         ] = true;
        p[GGML_OP_CONV_TRANSPOSE_1D      ] = true;
        p[GGML_OP_CONV_TRANSPOSE_2D      ] = true;
        p[GGML_OP_FLASH_ATTN_BACK        ] = true;
        p[GGML_OP_CROSS_ENTROPY_LOSS     ] = true;
        p[GGML_OP_ADD_REL_POS            ] = true;
    }

    {   // FINALIZE
        bool * p = GGML_OP_HAS_FINALIZE;

        p[GGML_OP_CROSS_ENTROPY_LOSS     ] = true;
    }
}


//QNN cDSP and HTA backend would not be used currently, just focus on QNN CPU/GPU/HTP(aka DSP) backend currently
static struct ggml_backend_qnn_context g_qnn_mgr[GGML_QNN_MAX_DEVICES] = {
        [QNN_CPU]   = {.device = 0, .threads = 1, .name =   "qnn-cpu", .lib = "libQnnCpu.so", .instance = nullptr, .buffer_pool = nullptr, .backend = nullptr, .raw_interface = nullptr, .raw_system_interface = nullptr},
        [QNN_GPU]   = {.device = 1, .threads = 1, .name =   "qnn-gpu", .lib = "libQnnGpu.so", .instance = nullptr, .buffer_pool = nullptr, .backend = nullptr, .raw_interface = nullptr, .raw_system_interface = nullptr},
        [QNN_HTP]   = {.device = 2, .threads = 1, .name =   "qnn-htp(aka dsp)", .lib = "libQnnHtp.so", .instance = nullptr, .buffer_pool = nullptr, .backend = nullptr, .raw_interface = nullptr, .raw_system_interface = nullptr},
};



// =================================================================================================
//
//  internal helper functions
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


static inline int validate_opconfig_version(Qnn_OpConfig_t opConfig) {
    if (opConfig.version != QNN_OPCONFIG_VERSION_1) {
        QNN_LOG_WARN("validate_opconfig_version() op %s, got unsupported version %d\n",
              opConfig.v1.name,
              opConfig.version);
        return 1;
    }
    return 0;
}


static inline const char * get_qnn_oponfig_name(const Qnn_OpConfig_t & opConfig) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        return opConfig.v1.name;
    }
    return nullptr;
}


static inline const char * get_qnn_oponfig_name(const Qnn_OpConfig_t * opConfig) {
    return get_qnn_oponfig_name(*opConfig);
}


static inline const char * get_qnn_opconfig_packagename(const Qnn_OpConfig_t & opConfig) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        return opConfig.v1.packageName;
    }
    return nullptr;
}


static inline const char * get_qnn_opconfig_packagename(const Qnn_OpConfig_t * opConfig) {
    return get_qnn_opconfig_packagename(*opConfig);
}


static inline const char * get_qnn_opconfig_typename(const Qnn_OpConfig_t & opConfig) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        return opConfig.v1.typeName;
    }
    return nullptr;
}


static inline const char * get_qnn_opconfig_typename(const Qnn_OpConfig_t * opConfig) {
    return get_qnn_opconfig_typename(*opConfig);
}


static inline uint32_t get_qnn_opconfig_numparams(const Qnn_OpConfig_t & opConfig) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        return opConfig.v1.numOfParams;
    }
    return 0u;
}


static inline uint32_t get_qnn_opconfig_numparams(const Qnn_OpConfig_t * opConfig) {
    return get_qnn_opconfig_numparams(*opConfig);
}


static inline const Qnn_Param_t * get_qnn_opconfig_params(const Qnn_OpConfig_t & opConfig) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        return opConfig.v1.params;
    }
    return nullptr;
}


static inline const Qnn_Param_t * get_qnn_opconfig_params(const Qnn_OpConfig_t * opConfig) {
    return get_qnn_opconfig_params(*opConfig);
}


static inline uint32_t get_qnn_opconfig_numinputs(const Qnn_OpConfig_t & opConfig) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        return opConfig.v1.numOfInputs;
    }
    return 0u;
}


static inline uint32_t get_qnn_opconfig_numinputs(const Qnn_OpConfig_t * opConfig) {
    return get_qnn_opconfig_numinputs(*opConfig);
}


static inline const Qnn_Tensor_t * get_qnn_opconfig_inputs(const Qnn_OpConfig_t & opConfig) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        return opConfig.v1.inputTensors;
    }
    return nullptr;
}


static inline const Qnn_Tensor_t * get_qnn_opconfig_inputs(const Qnn_OpConfig_t * opConfig) {
    return get_qnn_opconfig_inputs(*opConfig);
}


static inline uint32_t get_qnn_opconfig_numoutputs(const Qnn_OpConfig_t & opConfig) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        return opConfig.v1.numOfOutputs;
    }
    return 0u;
}


static inline uint32_t get_qnn_opconfig_numoutputs(const Qnn_OpConfig_t * opConfig) {
    return get_qnn_opconfig_numoutputs(*opConfig);
}


static inline const Qnn_Tensor_t * get_qnn_opconfig_outputs(const Qnn_OpConfig_t & opConfig) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        return opConfig.v1.outputTensors;
    }
    return nullptr;
}


static inline const Qnn_Tensor_t * get_qnn_opconfig_outputs(const Qnn_OpConfig_t * opConfig) {
    return get_qnn_opconfig_outputs(*opConfig);
}


static inline void set_qnn_opconfig_name(Qnn_OpConfig_t & opConfig, const char * name) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        opConfig.v1.name = name;
    }
}


static inline void set_qnn_opconfig_name(Qnn_OpConfig_t * opConfig, const char * name) {
    set_qnn_opconfig_name(*opConfig, name);
}


static inline void set_qnn_opconfig_packagename(Qnn_OpConfig_t & opConfig, const char * packageName) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        opConfig.v1.packageName = packageName;
    }
}


static inline void set_qnn_opconfig_packagename(Qnn_OpConfig_t * opConfig, const char * packageName) {
    set_qnn_opconfig_packagename(*opConfig, packageName);
}


static inline void set_qnn_opconfig_typename(Qnn_OpConfig_t & opConfig, const char * typeName) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        opConfig.v1.typeName = typeName;
    }
}


static inline void set_qnn_opconfig_typename(Qnn_OpConfig_t * opConfig, const char * typeName) {
    set_qnn_opconfig_typename(*opConfig, typeName);
}


static inline void set_qnn_opconfig_params(Qnn_OpConfig_t & opConfig,
                                 uint32_t numOfParams,
                                 Qnn_Param_t * params) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        opConfig.v1.numOfParams = numOfParams;
        opConfig.v1.params      = params;
    }
}


static inline void set_qnn_opconfig_params(Qnn_OpConfig_t * opConfig,
                                 uint32_t numOfParams,
                                 Qnn_Param_t * params) {
    set_qnn_opconfig_params(*opConfig, numOfParams, params);
}


static inline void set_qnn_opconfig_inputs(Qnn_OpConfig_t & opConfig,
                                 uint32_t numOfInputs,
                                 Qnn_Tensor_t * inputTensors) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        opConfig.v1.numOfInputs  = numOfInputs;
        opConfig.v1.inputTensors = inputTensors;
    }
}


static inline void set_qnn_opconfig_inputs(Qnn_OpConfig_t * opConfig,
                                 uint32_t numOfInputs,
                                 Qnn_Tensor_t * inputTensors) {
    set_qnn_opconfig_inputs(*opConfig, numOfInputs, inputTensors);
}


static inline void set_qnn_opconfig_outputs(Qnn_OpConfig_t & opConfig,
                                  uint32_t numOfOutputs,
                                  Qnn_Tensor_t * outputTensors) {
    if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
        opConfig.v1.numOfOutputs  = numOfOutputs;
        opConfig.v1.outputTensors = outputTensors;
    }
}


static inline void set_qnn_opconfig_outputs(Qnn_OpConfig_t * opConfig,
                                  uint32_t numOfOutputs,
                                  Qnn_Tensor_t * outputTensors) {
    set_qnn_opconfig_outputs(*opConfig, numOfOutputs, outputTensors);
}


static inline uint32_t get_qnn_tensorid(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.id;
    }
    return 0u;
}


static inline uint32_t get_qnn_tensorid(const Qnn_Tensor_t * tensor) { return get_qnn_tensorid(*tensor); }


static inline const char * get_qnn_tensorname(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.name;
    }
    return nullptr;
}


static inline const char * get_qnn_tensorname(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensorname(*tensor);
}


static inline Qnn_TensorType_t get_qnn_tensortype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.type;
    }
    return QNN_TENSOR_TYPE_UNDEFINED;
}


static inline Qnn_TensorType_t get_qnn_tensortype(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensortype(*tensor);
}


static inline Qnn_TensorDataFormat_t get_qnn_tensor_dataformat(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataFormat;
    }
    return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}


static inline Qnn_TensorDataFormat_t get_qnn_tensor_dataformat(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_dataformat(*tensor);
}


static inline Qnn_DataType_t get_qnn_tensor_datatype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dataType;
    }
    return QNN_DATATYPE_UNDEFINED;
}


static inline Qnn_DataType_t get_qnn_tensor_datatype(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_datatype(*tensor);
}


static inline Qnn_QuantizeParams_t get_qnn_tensor_quantparams(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.quantizeParams;
    }
    return QNN_QUANTIZE_PARAMS_INIT;
}


static inline Qnn_QuantizeParams_t get_qnn_tensor_quantparams(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_quantparams(*tensor);
}


static inline uint32_t get_qnn_tensor_rank(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.rank;
    }
    return 0u;
}


static inline uint32_t get_qnn_tensor_rank(const Qnn_Tensor_t * tensor) { return get_qnn_tensor_rank(*tensor); }


static inline uint32_t * get_qnn_tensor_dimensions(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.dimensions;
    }
    return nullptr;
}


static inline uint32_t * get_qnn_tensor_dimensions(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_dimensions(*tensor);
}


static inline Qnn_TensorMemType_t get_qnn_tensor_memtype(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.memType;
    }
    return QNN_TENSORMEMTYPE_UNDEFINED;
}


static inline Qnn_TensorMemType_t get_qnn_tensor_memtype(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_memtype(*tensor);
}


static inline Qnn_ClientBuffer_t get_qnn_tensor_clientbuf(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.clientBuf;
    }
    return QNN_CLIENT_BUFFER_INIT;
}


static inline Qnn_ClientBuffer_t get_qnn_tensor_clientbuf(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_clientbuf(*tensor);
}


static inline Qnn_MemHandle_t get_qnn_tensor_memhandle(const Qnn_Tensor_t & tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        return tensor.v1.memHandle;
    }
    return nullptr;
}


static inline Qnn_MemHandle_t get_qnn_tensor_memhandle(const Qnn_Tensor_t * tensor) {
    return get_qnn_tensor_memhandle(*tensor);
}


static inline void set_qnn_tensor_id(Qnn_Tensor_t & tensor, uint32_t id) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.id = id;
    }
}


static inline void set_qnn_tensor_id(Qnn_Tensor_t * tensor, uint32_t id) { set_qnn_tensor_id(*tensor, id); }


static inline void set_qnn_tensor_name(Qnn_Tensor_t & tensor, const char * name) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.name = name;
    }
}


static inline void set_qnn_tensor_name(Qnn_Tensor_t * tensor, const char * name) {
    set_qnn_tensor_name(*tensor, name);
}


static inline void set_qnn_tensor_type(Qnn_Tensor_t & tensor, Qnn_TensorType_t type) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.type = type;
    }
}


static inline void set_qnn_tensor_type(Qnn_Tensor_t * tensor, Qnn_TensorType_t type) {
    set_qnn_tensor_type(*tensor, type);
}


static inline void set_qnn_tensor_dataformat(Qnn_Tensor_t & tensor, Qnn_TensorDataFormat_t format) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dataFormat = format;
    }
}


static inline void set_qnn_tensor_dataformat(Qnn_Tensor_t * tensor, Qnn_TensorDataFormat_t format) {
    set_qnn_tensor_dataformat(*tensor, format);
}


static inline void set_qnn_tensor_datatype(Qnn_Tensor_t & tensor, Qnn_DataType_t dataType) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dataType = dataType;
    }
}


static inline void set_qnn_tensor_datatype(Qnn_Tensor_t * tensor, Qnn_DataType_t dataType) {
    set_qnn_tensor_datatype(*tensor, dataType);
}


static inline void set_qnn_tensor_quantparams(Qnn_Tensor_t & tensor, Qnn_QuantizeParams_t params) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.quantizeParams = params;
    }
}


static inline void set_qnn_tensor_quantparams(Qnn_Tensor_t * tensor, Qnn_QuantizeParams_t params) {
    set_qnn_tensor_quantparams(*tensor, params);
}


static inline void set_qnn_tensor_rank(Qnn_Tensor_t & tensor, uint32_t rank) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.rank = rank;
    }
}


static inline void set_qnn_tensor_rank(Qnn_Tensor_t * tensor, uint32_t rank) {
    set_qnn_tensor_rank(*tensor, rank);
}


static inline void set_qnn_tensor_dimensions(Qnn_Tensor_t & tensor, uint32_t * dims) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.dimensions = dims;
    }
}


static inline void set_qnn_tensor_dimensions(Qnn_Tensor_t * tensor, uint32_t * dims) {
    set_qnn_tensor_dimensions(*tensor, dims);
}


static inline void set_qnn_tensor_memtype(Qnn_Tensor_t & tensor, Qnn_TensorMemType_t memType) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memType = memType;
    }
}


static inline void set_qnn_tensor_memtype(Qnn_Tensor_t * tensor, Qnn_TensorMemType_t memType) {
    set_qnn_tensor_memtype(*tensor, memType);
}


static inline void set_qnn_tensor_clientbuf(Qnn_Tensor_t & tensor, Qnn_ClientBuffer_t clientBuf) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.clientBuf = clientBuf;
    }
}


static inline void set_qnn_tensor_clientbuf(Qnn_Tensor_t * tensor, Qnn_ClientBuffer_t clientBuf) {
    set_qnn_tensor_clientbuf(*tensor, clientBuf);
}


static inline void set_qnn_tensor_memhandle(Qnn_Tensor_t & tensor, Qnn_MemHandle_t handle) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.memHandle = handle;
    }
}


static inline void set_qnn_tensor_memhandle(Qnn_Tensor_t * tensor, Qnn_MemHandle_t handle) {
    set_qnn_tensor_memhandle(*tensor, handle);
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
        Qnn_ClientBuffer_t clientBuf = {nullptr, 0};
        QNN_TENSOR_SET_CLIENT_BUF(dst, clientBuf);
    } else if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_MEMHANDLE) {
        QNN_TENSOR_SET_MEM_HANDLE(dst, nullptr);
    } else {
        return 1;
    }

    Qnn_QuantizeParams_t srcQParam      = QNN_TENSOR_GET_QUANT_PARAMS(src);
    Qnn_QuantizationEncoding_t encoding = srcQParam.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        // need to allocate and copy memory for scaleOffset as it is a pointer array
        Qnn_QuantizeParams_t srcQParamCpy      = srcQParam;
        Qnn_AxisScaleOffset_t &axisScaleOffset = srcQParamCpy.axisScaleOffsetEncoding;
        Qnn_ScaleOffset_t **scaleOffset        = &axisScaleOffset.scaleOffset;
        size_t scaleOffsetSize = axisScaleOffset.numScaleOffsets * sizeof(Qnn_ScaleOffset_t);
        *scaleOffset           = (Qnn_ScaleOffset_t *)malloc(scaleOffsetSize);
        memscpy(*scaleOffset,
                scaleOffsetSize,
                srcQParam.axisScaleOffsetEncoding.scaleOffset,
                scaleOffsetSize);
        QNN_TENSOR_SET_QUANT_PARAMS(dst, srcQParamCpy);
    } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
        // need to allocate and copy memory for scaleOffset as it is a pointer array
        Qnn_QuantizeParams_t srcQParamCpy          = srcQParam;
        Qnn_BwAxisScaleOffset_t &bwAxisScaleOffset = srcQParamCpy.bwAxisScaleOffsetEncoding;
        size_t scaleSize                           = bwAxisScaleOffset.numElements * sizeof(float);
        float **scales                             = &bwAxisScaleOffset.scales;
        int32_t **offsets                          = &bwAxisScaleOffset.offsets;
        *scales                                    = (float *)malloc(scaleSize);
        memscpy(*scales, scaleSize, srcQParam.bwAxisScaleOffsetEncoding.scales, scaleSize);

        // Only copy offsets if present, nullptr implies all offsets are 0
        if (bwAxisScaleOffset.offsets != nullptr) {
            size_t offsetSize = bwAxisScaleOffset.numElements * sizeof(int32_t);
            *offsets          = (int32_t *)malloc(offsetSize);
            memscpy(*offsets, offsetSize, srcQParam.bwAxisScaleOffsetEncoding.offsets, offsetSize);
        }
        QNN_TENSOR_SET_QUANT_PARAMS(dst, srcQParamCpy);
    } else {
        QNN_TENSOR_SET_QUANT_PARAMS(dst, srcQParam);
    }

    // need to allocate and copy memory for all the pointer members
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

    if (nullptr == QNN_TENSOR_GET_NAME(tensor)) {
        QNN_LOG_INFO("it should not happen, pls check");
    } else {
        //QNN_LOG_DEBUG("QNN tensor name %s", QNN_TENSOR_GET_NAME(tensor));
        free((void *) QNN_TENSOR_GET_NAME(tensor));
    }
    if (nullptr == QNN_TENSOR_GET_DIMENSIONS(tensor)) {
        QNN_LOG_INFO("it should not happen, pls check");
    } else {
        //TODO:why crash in here? why pointer changed with mul_mat?
        //memory leak after comment above line
        //free(QNN_TENSOR_GET_DIMENSIONS(tensor));
    }

    return err;
}


static int free_qnn_tensors(Qnn_Tensor_t *& tensors, uint32_t numTensors) {
    int err = 0;

    // free all pointer allocations in struct
    for (size_t i = 0; i < numTensors; i++) {
        free_qnn_tensor(tensors[i]);
    }
    free(tensors);

    return err;
}


static float ggml_tensor_sum_elements(const ggml_tensor * tensor) {
    double sum = 0;
    float  value = 0;
    std::ostringstream tmposs;
    if (tensor->type == GGML_TYPE_F32) {
        for (int h = 0; h < tensor->ne[3]; h++) {
            for (int i = 0; i < tensor->ne[2]; i++) {
                for (int j = 0; j < tensor->ne[1]; j++) {
                    for (int k = 0; k < tensor->ne[0]; k++) {
                        value = ((float *) tensor->data)[h * tensor->ne[2] + i * tensor->ne[1] + j * tensor->ne[0] + k];
                        sum += value;
                        //QNN_LOG_DEBUG("[%d][%d][%d][%d]%.2f \t", h, i, j, k, value);
                        tmposs << std::setw(8) << std::fixed << std::setprecision(2) << value << "\t";
                    }
                    if (strlen(tmposs.str().c_str()) > 4000) {

                    } else {
                        QNN_LOG_DEBUG("%s", tmposs.str().c_str());
                    }
                    tmposs.clear();
                    tmposs.str("");
                    QNN_LOG_DEBUG("\n");
                }
            }
        }
    }
    QNN_LOG_DEBUG("\n");
    return sum;
}


static void ggml_dump_tensor(const ggml_tensor * tensor, const char * name) {
    QNN_LOG_DEBUG("dump ggml tensor %s\n", name);
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n", name,
          tensor->type, ggml_type_name(tensor->type),
          tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[0], tensor->nb[1], tensor->nb[2]);
    float sum = ggml_tensor_sum_elements(tensor);

    //QNN_LOG_DEBUG("\n");
    //QNN_LOG_DEBUG("Sum of tensor %s is %6.2f\n", name, sum);
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


//TODO:
//ref:explanation of k-quants, https://github.com/ggerganov/llama.cpp/pull/1684
static Qnn_DataType_t qnn_datatype_from_ggml_datatype(enum ggml_type ggmltype) {
    switch (ggmltype) {
        case GGML_TYPE_Q4_0:
            return QNN_DATATYPE_UFIXED_POINT_4;
        case GGML_TYPE_Q4_1:
            return QNN_DATATYPE_SFIXED_POINT_4;
        case GGML_TYPE_Q8_0:
            return QNN_DATATYPE_UFIXED_POINT_8;
        case GGML_TYPE_Q8_1:
            return QNN_DATATYPE_SFIXED_POINT_8;
        case GGML_TYPE_F16:
            return QNN_DATATYPE_FLOAT_16;
        case GGML_TYPE_F32:
            return QNN_DATATYPE_FLOAT_32;

    }
    return QNN_DATATYPE_FLOAT_32;
}


//TODO:
static const char * qnn_opname_from_ggmlop(enum ggml_op ggmlop) {
    switch (ggmlop) {
        case GGML_OP_ADD:
            return QNN_OP_ELEMENT_WISE_ADD;
        case GGML_OP_MUL:
            return QNN_OP_ELEMENT_WISE_MULTIPLY;
        case GGML_OP_MUL_MAT:
            return QNN_OP_MAT_MUL;
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


static void qnn_xfree(void * ptr) {
    if (nullptr != ptr) {
        free(ptr);
        ptr = nullptr;
    }
}


static void * qnn_xmalloc(size_t size) {
    void * ptr;

    if (!size)
        size++;

    if ((ptr = calloc(1, size)) == nullptr) {
        QNN_LOG_WARN("malloc(%d) failed: %s\n",size, strerror(errno));
        return nullptr;
    }

    return ptr;
}


static void * qnn_xmalloc_aligned(size_t alignment, size_t size, void ** base) {
    char * ptr;

    *base = ptr = static_cast<char *>(qnn_xmalloc(size + alignment));

    while ((size_t) ptr % alignment)
        ptr++;

    return ptr;
}


static void buffer_pool_free (buf_element_t * element) {
    qnn_buf_t * self = (qnn_buf_t *) element->source;

    pthread_mutex_lock(&self->buffer_pool_mutex);

    element->next = self->buffer_pool_top;
    self->buffer_pool_top = element;

    self->buffer_pool_num_free++;
    if (self->buffer_pool_num_free > self->buffer_pool_capacity) {
        QNN_LOG_DEBUG("TOO MANY FREE\n");
    }

    pthread_cond_signal (&self->buffer_pool_cond_not_empty);

    pthread_mutex_unlock (&self->buffer_pool_mutex);
}


static buf_element_t * buffer_pool_alloc (qnn_buf_t * self) {
    buf_element_t * buf = nullptr;
    int i;

    pthread_mutex_lock (&self->buffer_pool_mutex);

    while (self->buffer_pool_num_free < 2) {
        pthread_cond_wait (&self->buffer_pool_cond_not_empty, &self->buffer_pool_mutex);
    }

    buf = self->buffer_pool_top;
    self->buffer_pool_top = self->buffer_pool_top->next;
    self->buffer_pool_num_free--;

    buf->content = buf->mem;
    buf->size = 0;
    buf->type = 0;

    pthread_mutex_unlock (&self->buffer_pool_mutex);

    return buf;
}


static buf_element_t * buffer_pool_try_alloc (qnn_buf_t * self) {
    buf_element_t * buf = nullptr;

    pthread_mutex_lock (&self->buffer_pool_mutex);

    if (self->buffer_pool_top) {
        buf = self->buffer_pool_top;
        self->buffer_pool_top = self->buffer_pool_top->next;
        self->buffer_pool_num_free--;
    } else {
        buf = nullptr;
    }

    pthread_mutex_unlock (&self->buffer_pool_mutex);

    if (buf) {
        buf->content = buf->mem;
        buf->size = 0;
    }

    return buf;
}


static void qnn_buf_buffer_put(qnn_buf_t * fifo, buf_element_t * element) {
    pthread_mutex_lock (&fifo->mutex);

    if (fifo->last)
        fifo->last->next = element;
    else
        fifo->first = element;

    fifo->last = element;
    element->next = nullptr;
    fifo->qnn_buf_size++;
    fifo->qnn_buf_data_size += element->size;

    LOGJ("put:index %d, fifo->size is %d, self->buffer_pool_num_free %d\n", element->id, fifo->qnn_buf_size, fifo->buffer_pool_num_free);
    pthread_cond_signal (&fifo->not_empty);

    pthread_mutex_unlock (&fifo->mutex);
}


static buf_element_t * qnn_buf_buffer_get (qnn_buf_t * fifo) {
    buf_element_t * buf = nullptr;

    pthread_mutex_lock (&fifo->mutex);
#if 0
    while (fifo->first == nullptr) {
        pthread_cond_wait (&fifo->not_empty, &fifo->mutex);
    }
#else
    if (fifo->first == nullptr) {
        pthread_mutex_unlock (&fifo->mutex);
        return nullptr;
    }
#endif

    buf = fifo->first;

    fifo->first = fifo->first->next;
    if (fifo->first==nullptr)
        fifo->last = nullptr;

    fifo->qnn_buf_size--;
    fifo->qnn_buf_data_size -= buf->size;

    pthread_mutex_unlock (&fifo->mutex);

    return buf;
}


static void qnn_buf_buffer_clear (qnn_buf_t * fifo) {
    buf_element_t * buf, * next, * prev;

    pthread_mutex_lock (&fifo->mutex);

    buf = fifo->first;
    prev = nullptr;

    while (buf != nullptr) {
        next = buf->next;
        if ((buf->type & BUF_MAJOR_MASK) !=  BUF_CONTROL_BASE) {
            if (prev)
                prev->next = next;
            else
                fifo->first = next;

            if (!next)
                fifo->last = prev;

            fifo->qnn_buf_size--;
            fifo->qnn_buf_data_size -= buf->size;

            buf->free_buffer(buf);
        } else {
            prev = buf;
        }

        buf = next;
    }

    QNN_LOG_DEBUG("free buffers after clear: %d\n", fifo->buffer_pool_num_free);
    pthread_mutex_unlock (&fifo->mutex);
}


static int qnn_buf_buffer_size (qnn_buf_t * self) {
    int size = 0;

    pthread_mutex_lock(&self->mutex);
    size = self->qnn_buf_size;
    pthread_mutex_unlock(&self->mutex);

    return size;
}


static uint32_t qnn_buf_buffer_data_size (qnn_buf_t * self) {
    uint32_t data_size;

    pthread_mutex_lock(&self->mutex);
    data_size = self->qnn_buf_data_size;
    pthread_mutex_unlock(&self->mutex);

    return data_size;
}


static int qnn_buf_buffer_num_free (qnn_buf_t * self) {
    int buffer_pool_num_free = 0;

    pthread_mutex_lock(&self->mutex);
    buffer_pool_num_free = self->buffer_pool_num_free;
    pthread_mutex_unlock(&self->mutex);

    return buffer_pool_num_free;
}


static void qnn_buf_buffer_dispose (qnn_buf_t * self) {
    buf_element_t * buf, * next;
    int received = 0;

    self->clear( self );
    buf = self->buffer_pool_top;

    while (buf != nullptr) {
        next = buf->next;
        qnn_xfree(buf);
        received++;

        buf = next;
    }

    while (received < self->buffer_pool_capacity) {
        buf = self->get(self);
        qnn_xfree(buf);
        received++;
    }

    qnn_xfree(self->buffer_pool_base);
    pthread_mutex_destroy(&self->mutex);
    pthread_cond_destroy(&self->not_empty);
    pthread_mutex_destroy(&self->buffer_pool_mutex);
    pthread_cond_destroy(&self->buffer_pool_cond_not_empty);
    qnn_xfree((void *)self->name);
    qnn_xfree (self);
}


static qnn_buf_t * qnn_buf_new(const char * name, int num_buffers, uint32_t buf_size) {
    int    i                = 0;
    int    alignment        = 4;
    qnn_buf_t * self        = nullptr;
    uint8_t  * multi_buffer = nullptr;

    self = (qnn_buf_t*)qnn_xmalloc(sizeof(qnn_buf_t));
    if (nullptr == self) {
        QNN_LOG_WARN("malloc memory failed\n");
        return nullptr;
    }

    self->name                = strdup(name);
    self->first               = nullptr;
    self->last                = nullptr;
    self->qnn_buf_size        = 0;
    self->put                 = qnn_buf_buffer_put;
    self->get                 = qnn_buf_buffer_get;
    self->clear               = qnn_buf_buffer_clear;
    self->size                = qnn_buf_buffer_size;
    self->num_free            = qnn_buf_buffer_num_free;
    self->data_size           = qnn_buf_buffer_data_size;
    self->destroy             = qnn_buf_buffer_dispose;
    pthread_mutex_init (&self->mutex, nullptr);
    pthread_cond_init (&self->not_empty, nullptr);


    if (buf_size % alignment != 0)
        buf_size += alignment - (buf_size % alignment);

    QNN_LOG_INFO("[%s]allocating %d Mbytes memory(alignment = %d)\n", name, (num_buffers * buf_size) / (1 << 20), alignment);

    multi_buffer = (uint8_t *)qnn_xmalloc_aligned (alignment, num_buffers * buf_size, &self->buffer_pool_base);
    if (nullptr == multi_buffer) {
        QNN_LOG_WARN("malloc memory failed\n");
        free(self);
        return nullptr;
    }

    self->buffer_pool_top       = nullptr;

    pthread_mutex_init (&self->buffer_pool_mutex, nullptr);
    pthread_cond_init (&self->buffer_pool_cond_not_empty, nullptr);

    self->buffer_pool_num_free  = 0;
    self->buffer_pool_capacity  = num_buffers;
    self->buffer_pool_buf_size  = buf_size;
    self->buffer_alloc          = buffer_pool_alloc;
    self->buffer_try_alloc      = buffer_pool_try_alloc;

    for (i = 0; i < num_buffers; i++) {
        buf_element_t * buf = nullptr;

        buf = (buf_element_t *)qnn_xmalloc(sizeof (buf_element_t));
        if (nullptr == buf) {
            QNN_LOG_WARN("malloc memory failed");
            free(multi_buffer);
            free(self);
            return nullptr;
        }

        buf->id          = i;
        buf->mem         = multi_buffer;
        multi_buffer     += buf_size;

        buf->max_size    = buf_size;
        buf->free_buffer = buffer_pool_free;
        buf->source      = self;

        buffer_pool_free(buf);
    }

    return self;
}


static const char * get_qnn_backend_name(int n_backend_type) {
    switch (n_backend_type) {
        case 0:
            return "QNN-CPU";
        case 1:
            return "QNN-GPU";
        case 2:
            return "QNN-HTP(DSP)";
        case 3:
            return "ggml";      //the default GGML backend, used to compare performance between QNN backend and the default GGML backend

#if 0 //QNN cDSP and HTA backend would not be used currently, focus on QNN CPU/GPU/HTP(aka DSP) backend currently
        case 3:
            return "QNN-cDSP";
        case 4:
            return "QNN-HTA";
#endif

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
            __android_log_print(level, "llamacpp", "%s", s_ggml_qnn_log_internal_buf);
#else
            printf("%s", buffer); //Qualcomm's QNN could running on Window over ARM
#endif
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
    void *_model_lib_handle = nullptr;

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


int32_t qnn_instance::rpcmem_to_fd(void *buf) {
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

    auto saver_initialize = load_qnn_functionpointers<_pfn_QnnSaver_initialize *>(
            _loaded_lib_handle[backend_id], "QnnSaver_initialize");
    if (nullptr != saver_initialize) {
        error = saver_initialize(saver_config);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to saver_initialize，error %d", QNN_GET_ERROR_CODE(error));
            return 7;
        }
    } else {
        QNN_LOG_WARN("saver_initialize is null\n");
    }

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
        QNN_LOG_WARN("can not pen QNN library %s, error: %s\n", system_lib_path.c_str(), dlerror());
        return 1;
    }

    auto *get_providers = reinterpret_cast<_pfn_QnnSystemInterface_getProviders *>(dlsym(
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
        LOGW("can not create QNN system contenxt\n");
    } else {
        QNN_LOG_DEBUG("initialize qnn system successfully\n");
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

    return 0;
}


static void ggml_qnn_logcallback(const char * fmt,
                                 QnnLog_Level_t level,
                                 uint64_t timestamp,
                                 va_list argp) {

    static std::mutex log_mutex;
    static unsigned char s_ggml_qnn_logbuf[GGML_QNN_LOGBUF_LEN];

    const char * levelStr = "";
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:
            levelStr = " ERROR ";
            break;
        case QNN_LOG_LEVEL_WARN:
            levelStr = "WARNING";
            break;
        case QNN_LOG_LEVEL_INFO:
            levelStr = "  INFO ";
            break;
        case QNN_LOG_LEVEL_DEBUG:
            levelStr = " DEBUG ";
            break;
        case QNN_LOG_LEVEL_VERBOSE:
            levelStr = "VERBOSE";
            break;
        case QNN_LOG_LEVEL_MAX:
            levelStr = "UNKNOWN";
            break;
    }

    double ms = (double) timestamp / 1000000.0;

    {
        std::lock_guard<std::mutex> lock(log_mutex);

        int len_content = 0;
        memset(s_ggml_qnn_logbuf, 0, GGML_QNN_LOGBUF_LEN);
        len_content = vsnprintf(reinterpret_cast<char *const>(s_ggml_qnn_logbuf), GGML_QNN_LOGBUF_LEN, fmt, argp);
        //QNN_LOG_DEBUG("%8.1fms [%-7s] %s ", ms, levelStr, s_ggml_qnn_logbuf);
    }
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

#if 1
    _qnn_interface.qnn_log_create(ggml_qnn_logcallback, _qnn_log_level, &_qnn_log_handle);
#else
    _qnn_raw_interface.logCreate(ggml_qnn_logcallback, _qnn_log_level, &_qnn_log_handle);
#endif
    if (nullptr == _qnn_log_handle) {
        QNN_LOG_WARN("why failed to initialize qnn log\n"); //DSP backend not work on Qualcomm SoC based low-end phone
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

    /*
    std::vector<const QnnDevice_Config_t*> temp_device_config;
    _qnn_interface.qnn_device_create(_qnn_log_handle, temp_device_config.empty() ? nullptr : temp_device_config.data(), &_qnn_device_handle);
    if (nullptr == _qnn_device_handle) {
        QNN_LOG_WARN("why failed to initialize qnn device\n");
        //return 6;
    }
    */

    if (ggml_qnn_profile_level::profile_off != _profile_level) {
        QNN_LOG_INFO("profiling turned on; level = %d", _profile_level);
        if (ggml_qnn_profile_level::profile_basic == _profile_level) {
            QNN_LOG_INFO("basic profiling requested. creating Qnn Profile object\n");
            if (QNN_PROFILE_NO_ERROR != _qnn_raw_interface.profileCreate(
                    _qnn_backend_handle, QNN_PROFILE_LEVEL_BASIC, &_qnn_profile_handle)) {
                QNN_LOG_WARN("unable to create profile handle in the backend\n");
                return 7;
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
        return 9;
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
        return 10;
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
        return 8;
    } else {
        QNN_LOG_DEBUG("initialize qnn context successfully\n");
    }

    QNN_LOG_DEBUG("leave qni_init\n");

    return 0;
}


//QNN SDK would/might/should release all allocated resource in SDK's internal
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
static bool ggml_qnn_can_handle_op(const struct ggml_tensor * src0, const struct ggml_tensor * src1,
                                 struct ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    //double check
    bool supported_op = ((dst->op == GGML_OP_ADD) || (dst->op == GGML_OP_MUL) || (dst->op == GGML_OP_MUL_MAT));
    if (!supported_op) {
        QNN_LOG_DEBUG("op %d(%s)not support", dst->op, ggml_op_name(dst->op));
        return false;
    }


    //make QNN SDK happy
    if (dst->op == GGML_OP_ADD) {
        return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
               (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
               (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) && ((ne00 > 1 && ne01 > 1 && ne10 > 1 && ne11 > 1)) &&
               (src0->rank == src1->rank);

    }

    if (dst->op == GGML_OP_MUL_MAT) {
#if 1 // log output have significant effect to performance but useful during development stage
        QNN_LOG_DEBUG("GGML_OP_MUL_MAT");
        QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s)  ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
              src0->name, src0->rank,
              src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
              src0->nb[0], src0->nb[1], src0->nb[2]);
        QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
              src1->name, src1->rank,
              src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
              src1->nb[0], src1->nb[1], src1->nb[2]);
        QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
              dst->name, dst->rank,
              dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0],
              dst->nb[1], dst->nb[2]);
#endif
    }

    //make QNN SDK happy
    return  (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
            (src0->type == src1->type) && (src0->type == dst->type) && ((ne00 > 1 && ne01 > 1 && ne10 > 1 && ne11 > 1));


}


static void ggml_qnn_add(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    bool graph_initialized                      = false;
    int64_t n_begin_time                        = 0LL;
    int64_t n_end_time                          = 0LL;
    int64_t n_durtion                           = 0LL;

    qnn_instance * instance                     = nullptr;
    struct ggml_backend_qnn_context * ctx       = nullptr;

    std::string graph_name                      = "ggml_op_qnn_add";
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * tensor_0                     = nullptr;
    Qnn_Tensor_t * tensor_1                     = nullptr;
    Qnn_Tensor_t * tensor_2                     = nullptr;

    Qnn_QuantizeParams_t quantize_param         = QNN_QUANTIZE_PARAMS_INIT;
    Qnn_OpConfig_t qnn_opconfig                 = QNN_OPCONFIG_INIT;
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
    ctx                                         = (struct ggml_backend_qnn_context *)g_qnn_backend->context;
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
#if 0  //it works fine with whisper.cpp and llama.cpp. comment them because focus on mulmat in llama.cpp inference since 04-23-2024
    QNN_LOG_DEBUG("call %s\n", __func__);
    QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s)  ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src0->name, src0->rank,
          src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
          src0->nb[0], src0->nb[1], src0->nb[2]);
    QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src1->name, src1->rank,
          src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
          src1->nb[0], src1->nb[1], src1->nb[2]);
    QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          dst->name, dst->rank,
          dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0],
          dst->nb[1], dst->nb[2]);
    QNN_LOG_DEBUG("%d, %d, %d, %d", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    QNN_LOG_DEBUG("tensor0 name %s", QNN_TENSOR_GET_NAME(tensor_0));
    QNN_LOG_DEBUG("tensor1 name %s", QNN_TENSOR_GET_NAME(tensor_1));
    QNN_LOG_DEBUG("tensor2 name %s", QNN_TENSOR_GET_NAME(tensor_2));
#endif

    QNN_VER_PTR(*tensor_0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_2)->type = QNN_TENSOR_TYPE_APP_READ;

    uint32_t dimensions_input_0[]   = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                     (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
    uint32_t dimensions_input_1[]   = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                     (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
    uint32_t dimensions_output[]    = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                     (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};


    src0_qnn_type                   = qnn_datatype_from_ggml_datatype(src0->type);
    src1_qnn_type                   = qnn_datatype_from_ggml_datatype(src1->type);
    dst_qnn_type                    = qnn_datatype_from_ggml_datatype(dst->type);

    std::string map_entry           = std::string(ggml_op_name(ggmlop));
    if (instance->_qnn_graph_map.find(map_entry) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
    }

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

        Qnn_Tensor_t tensor_inputs[] = {
                *tensor_0,
                *tensor_1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *tensor_2
        };
        Qnn_OpConfig_t opconfig = {
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
        error = qnn_raw_interface.graphAddNode(graph_handle, opconfig);
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

        //comment them because focus on mulmat in llama.cpp inference since 04-23-2024
        //QNN_LOG_DEBUG("%d, %d, %d, %d", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
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
    n_end_time = ggml_time_us();
    n_durtion = (n_end_time - n_begin_time) / 1000;
    //comment them because focus on mulmat in llama.cpp inference since 04-23-2024
    //QNN_LOG_DEBUG("duration of ggml_qnn_add : %lld milliseconds\n", n_durtion);
    //QNN_LOG_DEBUG("call %s done\n", __func__);
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

static void ggml_qnn_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    bool graph_initialized                      = false;
    int64_t n_begin_time                        = 0LL;
    int64_t n_end_time                          = 0LL;
    int64_t n_durtion                           = 0LL;

    qnn_instance * instance                     = nullptr;
    struct ggml_backend_qnn_context * ctx       = nullptr;

    std::string graph_name                      = "ggml_op_qnn_mul_mat";
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * tensor_0                     = nullptr;
    Qnn_Tensor_t * tensor_1                     = nullptr;
    Qnn_Tensor_t * tensor_2                     = nullptr;

    Qnn_QuantizeParams_t quantize_param         = QNN_QUANTIZE_PARAMS_INIT;
    Qnn_OpConfig_t qnn_opconfig                 = QNN_OPCONFIG_INIT;
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
    ctx                                         = (struct ggml_backend_qnn_context *)g_qnn_backend->context;
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
    QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s)  ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src0->name, src0->rank,
          src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
          src0->nb[0], src0->nb[1], src0->nb[2]);
    QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src1->name, src1->rank,
          src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
          src1->nb[0], src1->nb[1], src1->nb[2]);
    QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          dst->name, dst->rank,
          dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0],
          dst->nb[1], dst->nb[2]);
    QNN_LOG_DEBUG("%d, %d, %d, %d", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    QNN_LOG_DEBUG("tensor0 name %s", QNN_TENSOR_GET_NAME(tensor_0));
    QNN_LOG_DEBUG("tensor1 name %s", QNN_TENSOR_GET_NAME(tensor_1));
    QNN_LOG_DEBUG("tensor2 name %s", QNN_TENSOR_GET_NAME(tensor_2));

    QNN_VER_PTR(*tensor_0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_2)->type = QNN_TENSOR_TYPE_APP_READ;

    uint32_t dimensions_input_0[]   = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                       (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
    uint32_t dimensions_input_1[]   = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                       (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
    uint32_t dimensions_output[]    = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                       (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};

    src0_qnn_type                   = qnn_datatype_from_ggml_datatype(src0->type);
    src1_qnn_type                   = qnn_datatype_from_ggml_datatype(src1->type);
    dst_qnn_type                    = qnn_datatype_from_ggml_datatype(dst->type);

    std::string map_entry           = std::string(ggml_op_name(ggmlop));
    if (instance->_qnn_graph_map.find(map_entry) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
    }

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

        Qnn_Tensor_t tensor_inputs[] = {
                *tensor_0,
                *tensor_1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *tensor_2
        };
        Qnn_OpConfig_t opconfig = {
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
        error = qnn_raw_interface.graphAddNode(graph_handle, opconfig);
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
    n_end_time = ggml_time_us();
    n_durtion = (n_end_time - n_begin_time) / 1000;
    QNN_LOG_DEBUG("duration of ggml_qnn_mul_mat : %lld milliseconds\n", n_durtion);
    QNN_LOG_DEBUG("call %s done\n", __func__);
}


//common function for GGML OPs using QNN API
static void ggml_qnn_hanlde_op(const enum ggml_op ggmlop, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    Qnn_ErrorHandle_t error                     = QNN_SUCCESS;
    bool graph_initialized                      = false;
    int64_t n_begin_time                        = 0LL;
    int64_t n_end_time                          = 0LL;
    int64_t n_durtion                           = 0LL;

    qnn_instance * instance                     = nullptr;
    struct ggml_backend_qnn_context * ctx       = nullptr;

    std::string qnn_graph_name                  = "ggml_qnn_graph";
    std::string qnn_opconfig_name               = "ggml_qnn_opconfig";
    const char * qnn_op_name                    = nullptr;
    Qnn_GraphHandle_t graph_handle              = nullptr;
    Qnn_Tensor_t * tensor_0                     = nullptr;
    Qnn_Tensor_t * tensor_1                     = nullptr;
    Qnn_Tensor_t * tensor_2                     = nullptr;

    Qnn_QuantizeParams_t quantize_param         = QNN_QUANTIZE_PARAMS_INIT;
    Qnn_OpConfig_t qnn_opconfig                 = QNN_OPCONFIG_INIT;
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
    ctx                                         = (struct ggml_backend_qnn_context *)g_qnn_backend->context;
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
    QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s)  ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src0->name, src0->rank,
          src0->type, ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2],
          src0->nb[0], src0->nb[1], src0->nb[2]);
    QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          src1->name, src1->rank,
          src1->type, ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2],
          src1->nb[0], src1->nb[1], src1->nb[2]);
    QNN_LOG_INFO("%15s: rank = %d, type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
          dst->name, dst->rank,
          dst->type, ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0],
          dst->nb[1], dst->nb[2]);
    QNN_LOG_DEBUG("%d, %d, %d, %d", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    QNN_LOG_DEBUG("tensor0 name %s", QNN_TENSOR_GET_NAME(tensor_0));
    QNN_LOG_DEBUG("tensor1 name %s", QNN_TENSOR_GET_NAME(tensor_1));
    QNN_LOG_DEBUG("tensor2 name %s", QNN_TENSOR_GET_NAME(tensor_2));

    QNN_VER_PTR(*tensor_0)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_1)->type = QNN_TENSOR_TYPE_APP_WRITE;
    QNN_VER_PTR(*tensor_2)->type = QNN_TENSOR_TYPE_APP_READ;

    uint32_t dimensions_input_0[]   = {(uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                       (uint32_t) src0->ne[2], (uint32_t) src0->ne[3]};
    uint32_t dimensions_input_1[]   = {(uint32_t) src1->ne[0], (uint32_t) src1->ne[1],
                                       (uint32_t) src1->ne[2], (uint32_t) src1->ne[3]};
    uint32_t dimensions_output[]    = {(uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
                                       (uint32_t) dst->ne[2], (uint32_t) dst->ne[3]};

    std::string map_entry           = std::string(ggml_op_name(ggmlop));
    if (instance->_qnn_graph_map.find(map_entry) != instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        auto & graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
    }

    if (!graph_initialized) {
        qnn_graph_name = qnn_graph_name + "_" + ggml_op_name(ggmlop) + std::to_string(ctx->threads) + src0->name + "_" + src1->name;
        qnn_opconfig_name = qnn_opconfig_name + "_" + ggml_op_name(ggmlop) + std::to_string(ctx->threads) + src0->name + "_" + src1->name;
        QNN_LOG_DEBUG("qnn graph name %s", qnn_graph_name.c_str());
        QNN_LOG_DEBUG("qnn opconfig name %s", qnn_opconfig_name.c_str());
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

        Qnn_Tensor_t tensor_inputs[] = {
                *tensor_0,
                *tensor_1
        };
        Qnn_Tensor_t tensor_outputs[] = {
                *tensor_2
        };
        Qnn_OpConfig_t opconfig = {
                (Qnn_OpConfigVersion_t) 1, .v1 = {
                        qnn_opconfig_name.c_str(),
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
        error = qnn_raw_interface.graphAddNode(graph_handle, opconfig);
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
    n_end_time = ggml_time_us();
    n_durtion = (n_end_time - n_begin_time) / 1000;
    QNN_LOG_DEBUG("duration of ggml_qnn_%s : %lld milliseconds\n", ggml_op_name(ggmlop), n_durtion);
    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_repeat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_get_rows(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_acc(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}





static void ggml_qnn_div(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_gelu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_silu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_gelu_quick(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_tanh(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_relu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_hardsigmoid(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_hardswish(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_leaky_relu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_sqr(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_group_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_concat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_upscale(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_pad(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_rms_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_cpy(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_dup(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_qnn_cpy(src0, dst, nullptr);
    (void) src1;
}


static void ggml_qnn_mul_mat_id(const ggml_tensor * src0,
                                const ggml_tensor * src1,
                                ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_scale(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_clamp(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_diag_mask_inf(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_soft_max(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_rope(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);

}


static void ggml_qnn_alibi(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_pool2d(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_im2col(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_sum_rows(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_argsort(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


static void ggml_qnn_nop(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    (void) src0;
    (void) src1;
    (void) dst;
    QNN_LOG_DEBUG("call %s\n", __func__);

    QNN_LOG_DEBUG("call %s done\n", __func__);
}


bool ggml_qnn_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    ggml_qnn_func_t func                = nullptr;
    ggml_qnn_func_common_t  func_common = nullptr;

    bool supported_op                   = false;

    bool use_hwaccel                    = false;

    //begin sanity check
    if (nullptr == g_qnn_backend) {
        QNN_LOG_ERROR("pls check why qnn subsystem not initialized");
        return false;
    }

    //this is special scenario for UT function qnn_ggml_op
    //borrow some advantages from PyTorch:the user or the upper layer codes could specify whether a GGML OP(such as add/mul/mulmat) is accelerated by a specify backend)
    //otherwise ggml-qnn.cpp don't known whether current caller is whisper.cpp or other scenario(for example, JNI function...)

    //in the all, use_hwaccel is different with supported_op
    //this feature is heavily depend on PR in upstream whisper.cpp https://github.com/ggerganov/whisper.cpp/pull/2073
    use_hwaccel = (tensor->src[0]->backend == GGML_BACKEND_TYPE_GPU);

    supported_op = ((tensor->op == GGML_OP_ADD) || (tensor->op == GGML_OP_MUL) || (tensor->op == GGML_OP_MUL_MAT));
    //supported_op = (tensor->op == GGML_OP_ADD); //works very good with whisper.cpp(asr result is correct)

    if ((!use_hwaccel) && (!supported_op)) {
        //TODO: should be removed because this is a workaround method during development stage
        ggml_compute_forward(params, tensor);
        return false;
    }

    if ((!use_hwaccel) && (!ggml_qnn_can_handle_op(tensor->src[0], tensor->src[1], tensor))) {
        //TODO: should be removed because this is a workaround method during development stage
        ggml_compute_forward(params, tensor);
        return false;
    }
    //end sanity check

    switch (tensor->op) {
        case GGML_OP_ADD:
            func = ggml_qnn_add;
            //func_common = ggml_qnn_hanlde_op;
            break;

        case GGML_OP_MUL:
            func_common = ggml_qnn_hanlde_op;
            break;

        case GGML_OP_MUL_MAT:
            func = ggml_qnn_mul_mat;
            //func_common = ggml_qnn_hanlde_op;
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
        case GGML_OP_ALIBI:
            func = ggml_qnn_alibi;
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


    //ok, real show time in Qualcomm's QNN internal
    if (nullptr != func)
        func(tensor->src[0], tensor->src[1], tensor);
    if (nullptr != func_common)
        func_common(tensor->op, tensor->src[0], tensor->src[1], tensor);

    return true;
}


struct ggml_backend_qnn_buffer_context {
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

        std::map<std::string, std::tuple<Qnn_GraphHandle_t, Qnn_Tensor_t *, Qnn_Tensor_t *, Qnn_Tensor_t *>>::iterator graph_it;
        struct ggml_backend_qnn_context * ctx = (struct ggml_backend_qnn_context *) g_qnn_backend->context;
        QNN_INTERFACE_VER_TYPE qnn_raw_interface = ctx->instance->get_qnn_raw_interface();
        for (graph_it = backend_ctx->instance->_qnn_graph_map.begin(); graph_it != backend_ctx->instance->_qnn_graph_map.end(); graph_it++) {
            auto & graph_item = graph_it->second;
            Qnn_GraphHandle_t & graph_handle = std::get<0>(graph_item);
            QNN_LOG_DEBUG("graph type:%s", graph_it->first.c_str());
        }
        backend_ctx->instance->_qnn_graph_map.clear();

        sub_buffers.clear();
        qnn_tensors.clear();
    }
    void * buffer       = nullptr;

    struct ggml_backend_qnn_context * backend_ctx = nullptr;

    size_t buffer_size  = 0;
    std::vector<void *> sub_buffers;
    std::vector<Qnn_Tensor_t *> qnn_tensors;
};

static const char * ggml_backend_qnn_buffer_get_name(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return "QNN";
}


GGML_CALL static bool ggml_backend_buffer_is_qnn(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_qnn_buffer_get_name;
}


static void ggml_backend_qnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *) buffer->context;
    delete ctx;
}


//TODO:not used
static void * ggml_backend_qnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *) buffer->context;

    return ctx->buffer;
}


static void ggml_backend_qnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *)buffer->context;

    /*
    if (tensor->view_src != nullptr && tensor->view_offs == 0) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        tensor->backend = tensor->view_src->backend;
        tensor->extra = tensor->view_src->extra;
        return;
    }
    */

    uint32_t dimensions[] = {(uint32_t) tensor->ne[0], (uint32_t) tensor->ne[1], (uint32_t) tensor->ne[2], (uint32_t) tensor->ne[3]};
    //TODO:only support FP32 & FP16
    Qnn_DataType_t  qnn_data_type = QNN_DATATYPE_FLOAT_32;
    Qnn_TensorType_t qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;


    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
    } else if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
    }
    Qnn_Tensor_t  qnn_tensor = {
            .version= QNN_TENSOR_VERSION_1,
            {.v1= {
                    .id=0,
                    .name= tensor->name,
                    .type= qnn_tensor_type,
                    .dataFormat= QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                    .dataType= qnn_data_type,
                    .quantizeParams= {QNN_DEFINITION_UNDEFINED,
                                      QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                      {.scaleOffsetEncoding= {.scale= 0.0000000000000000f, .offset= 0}}},
                    .rank= ggml_get_tensor_rank(tensor),
                    .dimensions=dimensions,
                    .memType= QNN_TENSORMEMTYPE_RAW,
                    {.clientBuf= {.data=nullptr,
                            .dataSize=0}}}}
    };
    Qnn_Tensor_t  * p_qnn_tensor = (Qnn_Tensor_t *)malloc(sizeof(Qnn_Tensor_t));
    if (nullptr == p_qnn_tensor) {
        QNN_LOG_WARN("init tensor failed");
        return;
    }
    Qnn_Tensor_t tensor_copy;
    error = deep_copy_qnn_tensors(qnn_tensor, *p_qnn_tensor);
    if (error != QNN_SUCCESS) {
        free(p_qnn_tensor);
        QNN_LOG_DEBUG("init tensor failed");
        return;
    }
    tensor->extra = p_qnn_tensor;
    ctx->qnn_tensors.push_back(p_qnn_tensor);

    if (ggml_is_quantized(tensor->type)) {
        //TODO
        QNN_LOG_DEBUG("is quantized");
    }
}


static void ggml_backend_qnn_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);

    memcpy((char *)tensor->data + offset, data, size);
}


static void ggml_backend_qnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *)tensor->data + offset, size);
}


static bool ggml_backend_qnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}


static void ggml_backend_qnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *) buffer->context;

    memset(ctx->buffer, value, ctx->buffer_size);
}



static void ggml_backend_qnn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_qnn_buffer_context * ctx = (ggml_backend_qnn_buffer_context *) buffer->context;
    for (auto * sub_buffer : ctx->sub_buffers) {
        free(sub_buffer);
    }
    ctx->sub_buffers.clear();
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


static const char * ggml_backend_qnn_buffer_type_name(ggml_backend_buffer_type_t buft) {
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


static ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_qnn_buffer_context * ctx = new ggml_backend_qnn_buffer_context;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    //TODO:use pre-allocated buffer in internal memory pool
    ctx->buffer = ggml_qnn_host_malloc(size_aligned);
    ctx->buffer_size = size_aligned;

    ctx->backend_ctx = &g_qnn_mgr[g_current_device];

    if (nullptr == ctx->buffer) {
        QNN_LOG_WARN("%s: failed to allocate %.2f MiB\n", __func__, size / (1 << 20));
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_qnn_buffer_interface, ctx, size);
}


static size_t ggml_backend_qnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}


//TODO: this value is an experimental value
static size_t ggml_backend_qnn_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);

    return (38 * 1024 * 1024);
}


static bool ggml_backend_qnn_buffer_type_supports_backend(ggml_backend_buffer_type_t buft,
                                                          ggml_backend_t backend) {
    GGML_UNUSED(buft);

    return ggml_backend_is_qnn(backend) || ggml_backend_is_cpu(backend);
}


// attention here because Qualcomm's QNN SDK is a highly well-designed SDK
//
// refer to https://developer.qualcomm.com/sites/default/files/attachments/qnn_software_stack.png
//          https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html
static bool ggml_backend_qnn_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static ggml_backend_buffer_type_i ggml_backend_qnn_buffer_type_interface = {
        /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
        /* .alloc_buffer     = */ ggml_backend_qnn_buffer_type_alloc_buffer,
        /* .get_alignment    = */ ggml_backend_qnn_buffer_type_get_alignment,
        /* .get_max_size     = */ ggml_backend_qnn_buffer_type_get_max_size,
        /* .get_alloc_size   = */ nullptr,
        /* .supports_backend = */ ggml_backend_qnn_buffer_type_supports_backend,
        /* .is_host          = */ ggml_backend_qnn_buffer_is_host
};


static const char * ggml_backend_qnn_name(ggml_backend_t backend) {
    return "QNN";
}


static void ggml_backend_qnn_free(ggml_backend_t backend) {
    QNN_LOG_INFO("enter %s", __func__ );
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;
    QNN_LOG_DEBUG("idx %d, name:%s", ctx->device, g_qnn_mgr[ctx->device].name);

    qnn_instance * instance = (qnn_instance*)g_qnn_mgr[ctx->device].instance;
    if (instance != nullptr) {
        instance->qnn_finalize();
        delete instance;
        g_qnn_mgr[ctx->device].instance = nullptr;
    }

    qnn_buf_t * buffer_pool = (qnn_buf_t*)g_qnn_mgr[ctx->device].buffer_pool;
    if (buffer_pool != nullptr) {
        buffer_pool->destroy(buffer_pool);
        g_qnn_mgr[ctx->device].buffer_pool = nullptr;
    }

    if (g_qnn_mgr[ctx->device].backend      != nullptr) {
        delete backend;
        g_qnn_backend = nullptr;
        g_qnn_mgr[ctx->device].backend = nullptr;
    }
    QNN_LOG_INFO("leave %s", __func__ );
}


static ggml_backend_buffer_type_t ggml_backend_qnn_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_qnn_context * ctx = (ggml_backend_qnn_context *) backend->context;

    return ggml_backend_qnn_buffer_type(ctx->device);
}


#if 0
static bool ggml_backend_qnn_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    GGML_UNUSED(backend);

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID: {
            struct ggml_tensor *a;
            struct ggml_tensor *b;
            if (op->op == GGML_OP_MUL_MAT) {
                a = op->src[0];
                b = op->src[1];
            } else {
                a = op->src[2];
                b = op->src[1];
            }
            if (a->ne[3] != b->ne[3]) {
                return false;
            }
            ggml_type a_type = a->type;
            if (a_type == GGML_TYPE_IQ4_NL || a_type == GGML_TYPE_IQ2_S ||
                a_type == GGML_TYPE_IQ4_XS) {
                return false;
            }
            return true;
        }
            break;
        case GGML_OP_GET_ROWS: {
            switch (op->src[0]->type) {
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return true;
                default:
                    return false;
            }
        }
            break;
        case GGML_OP_CPY: {
            ggml_type src0_type = op->src[0]->type;
            ggml_type src1_type = op->src[1]->type;
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                return true;
            }
            if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                return true;
            }
            if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                return true;
            }
            if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        }
            break;
        case GGML_OP_CONCAT: {
            ggml_type src0_type = op->src[0]->type;
            return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
        }
            break;
        case GGML_OP_DUP:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_REPEAT:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ALIBI:
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_LEAKY_RELU:
            return true;
        default:
            return false;
    }
}
# else
static bool ggml_backend_qnn_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    GGML_UNUSED(backend);

    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return true;
        default:
            return false;
    }
}
#endif


static ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    enum ggml_status result         = GGML_STATUS_SUCCESS;
    int node_n                      = -1;
    int task_phase                  = GGML_TASK_TYPE_FINALIZE;
    ggml_backend_qnn_context * ctx  = (ggml_backend_qnn_context *) backend->context;

    struct ggml_cplan plan          = ggml_graph_plan(cgraph, 1);

    buf_element_t * qnn_buf = nullptr;

    if (plan.work_size > 0) {
        //plan.work_data = static_cast<uint8_t *>(malloc(plan.work_size));
        plan.work_data = static_cast<uint8_t *>(ctx->buffer_pool->buffer_pool_base);
        if (plan.work_data == nullptr) {
            QNN_LOG_ERROR("malloc failed");
            return GGML_STATUS_FAILED;
        }
    }
    struct ggml_cplan * cplan = &plan;
    GGML_ASSERT(cplan->n_threads > 0);
    if (cplan->work_size > 0) {
        GGML_ASSERT(cplan->work_data);
    }

    while (true) {
        if (cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            result = GGML_STATUS_ABORTED;
            break;
        }
        struct ggml_compute_params params = {
                /*.type  =*/ GGML_TASK_TYPE_FINALIZE,
                /*.ith   =*/ 0,
                /*.nth   =*/ 0,
                /*.wsize =*/ cplan->work_size,
                /*.wdata =*/ cplan->work_data,
        };

        if (node_n != -1) {
            /* FINALIZE */
            struct ggml_tensor * node = cgraph->nodes[node_n];
            if (GGML_OP_HAS_FINALIZE[node->op]) {
                params.nth = 1;
                ggml_qnn_compute_forward(&params, node);
            }
        }

        while (++node_n < cgraph->n_nodes) {
            struct ggml_tensor * node = cgraph->nodes[node_n];
            params.nth = 1;
            if (GGML_OP_HAS_INIT[node->op]) {
                params.type = GGML_TASK_TYPE_INIT;
                ggml_qnn_compute_forward(&params, node);
            }
            params.type = GGML_TASK_TYPE_COMPUTE;
            ggml_qnn_compute_forward(&params, node);
            if (GGML_OP_HAS_FINALIZE[node->op]) {
                params.type = GGML_TASK_TYPE_FINALIZE;
                ggml_qnn_compute_forward(&params, node);
            }
            if (cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
                result = GGML_STATUS_ABORTED;
                break;
            }
        }
        task_phase = GGML_TASK_TYPE_INIT;
        if (node_n >= cgraph->n_nodes) {
            //QNN_LOG_INFO("node_n %d", node_n);
            //QNN_LOG_INFO("cgraph->n_nodes %d", cgraph->n_nodes);
            break;
        }
    }

    //free(plan.work_data);

    return result;
}


struct ggml_compute_state_shared {
    const struct ggml_cgraph * cgraph;
    const struct ggml_cplan  * cplan;

    int64_t perf_node_start_cycles;
    int64_t perf_node_start_time_us;

    const int n_threads;

    // synchronization primitives
    atomic_int n_active;  // num active threads
    atomic_int node_n;    // active graph node
    atomic_int node_task; // active graph node task phase

    ggml_abort_callback abort_callback; // abort ggml_graph_compute when true
    void * abort_callback_data;
};

struct ggml_compute_state {
    pthread_t thrd;
    int ith;
    struct ggml_compute_state_shared * shared;
    enum ggml_status ec;
};


#ifdef GGML_PERF
#define ggml_perf_time_ms()       ggml_time_ms()
#define ggml_perf_time_us()       ggml_time_us()
#define ggml_perf_cycles()        ggml_cycles()
#define ggml_perf_cycles_per_ms() ggml_cycles_per_ms()
#else
#define ggml_perf_time_ms()       0
#define ggml_perf_time_us()       0
#define ggml_perf_cycles()        0
#define ggml_perf_cycles_per_ms() 0
#endif
#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


static void ggml_graph_compute_perf_stats_node(struct ggml_tensor * node, const struct ggml_compute_state_shared * st) {
    int64_t cycles_cur  = ggml_perf_cycles()  - st->perf_node_start_cycles;
    int64_t time_us_cur = ggml_perf_time_us() - st->perf_node_start_time_us;

    node->perf_runs++;
    node->perf_cycles  += cycles_cur;
    node->perf_time_us += time_us_cur;
}


static void ggml_graph_compute_thread_sync_node(int * node_n, struct ggml_compute_state * state, const bool do_yield) {
    // wait for other threads to finish
    const int last_node_n = * node_n;

    while (true) {
        if (do_yield) {
            sched_yield();
        }

        * node_n = atomic_load(&state->shared->node_n);
        if (* node_n != last_node_n) break;
    }
}


static void ggml_graph_compute_thread_sync_task(int * task_phase, struct ggml_compute_state * state, const bool do_yield) {
    // wait for other threads to finish
    const int last_task_phase = * task_phase;

    while (true) {
        if (do_yield) {
            sched_yield();
        }

        * task_phase = atomic_load(&state->shared->node_task);
        if (* task_phase != last_task_phase) break;
    }
}


static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads, int n_cur_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ACC: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU: {
            n_tasks = 1;
        }
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID: {
                    n_tasks = 1;
                }
                    break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU: {
                    n_tasks = n_threads;
                }
                    break;
                default:
                    GGML_ASSERT(false);
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_MUL_MAT: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_MUL_MAT_ID: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_OUT_PROD: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_GET_ROWS: {
            n_tasks = MIN(n_cur_threads, ggml_nelements(node->src[1]));
        }
            break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_CONT:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG: {
            n_tasks = 1;
        }
            break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_ALIBI: {
            n_tasks = 1;
        }
            break;
        case GGML_OP_CLAMP: {
            n_tasks = 1;
        }
            break;
        case GGML_OP_SOFT_MAX: {
            n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
        }
            break;
        case GGML_OP_CONV_TRANSPOSE_1D: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_IM2COL: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_CONV_TRANSPOSE_2D: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D: {
            n_tasks = 1;
        }
            break;
        case GGML_OP_UPSCALE: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_PAD: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_ARANGE: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_TIMESTEP_EMBEDDING: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_ARGSORT: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_FLASH_ATTN: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_FLASH_FF: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_FLASH_ATTN_BACK: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
        case GGML_OP_MAP_UNARY:
        case GGML_OP_MAP_BINARY:
        case GGML_OP_MAP_CUSTOM1_F32:
        case GGML_OP_MAP_CUSTOM2_F32:
        case GGML_OP_MAP_CUSTOM3_F32: {
            n_tasks = 1;
        }
            break;
        case GGML_OP_MAP_CUSTOM1: {
            QNN_LOG_ERROR("not support");
        }
            break;
        case GGML_OP_MAP_CUSTOM2: {
            QNN_LOG_ERROR("not support");
        }
            break;
        case GGML_OP_MAP_CUSTOM3: {
            QNN_LOG_ERROR("not support");
        }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
            n_tasks = n_threads;
        }
            break;
        case GGML_OP_NONE: {
            n_tasks = 1;
        }
            break;
        case GGML_OP_COUNT: {
            GGML_ASSERT(false);
        }
            break;
        default: {
            QNN_LOG_WARN("%s: op not implemented: ", __func__);
            if (node->op < GGML_OP_COUNT) {
                QNN_LOG_DEBUG("%s\n", ggml_op_name(node->op));
            } else {
                QNN_LOG_DEBUG("%d\n", node->op);
            }
            GGML_ASSERT(false);
        }
            break;
    }

    assert(n_tasks > 0);

    return n_tasks;
}


static void * ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;

    const struct ggml_cgraph * cgraph = state->shared->cgraph;
    const struct ggml_cplan  * cplan  = state->shared->cplan;

    const int   n_threads   = state->shared->n_threads;

    int node_n     = -1;
    int task_phase = GGML_TASK_TYPE_FINALIZE;

    while (true) {
        if (cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->shared->node_n += 1;
            state->ec = GGML_STATUS_ABORTED;
            return 0;
        }

        if (atomic_fetch_sub(&state->shared->n_active, 1) == 1) {
            // all other threads are finished and spinning
            // do finalize and init here so we don't have synchronize again
            struct ggml_compute_params params = {
                    /*.type  =*/ GGML_TASK_TYPE_FINALIZE,
                    /*.ith   =*/ 0,
                    /*.nth   =*/ 0,
                    /*.wsize =*/ cplan->work_size,
                    /*.wdata =*/ cplan->work_data,
            };

            if (node_n != -1) {
                /* FINALIZE */
                struct ggml_tensor * node = cgraph->nodes[node_n];
                if (GGML_OP_HAS_FINALIZE[node->op]) {
                    params.nth = ggml_get_n_tasks(node, n_threads, state->shared->n_threads);
                    ggml_qnn_compute_forward(&params, node);
                }
                ggml_graph_compute_perf_stats_node(node, state->shared);
            }

            // distribute new work or execute it direct if 1T
            while (++node_n < cgraph->n_nodes) {
                //QNN_LOG_INFO("%s: %d/%d\n", __func__, node_n, cgraph->n_nodes);
                struct ggml_tensor * node = cgraph->nodes[node_n];
                const int n_tasks = ggml_get_n_tasks(node, n_threads, state->shared->n_threads);

                state->shared->perf_node_start_cycles  = ggml_perf_cycles();
                state->shared->perf_node_start_time_us = ggml_perf_time_us();

                params.nth = n_tasks;

                if (n_tasks == 1) {
                    /* INIT */
                    if (GGML_OP_HAS_INIT[node->op]) {
                        params.type = GGML_TASK_TYPE_INIT;
                        ggml_qnn_compute_forward(&params, node);
                    }

                    // TODO: maybe push node_n to the atomic but if other threads see n_tasks is 1,
                    // they do something more efficient than spinning (?)
                    params.type = GGML_TASK_TYPE_COMPUTE;
                    ggml_qnn_compute_forward(&params, node);

                    if (GGML_OP_HAS_FINALIZE[node->op]) {
                        params.type = GGML_TASK_TYPE_FINALIZE;
                        ggml_qnn_compute_forward(&params, node);
                    }

                    ggml_graph_compute_perf_stats_node(node, state->shared);
                } else {
                    break;
                }

                if (cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
                    break;
                }
            }

            task_phase = GGML_TASK_TYPE_INIT;
            atomic_store(&state->shared->n_active,  n_threads);
            atomic_store(&state->shared->node_n,    node_n);
            atomic_store(&state->shared->node_task, task_phase);
        } else {
            ggml_graph_compute_thread_sync_node(&node_n,     state, false);
            ggml_graph_compute_thread_sync_task(&task_phase, state, false);
        }

        // check if we should stop
        if (node_n >= cgraph->n_nodes) break;

        /* INIT & COMPUTE */
        struct ggml_tensor * node = cgraph->nodes[node_n];
        const int n_tasks = ggml_get_n_tasks(node, n_threads, state->shared->n_threads);

        struct ggml_compute_params params = {
                /*.type  =*/ GGML_TASK_TYPE_INIT,
                /*.ith   =*/ state->ith,
                /*.nth   =*/ n_tasks,
                /*.wsize =*/ cplan->work_size,
                /*.wdata =*/ cplan->work_data,
        };

        if (state->ith < n_tasks) {
            if (GGML_OP_HAS_INIT[node->op]) {
                ggml_qnn_compute_forward(&params, node);
            }
        }

        if (atomic_fetch_sub(&state->shared->n_active, 1) == 1) {
            task_phase = GGML_TASK_TYPE_COMPUTE;
            atomic_store(&state->shared->n_active,  n_threads);
            atomic_store(&state->shared->node_task, task_phase);
        }
        else {
            const bool do_yield = node_n < 0 || cgraph->nodes[node_n]->op == GGML_OP_MUL_MAT;
            ggml_graph_compute_thread_sync_task(&task_phase, state, do_yield);
        }

        if (state->ith < n_tasks) {
            params.type = GGML_TASK_TYPE_COMPUTE;
            ggml_qnn_compute_forward(&params, node);
        }

        if (atomic_fetch_sub(&state->shared->n_active, 1) == 1) {
            task_phase = GGML_TASK_TYPE_FINALIZE;
            atomic_store(&state->shared->n_active,  n_threads);
            atomic_store(&state->shared->node_task, task_phase);
        }
        else {
            ggml_graph_compute_thread_sync_task(&task_phase, state, false);
        }
    }

    return 0;
}


static ggml_status ggml_backend_qnn_graph_compute_multithread(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_qnn_context * ctx  = (ggml_backend_qnn_context *) backend->context;

    int num_threads = ctx->threads;

    if (QNN_GPU == ctx->device || QNN_HTP == ctx->device) {
        //TODO:multithreading not supported using QNN GPU/HTP(aka DSP) backend
        num_threads = 1;
    }
    struct ggml_cplan plan          = ggml_graph_plan(cgraph, num_threads);


    if (plan.work_size > 0) {
        //QNN_LOG_INFO("work size %d(%d MB)", plan.work_size, plan.work_size / (1 << 20));
        plan.work_data = static_cast<uint8_t *>(malloc(plan.work_size));
        if (plan.work_data == nullptr) {
            QNN_LOG_ERROR("malloc failed");
            return GGML_STATUS_FAILED;
        }
    }

    struct ggml_cplan * cplan = &plan;
    GGML_ASSERT(cplan->n_threads > 0);
    if (cplan->work_size > 0) {
        GGML_ASSERT(cplan->work_data);
    }

    //QNN_LOG_DEBUG("cgraph %p, cplan %p, work size %d, work data %p", cgraph, cplan, cplan->work_size, cplan->work_data);
    const int n_threads = cplan->n_threads;

    struct ggml_compute_state_shared state_shared = {
            /*.cgraph                  =*/ cgraph,
            /*.cgraph_plan             =*/ cplan,
            /*.perf_node_start_cycles  =*/ 0,
            /*.perf_node_start_time_us =*/ 0,
            /*.n_threads               =*/ n_threads,
            /*.n_active                =*/ n_threads,
            /*.node_n                  =*/ -1,
            /*.node_task               =*/ GGML_TASK_TYPE_FINALIZE,
            /*.abort_callback          =*/ nullptr,
            /*.abort_callback_data     =*/ nullptr,
    };
    struct ggml_compute_state * workers = (struct ggml_compute_state*)alloca(sizeof(struct ggml_compute_state) * n_threads);
    if (nullptr == workers) {
        QNN_LOG_ERROR("malloc failed");
        if (plan.work_data != nullptr) {
            free(plan.work_data);
        }
        return GGML_STATUS_FAILED;
    }

    // create thread pool
    if (n_threads > 1) {
        for (int j = 1; j < n_threads; ++j) {
            workers[j] = (struct ggml_compute_state) {
                    .thrd   = 0,
                    .ith = j,
                    .shared = &state_shared,
                    .ec = GGML_STATUS_SUCCESS,
            };

            const int rc = pthread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
            GGML_ASSERT(rc == 0);
        }
    }

    workers[0].ith = 0;
    workers[0].shared = &state_shared;
    workers[0].ec = GGML_STATUS_SUCCESS;

    // this is a work thread too
    ggml_graph_compute_thread(&workers[0]);
    enum ggml_status compute_status = workers[0].ec;

    // join or kill thread pool
    if (n_threads > 1) {
        for (int j = 1; j < n_threads; j++) {
            const int rc = pthread_join(workers[j].thrd, NULL);
            GGML_ASSERT(rc == 0);
            if (workers[j].ec != GGML_STATUS_SUCCESS)
                compute_status = workers[j].ec;
        }
    }

    if (plan.work_data != nullptr) {
        free(plan.work_data);
    }

    return compute_status;
}


static bool ggml_backend_qnn_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    GGML_UNUSED(backend);

    const int min_batch_size = 32;

    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;
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
        /* .graph_compute           = */ ggml_backend_qnn_graph_compute_multithread,
        /* .supports_op             = */ ggml_backend_qnn_supports_op,
        /* .offload_op              = */ nullptr,
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


void ggml_backend_qnn_get_device_description(int device, char * description, size_t description_size) {
    if (nullptr == description || 0 == description_size) {
        QNN_LOG_WARN("invalid param");
        return;
    }

    if (device >= GGML_QNN_MAX_DEVICES) {
        QNN_LOG_WARN("invalid param");
        return;
    }

    snprintf(description, description_size, "%s", g_qnn_mgr[device].name);
    QNN_LOG_DEBUG("description:%s", description);
}


ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(size_t device_index) {
    if (device_index >= GGML_QNN_MAX_DEVICES) {
        QNN_LOG_DEBUG("ggml_backend_qnn_buffer_type error: device_index:%d is out of range [0, %d]\n",
               device_index, GGML_QNN_MAX_DEVICES - 1);
        return nullptr;
    }

    static struct ggml_backend_buffer_type ggml_backend_buffer_type_qnn = {
            /* .iface   = */ {
                /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
                /* .alloc_buffer     = */ ggml_backend_qnn_buffer_type_alloc_buffer,
                /* .get_alignment    = */ ggml_backend_qnn_buffer_type_get_alignment,
                /* .get_max_size     = */ ggml_backend_qnn_buffer_type_get_max_size,
                /* .get_alloc_size   = */ nullptr,// defaults to ggml_nbytes
                /* .supports_backend = */ ggml_backend_qnn_buffer_type_supports_backend,
                /* .is_host          = */ ggml_backend_qnn_buffer_is_host
            },
            /* .context = */ nullptr,
    };

    return &ggml_backend_buffer_type_qnn;
}


/**
 *
 * @param device            0: QNN_CPU 1: QNN_GPU 2: QNN_HTP(aka DSP)
 * @param qnn_lib_path      qnn library path, such as "/data/data/com.ggml.llamacpp/" on Android which can got by JNI from Java layer
 * @return
 */
ggml_backend_t ggml_backend_qnn_init(size_t device, const char * qnn_lib_path) {
    int result = 0;

    if (nullptr == qnn_lib_path)
        return nullptr;

    QNN_LOG_DEBUG("device %d", device);
    QNN_LOG_DEBUG("qnn_lib_path %s", qnn_lib_path);
    if (device >= GGML_QNN_MAX_DEVICES) {
        QNN_LOG_ERROR("invalid device %d", device);
        return nullptr;
    }

    if (nullptr != g_qnn_mgr[device].backend) {
        QNN_LOG_ERROR("qnn backend %d(%s) already loaded, it should not happened, pls check why?", device, get_qnn_backend_name(device));
        if (device == g_current_device) {
            g_qnn_backend = g_qnn_mgr[device].backend;
            QNN_LOG_INFO("re-use cached backend %d(%s)", device, get_qnn_backend_name(device));
            return g_qnn_mgr[device].backend;
        } else {
            QNN_LOG_INFO("delete previous backend %d(%s)", device, get_qnn_backend_name(device));
            ggml_backend_qnn_free(g_qnn_backend);
        }
    }

    static bool is_first_call = true;
    if (is_first_call) {
        ggml_setup_op_has_task_pass();
        is_first_call = false;
    }

    if (QNN_HTP == device) {
        std::string path = qnn_lib_path;
        if (0 == setenv("LD_LIBRARY_PATH",
                        (path +
                         ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images").c_str(),
                        1)) {
            QNN_LOG_INFO("QNN DSP backend setenv successfully");
        } else {
            QNN_LOG_ERROR("QNN DSP backend setenv failure");
        }
        if (0 == setenv("ADSP_LIBRARY_PATH",
                        (path +
                         ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp").c_str(),
                        1)) {
            QNN_LOG_INFO("QNN DSP backend setenv successfully");
        } else {
            QNN_LOG_ERROR("QNN DSP backend setenv failure");
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

    std::string device_name = GGML_QNN_NAME + std::string("_") + std::to_string(device) + std::string("_") + get_qnn_backend_name(device);
    QNN_LOG_INFO("qnn device name %s", device_name.c_str());
    instance->init_qnn_graph(device_name.c_str(), false);
    g_qnn_mgr[device].instance                  = instance;
    g_qnn_mgr[device].raw_interface             = instance->get_qnn_raw_interface();
    g_qnn_mgr[device].raw_system_interface      = instance->get_qnn_raw_system_interface();
    //TODO:refine internal buffer management
    g_qnn_mgr[device].buffer_pool               = qnn_buf_new(get_qnn_backend_name(device), GGML_QNN_MAX_BUFFERS, (1 << 20));
    GGML_ASSERT(g_qnn_mgr[device].buffer_pool != nullptr);

    ggml_backend_t qnn_backend = new ggml_backend{
                /* .guid      = */ ggml_backend_qnn_guid(),
                /* .iface     = */ ggml_backend_qnn_interface,
                /* .context   = */ &g_qnn_mgr[device]
    };
    g_qnn_mgr[device].backend   = qnn_backend;
    g_qnn_backend = g_qnn_mgr[device].backend;
    g_current_device = device;

    return qnn_backend;
}


extern "C" int ggml_backend_qnn_reg_devices();


int ggml_backend_qnn_reg_devices() {
    for (size_t idx = 0; idx < GGML_QNN_MAX_DEVICES; idx++) {
        int id = g_qnn_mgr[idx].device;
        char name[GGML_MAX_NAME];
        ggml_backend_qnn_get_device_description(idx, name, GGML_MAX_NAME);
        ggml_backend_register(name, ggml_backend_qnn_reg_init, ggml_backend_qnn_buffer_type(idx),
                              (void *) (intptr_t)idx);
    }

    return GGML_QNN_MAX_DEVICES;
}
