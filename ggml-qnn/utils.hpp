#pragma once

#include "QnnTypes.h"

#include "ggml.h"

#include "qnn-types.hpp"

namespace qnn {

    // TODO: mapping more ggml data type to QNN data type
    // ref:explanation of k-quants, https://github.com/ggerganov/llama.cpp/pull/1684
    Qnn_DataType_t datatype_from_ggml_datatype(enum ggml_type ggmltype) {
        switch (ggmltype) {
        case GGML_TYPE_F16:
            return QNN_DATATYPE_FLOAT_16;
        case GGML_TYPE_F32:
            return QNN_DATATYPE_FLOAT_32;
        case GGML_TYPE_I8:
            return QNN_DATATYPE_INT_8;
        case GGML_TYPE_Q8_0:
            return QNN_DATATYPE_SFIXED_POINT_8;
        case GGML_TYPE_Q4_0:
            return QNN_DATATYPE_SFIXED_POINT_4;
        default:
            break;
        }
        return QNN_DATATYPE_UNDEFINED;
    }


    uint32_t get_ggml_tensor_rank(const ggml_tensor* tensor) {
        uint32_t rank = 0;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if ((0 != tensor->ne[i]) && (1 != tensor->ne[i])) {
                rank++;
            }
        }
        return rank;
    }


    const char* get_backend_name(int n_backend_type) {
        switch (n_backend_type) {
        case QNN_BACKEND_CPU:
            return "QNN-CPU";
        case QNN_BACKEND_GPU:
            return "QNN-GPU";
        case QNN_BACKEND_NPU:
            return "QNN-NPU";
        case QNN_BACKEND_GGML:
            return "ggml"; //"fake" QNN backend, used for compare performance between QNN backend and original GGML
        default:
            return "unknown";
        }
    }

    const char* get_chipset_desc(uint32_t chipset_id) {
        switch (chipset_id) {
        case SM8450:
            return "SM8450";
        case SM8475:
            return "SM8475";
        case SM8550:
            return "SM8550";
        case SM8650:
            return "SM8650";
        default:
            return "unknown";
        }
    }

    const char* get_htparch_desc(size_t htp_arch) {
        switch (htp_arch) {
        case V68:
            return "QCOM_HTP_V68";
        case V69:
            return "QCOM_HTP_V69";
        case V73:
            return "QCOM_HTP_V73";
        case V75:
            return "QCOM_HTP_V75";
        default:
            return "unknown";
        }
    }

    template <typename Fn> Fn load_qnn_functionpointers(void* handle, const char* function_name) {
        return reinterpret_cast<Fn>(dlsym(handle, function_name));
    }

    intptr_t align_to(size_t alignment, intptr_t offset) {
        return offset % alignment == 0
            ? offset
            : offset + (static_cast<intptr_t>(alignment) -
                offset % static_cast<intptr_t>(alignment));
    }

    uint32_t get_ggml_tensor_data_size(const ggml_tensor* tensor) {
        /*
        size_t data_size = ggml_row_size(tensor->type, tensor->ne[0]);
        size_t n_dims = qnn_get_ggml_tensor_rank(tensor);
        for (int i = 1; i < n_dims; i++) {
            data_size *= tensor->ne[i];
        }

        return data_size;
        */
        return ggml_nbytes(tensor);
    }


    // =================================================================================================
    //
    //  QNN backend internal helper functions
    //
    // =================================================================================================
    // TODO: only support GGML_OP_ADD/GGML_OP_MUL/GGML_OP_MUL_MAT
    const char* opname_from_ggmlop(enum ggml_op ggmlop) {
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

    inline int validate_tensor_version(Qnn_Tensor_t tensor) {
        if (tensor.version != QNN_TENSOR_VERSION_1) {
            QNN_LOG_WARN(
                "validate_tensor_version() tensor %s, got unsupported version %d\n",
                tensor.v1.name, tensor.version);
            return 1;
        }
        return 0;
    }

    inline uint32_t get_qnn_tensorid(const Qnn_Tensor_t& tensor) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            return tensor.v1.id;
        }

        return 0u;
    }

    inline const char* get_qnn_tensorname(const Qnn_Tensor_t& tensor) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            return tensor.v1.name;
        }
        return nullptr;
    }

    inline Qnn_TensorType_t get_qnn_tensortype(const Qnn_Tensor_t& tensor) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            return tensor.v1.type;
        }
        return QNN_TENSOR_TYPE_UNDEFINED;
    }

    inline Qnn_TensorDataFormat_t
        get_qnn_tensor_dataformat(const Qnn_Tensor_t& tensor) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            return tensor.v1.dataFormat;
        }
        return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    }

    inline Qnn_DataType_t
        get_qnn_tensor_datatype(const Qnn_Tensor_t& tensor) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            return tensor.v1.dataType;
        }
        return QNN_DATATYPE_UNDEFINED;
    }

    inline Qnn_QuantizeParams_t
        get_qnn_tensor_quantparams(const Qnn_Tensor_t& tensor) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            return tensor.v1.quantizeParams;
        }
        return QNN_QUANTIZE_PARAMS_INIT;
    }

    inline uint32_t get_qnn_tensor_rank(const Qnn_Tensor_t& tensor) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            return tensor.v1.rank;
        }
        return 0u;
    }

    inline uint32_t* get_qnn_tensor_dimensions(const Qnn_Tensor_t& tensor) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            return tensor.v1.dimensions;
        }
        return nullptr;
    }

    inline Qnn_TensorMemType_t get_qnn_tensor_memtype(const Qnn_Tensor_t& tensor) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            return tensor.v1.memType;
        }
        return QNN_TENSORMEMTYPE_UNDEFINED;
    }

    inline void set_qnn_tensor_id(Qnn_Tensor_t& tensor, uint32_t id) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.id = id;
        }
    }

    inline void set_qnn_tensor_name(Qnn_Tensor_t& tensor, const char* name) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.name = name;
        }
    }

    inline void set_qnn_tensor_type(Qnn_Tensor_t& tensor, Qnn_TensorType_t type) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.type = type;
        }
    }

    inline void set_qnn_tensor_dataformat(Qnn_Tensor_t& tensor, Qnn_TensorDataFormat_t format) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.dataFormat = format;
        }
    }

    inline void set_qnn_tensor_datatype(Qnn_Tensor_t& tensor, Qnn_DataType_t dataType) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.dataType = dataType;
        }
    }

    inline void set_qnn_tensor_quantparams(Qnn_Tensor_t& tensor, Qnn_QuantizeParams_t params) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.quantizeParams = params;
        }
    }

    inline void set_qnn_tensor_rank(Qnn_Tensor_t& tensor, uint32_t rank) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.rank = rank;
        }
    }

    inline void set_qnn_tensor_dimensions(Qnn_Tensor_t& tensor, uint32_t* dims) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.dimensions = dims;
        }
    }

    inline void set_qnn_tensor_memtype(Qnn_Tensor_t& tensor, Qnn_TensorMemType_t mem_type) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.memType = mem_type;
        }
    }

    inline void set_qnn_tensor_clientbuf(Qnn_Tensor_t& tensor, Qnn_ClientBuffer_t client_buf) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.clientBuf = client_buf;
        }
    }

    inline void set_qnn_tensor_memhandle(Qnn_Tensor_t& tensor, Qnn_MemHandle_t handle) {
        if (tensor.version == QNN_TENSOR_VERSION_1) {
            tensor.v1.memHandle = handle;
        }
    }
}


#define VALIDATE(value, status)                                                \
    do {                                                                       \
        status = value;                                                        \
        if (status != QNN_SUCCESS) {                                           \
            QNN_LOG_WARN("%s expected QNN_SUCCESS\n", #value);                 \
            return status;                                                     \
        }                                                                      \
    } while (0)

#define QNN_TENSOR_GET_ID(tensor)                   qnn::get_qnn_tensorid(tensor)
#define QNN_TENSOR_GET_NAME(tensor)                 qnn::get_qnn_tensorname(tensor)
#define QNN_TENSOR_GET_TYPE(tensor)                 qnn::get_qnn_tensortype(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor)          qnn::get_qnn_tensor_dataformat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor)            qnn::get_qnn_tensor_datatype(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor)         qnn::get_qnn_tensor_quantparams(tensor)
#define QNN_TENSOR_GET_RANK(tensor)                 qnn::get_qnn_tensor_rank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor)           qnn::get_qnn_tensor_dimensions(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor)             qnn::get_qnn_tensor_memtype(tensor)

#define QNN_TENSOR_SET_ID(tensor, value)            qnn::set_qnn_tensor_id(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value)          qnn::set_qnn_tensor_name(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value)          qnn::set_qnn_tensor_type(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value)   qnn::set_qnn_tensor_dataformat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value)     qnn::set_qnn_tensor_datatype(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value)  qnn::set_qnn_tensor_quantparams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value)          qnn::set_qnn_tensor_rank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value)    qnn::set_qnn_tensor_dimensions(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value)      qnn::set_qnn_tensor_memtype(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value)    qnn::set_qnn_tensor_clientbuf(tensor, value)
#define QNN_TENSOR_SET_MEM_HANDLE(tensor, value)    qnn::set_qnn_tensor_memhandle(tensor, value)
#define VALIDATE_TENSOR_VERSION(tensor, err)        VALIDATE(qnn::validate_tensor_version(tensor), err)
