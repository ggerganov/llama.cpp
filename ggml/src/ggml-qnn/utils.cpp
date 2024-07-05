
#include "utils.hpp"

#include "ggml-qnn.h"

#include "qnn-types.hpp"

namespace {

size_t memscpy(void *dst, size_t dst_size, const void *src, size_t copy_size) {
    if (!dst || !src || !dst_size || !copy_size) return 0;

    size_t min_size = dst_size < copy_size ? dst_size : copy_size;

    memcpy(dst, src, min_size);

    return min_size;
}

} // namespace

namespace qnn {

// TODO: mapping more ggml data type to QNN data type
// ref:explanation of k-quants, https://github.com/ggerganov/llama.cpp/pull/1684
Qnn_DataType_t device_datatype_from_ggml_datatype(ggml_type ggml_type) {
    switch (ggml_type) {
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

Qnn_TensorType_t device_tensortype_from_ggml_tensor(ggml_tensor *ggml_tensor) {
    Qnn_TensorType_t qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;

    if (ggml_tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
    } else if (ggml_tensor->flags & GGML_TENSOR_FLAG_OUTPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
    }

    return qnn_tensor_type;
}

uint32_t get_ggml_tensor_rank(const ggml_tensor *tensor) {
    uint32_t rank = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if ((0 != tensor->ne[i]) && (1 != tensor->ne[i])) {
            rank++;
        }
    }
    return rank;
}

const char *get_backend_name(int n_backend_type) {
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

const char *get_chipset_desc(uint32_t chipset_id) {
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

const char *get_htparch_desc(size_t htp_arch) {
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

intptr_t align_to(size_t alignment, intptr_t offset) {
    return offset % alignment == 0
               ? offset
               : offset + (static_cast<intptr_t>(alignment) - offset % static_cast<intptr_t>(alignment));
}

uint32_t get_ggml_tensor_data_size(const ggml_tensor *tensor) {
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
const char *opname_from_ggmlop(enum ggml_op ggmlop) {
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

void device_tensor_init(Qnn_Tensor_t &tensor, uint32_t rank, Qnn_TensorMemType_t mem_type, const char *tensor_name,
                        Qnn_TensorType_t qnn_tensor_type, Qnn_DataType_t qnn_data_type, uint32_t *dimensions) {
    tensor = QNN_TENSOR_INIT;
    tensor = { .version = QNN_TENSOR_VERSION_1,
               { .v1 = { .id = 0,
                         .name = tensor_name,
                         .type = qnn_tensor_type,
                         .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                         .dataType = qnn_data_type,
                         .quantizeParams = { QNN_DEFINITION_UNDEFINED,
                                             QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                             { .scaleOffsetEncoding = { .scale = 0.0000000000000000f, .offset = 0 } } },
                         .rank = rank,
                         .dimensions = dimensions,
                         .memType = mem_type,
                         { .clientBuf = {} } } } };
}

Qnn_ErrorHandle_t device_tensor_deep_copy(const Qnn_Tensor_t &src, Qnn_Tensor_t &dst) {
    Qnn_ErrorHandle_t err = validate_tensor_version(src);
    if (err != QNN_SUCCESS) {
        QNN_LOG_WARN("validate_tensor_version expected QNN_SUCCESS\n");
        return err;
    }

    dst.version = src.version;
    QNN_TENSOR_SET_NAME(dst, ::strndup(QNN_TENSOR_GET_NAME(src), std::string(QNN_TENSOR_GET_NAME(src)).size()));
    if (nullptr == QNN_TENSOR_GET_NAME(dst)) {
        return (Qnn_ErrorHandle_t)1;
    }
    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
    QNN_TENSOR_SET_MEM_TYPE(dst, QNN_TENSOR_GET_MEM_TYPE(src));

    if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_RAW) {
        Qnn_ClientBuffer_t client_buf = { nullptr, 0 };
        QNN_TENSOR_SET_CLIENT_BUF(dst, client_buf);
    } else if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_MEMHANDLE) {
        QNN_TENSOR_SET_MEM_HANDLE(dst, nullptr);
    } else {
        return (Qnn_ErrorHandle_t)1;
    }

    Qnn_QuantizeParams_t src_qparam = QNN_TENSOR_GET_QUANT_PARAMS(src);
    Qnn_QuantizationEncoding_t encoding = src_qparam.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        Qnn_QuantizeParams_t src_qparam_cpy = src_qparam;
        Qnn_AxisScaleOffset_t &axis_scale_offset = src_qparam_cpy.axisScaleOffsetEncoding;
        Qnn_ScaleOffset_t **scaleOffset = &axis_scale_offset.scaleOffset;
        size_t scaleOffsetSize = axis_scale_offset.numScaleOffsets * sizeof(Qnn_ScaleOffset_t);
        *scaleOffset = (Qnn_ScaleOffset_t *)malloc(scaleOffsetSize);
        memscpy(*scaleOffset, scaleOffsetSize, src_qparam.axisScaleOffsetEncoding.scaleOffset, scaleOffsetSize);
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam_cpy);
    } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
        Qnn_QuantizeParams_t src_qparam_cpy = src_qparam;
        Qnn_BwAxisScaleOffset_t &bwaxis_scale_offset = src_qparam_cpy.bwAxisScaleOffsetEncoding;
        size_t scaleSize = bwaxis_scale_offset.numElements * sizeof(float);
        float **scales = &bwaxis_scale_offset.scales;
        int32_t **offsets = &bwaxis_scale_offset.offsets;
        *scales = (float *)malloc(scaleSize);
        memscpy(*scales, scaleSize, src_qparam.bwAxisScaleOffsetEncoding.scales, scaleSize);

        if (bwaxis_scale_offset.offsets != nullptr) {
            size_t offsetSize = bwaxis_scale_offset.numElements * sizeof(int32_t);
            *offsets = (int32_t *)malloc(offsetSize);
            memscpy(*offsets, offsetSize, src_qparam.bwAxisScaleOffsetEncoding.offsets, offsetSize);
        }
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam_cpy);
    } else {
        QNN_TENSOR_SET_QUANT_PARAMS(dst, src_qparam);
    }

    uint32_t rank = QNN_TENSOR_GET_RANK(src);
    QNN_TENSOR_SET_RANK(dst, rank);
    size_t dim_size = rank * sizeof(uint32_t);
    uint32_t *dimensions = (uint32_t *)malloc(dim_size);
    if (dimensions == nullptr) {
        QNN_LOG_WARN(
            "deep_copy_qnn_tensors() allocation error while copying "
            "tensor %s\n",
            QNN_TENSOR_GET_NAME(src));
        return (Qnn_ErrorHandle_t)1;
    }
    memscpy(dimensions, dim_size, QNN_TENSOR_GET_DIMENSIONS(src), dim_size);
    QNN_TENSOR_SET_DIMENSIONS(dst, dimensions);

    return err;
}

void device_tensor_free(Qnn_Tensor_t &tensor) {
    if (validate_tensor_version(tensor) != QNN_SUCCESS) {
        QNN_LOG_WARN("validate_tensor_version expected QNN_SUCCESS\n");
        return;
    }

    free((void *)QNN_TENSOR_GET_NAME(tensor));
    free(QNN_TENSOR_GET_DIMENSIONS(tensor));
}

} // namespace qnn
