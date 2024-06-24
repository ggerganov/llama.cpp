
#include "utils.hpp"

#include "ggml-qnn.h"
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

}
