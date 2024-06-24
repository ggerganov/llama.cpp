#pragma once

#include "ggml.h"
#include "backend.hpp"

namespace qnn {

    typedef void (*ggml_qnn_op_t)(ggml_backend_qnn_context* ctx,
        const ggml_tensor* src0,
        const ggml_tensor* src1,
        ggml_tensor* dst);

    typedef const ggml_qnn_op_t(&ggml_qnn_op_array_t)[GGML_OP_COUNT];

    ggml_qnn_op_array_t ggml_qnn_op_array();

}
