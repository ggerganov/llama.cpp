
#include "backend-ops.hpp"

#include <memory>

#include "graph.hpp"
#include "logger.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#ifndef NDEBUG

namespace {

bool qnn_is_valid_params(ggml_backend_qnn_context *ctx, const ggml_tensor *src, ggml_tensor *dst) {
    if (!ctx || !src || !dst) {
        QNN_LOG_WARN("invalid params\n");
        return false;
    }

    auto instance = ctx->instance;
    auto *tensor0 = qnn::ggml_qnn_tensor::from_ggml_tensor(src);
    auto *tensor1 = qnn::ggml_qnn_tensor::from_ggml_tensor(dst);
    if (!instance || !tensor0 || !tensor1) {
        QNN_LOG_WARN("invalid tensors\n");
        return false;
    }

    return true;
}

bool qnn_is_valid_params(ggml_backend_qnn_context *ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                         ggml_tensor *dst) {
    if (!ctx || !src0 || !src1 || !dst) {
        QNN_LOG_WARN("invalid params\n");
        return false;
    }

    auto instance = ctx->instance;
    auto *tensor0 = qnn::ggml_qnn_tensor::from_ggml_tensor(src0);
    auto *tensor1 = qnn::ggml_qnn_tensor::from_ggml_tensor(src1);
    auto *tensor2 = qnn::ggml_qnn_tensor::from_ggml_tensor(dst);
    if (!instance || !tensor0 || !tensor1 || !tensor2) {
        QNN_LOG_WARN("invalid tensors\n");
        return false;
    }

    return true;
}

} // namespace

#define CHECK_PARAMS(ctx, ...)                      \
    if (!qnn_is_valid_params((ctx), __VA_ARGS__)) { \
        return false;                               \
    }

#else
#define CHECK_PARAMS(ctx, ...)
#endif

namespace {

void print_ggml_tensor(const ggml_tensor *tensor) {
    QNN_LOG_DEBUG("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi)\n",
                  tensor->name, tensor->type, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2],
                  tensor->nb[0], tensor->nb[1], tensor->nb[2]);
}

template <size_t _InputSize, size_t _OutputSize>
bool qnn_bind_tensors_to_graph(qnn::ggml_qnn_graph<_InputSize, _OutputSize> *graph, const std::string &op_name,
                               const std::array<const ggml_tensor *, _InputSize> &inputs,
                               const std::array<ggml_tensor *, _OutputSize> &outputs) {
    std::array<Qnn_Tensor_t, _InputSize> qnn_input_tensors;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto tensor = qnn::ggml_qnn_tensor::from_ggml_tensor(inputs[i]);
        if (!tensor || !tensor->bind_to_graph(*graph)) {
            return false;
        }

        qnn_input_tensors[i] = tensor->get_qnn_tensor();
    }

    std::array<Qnn_Tensor_t, _OutputSize> qnn_output_tensors;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = qnn::ggml_qnn_tensor::from_ggml_tensor(outputs[i]);
        if (!tensor || !tensor->bind_to_graph(*graph)) {
            return false;
        }

        qnn_output_tensors[i] = tensor->get_qnn_tensor();
    }

    if (!graph->add_nodes(op_name, qnn_input_tensors, qnn_output_tensors)) {
        return false;
    }

    return true;
}

template <size_t _InputSize, size_t _OutputSize>
bool execute_graph(qnn::ggml_qnn_graph<_InputSize, _OutputSize> *graph,
                   const std::array<const ggml_tensor *, _InputSize> &inputs,
                   const std::array<ggml_tensor *, _OutputSize> &outputs) {

    std::array<Qnn_Tensor_t, _InputSize> qnn_input_tensors;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto tensor = qnn::ggml_qnn_tensor::from_ggml_tensor(inputs[i]);
        if (!tensor || !tensor->write_to_qnn_tensor()) {
            QNN_LOG_WARN("write_to_qnn_tensor failed\n");
            return false;
        }

        qnn_input_tensors[i] = tensor->get_qnn_tensor();
    }

    std::array<Qnn_Tensor_t, _OutputSize> qnn_output_tensors;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = qnn::ggml_qnn_tensor::from_ggml_tensor(outputs[i]);
        if (!tensor) {
            return false;
        }

        qnn_output_tensors[i] = tensor->get_qnn_tensor();
    }

    if (!graph->execute(qnn_input_tensors, qnn_output_tensors)) {
        QNN_LOG_WARN("execute failed\n");
        return false;
    }

    for (auto &output : outputs) {
        auto tensor = qnn::ggml_qnn_tensor::from_ggml_tensor(output);
        if (!tensor || !tensor->read_from_qnn_tensor()) {
            QNN_LOG_WARN("read_from_qnn_tensors failed\n");
            return false;
        }
    }

    return true;
}

qnn::ggml_qnn_unary_graph_cache_t &get_qnn_graph_cache(ggml_backend_qnn_context *ctx,
                                                       const std::array<const ggml_tensor *, 1> &inputs,
                                                       const std::array<ggml_tensor *, 1> &outputs) {
    GGML_UNUSED(inputs);
    GGML_UNUSED(outputs);
    return ctx->qnn_unary_graph_cache;
}

qnn::ggml_qnn_binary_graph_cache_t &get_qnn_graph_cache(ggml_backend_qnn_context *ctx,
                                                        const std::array<const ggml_tensor *, 2> &inputs,
                                                        const std::array<ggml_tensor *, 1> &outputs) {
    GGML_UNUSED(inputs);
    GGML_UNUSED(outputs);
    return ctx->qnn_binary_graph_cache;
}

template <size_t _InputSize, size_t _OutputSize>
qnn::ggml_qnn_graph<_InputSize, _OutputSize> *get_qnn_graph_from_cache(
    ggml_backend_qnn_context *ctx, ggml_op op, const std::string &qnn_op,
    const std::array<const ggml_tensor *, _InputSize> &inputs, const std::array<ggml_tensor *, _OutputSize> &outputs) {
    using graph_t = qnn::ggml_qnn_graph<_InputSize, _OutputSize>;

    auto &graph_cache = get_qnn_graph_cache(ctx, inputs, outputs);
    const std::string graph_key(ggml_op_name(op));
    auto it = graph_cache.find(graph_key);
    graph_t *graph_ptr = nullptr;
    if (it != graph_cache.end()) {
        graph_ptr = it->second.get();
    } else {
        std::string graph_name = graph_key + "_" + std::to_string(ctx->threads);
        for (auto &input : inputs) {
            graph_name += "_";
            graph_name += input->name;
        }
        auto graph =
            std::make_unique<graph_t>(graph_name, (QNNBackend)(ctx->device), ctx->instance->get_qnn_context_handle(),
                                      ctx->qnn_interface, ctx->socinfo.vtcm_size_in_mb);

        if (!graph->is_valid()) {
            return nullptr;
        }

        if (!qnn_bind_tensors_to_graph<_InputSize, _OutputSize>(graph.get(), qnn_op.c_str(), inputs, outputs)) {
            return nullptr;
        }

        graph_ptr = graph.get();
        graph_cache[graph_key] = std::move(graph);
    }

    return graph_ptr;
}

constexpr const char *kGgmlOpToQnnOp[] = {
    nullptr,                         // GGML_OP_NONE
    nullptr,                         // GGML_OP_DUP
    QNN_OP_ELEMENT_WISE_ADD,         // GGML_OP_ADD
    nullptr,                         // GGML_OP_ADD1
    nullptr,                         // GGML_OP_ACC
    QNN_OP_ELEMENT_WISE_SUBTRACT,    // GGML_OP_SUB
    QNN_OP_ELEMENT_WISE_MULTIPLY,    // GGML_OP_MUL
    QNN_OP_ELEMENT_WISE_DIVIDE,      // GGML_OP_DIV
    nullptr,                         // GGML_OP_SQR
    QNN_OP_ELEMENT_WISE_SQUARE_ROOT, // GGML_OP_SQRT
    QNN_OP_ELEMENT_WISE_LOG,         // GGML_OP_LOG
    nullptr,                         // GGML_OP_SUM
    nullptr,                         // GGML_OP_SUM_ROWS
    nullptr,                         // GGML_OP_MEAN
    nullptr,                         // GGML_OP_ARGMAX
    nullptr,                         // GGML_OP_REPEAT
    nullptr,                         // GGML_OP_REPEAT_BACK
    nullptr,                         // GGML_OP_CONCAT
    nullptr,                         // GGML_OP_SILU_BACK
    nullptr,                         // GGML_OP_NORM
    nullptr,                         // GGML_OP_RMS_NORM
    nullptr,                         // GGML_OP_RMS_NORM_BACK
    nullptr,                         // GGML_OP_GROUP_NORM

    QNN_OP_MAT_MUL, // GGML_OP_MUL_MAT
    nullptr,        // GGML_OP_MUL_MAT_ID
    nullptr,        // GGML_OP_OUT_PROD

    nullptr, // GGML_OP_SCALE
    nullptr, // GGML_OP_SET
    nullptr, // GGML_OP_CPY
    nullptr, // GGML_OP_CONT
    nullptr, // GGML_OP_RESHAPE
    nullptr, // GGML_OP_VIEW
    nullptr, // GGML_OP_PERMUTE
    nullptr, // GGML_OP_TRANSPOSE
    nullptr, // GGML_OP_GET_ROWS
    nullptr, // GGML_OP_GET_ROWS_BACK
    nullptr, // GGML_OP_DIAG
    nullptr, // GGML_OP_DIAG_MASK_INF
    nullptr, // GGML_OP_DIAG_MASK_ZERO
    nullptr, // GGML_OP_SOFT_MAX
    nullptr, // GGML_OP_SOFT_MAX_BACK
    nullptr, // GGML_OP_ROPE
    nullptr, // GGML_OP_ROPE_BACK
    nullptr, // GGML_OP_CLAMP
    nullptr, // GGML_OP_CONV_TRANSPOSE_1D
    nullptr, // GGML_OP_IM2COL
    nullptr, // GGML_OP_CONV_TRANSPOSE_2D
    nullptr, // GGML_OP_POOL_1D
    nullptr, // GGML_OP_POOL_2D
    nullptr, // GGML_OP_UPSCALE
    nullptr, // GGML_OP_PAD
    nullptr, // GGML_OP_ARANGE
    nullptr, // GGML_OP_TIMESTEP_EMBEDDING
    nullptr, // GGML_OP_ARGSORT
    nullptr, // GGML_OP_LEAKY_RELU

    nullptr, // GGML_OP_FLASH_ATTN_EXT
    nullptr, // GGML_OP_FLASH_ATTN_BACK
    nullptr, // GGML_OP_SSM_CONV
    nullptr, // GGML_OP_SSM_SCAN
    nullptr, // GGML_OP_WIN_PART
    nullptr, // GGML_OP_WIN_UNPART
    nullptr, // GGML_OP_GET_REL_POS
    nullptr, // GGML_OP_ADD_REL_POS

    nullptr, // GGML_OP_UNARY

    nullptr, // GGML_OP_MAP_UNARY
    nullptr, // GGML_OP_MAP_BINARY

    nullptr, // GGML_OP_MAP_CUSTOM1_F32
    nullptr, // GGML_OP_MAP_CUSTOM2_F32
    nullptr, // GGML_OP_MAP_CUSTOM3_F32

    nullptr, // GGML_OP_MAP_CUSTOM1
    nullptr, // GGML_OP_MAP_CUSTOM2
    nullptr, // GGML_OP_MAP_CUSTOM3

    nullptr, // GGML_OP_CROSS_ENTROPY_LOSS
    nullptr, // GGML_OP_CROSS_ENTROPY_LOSS_BACK
};

static_assert(sizeof(kGgmlOpToQnnOp) / sizeof(kGgmlOpToQnnOp[0]) == GGML_OP_COUNT,
              "GGML_OP_COUNT does not match the size of the ops table");

template <ggml_op _GgmlOp>
bool qnn_binary_op_impl(ggml_backend_qnn_context *ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                        ggml_tensor *dst) {
    static_assert(kGgmlOpToQnnOp[_GgmlOp] != nullptr, "GGML_OP does not have a corresponding QNN_OP");

    CHECK_PARAMS(ctx, src0, src1, dst);

    qnn::qnn_perf perf(ggml_op_name(_GgmlOp));
    perf.start();

    bool succeed = false;
    qnn::ggml_qnn_graph_binary *graph_ptr =
        get_qnn_graph_from_cache<2, 1>(ctx, _GgmlOp, kGgmlOpToQnnOp[_GgmlOp], { src0, src1 }, { dst });
    if (graph_ptr) {
        succeed = execute_graph<2, 1>(graph_ptr, { src0, src1 }, { dst });
    }

    if (!succeed) {
        print_ggml_tensor(src0);
        print_ggml_tensor(src1);
        print_ggml_tensor(dst);
    }

    return succeed;
}

template <ggml_op _GgmlOp>
bool qnn_unary_op_impl(ggml_backend_qnn_context *ctx, const ggml_tensor *src, ggml_tensor *dst) {
    static_assert(kGgmlOpToQnnOp[_GgmlOp] != nullptr, "GGML_OP does not have a corresponding QNN_OP");

    CHECK_PARAMS(ctx, src, dst);

    qnn::qnn_perf perf(ggml_op_name(_GgmlOp));
    perf.start();

    bool succeed = false;
    auto *graph_ptr = get_qnn_graph_from_cache<1, 1>(ctx, _GgmlOp, kGgmlOpToQnnOp[_GgmlOp], { src }, { dst });
    if (graph_ptr) {
        succeed = execute_graph<1, 1>(graph_ptr, { src }, { dst });
    }

    if (!succeed) {
        print_ggml_tensor(src);
        print_ggml_tensor(dst);
    }

    return succeed;
}

} // namespace

qnn::ggml_qnn_unary_op_array_t qnn::ggml_qnn_unary_op_array() {
    static constexpr const qnn::ggml_qnn_unary_op_t kQnnOpsTable[] = {
        nullptr,                         // GGML_OP_NONE
        nullptr,                         // GGML_OP_DUP
        nullptr,                         // GGML_OP_ADD
        nullptr,                         // GGML_OP_ADD1
        nullptr,                         // GGML_OP_ACC
        nullptr,                         // GGML_OP_SUB
        nullptr,                         // GGML_OP_MUL
        nullptr,                         // GGML_OP_DIV
        nullptr,                         // GGML_OP_SQR
        qnn_unary_op_impl<GGML_OP_SQRT>, // GGML_OP_SQRT
        qnn_unary_op_impl<GGML_OP_LOG>,  // GGML_OP_LOG
        nullptr,                         // GGML_OP_SUM
        nullptr,                         // GGML_OP_SUM_ROWS
        nullptr,                         // GGML_OP_MEAN
        nullptr,                         // GGML_OP_ARGMAX
        nullptr,                         // GGML_OP_REPEAT
        nullptr,                         // GGML_OP_REPEAT_BACK
        nullptr,                         // GGML_OP_CONCAT
        nullptr,                         // GGML_OP_SILU_BACK
        nullptr,                         // GGML_OP_NORM
        nullptr,                         // GGML_OP_RMS_NORM
        nullptr,                         // GGML_OP_RMS_NORM_BACK
        nullptr,                         // GGML_OP_GROUP_NORM

        nullptr, // GGML_OP_MUL_MAT
        nullptr, // GGML_OP_MUL_MAT_ID
        nullptr, // GGML_OP_OUT_PROD

        nullptr, // GGML_OP_SCALE
        nullptr, // GGML_OP_SET
        nullptr, // GGML_OP_CPY
        nullptr, // GGML_OP_CONT
        nullptr, // GGML_OP_RESHAPE
        nullptr, // GGML_OP_VIEW
        nullptr, // GGML_OP_PERMUTE
        nullptr, // GGML_OP_TRANSPOSE
        nullptr, // GGML_OP_GET_ROWS
        nullptr, // GGML_OP_GET_ROWS_BACK
        nullptr, // GGML_OP_DIAG
        nullptr, // GGML_OP_DIAG_MASK_INF
        nullptr, // GGML_OP_DIAG_MASK_ZERO
        nullptr, // GGML_OP_SOFT_MAX
        nullptr, // GGML_OP_SOFT_MAX_BACK
        nullptr, // GGML_OP_ROPE
        nullptr, // GGML_OP_ROPE_BACK
        nullptr, // GGML_OP_CLAMP
        nullptr, // GGML_OP_CONV_TRANSPOSE_1D
        nullptr, // GGML_OP_IM2COL
        nullptr, // GGML_OP_CONV_TRANSPOSE_2D
        nullptr, // GGML_OP_POOL_1D
        nullptr, // GGML_OP_POOL_2D
        nullptr, // GGML_OP_UPSCALE
        nullptr, // GGML_OP_PAD
        nullptr, // GGML_OP_ARANGE
        nullptr, // GGML_OP_TIMESTEP_EMBEDDING
        nullptr, // GGML_OP_ARGSORT
        nullptr, // GGML_OP_LEAKY_RELU

        nullptr, // GGML_OP_FLASH_ATTN_EXT
        nullptr, // GGML_OP_FLASH_ATTN_BACK
        nullptr, // GGML_OP_SSM_CONV
        nullptr, // GGML_OP_SSM_SCAN
        nullptr, // GGML_OP_WIN_PART
        nullptr, // GGML_OP_WIN_UNPART
        nullptr, // GGML_OP_GET_REL_POS
        nullptr, // GGML_OP_ADD_REL_POS

        nullptr, // GGML_OP_UNARY

        nullptr, // GGML_OP_MAP_UNARY
        nullptr, // GGML_OP_MAP_BINARY

        nullptr, // GGML_OP_MAP_CUSTOM1_F32
        nullptr, // GGML_OP_MAP_CUSTOM2_F32
        nullptr, // GGML_OP_MAP_CUSTOM3_F32

        nullptr, // GGML_OP_MAP_CUSTOM1
        nullptr, // GGML_OP_MAP_CUSTOM2
        nullptr, // GGML_OP_MAP_CUSTOM3

        nullptr, // GGML_OP_CROSS_ENTROPY_LOSS
        nullptr, // GGML_OP_CROSS_ENTROPY_LOSS_BACK
    };

    static_assert(sizeof(kQnnOpsTable) / sizeof(kQnnOpsTable[0]) == GGML_OP_COUNT,
                  "GGML_OP_COUNT does not match the size of the ops table");
    return kQnnOpsTable;
}

qnn::ggml_qnn_binary_op_array_t qnn::ggml_qnn_binary_op_array() {
    static constexpr const qnn::ggml_qnn_binary_op_t kQnnOpsTable[] = {
        nullptr,                         // GGML_OP_NONE
        nullptr,                         // GGML_OP_DUP
        qnn_binary_op_impl<GGML_OP_ADD>, // GGML_OP_ADD
        nullptr,                         // GGML_OP_ADD1
        nullptr,                         // GGML_OP_ACC
        qnn_binary_op_impl<GGML_OP_SUB>, // GGML_OP_SUB
        qnn_binary_op_impl<GGML_OP_MUL>, // GGML_OP_MUL
        qnn_binary_op_impl<GGML_OP_DIV>, // GGML_OP_DIV
        nullptr,                         // GGML_OP_SQR
        nullptr,                         // GGML_OP_SQRT
        nullptr,                         // GGML_OP_LOG
        nullptr,                         // GGML_OP_SUM
        nullptr,                         // GGML_OP_SUM_ROWS
        nullptr,                         // GGML_OP_MEAN
        nullptr,                         // GGML_OP_ARGMAX
        nullptr,                         // GGML_OP_REPEAT
        nullptr,                         // GGML_OP_REPEAT_BACK
        nullptr,                         // GGML_OP_CONCAT
        nullptr,                         // GGML_OP_SILU_BACK
        nullptr,                         // GGML_OP_NORM
        nullptr,                         // GGML_OP_RMS_NORM
        nullptr,                         // GGML_OP_RMS_NORM_BACK
        nullptr,                         // GGML_OP_GROUP_NORM

        qnn_binary_op_impl<GGML_OP_MUL_MAT>, // GGML_OP_MUL_MAT
        nullptr,                             // GGML_OP_MUL_MAT_ID
        nullptr,                             // GGML_OP_OUT_PROD

        nullptr, // GGML_OP_SCALE
        nullptr, // GGML_OP_SET
        nullptr, // GGML_OP_CPY
        nullptr, // GGML_OP_CONT
        nullptr, // GGML_OP_RESHAPE
        nullptr, // GGML_OP_VIEW
        nullptr, // GGML_OP_PERMUTE
        nullptr, // GGML_OP_TRANSPOSE
        nullptr, // GGML_OP_GET_ROWS
        nullptr, // GGML_OP_GET_ROWS_BACK
        nullptr, // GGML_OP_DIAG
        nullptr, // GGML_OP_DIAG_MASK_INF
        nullptr, // GGML_OP_DIAG_MASK_ZERO
        nullptr, // GGML_OP_SOFT_MAX
        nullptr, // GGML_OP_SOFT_MAX_BACK
        nullptr, // GGML_OP_ROPE
        nullptr, // GGML_OP_ROPE_BACK
        nullptr, // GGML_OP_CLAMP
        nullptr, // GGML_OP_CONV_TRANSPOSE_1D
        nullptr, // GGML_OP_IM2COL
        nullptr, // GGML_OP_CONV_TRANSPOSE_2D
        nullptr, // GGML_OP_POOL_1D
        nullptr, // GGML_OP_POOL_2D
        nullptr, // GGML_OP_UPSCALE
        nullptr, // GGML_OP_PAD
        nullptr, // GGML_OP_ARANGE
        nullptr, // GGML_OP_TIMESTEP_EMBEDDING
        nullptr, // GGML_OP_ARGSORT
        nullptr, // GGML_OP_LEAKY_RELU

        nullptr, // GGML_OP_FLASH_ATTN_EXT
        nullptr, // GGML_OP_FLASH_ATTN_BACK
        nullptr, // GGML_OP_SSM_CONV
        nullptr, // GGML_OP_SSM_SCAN
        nullptr, // GGML_OP_WIN_PART
        nullptr, // GGML_OP_WIN_UNPART
        nullptr, // GGML_OP_GET_REL_POS
        nullptr, // GGML_OP_ADD_REL_POS

        nullptr, // GGML_OP_UNARY

        nullptr, // GGML_OP_MAP_UNARY
        nullptr, // GGML_OP_MAP_BINARY

        nullptr, // GGML_OP_MAP_CUSTOM1_F32
        nullptr, // GGML_OP_MAP_CUSTOM2_F32
        nullptr, // GGML_OP_MAP_CUSTOM3_F32

        nullptr, // GGML_OP_MAP_CUSTOM1
        nullptr, // GGML_OP_MAP_CUSTOM2
        nullptr, // GGML_OP_MAP_CUSTOM3

        nullptr, // GGML_OP_CROSS_ENTROPY_LOSS
        nullptr, // GGML_OP_CROSS_ENTROPY_LOSS_BACK
    };

    static_assert(sizeof(kQnnOpsTable) / sizeof(kQnnOpsTable[0]) == GGML_OP_COUNT,
                  "GGML_OP_COUNT does not match the size of the ops table");
    return kQnnOpsTable;
}
