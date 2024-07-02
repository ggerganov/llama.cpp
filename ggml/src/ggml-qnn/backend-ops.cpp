
#include "backend-ops.hpp"

#include "utils.hpp"
#include "logger.hpp"
#include "tensor.hpp"


static bool qnn_is_valid_params(ggml_backend_qnn_context* ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
    if ((nullptr == ctx) || (nullptr == src0) || (nullptr == src1) || (nullptr == dst)) {
        QNN_LOG_WARN("invalid params\n");
        return false;
    }

    qnn::qnn_instance* instance = nullptr;
    Qnn_Tensor_t* tensor_0 = nullptr;
    Qnn_Tensor_t* tensor_1 = nullptr;
    Qnn_Tensor_t* tensor_2 = nullptr;
    tensor_0 = (Qnn_Tensor_t*)src0->extra;
    tensor_1 = (Qnn_Tensor_t*)src1->extra;
    tensor_2 = (Qnn_Tensor_t*)dst->extra;
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

//TODO: this function can be removed later because there are duplicated codes with ggml_qnn_mul_mat
//      keep it for illustrate how to implement a specified GGMPL OP using QNN API + QNN RPC
static void ggml_qnn_add(ggml_backend_qnn_context* ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
    Qnn_ErrorHandle_t  error = QNN_SUCCESS;
    bool               graph_initialized = false;
    qnn::qnn_instance* instance = nullptr;
    std::string        graph_name = "ggml_op_qnn_add";
    Qnn_GraphHandle_t  graph_handle = nullptr;
    Qnn_Param_t        qnn_params[] = {};
    enum ggml_op       ggmlop = GGML_OP_ADD;

    CHECK_PARAMS(ctx, src0, src1, dst);
    instance = ctx->instance;
    auto qnn_raw_interface = ctx->raw_interface;

    qnn::qnn_perf perf("ggml_qnn_add");
    perf.start();

    std::string map_entry = std::string(ggml_op_name(ggmlop));
    if (instance->_qnn_graph_map.find(map_entry) !=
        instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        auto& graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
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

            const QnnGraph_Config_t* p_graphconfig[] = { &graph_hvx_config,
                                                         &graph_dlbc_config,
                                                         &graph_vtcm_config,
                                                         &graph_opt_config,
                                                         NULL };
            error = qnn_raw_interface.graphCreate(
                instance->get_qnn_context_handle(), graph_name.c_str(), p_graphconfig,
                &graph_handle);
        }
        else {
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
        else {
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

        Qnn_Tensor_t   tensor_inputs[] = { *tensor_input0.get_qnn_tensor(), *tensor_input1.get_qnn_tensor() };
        Qnn_Tensor_t   tensor_outputs[] = { *tensor_output.get_qnn_tensor() };
        Qnn_OpConfig_t op_config = {
            (Qnn_OpConfigVersion_t)1,
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
    }
    else {
        auto& graph_item = instance->_qnn_graph_map[map_entry];
        qnn::ggml_qnn_tensor_input tensor_input0(src0, std::get<1>(graph_item), ctx);
        qnn::ggml_qnn_tensor_input tensor_input1(src1, std::get<2>(graph_item), ctx);
        qnn::ggml_qnn_tensor_output tensor_output(dst, std::get<3>(graph_item), ctx);

        Qnn_Tensor_t tensor_inputs[] = { *tensor_input0.get_qnn_tensor(), *tensor_input1.get_qnn_tensor() };
        Qnn_Tensor_t tensor_outputs[] = { *tensor_output.get_qnn_tensor() };
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
static void ggml_qnn_mul_mat(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
    Qnn_ErrorHandle_t  error = QNN_SUCCESS;
    bool               graph_initialized = false;
    qnn::qnn_instance* instance = nullptr;
    std::string        graph_name = "ggml_op_qnn_mul_mat";
    Qnn_GraphHandle_t  graph_handle = nullptr;
    Qnn_Param_t        qnn_params[] = {};
    enum ggml_op       ggmlop = GGML_OP_MUL_MAT;

    CHECK_PARAMS(ctx, src0, src1, dst);
    instance = ctx->instance;
    auto qnn_raw_interface = ctx->raw_interface;

    qnn::qnn_perf perf("ggml_qnn_mul_mat");
    perf.start();

    std::string map_entry = std::string(ggml_op_name(ggmlop));
    if (instance->_qnn_graph_map.find(map_entry) !=
        instance->_qnn_graph_map.end()) {
        graph_initialized = true;
        auto& graph_item = instance->_qnn_graph_map[map_entry];
        graph_handle = std::get<0>(graph_item);
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

            const QnnGraph_Config_t* p_graphconfig[] = { &graph_hvx_config,
                                                         &graph_dlbc_config,
                                                         &graph_vtcm_config,
                                                         &graph_opt_config,
                                                         NULL };
            error = qnn_raw_interface.graphCreate(
                instance->get_qnn_context_handle(), graph_name.c_str(), p_graphconfig,
                &graph_handle);
        }
        else {
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

        Qnn_Tensor_t   tensor_inputs[] = { *tensor_input0.get_qnn_tensor(), *tensor_input1.get_qnn_tensor() };
        Qnn_Tensor_t   tensor_outputs[] = { *tensor_output.get_qnn_tensor() };
        Qnn_OpConfig_t op_config = {
                (Qnn_OpConfigVersion_t)1,
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
    }
    else {
        auto& graph_item = instance->_qnn_graph_map[map_entry];
        qnn::ggml_qnn_tensor_input tensor_input0(src0, std::get<1>(graph_item), ctx);
        qnn::ggml_qnn_tensor_input tensor_input1(src1, std::get<2>(graph_item), ctx);
        qnn::ggml_qnn_tensor_output tensor_output(dst, std::get<3>(graph_item), ctx);

        Qnn_Tensor_t tensor_inputs[] = { *tensor_input0.get_qnn_tensor(), *tensor_input1.get_qnn_tensor() };
        Qnn_Tensor_t tensor_outputs[] = { *tensor_output.get_qnn_tensor() };
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

static void ggml_qnn_repeat(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_get_rows(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_acc(ggml_backend_qnn_context* ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_div(ggml_backend_qnn_context* ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_gelu(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_silu(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_gelu_quick(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_tanh(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_relu(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_hardsigmoid(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_hardswish(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_leaky_relu(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_sqr(ggml_backend_qnn_context* ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_norm(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_group_norm(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_concat(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_upscale(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_pad(ggml_backend_qnn_context* ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_rms_norm(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_cpy(ggml_backend_qnn_context* ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_dup(ggml_backend_qnn_context* ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
    ggml_qnn_cpy(ctx, src0, dst, nullptr);
    (void)src1;
}

static void ggml_qnn_mul_mat_id(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_scale(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_clamp(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_diag_mask_inf(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
}

static void ggml_qnn_soft_max(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_rope(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
}

static void ggml_qnn_pool2d(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_im2col(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
}

static void ggml_qnn_sum_rows(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
}

static void ggml_qnn_argsort(ggml_backend_qnn_context* ctx,
    const ggml_tensor* src0, const ggml_tensor* src1,
    ggml_tensor* dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
}

static void ggml_qnn_nop(ggml_backend_qnn_context* ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
    (void)src0;
    (void)src1;
    (void)dst;
}

qnn::ggml_qnn_op_array_t qnn::ggml_qnn_op_array() {
    static constexpr const qnn::ggml_qnn_op_t kQnnOpsTable[GGML_OP_COUNT] = {
        nullptr, // GGML_OP_NONE
        nullptr, // GGML_OP_DUP
        ggml_qnn_add, // GGML_OP_ADD
        nullptr, // GGML_OP_ADD1
        nullptr, // GGML_OP_ACC
        nullptr, // GGML_OP_SUB
        nullptr, // GGML_OP_MUL
        nullptr, // GGML_OP_DIV
        nullptr, // GGML_OP_SQR
        nullptr, // GGML_OP_SQRT
        nullptr, // GGML_OP_LOG
        nullptr, // GGML_OP_SUM
        nullptr, // GGML_OP_SUM_ROWS
        nullptr, // GGML_OP_MEAN
        nullptr, // GGML_OP_ARGMAX
        nullptr, // GGML_OP_REPEAT
        nullptr, // GGML_OP_REPEAT_BACK
        nullptr, // GGML_OP_CONCAT
        nullptr, // GGML_OP_SILU_BACK
        nullptr, // GGML_OP_NORM
        nullptr, // GGML_OP_RMS_NORM
        nullptr, // GGML_OP_RMS_NORM_BACK
        nullptr, // GGML_OP_GROUP_NORM

        ggml_qnn_mul_mat, // GGML_OP_MUL_MAT
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

    return kQnnOpsTable;
}
