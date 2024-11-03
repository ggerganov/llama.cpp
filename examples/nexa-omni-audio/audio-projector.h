#pragma once

#include "ggml.h"
#include "common-nexa.h"

#include <vector>

//
// Audio Projector
//

struct audio_projector : public NexaBaseModel
{

    audio_projector() : NexaBaseModel()
    {
        this->hparam_names = {
            "max_source_positions",
            "d_model",
        };
        this->tensor_names = {
            "multi_modal_projector.linear.weight",
            "multi_modal_projector.linear.bias",
        };
    }

    struct ggml_cgraph *build_graph() override
    {
        const int MAX_NODES = 64;
        size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
        static std::vector<uint8_t> buf(buf_size);

        // Create temporary GGML context for building the graph
        struct ggml_init_params params = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true, // Memory will be allocated later
        };
        struct ggml_context *ctx0 = ggml_init(params);
        struct ggml_cgraph *gf = ggml_new_graph_custom(ctx0, MAX_NODES, false); // Create new graph

        // Create input tensor
        struct ggml_tensor *input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                                                       std::get<int32_t>(hparams["d_model"]),
                                                       std::get<int32_t>(hparams["max_source_positions"]) / 2);
        ggml_set_name(input, "input");
        ggml_set_input(input); // Mark tensor as input

        // weight * input + bias
        struct ggml_tensor *cur = ggml_mul_mat(ctx0, tensors["multi_modal_projector.linear.weight"], input);
        cur = ggml_add(ctx0, cur, tensors["multi_modal_projector.linear.bias"]);

        // Set the final output
        ggml_set_name(cur, "output");
        ggml_set_output(cur);

        ggml_build_forward_expand(gf, cur); // Expand graph with operations

        ggml_free(ctx0); // Free temporary context

        return gf;
    }
};

struct ggml_tensor *audio_projector_inference(audio_projector &model, std::vector<float> &audio_feature_data);

struct ggml_tensor *audio_projector_inference(audio_projector &model, struct ggml_tensor *audio_feature_tensor);