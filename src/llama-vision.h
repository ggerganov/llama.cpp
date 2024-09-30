#pragma once

#include "ggml.h"

#include <vector>

enum vision_arch {
    VISION_ARCH_LLAVA,
    VISION_ARCH_UNKNOWN,
};

enum clip_projector_type {
    CLIP_PROJECTOR_TYPE_MLP,
};

enum mm_patch_merge {
    MM_PATCH_MERGE_FLAT,
    MM_PATCH_MERGE_SPATIAL_UNPAD,
};

struct clip_hparams {
    vision_arch arch = VISION_ARCH_UNKNOWN;

    uint32_t image_size;
    uint32_t patch_size;
    uint32_t hidden_size;
    uint32_t n_intermediate;
    uint32_t projection_dim;
    uint32_t n_head;
    uint32_t n_layer;
    uint32_t max_pos_embd;

    float eps;

    clip_projector_type proj_type = CLIP_PROJECTOR_TYPE_MLP;
    mm_patch_merge mm_patch_merge_type = MM_PATCH_MERGE_FLAT;

    std::array<float, 3> image_mean;
    std::array<float, 3> image_std;

    std::array<int32_t, 32> image_grid_pinpoints;
    int32_t image_crop_resolution;
};

struct clip_layer {
    // attention
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    struct ggml_tensor * output_w;
    struct ggml_tensor * output_b;

    // layernorm 1
    struct ggml_tensor * norm_in_w;
    struct ggml_tensor * norm_in_b;

    // ff
    struct ggml_tensor * ffn_up_w;
    struct ggml_tensor * ffn_up_b;

    struct ggml_tensor * ffn_down_w;
    struct ggml_tensor * ffn_down_b;

    // layernorm 2
    struct ggml_tensor * norm_out_w;
    struct ggml_tensor * norm_out_b;
};

struct clip_vision_model {
    struct clip_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding;
    struct ggml_tensor * patch_embeddings;
    struct ggml_tensor * patch_bias;
    struct ggml_tensor * position_embeddings;

    struct ggml_tensor * pre_norm_w;
    struct ggml_tensor * pre_norm_b;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_norm_w;
    struct ggml_tensor * post_norm_b;

    struct ggml_tensor * projection;

    // LLaVA projection
    struct ggml_tensor * mm_a_w = NULL;
    struct ggml_tensor * mm_a_b = NULL;
    struct ggml_tensor * mm_b_w = NULL;
    struct ggml_tensor * mm_b_b = NULL;

    struct ggml_tensor * image_newline = NULL;
};

struct clip_context {
    struct ggml_context * ctx_ggml;
    clip_vision_model model;

    int32_t n_output;
    float * output;
};
