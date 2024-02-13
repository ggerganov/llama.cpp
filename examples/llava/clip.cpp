// NOTE: This is modified from clip.cpp only for LLaVA,
// so there might be still unnecessary artifacts hanging around
// I'll gradually clean and extend it

#include "clip.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <cinttypes>

static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

//
// key constants
//

#define KEY_FTYPE "general.file_type"
#define KEY_NAME "general.name"
#define KEY_DESCRIPTION "general.description"
#define KEY_HAS_TEXT_ENC "clip.has_text_encoder"
#define KEY_HAS_VIS_ENC "clip.has_vision_encoder"
#define KEY_HAS_LLAVA_PROJ "clip.has_llava_projector"
#define KEY_USE_GELU "clip.use_gelu"
#define KEY_N_EMBD "clip.%s.embedding_length"
#define KEY_N_FF "clip.%s.feed_forward_length"
#define KEY_N_BLOCK "clip.%s.block_count"
#define KEY_N_HEAD "clip.%s.attention.head_count"
#define KEY_LAYER_NORM_EPS "clip.%s.attention.layer_norm_epsilon"
#define KEY_PROJ_DIM "clip.%s.projection_dim"
#define KEY_TOKENS "tokenizer.ggml.tokens"
#define KEY_N_POSITIONS "clip.text.context_length"
#define KEY_IMAGE_SIZE "clip.vision.image_size"
#define KEY_PATCH_SIZE "clip.vision.patch_size"
#define KEY_IMAGE_MEAN "clip.vision.image_mean"
#define KEY_IMAGE_STD "clip.vision.image_std"
#define KEY_PROJ_TYPE "clip.projector_type"

//
// tensor name constants
//

#define TN_TOKEN_EMBD "%s.token_embd.weight"
#define TN_POS_EMBD "%s.position_embd.weight"
#define TN_CLASS_EMBD "v.class_embd"
#define TN_PATCH_EMBD "v.patch_embd.weight"
#define TN_ATTN_K "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT "%s.blk.%d.attn_out.%s"
#define TN_FFN_DOWN "%s.blk.%d.ffn_down.%s"
#define TN_FFN_UP "%s.blk.%d.ffn_up.%s"
#define TN_LN_1 "%s.blk.%d.ln1.%s"
#define TN_LN_2 "%s.blk.%d.ln2.%s"
#define TN_LN_PRE "%s.pre_ln.%s"
#define TN_LN_POST "%s.post_ln.%s"
#define TN_TEXT_PROJ "text_projection.weight"
#define TN_VIS_PROJ "visual_projection.weight"
#define TN_LLAVA_PROJ "mm.%d.%s"
#define TN_MVLM_PROJ_MLP "mm.model.mlp.%d.%s"
#define TN_MVLM_PROJ_BLOCK "mm.model.mb_block.%d.block.%d.%s"


enum projector_type {
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_MLP_NORM,
    PROJECTOR_TYPE_LDP,
    PROJECTOR_TYPE_UNKNOWN,
};

static std::map<projector_type, std::string> PROJECTOR_TYPE_NAMES = {
    { PROJECTOR_TYPE_MLP,           "mlp"     },
    { PROJECTOR_TYPE_LDP,          "ldp"    },
};


//
// utilities to get data from a gguf file
//

static int get_key_idx(const gguf_context * ctx, const char * key) {
    int i = gguf_find_key(ctx, key);
    if (i == -1) {
        fprintf(stderr, "key %s not found in file\n", key);
        throw std::runtime_error(format("Missing required key: %s", key));
    }

    return i;
}

static uint32_t get_u32(const gguf_context * ctx, const std::string & key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_u32(ctx, i);
}

static float get_f32(const gguf_context * ctx, const std::string & key) {
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_f32(ctx, i);
}

static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        throw std::runtime_error(format("%s: unable to find tensor %s\n", __func__, name.c_str()));
    }

    return cur;
}

static std::string get_ftype(int ftype) {
    return ggml_type_name(static_cast<ggml_type>(ftype));
}

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const bool *)data)[i] ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}


static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    std::string result;
    for (size_t pos = 0; ; pos += search.length()) {
        auto new_pos = s.find(search, pos);
        if (new_pos == std::string::npos) {
            result += s.substr(pos, s.size() - pos);
            break;
        }
        result += s.substr(pos, new_pos - pos) + replace;
        pos = new_pos;
    }
    s = std::move(result);
}

static std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replace_all(val, "\\", "\\\\");
                        replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

static void print_tensor_info(const ggml_tensor* tensor, const char* prefix = "") {
    size_t tensor_size = ggml_nbytes(tensor);
    printf("%s: n_dims = %d, name = %s, tensor_size=%zu, shape:[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], type = %s\n",
            prefix, ggml_n_dims(tensor), tensor->name, tensor_size,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], ggml_type_name(tensor->type));
}

static projector_type clip_projector_type_from_string(const std::string & name) {
    for (const auto & kv : PROJECTOR_TYPE_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }
    return PROJECTOR_TYPE_UNKNOWN;
}

//
// image data
//

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

//
// clip layers
//

struct clip_layer {
    // attention
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    struct ggml_tensor * o_w;
    struct ggml_tensor * o_b;

    // layernorm 1
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_1_b;

    // ff
    struct ggml_tensor * ff_i_w;
    struct ggml_tensor * ff_i_b;

    struct ggml_tensor * ff_o_w;
    struct ggml_tensor * ff_o_b;

    // layernorm 2
    struct ggml_tensor * ln_2_w;
    struct ggml_tensor * ln_2_b;
};

struct clip_vision_model {
    struct clip_vision_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding;
    struct ggml_tensor * patch_embeddings;
    struct ggml_tensor * position_embeddings;

    struct ggml_tensor * pre_ln_w;
    struct ggml_tensor * pre_ln_b;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;

    // LLaVA projection
    struct ggml_tensor * mm_0_w = NULL;
    struct ggml_tensor * mm_0_b = NULL;
    struct ggml_tensor * mm_2_w = NULL;
    struct ggml_tensor * mm_2_b = NULL;

    // Yi type models with mlp+normalization projection
    struct ggml_tensor * mm_1_w = NULL; // Yi type models have 0, 1, 3, 4
    struct ggml_tensor * mm_1_b = NULL;
    struct ggml_tensor * mm_3_w = NULL;
    struct ggml_tensor * mm_3_b = NULL;
    struct ggml_tensor * mm_4_w = NULL;
    struct ggml_tensor * mm_4_b = NULL;

    // MobileVLM projection
    struct ggml_tensor * mm_model_mlp_1_w;
    struct ggml_tensor * mm_model_mlp_1_b;
    struct ggml_tensor * mm_model_mlp_3_w;
    struct ggml_tensor * mm_model_mlp_3_b;
    struct ggml_tensor * mm_model_block_1_block_0_0_w;
    struct ggml_tensor * mm_model_block_1_block_0_1_w;
    struct ggml_tensor * mm_model_block_1_block_0_1_b;
    struct ggml_tensor * mm_model_block_1_block_1_fc1_w;
    struct ggml_tensor * mm_model_block_1_block_1_fc1_b;
    struct ggml_tensor * mm_model_block_1_block_1_fc2_w;
    struct ggml_tensor * mm_model_block_1_block_1_fc2_b;
    struct ggml_tensor * mm_model_block_1_block_2_0_w;
    struct ggml_tensor * mm_model_block_1_block_2_1_w;
    struct ggml_tensor * mm_model_block_1_block_2_1_b;
    struct ggml_tensor * mm_model_block_2_block_0_0_w;
    struct ggml_tensor * mm_model_block_2_block_0_1_w;
    struct ggml_tensor * mm_model_block_2_block_0_1_b;
    struct ggml_tensor * mm_model_block_2_block_1_fc1_w;
    struct ggml_tensor * mm_model_block_2_block_1_fc1_b;
    struct ggml_tensor * mm_model_block_2_block_1_fc2_w;
    struct ggml_tensor * mm_model_block_2_block_1_fc2_b;
    struct ggml_tensor * mm_model_block_2_block_2_0_w;
    struct ggml_tensor * mm_model_block_2_block_2_1_w;
    struct ggml_tensor * mm_model_block_2_block_2_1_b;
};

struct clip_ctx {
    bool has_text_encoder    = false;
    bool has_vision_encoder  = false;
    bool has_llava_projector = false;

    struct clip_vision_model vision_model;
    projector_type proj_type = PROJECTOR_TYPE_MLP;

    float image_mean[3];
    float image_std[3];
    bool use_gelu = false;
    int32_t ftype = 1;

    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    // memory buffers to evaluate the model
    ggml_backend_buffer_t params_buffer = NULL;
    ggml_backend_buffer_t compute_buffer = NULL;
    ggml_backend_t backend = NULL;
    ggml_gallocr_t compute_alloc = NULL;
};

static ggml_cgraph * clip_image_build_graph(clip_ctx * ctx, const clip_image_f32_batch * imgs) {
    if (!ctx->has_vision_encoder) {
        printf("This gguf file seems to have no vision encoder\n");
        return nullptr;
    }

    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size = hparams.image_size;
    const int patch_size = hparams.patch_size;
    const int num_patches = ((image_size / patch_size) * (image_size / patch_size));
    const int num_positions = num_patches + 1;
    const int hidden_size = hparams.hidden_size;
    const int n_head = hparams.n_head;
    const int d_head = hidden_size / n_head;
    const int n_layer = hparams.n_layer;
    //const int n_intermediate = hparams.n_intermediate;
    //const int projection_dim = hparams.projection_dim;
    const float eps = hparams.eps;
    int batch_size = imgs->size;
    if (ctx->has_llava_projector) {
        GGML_ASSERT(batch_size == 1);
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size, image_size, 3, batch_size);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

    // concat class_embeddings and patch_embeddings
    struct ggml_tensor * embeddings = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, batch_size);
    ggml_set_name(embeddings, "embeddings");
    ggml_set_input(embeddings);

    embeddings = ggml_acc(ctx0, embeddings, model.class_embedding,
            embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], 0);

    embeddings = ggml_acc(ctx0, embeddings, inp,
            embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], model.class_embedding->nb[1]);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_positions);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    embeddings =
        ggml_add(ctx0, embeddings, ggml_get_rows(ctx0, model.position_embeddings, positions));

    // pre-layernorm
    {
        embeddings = ggml_norm(ctx0, embeddings, eps);
        ggml_set_name(embeddings, "pre_ln");

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.pre_ln_w), model.pre_ln_b);
    }

    // loop over layers
    for (int il = 0; il < n_layer - 1; il++) {
        struct ggml_tensor * cur = embeddings; // embeddings = residual, cur = hidden_states

        //const size_t nb_q_w = model.layers[il].q_w->nb[0];

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_1_w),
                           model.layers[il].ln_1_b);
        }

        // self-attention
        {

            struct ggml_tensor * Q =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].q_w, cur), model.layers[il].q_b);

            Q = ggml_scale_inplace(ctx0, Q, 1.0f / sqrt((float)d_head));
            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_positions, batch_size);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * K =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].k_w, cur), model.layers[il].k_b);

            K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * V =
                ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].v_w, cur), model.layers[il].v_b);

            V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, batch_size);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head * batch_size);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_soft_max_inplace(ctx0, KQ);
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_positions, n_head, batch_size);
            KQV = ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3));

            cur = ggml_cpy(ctx0, KQV, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, batch_size));
        }

        // attention output
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].o_w, cur), model.layers[il].o_b);

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_2_w), model.layers[il].ln_2_b);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ff_i_b);

        if (ctx->use_gelu) {
            cur = ggml_gelu_inplace(ctx0, cur);
        } else {
            cur = ggml_gelu_quick_inplace(ctx0, cur);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ff_o_b);

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);

        embeddings = cur;
    }

    // llava projector
    {
        embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);

        struct ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_patches);
        ggml_set_name(patches, "patches");
        ggml_set_input(patches);

        // shape [1, 576, 1024]
        // ne is whcn, ne = [1024, 576, 1, 1]
        embeddings = ggml_get_rows(ctx0, embeddings, patches);

        // print_tensor_info(embeddings, "embeddings");

        // llava projector
        if (ctx->proj_type == PROJECTOR_TYPE_MLP) {
            embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);

            embeddings = ggml_gelu(ctx0, embeddings);

            embeddings = ggml_mul_mat(ctx0, model.mm_2_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_2_b);

        } else if (ctx->proj_type == PROJECTOR_TYPE_MLP_NORM) {
            embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);
            // ggml_tensor_printf(embeddings, "mm_0_w",0,true,false);
            // First LayerNorm
            embeddings = ggml_norm(ctx0, embeddings, eps);
            embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_1_w),
                                model.mm_1_b);

            // GELU activation
            embeddings = ggml_gelu(ctx0, embeddings);

            // Second linear layer
            embeddings = ggml_mul_mat(ctx0, model.mm_3_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_3_b);

            // Second LayerNorm
            embeddings = ggml_norm(ctx0, embeddings, eps);
            embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_4_w),
                                model.mm_4_b);
        }
        else if (ctx->proj_type == PROJECTOR_TYPE_LDP) {
            // MobileVLM projector
            int n_patch = 24;
            struct ggml_tensor * mlp_1 = ggml_mul_mat(ctx0, model.mm_model_mlp_1_w, embeddings);
            mlp_1 = ggml_add(ctx0, mlp_1, model.mm_model_mlp_1_b);
            mlp_1 = ggml_gelu(ctx0, mlp_1);
            struct ggml_tensor * mlp_3 = ggml_mul_mat(ctx0, model.mm_model_mlp_3_w, mlp_1);
            mlp_3 = ggml_add(ctx0, mlp_3, model.mm_model_mlp_3_b);
            // mlp_3 shape = [1, 576, 2048], ne = [2048, 576, 1, 1]

            // block 1
            struct ggml_tensor * block_1 = nullptr;
            {
                // transpose from [1, 576, 2048] --> [1, 2048, 576] --> [1, 2048, 24, 24]
                mlp_3 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_3, 1, 0, 2, 3));
                mlp_3 = ggml_reshape_4d(ctx0, mlp_3, n_patch, n_patch, mlp_3->ne[1], mlp_3->ne[2]);
                // stride = 1, padding = 1, bias is nullptr
                block_1 = ggml_conv_depthwise_2d(ctx0, model.mm_model_block_1_block_0_0_w, mlp_3, 1, 1, 1, 1, 1, 1);

                // layer norm
                // // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_0_1_w), model.mm_model_block_1_block_0_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));

                // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                // hardswish
                struct ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                // pointwise conv
                block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_1_fc1_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc1_b);
                block_1 = ggml_relu(ctx0, block_1);
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_1_fc2_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc2_b);
                block_1 = ggml_hardsigmoid(ctx0, block_1);
                // block_1_hw shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1], block_1 shape = [1, 2048], ne = [2048, 1, 1, 1]
                block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                int w = block_1->ne[0], h = block_1->ne[1];
                block_1 = ggml_reshape_3d(ctx0, block_1, w*h, block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));

                // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_2_0_w, block_1);
                block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);

                // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_2_1_w), model.mm_model_block_1_block_2_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                // block1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                // residual
                block_1 = ggml_add(ctx0, mlp_3, block_1);
            }

            // block_2
            {
                // stride = 2
                block_1 = ggml_conv_depthwise_2d(ctx0, model.mm_model_block_2_block_0_0_w, block_1, 2, 2, 1, 1, 1, 1);

                // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                // layer norm
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_0_1_w), model.mm_model_block_2_block_0_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                // hardswish
                struct ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                // not sure the parameters is right for globalAvgPooling
                block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                // pointwise conv
                block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_1_fc1_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc1_b);
                block_1 = ggml_relu(ctx0, block_1);
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_1_fc2_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc2_b);
                block_1 = ggml_hardsigmoid(ctx0, block_1);

                // block_1_hw shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1], block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                int w = block_1->ne[0], h = block_1->ne[1];
                block_1 = ggml_reshape_3d(ctx0, block_1, w*h, block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));
                // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_2_0_w, block_1);
                block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);


                // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_2_1_w), model.mm_model_block_2_block_2_1_b);
                block_1 = ggml_reshape_3d(ctx0, block_1, block_1->ne[0], block_1->ne[1] * block_1->ne[2], block_1->ne[3]);
                // block_1 shape = [1, 144, 2048], ne = [2048, 144, 1]
            }
            embeddings = block_1;
        }
        else {
            GGML_ASSERT(false);
        }
    }

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    ggml_free(ctx0);

    return gf;
}

// read and create ggml_context containing the tensors and their data
struct clip_ctx * clip_model_load(const char * fname, const int verbosity = 1) {
    struct ggml_context * meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname, params);
    if (!ctx) {
        throw std::runtime_error(format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname));
    }

    if (verbosity >= 1) {
        const int n_tensors = gguf_get_n_tensors(ctx);
        const int n_kv = gguf_get_n_kv(ctx);
        const int ftype = get_u32(ctx, KEY_FTYPE);
        const std::string ftype_str = get_ftype(ftype);
        const int idx_desc = get_key_idx(ctx, KEY_DESCRIPTION);
        const std::string description = gguf_get_val_str(ctx, idx_desc);
        const int idx_name = gguf_find_key(ctx, KEY_NAME);
        if (idx_name != -1) { // make name optional temporarily as some of the uploaded models missing it due to a bug
            const std::string name = gguf_get_val_str(ctx, idx_name);
            printf("%s: model name:   %s\n", __func__, name.c_str());
        }
        printf("%s: description:  %s\n", __func__, description.c_str());
        printf("%s: GGUF version: %d\n", __func__, gguf_get_version(ctx));
        printf("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx));
        printf("%s: n_tensors:    %d\n", __func__, n_tensors);
        printf("%s: n_kv:         %d\n", __func__, n_kv);
        printf("%s: ftype:        %s\n", __func__, ftype_str.c_str());
        printf("\n");
    }
    const int n_tensors = gguf_get_n_tensors(ctx);

    // kv
    const int n_kv = gguf_get_n_kv(ctx);
    printf("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n",
        __func__, n_kv, n_tensors, fname);
    {
        std::map<enum ggml_type, uint32_t> n_type;

        for (int i = 0; i < n_tensors; i++) {
            enum ggml_type type = gguf_get_tensor_type(ctx, i);

            n_type[type]++;
        }

        printf("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);
        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(ctx, i);
            const enum gguf_type type   = gguf_get_kv_type(ctx, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%d]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(ctx, i)), gguf_get_arr_n(ctx, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(ctx, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            printf("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            printf("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    // data
    size_t model_size = 0;
    {
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);
            enum ggml_type type = gguf_get_tensor_type(ctx, i);
            struct ggml_tensor * cur = ggml_get_tensor(meta, name);
            size_t tensor_size = ggml_nbytes(cur);
            model_size += tensor_size;
            if (verbosity >= 3) {
                printf("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                       __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    clip_ctx * new_clip = new clip_ctx;

    // update projector type
    {
        int idx = gguf_find_key(ctx, KEY_PROJ_TYPE);
        if (idx != -1) {
            const std::string proj_type = gguf_get_val_str(ctx, idx);
            new_clip->proj_type = clip_projector_type_from_string(proj_type);
        }
        else {
            new_clip->proj_type = PROJECTOR_TYPE_MLP;
        }
        if (new_clip->proj_type == PROJECTOR_TYPE_MLP) {
            if (gguf_find_tensor(ctx, format(TN_LLAVA_PROJ, 3, "weight").c_str()) != -1) {
                new_clip->proj_type = PROJECTOR_TYPE_MLP_NORM;
            }
        }
    }

#ifdef GGML_USE_CUBLAS
    new_clip->backend = ggml_backend_cuda_init(0);
    printf("%s: CLIP using CUDA backend\n", __func__);
#endif

#ifdef GGML_USE_METAL
    new_clip->backend = ggml_backend_metal_init();
    printf("%s: CLIP using Metal backend\n", __func__);
#endif


    if (!new_clip->backend) {
        new_clip->backend = ggml_backend_cpu_init();
        printf("%s: CLIP using CPU backend\n", __func__);
    }

    // model size and capabilities
    {
        int idx = get_key_idx(ctx, KEY_HAS_TEXT_ENC);
        new_clip->has_text_encoder = gguf_get_val_bool(ctx, idx);

        idx = get_key_idx(ctx, KEY_HAS_VIS_ENC);
        new_clip->has_vision_encoder = gguf_get_val_bool(ctx, idx);

        idx = gguf_find_key(ctx, KEY_HAS_LLAVA_PROJ);
        if (idx != -1) {
            new_clip->has_llava_projector = gguf_get_val_bool(ctx, idx);
        }

        GGML_ASSERT(new_clip->has_llava_projector); // see monatis/clip.cpp for image and/or text encoding for semantic search
        GGML_ASSERT(new_clip->has_vision_encoder);
        GGML_ASSERT(!new_clip->has_text_encoder);

        idx = get_key_idx(ctx, KEY_USE_GELU);
        new_clip->use_gelu = gguf_get_val_bool(ctx, idx);

        if (verbosity >= 1) {
            printf("%s: text_encoder:   %d\n", __func__, new_clip->has_text_encoder);
            printf("%s: vision_encoder: %d\n", __func__, new_clip->has_vision_encoder);
            printf("%s: llava_projector:  %d\n", __func__, new_clip->has_llava_projector);
            printf("%s: model size:     %.2f MB\n", __func__, model_size / 1024.0 / 1024.0);
            printf("%s: metadata size:  %.2f MB\n", __func__, ggml_get_mem_size(meta) / 1024.0 / 1024.0);
        }
    }

    printf("%s: params backend buffer size = % 6.2f MB (%i tensors)\n", __func__, model_size / (1024.0 * 1024.0), n_tensors);

    // load tensors
    {
        std::vector<uint8_t> read_buf;
        struct ggml_init_params params = {
            /*.mem_size =*/ (n_tensors + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };

        new_clip->ctx_data = ggml_init(params);
        if (!new_clip->ctx_data) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            clip_free(new_clip);
            return nullptr;
        }

        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            printf("cannot open model file for loading tensors\n");
            clip_free(new_clip);
            return nullptr;
        }

        // add tensors to context
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor * t = ggml_get_tensor(meta, name);
            struct ggml_tensor * cur = ggml_dup_tensor(new_clip->ctx_data, t);
            ggml_set_name(cur, name);
        }

        // alloc memory and offload data
        new_clip->params_buffer = ggml_backend_alloc_ctx_tensors(new_clip->ctx_data, new_clip->backend);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor * cur = ggml_get_tensor(new_clip->ctx_data, name);
            const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                printf("%s: failed to seek for tensor %s\n", __func__, name);
                clip_free(new_clip);
                return nullptr;
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(new_clip->params_buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        fin.close();
    }

    // vision model
    if (new_clip->has_vision_encoder) {
        // load vision model
        auto & vision_model = new_clip->vision_model;
        auto & hparams = vision_model.hparams;
        hparams.hidden_size    = get_u32(ctx, format(KEY_N_EMBD, "vision"));
        hparams.n_head         = get_u32(ctx, format(KEY_N_HEAD, "vision"));
        hparams.n_intermediate = get_u32(ctx, format(KEY_N_FF, "vision"));
        hparams.n_layer        = get_u32(ctx, format(KEY_N_BLOCK, "vision"));
        hparams.image_size     = get_u32(ctx, KEY_IMAGE_SIZE);
        hparams.patch_size     = get_u32(ctx, KEY_PATCH_SIZE);
        hparams.projection_dim = get_u32(ctx, format(KEY_PROJ_DIM, "vision"));
        hparams.eps            = get_f32(ctx, format(KEY_LAYER_NORM_EPS, "vision"));

        int idx_mean = get_key_idx(ctx, KEY_IMAGE_MEAN);
        int idx_std  = get_key_idx(ctx, KEY_IMAGE_STD);
        for (int i = 0; i < 3; ++i) {
            new_clip->image_mean[i] = *((const float *)gguf_get_arr_data(ctx, idx_mean));
            new_clip->image_std[i]  = *((const float *)gguf_get_arr_data(ctx, idx_std));
        }

        if (verbosity >= 2) {
            printf("\n%s: vision model hparams\n", __func__);
            printf("image_size         %d\n", hparams.image_size);
            printf("patch_size         %d\n", hparams.patch_size);
            printf("v_hidden_size      %d\n", hparams.hidden_size);
            printf("v_n_intermediate   %d\n", hparams.n_intermediate);
            printf("v_projection_dim   %d\n", hparams.projection_dim);
            printf("v_n_head           %d\n", hparams.n_head);
            printf("v_n_layer          %d\n", hparams.n_layer);
        }

        vision_model.patch_embeddings    = get_tensor(new_clip->ctx_data, TN_PATCH_EMBD);
        vision_model.class_embedding     = get_tensor(new_clip->ctx_data, TN_CLASS_EMBD);
        vision_model.position_embeddings = get_tensor(new_clip->ctx_data, format(TN_POS_EMBD, "v"));
        vision_model.pre_ln_w            = get_tensor(new_clip->ctx_data, format(TN_LN_PRE, "v", "weight"));
        vision_model.pre_ln_b            = get_tensor(new_clip->ctx_data, format(TN_LN_PRE, "v", "bias"));

        // LLaVA projection
        if (new_clip->proj_type == PROJECTOR_TYPE_MLP || new_clip->proj_type == PROJECTOR_TYPE_MLP_NORM) {
            vision_model.mm_0_w              = get_tensor(new_clip->ctx_data, format(TN_LLAVA_PROJ, 0, "weight"));
            vision_model.mm_0_b              = get_tensor(new_clip->ctx_data, format(TN_LLAVA_PROJ, 0, "bias"));
            try {
                // Yi-type llava
                vision_model.mm_1_w = get_tensor(new_clip->ctx_data, format(TN_LLAVA_PROJ, 1, "weight"));
                vision_model.mm_1_b = get_tensor(new_clip->ctx_data, format(TN_LLAVA_PROJ, 1, "bias"));
            } catch (std::runtime_error & e) {  }
            try {
                // missing in Yi-type llava
                vision_model.mm_2_w              = get_tensor(new_clip->ctx_data, format(TN_LLAVA_PROJ, 2, "weight"));
                vision_model.mm_2_b              = get_tensor(new_clip->ctx_data, format(TN_LLAVA_PROJ, 2, "bias"));
            } catch (std::runtime_error & e) {  }
            try {
                // Yi-type llava
                vision_model.mm_3_w = get_tensor(new_clip->ctx_data, format(TN_LLAVA_PROJ, 3, "weight"));
                vision_model.mm_3_b = get_tensor(new_clip->ctx_data, format(TN_LLAVA_PROJ, 3, "bias"));
            } catch (std::runtime_error & e) {  }
            try {
                // Yi-type llava
                vision_model.mm_4_w = get_tensor(new_clip->ctx_data, format(TN_LLAVA_PROJ, 4, "weight"));
                vision_model.mm_4_b = get_tensor(new_clip->ctx_data, format(TN_LLAVA_PROJ, 4, "bias"));
            } catch (std::runtime_error & e) {  }
        }
        else if (new_clip->proj_type == PROJECTOR_TYPE_LDP) {
            // MobileVLM projection
            vision_model.mm_model_mlp_1_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 1, "weight"));
            vision_model.mm_model_mlp_1_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 1, "bias"));
            vision_model.mm_model_mlp_3_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 3, "weight"));
            vision_model.mm_model_mlp_3_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 3, "bias"));
            vision_model.mm_model_block_1_block_0_0_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 0, "0.weight"));
            vision_model.mm_model_block_1_block_0_1_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.weight"));
            vision_model.mm_model_block_1_block_0_1_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.bias"));
            vision_model.mm_model_block_1_block_1_fc1_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.weight"));
            vision_model.mm_model_block_1_block_1_fc1_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.bias"));
            vision_model.mm_model_block_1_block_1_fc2_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.weight"));
            vision_model.mm_model_block_1_block_1_fc2_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.bias"));
            vision_model.mm_model_block_1_block_2_0_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 2, "0.weight"));
            vision_model.mm_model_block_1_block_2_1_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.weight"));
            vision_model.mm_model_block_1_block_2_1_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.bias"));
            vision_model.mm_model_block_2_block_0_0_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 0, "0.weight"));
            vision_model.mm_model_block_2_block_0_1_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.weight"));
            vision_model.mm_model_block_2_block_0_1_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.bias"));
            vision_model.mm_model_block_2_block_1_fc1_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.weight"));
            vision_model.mm_model_block_2_block_1_fc1_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.bias"));
            vision_model.mm_model_block_2_block_1_fc2_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.weight"));
            vision_model.mm_model_block_2_block_1_fc2_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.bias"));
            vision_model.mm_model_block_2_block_2_0_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 2, "0.weight"));
            vision_model.mm_model_block_2_block_2_1_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.weight"));
            vision_model.mm_model_block_2_block_2_1_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.bias"));
        }
        else {
            std::string proj_type = PROJECTOR_TYPE_NAMES[new_clip->proj_type];
            throw std::runtime_error(format("%s: don't support projector with: %s currently\n", __func__, proj_type.c_str()));
        }

        vision_model.layers.resize(hparams.n_layer);
        for (int il = 0; il < hparams.n_layer; ++il) {
            auto & layer = vision_model.layers[il];
            layer.k_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_K,      "v", il, "weight"));
            layer.q_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_Q,      "v", il, "weight"));
            layer.v_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_V,      "v", il, "weight"));
            layer.o_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_OUTPUT, "v", il, "weight"));
            layer.ln_1_w = get_tensor(new_clip->ctx_data, format(TN_LN_1,        "v", il, "weight"));
            layer.ln_2_w = get_tensor(new_clip->ctx_data, format(TN_LN_2,        "v", il, "weight"));
            layer.ff_i_w = get_tensor(new_clip->ctx_data, format(TN_FFN_DOWN,    "v", il, "weight"));
            layer.ff_o_w = get_tensor(new_clip->ctx_data, format(TN_FFN_UP,      "v", il, "weight"));
            layer.k_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_K,      "v", il, "bias"));
            layer.q_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_Q,      "v", il, "bias"));
            layer.v_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_V,      "v", il, "bias"));
            layer.o_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_OUTPUT, "v", il, "bias"));
            layer.ln_1_b = get_tensor(new_clip->ctx_data, format(TN_LN_1,        "v", il, "bias"));
            layer.ln_2_b = get_tensor(new_clip->ctx_data, format(TN_LN_2,        "v", il, "bias"));
            layer.ff_i_b = get_tensor(new_clip->ctx_data, format(TN_FFN_DOWN,    "v", il, "bias"));
            layer.ff_o_b = get_tensor(new_clip->ctx_data, format(TN_FFN_UP,      "v", il, "bias"));
        }
    }

    ggml_free(meta);

    new_clip->ctx_gguf = ctx;

    // measure mem requirement and allocate
    {
        new_clip->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
        new_clip->compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(new_clip->backend));
        clip_image_f32_batch batch;
        batch.size = 1;
        ggml_cgraph * gf = clip_image_build_graph(new_clip, &batch);
        ggml_gallocr_reserve(new_clip->compute_alloc, gf);
        size_t compute_memory_buffer_size = ggml_gallocr_get_buffer_size(new_clip->compute_alloc, 0);
        printf("%s: compute allocated memory: %.2f MB\n", __func__, compute_memory_buffer_size /1024.0/1024.0);
    }

    return new_clip;
}

struct clip_image_u8 * clip_image_u8_init() {
    return new clip_image_u8();
}

struct clip_image_f32 * clip_image_f32_init() {
    return new clip_image_f32();
}

void clip_image_u8_free (struct clip_image_u8  * img) { delete img; }
void clip_image_f32_free(struct clip_image_f32 * img) { delete img; }

static void build_clip_img_from_data(const stbi_uc * data, int nx, int ny, clip_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    memcpy(img->buf.data(), data, img->buf.size());
}

bool clip_image_load_from_file(const char * fname, clip_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load(fname, &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load image '%s'\n", __func__, fname);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img) {
    int nx, ny, nc;
    auto * data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to decode image bytes\n", __func__);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

// normalize: x = (x - mean) / std
// TODO: implement bicubic interpolation instead of linear.
bool clip_image_preprocess(struct clip_ctx * ctx, const clip_image_u8 * img, clip_image_f32 * res, const bool pad2square) {
    if (!ctx->has_vision_encoder) {
        printf("This gguf file seems to have no vision encoder\n");
        return false;
    }

    // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
    // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

    clip_image_u8 * temp = clip_image_u8_init(); // we will keep the input image data here temporarily
    if (pad2square && img->nx != img->ny) {
        int longer_side = std::max(img->nx, img->ny);
        temp->nx = longer_side;
        temp->ny = longer_side;
        temp->buf.resize(3 * longer_side * longer_side);
        const uint8_t bc[3] = {122, 116, 104}; // background color in RGB from LLaVA

        // fill with background color
        for (size_t i = 0; i < temp->buf.size(); i++) {
            temp->buf[i] = bc[i % 3];
        }

        // copy from the input image
        for (int y = 0; y < img->ny; y++) {
            for (int x = 0; x < img->nx; x++) {
                const int i = 3 * (y * img->nx + x);
                const int j = 3 * (y * temp->nx + x);
                temp->buf[j]   = img->buf[i];
                temp->buf[j+1] = img->buf[i+1];
                temp->buf[j+2] = img->buf[i+2];
            }
        }
    } else {
        temp->nx = img->nx;
        temp->ny = img->ny;
        temp->buf.resize(img->buf.size());
        memcpy(temp->buf.data(), img->buf.data(), temp->buf.size());
    }

    const int nx = temp->nx;
    const int ny = temp->ny;

    const int nx2 = ctx->vision_model.hparams.image_size;
    const int ny2 = ctx->vision_model.hparams.image_size;

    res->nx = nx2;
    res->ny = ny2;
    res->buf.resize(3 * nx2 * ny2);

    const float scale = std::max(nx, ny) / (float)ctx->vision_model.hparams.image_size;

    const int nx3 = int(nx / scale + 0.5f);
    const int ny3 = int(ny / scale + 0.5f);

    const auto & m3 = ctx->image_mean; // {0.48145466f, 0.4578275f, 0.40821073f};
    const auto & s3 = ctx->image_std;  // {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                // linear interpolation
                const float sx = (x + 0.5f) * scale - 0.5f;
                const float sy = (y + 0.5f) * scale - 0.5f;

                const int x0 = std::max(0, (int)std::floor(sx));
                const int y0 = std::max(0, (int)std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = temp->buf[j00];
                const float v01 = temp->buf[j01];
                const float v10 = temp->buf[j10];
                const float v11 = temp->buf[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res->buf[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }
    clip_image_u8_free(temp);

    return true;
}

void clip_free(clip_ctx * ctx) {
    ggml_free(ctx->ctx_data);
    gguf_free(ctx->ctx_gguf);

    delete ctx;
}

bool clip_image_encode(struct clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    if (!ctx->has_vision_encoder) {
        printf("This gguf file seems to have no vision encoder\n");
        return false;
    }

    clip_image_f32_batch imgs{};
    imgs.size = 1;
    imgs.data = img;
    return clip_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool clip_image_batch_encode(clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs, float * vec) {
    if (!ctx->has_vision_encoder) {
        printf("This gguf file seems to have no vision encoder\n");
        return false;
    }

    int batch_size = imgs->size;
    if(ctx->has_llava_projector) {
        GGML_ASSERT(batch_size == 1); // TODO: support multiple images
    }

    // build the inference graph
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs);
    ggml_gallocr_alloc_graph(ctx->compute_alloc, gf);

    // set inputs
    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;
    const int image_size = hparams.image_size;
    const int patch_size = hparams.patch_size;
    const int num_patches = ((image_size / patch_size) * (image_size / patch_size));
    const int num_positions = num_patches + 1;

    {
        struct ggml_tensor * inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        float * data = (float *)malloc(ggml_nbytes(inp_raw));

        for (size_t i = 0; i < imgs->size; i++) {
            const int nx = imgs->data[i].nx;
            const int ny = imgs->data[i].ny;
            GGML_ASSERT(nx == image_size && ny == image_size);

            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < 3; k++) {
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            data[(b * 3 * n) + k * n + y * nx + x] = imgs->data[b].buf[3 * (y * nx + x) + k];
                        }
                    }
                }
            }
        }
        ggml_backend_tensor_set(inp_raw, data, 0, ggml_nbytes(inp_raw));
        free(data);
    }

    {
        struct ggml_tensor * embeddings = ggml_graph_get_tensor(gf, "embeddings");

        void* zero_mem = malloc(ggml_nbytes(embeddings));
        memset(zero_mem, 0, ggml_nbytes(embeddings));
        ggml_backend_tensor_set(embeddings, zero_mem, 0, ggml_nbytes(embeddings));
        free(zero_mem);
    }

    {
        struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "positions");

        int* positions_data = (int*)malloc(ggml_nbytes(positions));
        for (int i = 0; i < num_positions; i++) {
            positions_data[i] = i;
        }
        ggml_backend_tensor_set(positions, positions_data, 0, ggml_nbytes(positions));
        free(positions_data);
    }

    {
        struct ggml_tensor * patches = ggml_graph_get_tensor(gf, "patches");
        int* patches_data = (int*)malloc(ggml_nbytes(patches));
        for (int i = 0; i < num_patches; i++) {
            patches_data[i] = i + 1;
        }
        ggml_backend_tensor_set(patches, patches_data, 0, ggml_nbytes(patches));
        free(patches_data);
    }

    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(ctx->backend)) {
        ggml_backend_metal_set_n_cb(ctx->backend, n_threads);
    }
#endif

    ggml_backend_graph_compute(ctx->backend, gf);

    // the last node is the embedding tensor
    struct ggml_tensor * embeddings = gf->nodes[gf->n_nodes - 1];

    // copy the embeddings to the location passed by the user
    ggml_backend_tensor_get(embeddings, vec, 0, ggml_nbytes(embeddings));
    return true;
}

bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype) {

    ggml_type type = GGML_TYPE_Q4_1;

    assert(itype < GGML_TYPE_COUNT);
    type = static_cast<ggml_type>(itype);

    auto * ctx_clip = clip_model_load(fname_inp, 2);

    const auto & ctx_src = ctx_clip->ctx_gguf;
    const auto & ctx_data = ctx_clip->ctx_data;

    auto * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_src);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", itype);

    auto fout = std::ofstream(fname_out, std::ios::binary);

    const int n_tensors = gguf_get_n_tensors(ctx_src);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
        gguf_add_tensor(ctx_out, cur);
    }

    const size_t meta_size = gguf_get_meta_size(ctx_out);
    for (size_t i = 0; i < meta_size; ++i) {
        fout.put(0);
    }

    // regexes of tensor names to be quantized
    const std::vector<std::string> k_names = {
        ".*weight",
    };

    std::vector<uint8_t> work(512);
    std::vector<float> conv_buf(512);
    std::vector<int64_t> hist_all(1 << 4, 0);
    size_t total_size_org = 0;
    size_t total_size_new = 0;

    for (int i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(ctx_src, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name.c_str());

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;

        bool quantize = false;
        for (const auto & s : k_names) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (ggml_n_dims(cur) == 2);

        if (quantize) {
            new_type = type;
            if (new_type >= GGML_TYPE_Q2_K && name.find("embd") != std::string::npos) {
                new_type = GGML_TYPE_Q8_0; // ggml_get_rows needs non K type
                // fprintf(stderr, "%s: quantizing %s to %s\n", __func__, name.c_str(), ggml_type_name(new_type));
            }
            const size_t n_elms = ggml_nelements(cur);
            float * f32_data;

            switch (cur->type) {
            case GGML_TYPE_F32:
                f32_data = (float *)cur->data;
                break;
            case GGML_TYPE_F16:
                if (conv_buf.size() < n_elms) {
                    conv_buf.resize(n_elms);
                }
                for (size_t j = 0; j < n_elms; ++j) {
                    conv_buf[j] = ggml_fp16_to_fp32(((ggml_fp16_t *)cur->data)[j]);
                }
                f32_data = (float *)conv_buf.data();
                break;
            default:
                printf("Please use an input file in f32 or f16\n");
                return false;
            }

            if (work.size() < n_elms * 4) {
                work.resize(n_elms * 4);
            }
            new_data = work.data();

            std::vector<int64_t> hist_cur(1 << 4, 0);

            switch (new_type) {
                case GGML_TYPE_Q4_0: {
                    new_size = ggml_quantize_q4_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                case GGML_TYPE_Q4_1: {
                    new_size = ggml_quantize_q4_1(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                case GGML_TYPE_Q5_0: {
                    new_size = ggml_quantize_q5_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                case GGML_TYPE_Q5_1: {
                    new_size = ggml_quantize_q5_1(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                case GGML_TYPE_Q8_0: {
                    new_size = ggml_quantize_q8_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                case GGML_TYPE_Q2_K: {
                    new_size = ggml_quantize_q2_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                case GGML_TYPE_Q3_K: {
                    new_size = ggml_quantize_q3_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                case GGML_TYPE_Q4_K: {
                    new_size = ggml_quantize_q4_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                case GGML_TYPE_Q5_K: {
                    new_size = ggml_quantize_q5_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                case GGML_TYPE_Q6_K: {
                    new_size = ggml_quantize_q6_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                default: {
                    fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, new_type);
                    return false;
                }
            }

            for (size_t j = 0; j < hist_cur.size(); ++j) {
                hist_all[j] += hist_cur[j];
            }
        } else {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }
        const size_t orig_size = ggml_nbytes(cur);
        total_size_org += orig_size;
        total_size_new += new_size;
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data, new_size);
        fout.write((const char *)new_data, new_size);
        size_t pad = GGML_PAD(new_size, gguf_get_alignment(ctx_out)) - new_size;
        for (size_t j = 0; j < pad; ++j) {
            fout.put(0);
        }

        printf("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), ggml_n_dims(cur), quantize,
               orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
    }

    // go back to beginning of file and write the updated metadata
    fout.seekp(0, std::ios::beg);
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(ctx_out, meta.data());
    fout.write((const char *)meta.data(), meta_size);

    fout.close();

    clip_free(ctx_clip);
    gguf_free(ctx_out);

    {
        printf("%s: original  size = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        printf("%s: quantized size = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);

        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); ++i) {
            sum_all += hist_all[i];
        }

        printf("%s: hist: ", __func__);
        for (size_t i = 0; i < hist_all.size(); ++i) {
            printf("%5.3f ", hist_all[i] / (float)sum_all);
        }
        printf("\n");
    }

    return true;
}

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    if (ctx->proj_type == PROJECTOR_TYPE_LDP) {
        return ctx->vision_model.mm_model_block_1_block_2_1_b->ne[0];
    }
    else if (ctx->proj_type == PROJECTOR_TYPE_MLP) {
        return ctx->vision_model.mm_2_b->ne[0];
    } else if (ctx->proj_type == PROJECTOR_TYPE_MLP_NORM) {
        return ctx->vision_model.mm_3_b->ne[0];
    }
    else {
        std::string proj_type = PROJECTOR_TYPE_NAMES[ctx->proj_type];
        throw std::runtime_error(format("%s: don't support projector with: %s currently\n", __func__, proj_type.c_str()));
    }
}

int clip_n_patches(const struct clip_ctx * ctx) {
    auto & params = ctx->vision_model.hparams;
    int n_patches = (params.image_size / params.patch_size) * (params.image_size / params.patch_size);
    if (ctx->proj_type == PROJECTOR_TYPE_LDP) {
        n_patches /= 4;
    }
    return n_patches;
}

size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    return clip_n_patches(ctx) * clip_n_mmproj_embd(ctx) * sizeof(float);
}
