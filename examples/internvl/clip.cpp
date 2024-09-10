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
#include <limits>

#include "clip.h"
#include "log.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

#define KEY_MM_PATCH_MERGE_TYPE   "clip.vision.mm_patch_merge_type"
#define KEY_IMAGE_GRID_PINPOINTS  "clip.vision.image_grid_pinpoints"
#define KEY_IMAGE_CROP_RESOLUTION "clip.vision.image_crop_resolution"

#define TN_PATCH_EMBD "v.patch_embedding.%s"
#define TN_POS_EMBD "v.position_embedding"
#define TN_CLASS_EMBD "v.class_embedding"

#define TN_ATTN_IN "v.blk.%d.attn.%s.%s"
#define TN_ATTN_PROJ "v.blk.%d.attn.proj.%s"
#define TN_LN_1 "v.blk.%d.ln1.%s"
#define TN_LN_2 "v.blk.%d.ln2.%s"
#define TN_MLP_1 "v.blk.%d.mlp.fc1.%s"
#define TN_MLP_2 "v.blk.%d.mlp.fc2.%s"
#define TN_LS_1 "v.blk.%d.ls1"
#define TN_LS_2 "v.blk.%d.ls2"

#define TN_MLP1_LN "mlp1.0.%s"
#define TN_MLP1_1 "mlp1.1.%s"
#define TN_MLP1_3 "mlp1.3.%s"

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
    LOG_TEE("%s: n_dims = %d, name = %s, tensor_size=%zu, shape:[%d, %d, %d, %d], type: %d\n",
            prefix, ggml_n_dims(tensor), tensor->name, tensor_size,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->type);
}

static void read_data(float* result, int size, std::string filepath,
                                    char delim) {
  std::ifstream file_handle(filepath.c_str());
  std::string line;
  int curr_count = 0;
  if (file_handle.is_open()) {
    while (getline(file_handle, line)) {
      std::stringstream ss(line);
      while (getline(ss, line, delim)) {  // split line content
        result[curr_count++] = std::stof(line);
      }
    }
    file_handle.close();
    assert(size == curr_count);
  } else {
    LOG_TEE("file:%s can't open normally!\n", filepath.c_str());
  }
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


struct clip_layer {
    // attention
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    struct ggml_tensor * proj_w;
    struct ggml_tensor * proj_b;

    struct ggml_tensor * ls_1;
    struct ggml_tensor * ls_2;

    struct ggml_tensor * mlp_1_w;
    struct ggml_tensor * mlp_1_b;

    struct ggml_tensor * mlp_2_w;
    struct ggml_tensor * mlp_2_b;

    // layernorm 1
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_1_b;

    // layernorm 2
    struct ggml_tensor * ln_2_w;
    struct ggml_tensor * ln_2_b;
};

struct clip_vision_model {
    struct clip_vision_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding;
    struct ggml_tensor * patch_embeddings_w; // conv1.weight
    struct ggml_tensor * patch_embeddings_b;
    struct ggml_tensor * position_embeddings; // positional_embd

    std::vector<clip_layer> layers;

    // struct ggml_tensor * slice_get_rows;

    struct ggml_tensor * mlp1_ln_w;
    struct ggml_tensor * mlp1_ln_b;

    struct ggml_tensor * mlp1_1_w;
    struct ggml_tensor * mlp1_1_b;

    struct ggml_tensor * mlp1_3_w;
    struct ggml_tensor * mlp1_3_b;
};

struct clip_ctx {
    bool has_text_encoder    = false;
    bool has_vision_encoder  = false;
    bool has_llava_projector = false;

    struct clip_vision_model vision_model;

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
        LOG_TEE("This gguf file seems to have no vision encoder\n");
        return nullptr;
    }

    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size           = hparams.image_size;
    const int patch_size           = hparams.patch_size;
    const int tgt_size             = image_size / patch_size;
    const int num_patches          = tgt_size * tgt_size;
    const int num_positions        = num_patches + 1;
    const int hidden_size          = hparams.hidden_size;
    const int n_head               = hparams.n_head;
    const int d_head               = hidden_size / n_head;
    const int n_layer              = hparams.n_layer;
    const float eps                = hparams.eps;
    const int n_intermediate       = hparams.n_intermediate;
    const float scale_factor       = hparams.scale_factor;

    const int batch_size = imgs->size;

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

    struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_w, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    
    inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

    inp = ggml_add(ctx0, inp, model.patch_embeddings_b);

    struct ggml_tensor * expanded_class_embedding = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, 1, batch_size);

    inp = ggml_concat(ctx0, ggml_repeat(ctx0, model.class_embedding, expanded_class_embedding), inp, 1);

    struct ggml_tensor * embeddings = ggml_add(ctx0, inp, model.position_embeddings);
    ggml_set_name(embeddings, "embeddings");

    // LOG_TEE("%d %d %d %d\n", embeddings->ne[0], embeddings->ne[1], embeddings->ne[2], embeddings->ne[3]);

    for (int il = 0; il < n_layer; il++) {
        struct ggml_tensor * cur = embeddings; // embeddings = residual, cur = hidden_states

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_1_w), model.layers[il].ln_1_b);
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
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cont_3d(ctx0, KQV, hidden_size, num_positions, batch_size);

            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].proj_w, cur), model.layers[il].proj_b);
        }
        cur = ggml_mul(ctx0, cur, model.layers[il].ls_1);

        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur;

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_2_w), model.layers[il].ln_2_b);
        }

        // mlp
        {
            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].mlp_1_w, cur), model.layers[il].mlp_1_b);
            
            if (ctx->use_gelu) {
                cur = ggml_gelu_inplace(ctx0, cur);
            } else {
                cur = ggml_gelu_quick_inplace(ctx0, cur);
            }

            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].mlp_2_w, cur), model.layers[il].mlp_2_b);
        }
        cur = ggml_mul(ctx0, cur, model.layers[il].ls_2);

        cur = ggml_add(ctx0, embeddings, cur);

        embeddings = cur;
    }

    embeddings = ggml_cont(ctx0, ggml_permute(ctx0, embeddings, 0, 2, 1, 3));

    embeddings = ggml_reshape_2d(ctx0, embeddings, batch_size * hidden_size, num_positions);

    struct ggml_tensor * slice_get_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_patches);
    ggml_set_name(slice_get_rows, "slice_get_rows");
    ggml_set_input(slice_get_rows);

    embeddings = ggml_get_rows(ctx0, embeddings, slice_get_rows);

    embeddings = ggml_reshape_3d(ctx0, embeddings, hidden_size, batch_size, num_patches);

    embeddings = ggml_cont(ctx0, ggml_permute(ctx0, embeddings, 0, 2, 1, 3));

    embeddings = ggml_reshape_4d(ctx0, embeddings, hidden_size, tgt_size, tgt_size, batch_size);
    
    // pixel shuffle
    {
        embeddings = ggml_reshape_4d(ctx0, embeddings, int(hidden_size / scale_factor), int(tgt_size * scale_factor), tgt_size, batch_size);
        embeddings = ggml_cont(ctx0, ggml_permute(ctx0, embeddings, 0, 2, 1, 3));
        embeddings = ggml_reshape_4d(ctx0, embeddings, int(hidden_size / (scale_factor * scale_factor)), int(tgt_size * scale_factor), int(tgt_size * scale_factor), batch_size);
        embeddings = ggml_cont(ctx0, ggml_permute(ctx0, embeddings, 0, 2, 1, 3));
    }

    embeddings = ggml_reshape_3d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1] * embeddings->ne[2], embeddings->ne[3]);
    
    // mlp1
    {
        embeddings = ggml_norm(ctx0, embeddings, eps);

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mlp1_ln_w), model.mlp1_ln_b);

        embeddings = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mlp1_1_w, embeddings), model.mlp1_1_b);

        if (ctx->use_gelu) {
            embeddings = ggml_gelu_inplace(ctx0, embeddings);
        } else {
            embeddings = ggml_gelu_quick_inplace(ctx0, embeddings);
        }

        embeddings = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mlp1_3_w, embeddings), model.mlp1_3_b);

        embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1] * embeddings->ne[2]);
    }
    
    ggml_build_forward_expand(gf, embeddings);

    ggml_free(ctx0);

    return gf;
}

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
            LOG_TEE("%s: model name:   %s\n", __func__, name.c_str());
        }
        LOG_TEE("%s: description:  %s\n", __func__, description.c_str());
        LOG_TEE("%s: GGUF version: %d\n", __func__, gguf_get_version(ctx));
        LOG_TEE("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx));
        LOG_TEE("%s: n_tensors:    %d\n", __func__, n_tensors);
        LOG_TEE("%s: n_kv:         %d\n", __func__, n_kv);
        LOG_TEE("%s: ftype:        %s\n", __func__, ftype_str.c_str());
        LOG_TEE("\n");
    }
    const int n_tensors = gguf_get_n_tensors(ctx);

    // kv
    const int n_kv = gguf_get_n_kv(ctx);
    LOG_TEE("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n",
        __func__, n_kv, n_tensors, fname);

    {
        std::map<enum ggml_type, uint32_t> n_type;

        for (int i = 0; i < n_tensors; i++) {
            enum ggml_type type = gguf_get_tensor_type(ctx, i);

            n_type[type]++;
        }

        LOG_TEE("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);
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

            LOG_TEE("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LOG_TEE("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
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
                LOG_TEE("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                       __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    clip_ctx * new_clip = new clip_ctx;

#ifdef GGML_USE_CUDA
    new_clip->backend = ggml_backend_cuda_init(0);
    LOG_TEE("%s: CLIP using CUDA backend\n", __func__);
#endif

#ifdef GGML_USE_METAL
    new_clip->backend = ggml_backend_metal_init();
    LOG_TEE("%s: CLIP using Metal backend\n", __func__);
#endif


    if (!new_clip->backend) {
        new_clip->backend = ggml_backend_cpu_init();
        LOG_TEE("%s: CLIP using CPU backend\n", __func__);
    }

    // model size and capabilities
    {
        new_clip->has_text_encoder = false;
        new_clip->has_vision_encoder = true;
        new_clip->has_llava_projector = false;

        // GGML_ASSERT(new_clip->has_llava_projector); // see monatis/clip.cpp for image and/or text encoding for semantic search
        GGML_ASSERT(new_clip->has_vision_encoder);
        GGML_ASSERT(!new_clip->has_text_encoder);

        new_clip->use_gelu = true;

        if (verbosity >= 1) {
            LOG_TEE("%s: text_encoder:   %d\n", __func__, new_clip->has_text_encoder);
            LOG_TEE("%s: vision_encoder: %d\n", __func__, new_clip->has_vision_encoder);
            LOG_TEE("%s: llava_projector:  %d\n", __func__, new_clip->has_llava_projector);
            LOG_TEE("%s: model size:     %.2f MB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_TEE("%s: metadata size:  %.2f MB\n", __func__, ggml_get_mem_size(meta) / 1024.0 / 1024.0);
        }
    }

    LOG_TEE("%s: params backend buffer size = % 6.2f MB (%i tensors)\n", __func__, model_size / (1024.0 * 1024.0), n_tensors);

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
            LOG_TEE("cannot open model file for loading tensors\n");
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
                LOG_TEE("%s: failed to seek for tensor %s\n", __func__, name);
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
        hparams.scale_factor   = sqrtf((float)hparams.hidden_size / hparams.n_intermediate);
        hparams.eps            = get_f32(ctx, format(KEY_LAYER_NORM_EPS, "vision"));

        int32_t image_grid[70] = {1, 1, 1, 2, 2, 1, 3, 1, 1, 3, 2, 2, 4, 1, 1, 4, 5, 1, 1, 5, 1, 6, 6, 1, 3, 2, 2, 3, 7, 1, 1, 7, 4, 2, 2, 4, 1, 8, 8, 1, 1, 9, 3, 3, 9, 1, 2, 5, 5, 2, 10, 1, 1, 10, 11, 1, 1, 11, 12, 1, 3, 4, 4, 3, 1, 12, 6, 2, 2, 6};

        for (int i = 0; i < 70; ++i) {
            hparams.image_grid_pinpoints[i] = image_grid[i];
        }
        hparams.image_grid_pinpoints[70] = 0;

        // strcpy(hparams.mm_patch_merge_type, "spatial_unpad");
        strcpy(hparams.mm_patch_merge_type, "flat");

        int idx_mean = get_key_idx(ctx, KEY_IMAGE_MEAN);
        int idx_std  = get_key_idx(ctx, KEY_IMAGE_STD);

        const float * mean_data = (const float *)gguf_get_arr_data(ctx, idx_mean);
        const float * std_data  = (const float *)gguf_get_arr_data(ctx, idx_std);

        for (int i=0; i<3; i++) {
            new_clip->image_mean[i] = mean_data[i];
            new_clip->image_std[i]  = std_data[i];
        }

        LOG_TEE("\n%s: vision model hparams\n", __func__);
        LOG_TEE("image_size         %d\n", hparams.image_size);
        LOG_TEE("patch_size         %d\n", hparams.patch_size);
        LOG_TEE("v_hidden_size      %d\n", hparams.hidden_size);
        LOG_TEE("v_n_intermediate   %d\n", hparams.n_intermediate);
        LOG_TEE("v_n_head           %d\n", hparams.n_head);
        LOG_TEE("v_n_layer          %d\n", hparams.n_layer);
        LOG_TEE("v_eps              %f\n", hparams.eps);
        LOG_TEE("v_image_mean       %f %f %f\n", new_clip->image_mean[0], new_clip->image_mean[1], new_clip->image_mean[2]);
        LOG_TEE("v_image_std        %f %f %f\n", new_clip->image_std[0], new_clip->image_std[1], new_clip->image_std[2]);
        LOG_TEE("v_image_grid_pinpoints: ");
        for (int i = 0; i < 72 && (hparams.image_grid_pinpoints[i] != 0); ++i) {
            LOG_TEE("%d ", hparams.image_grid_pinpoints[i]);
        }
        LOG_TEE("\n");
        LOG_TEE("v_mm_patch_merge_type: %s\n", hparams.mm_patch_merge_type);

        // for (auto * cur = ggml_get_first_tensor(new_clip->ctx_data); cur != NULL; cur = ggml_get_next_tensor(new_clip->ctx_data, cur)) {
        //     print_tensor_info(cur);
        // }


        try {
            vision_model.patch_embeddings_w  = get_tensor(new_clip->ctx_data, format(TN_PATCH_EMBD, "weight"));
            vision_model.patch_embeddings_b  = get_tensor(new_clip->ctx_data, format(TN_PATCH_EMBD, "bias"));
            vision_model.position_embeddings = get_tensor(new_clip->ctx_data, format(TN_POS_EMBD));
            vision_model.class_embedding = get_tensor(new_clip->ctx_data, format(TN_CLASS_EMBD));
        } catch(const std::exception& e) {
            fprintf(stderr, "%s: failed to load vision model tensors\n", __func__);
        }
        
        vision_model.layers.resize(hparams.n_layer);

        for (int il = 0; il < hparams.n_layer; ++il) {
            auto & layer = vision_model.layers[il];
            layer.q_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_IN,    il, "q", "weight"));
            layer.k_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_IN,    il, "k", "weight"));
            layer.v_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_IN,    il, "v", "weight"));
            layer.proj_w    = get_tensor(new_clip->ctx_data, format(TN_ATTN_PROJ, il, "weight"));
            layer.ln_1_w = get_tensor(new_clip->ctx_data, format(TN_LN_1,        il, "weight"));
            layer.ln_2_w = get_tensor(new_clip->ctx_data, format(TN_LN_2,        il, "weight"));
            layer.mlp_1_w = get_tensor(new_clip->ctx_data, format(TN_MLP_1,    il, "weight"));
            layer.mlp_2_w = get_tensor(new_clip->ctx_data, format(TN_MLP_2,      il, "weight"));
            layer.q_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_IN,    il, "q", "bias"));
            layer.k_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_IN,    il, "k", "bias"));
            layer.v_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_IN,    il, "v", "bias"));
            layer.proj_b    = get_tensor(new_clip->ctx_data, format(TN_ATTN_PROJ, il, "bias"));
            layer.ln_1_b = get_tensor(new_clip->ctx_data, format(TN_LN_1,        il, "bias"));
            layer.ln_2_b = get_tensor(new_clip->ctx_data, format(TN_LN_2,        il, "bias"));
            layer.mlp_1_b = get_tensor(new_clip->ctx_data, format(TN_MLP_1,    il, "bias"));
            layer.mlp_2_b = get_tensor(new_clip->ctx_data, format(TN_MLP_2,      il, "bias"));
            layer.ls_1  = get_tensor(new_clip->ctx_data, format(TN_LS_1, il));
            layer.ls_2  = get_tensor(new_clip->ctx_data, format(TN_LS_2, il));
        }

        vision_model.mlp1_ln_w  = get_tensor(new_clip->ctx_data, format(TN_MLP1_LN, "weight"));
        vision_model.mlp1_ln_b  = get_tensor(new_clip->ctx_data, format(TN_MLP1_LN, "bias"));
        vision_model.mlp1_1_w = get_tensor(new_clip->ctx_data, format(TN_MLP1_1, "weight"));
        vision_model.mlp1_1_b  = get_tensor(new_clip->ctx_data, format(TN_MLP1_1, "bias"));
        vision_model.mlp1_3_w  = get_tensor(new_clip->ctx_data, format(TN_MLP1_3, "weight"));
        vision_model.mlp1_3_b = get_tensor(new_clip->ctx_data, format(TN_MLP1_3, "bias"));
        // adaptor tensors
        // vision_model.visual_start_token = get_tensor(new_clip->ctx_data, TN_VISUAL_START_TOKEN);
        // vision_model.visual_end_token = get_tensor(new_clip->ctx_data, TN_VISUAL_END_TOKEN);
        // vision_model.text_embed_weight = get_tensor(new_clip->ctx_data, TN_TEXT_EMBED_WEIGHT);
    }

    ggml_free(meta);

    new_clip->ctx_gguf = ctx;

        // measure mem requirement and allocate
    {
        new_clip->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
        new_clip->compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(new_clip->backend));
        clip_image_f32_batch batch;
        auto & hparams = new_clip->vision_model.hparams;
        if (strcmp(hparams.mm_patch_merge_type, "flat") == 0)
            batch.size = 1;
        else
            batch.size = 13;
        ggml_cgraph * gf = clip_image_build_graph(new_clip, &batch);
        ggml_gallocr_reserve(new_clip->compute_alloc, gf);
        size_t compute_memory_buffer_size = ggml_gallocr_get_buffer_size(new_clip->compute_alloc, 0);
        LOG_TEE("%s: compute allocated memory: %.2f MB\n", __func__, compute_memory_buffer_size /1024.0/1024.0);
    }

    return new_clip;
}

struct clip_image_u8 * clip_image_u8_init() {
    return new clip_image_u8();
}

struct clip_image_f32 * clip_image_f32_init() {
    return new clip_image_f32();
}

void clip_image_u8_free(struct clip_image_u8  * img) { delete img; }
void clip_image_f32_free(struct clip_image_f32 * img) { delete img; }
void clip_image_u8_batch_free(struct clip_image_u8_batch  & batch) {
    if (batch.size > 0) {
        delete[] batch.data;
        batch.size = 0;
    }
}
void clip_image_f32_batch_free(struct clip_image_f32_batch  & batch) {
    if (batch.size > 0) {
        delete[] batch.data;
        batch.size = 0;
    }
}

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

// Linear interpolation between two points
inline float lerp(float s, float e, float t) {
    return s + (e - s) * t;
}
// Bilinear resize function
static void bilinear_resize(const clip_image_u8& src, clip_image_u8& dst, int target_width, int target_height) {
    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float x_ratio = static_cast<float>(src.nx - 1) / target_width;
    float y_ratio = static_cast<float>(src.ny - 1) / target_height;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            float px = x_ratio * x;
            float py = y_ratio * y;
            int x_floor = static_cast<int>(px);
            int y_floor = static_cast<int>(py);
            float x_lerp = px - x_floor;
            float y_lerp = py - y_floor;

            for (int c = 0; c < 3; c++) {
                float top = lerp(
                    static_cast<float>(src.buf[3 * (y_floor * src.nx + x_floor) + c]),
                    static_cast<float>(src.buf[3 * (y_floor * src.nx + (x_floor + 1)) + c]),
                    x_lerp
                );
                float bottom = lerp(
                    static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + x_floor) + c]),
                    static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + (x_floor + 1)) + c]),
                    x_lerp
                );
                dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(lerp(top, bottom, y_lerp));
            }
        }
    }
}

// Normalize image to float32 - careful with pytorch .to(model.device, dtype=torch.float16) - this sometimes reduces precision (32>16>32), sometimes not
static void normalize_image_u8_to_f32(const clip_image_u8* src, clip_image_f32* dst, const float mean[3], const float std[3]) {
    dst->nx = src->nx;
    dst->ny = src->ny;
    dst->buf.resize(src->buf.size());

    for (size_t i = 0; i < src->buf.size(); ++i) {
        int c = i % 3; // rgb
        dst->buf[i] = (static_cast<float>(src->buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

inline float clip(float x, float lower, float upper) {
    return std::max(lower, std::min(x, upper));
}

static bool bicubic_resize(const clip_image_u8 &img, clip_image_u8 &dst, int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)target_width;
    ty = (float)ny / (float)target_height;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (i = 0; i < target_height; i++) {
        for (j = 0; j < target_width; j++) {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            for (k = 0; k < 3; k++) {
                for (jj = 0; jj <= 3; jj++) {
                    d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                }
            }
        }
    }

    return true;
}

// llava-1.6 type of resize_and_pad (black)
static void resize_and_pad_image(const clip_image_u8& image, clip_image_u8 &image_output, const std::pair<int, int>& target_resolution) {
    int target_width = target_resolution.first;
    int target_height = target_resolution.second;

    float scale_w = static_cast<float>(target_width) / image.nx;
    float scale_h = static_cast<float>(target_height) / image.ny;

    int new_width, new_height;

    if (scale_w < scale_h) {
        new_width = target_width;
        new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
    } else {
        new_height = target_height;
        new_width = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
    }

    clip_image_u8 resized_image;
    // bilinear_resize(image, resized_image, new_width, new_height);
    bicubic_resize(image, resized_image, new_width, new_height);

    clip_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.resize(3 * target_width * target_height, 0); // Initialize with black

    // Calculate padding offsets
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;

    // Copy the resized image into the center of the padded buffer
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] = resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    image_output = std::move(padded_image);
}

/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static std::pair<int, int> select_best_resolution(const std::pair<int, int> & original_size, const std::vector<std::pair<int, int>> & possible_resolutions) {
    int original_width = original_size.first;
    int original_height = original_size.second;
    std::pair<int, int> best_fit;
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width = resolution.first;
        int height = resolution.second;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // fprintf(stderr, "resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

static std::pair<int, int> select_best_resolution_python(const std::pair<int, int> & original_size, const std::vector<std::pair<int, int>> & possible_resolutions) {
    float best_ratio_diff = INFINITY;
    std::pair<int, int> best_ratio;
    int original_width = original_size.first;
    int original_height = original_size.second;
    float area = (float)original_width * original_height;
    float aspect_ratio = (float)original_width / original_height;

    for (auto & resolution : possible_resolutions) {
        float target_aspect_ratio = (float)resolution.first / resolution.second;
        float ratio_diff = abs(target_aspect_ratio - aspect_ratio);
        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = resolution;
        }
        else if (ratio_diff == best_ratio_diff) {
            if (area > 0.5 * resolution.first * resolution.second) {
                best_ratio = resolution;
            }
        }
    }
    return best_ratio;

}

static std::vector<clip_image_u8*> divide_to_patches_u8(const clip_image_u8 & image, int patch_size) {
    std::vector<clip_image_u8*> patches;
    int width = image.nx;
    int height = image.ny;
    for (int i = 0; i < height; i += patch_size) {
        for (int j = 0; j < width; j += patch_size) {
            clip_image_u8 *patch = clip_image_u8_init();
            patch->nx = std::min(patch_size, width - j);
            patch->ny = std::min(patch_size, height - i);
            patch->buf.resize(3 * patch->nx * patch->ny);
            for (int y = 0; y < patch->ny; ++y) {
                for (int x = 0; x < patch->nx; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        patch->buf[3 * (y * patch->nx + x) + c] = image.buf[3 * ((i + y) * width + (j + x)) + c];
                    }
                }
            }
            patches.push_back(patch);
        }
    }
    return patches;
}

// returns the normalized float tensor for llava-1.5, for spatial_unpad with anyres processing for llava-1.6 it returns the normalized image patch tensors as a vector
// res_imgs memory is being allocated here, previous allocations will be freed if found
bool clip_image_preprocess(struct clip_ctx * ctx, const clip_image_u8 * img, clip_image_f32_batch & res_imgs) {
    bool pad_to_square = true;
    if (!ctx->has_vision_encoder) {
        LOG_TEE("This gguf file seems to have no vision encoder\n");
        return false;
    }
    auto & params = ctx->vision_model.hparams;
    // The model config actually contains all we need to decide on how to preprocess, here we automatically switch to the new llava-1.6 preprocessing
    if (strcmp(params.mm_patch_merge_type, "spatial_unpad") == 0) {
        pad_to_square = false;
    }
    // free the previous res_imgs if any set
    if (res_imgs.size > 0) {
        clip_image_f32_batch_free(res_imgs);
    }
    res_imgs.data = nullptr;
    res_imgs.size = 0;

    // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
    // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

    clip_image_u8 * temp = clip_image_u8_init(); // we will keep the input image data here temporarily
    if (pad_to_square && img->nx != img->ny) {
        params.batch_size = 1;
        res_imgs.size = 1;
        int longer_side = std::max(img->nx, img->ny);
        temp->nx = longer_side;
        temp->ny = longer_side;
        temp->buf.resize(3 * longer_side * longer_side);
        const uint8_t bc[3] = {122, 116, 104}; // background color in RGB from LLaVA (this is the mean rgb color * 255)

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
        if (params.image_grid_pinpoints[0] != 0) {
            // "spatial_unpad" with "anyres" processing for llava-1.6
            std::vector<std::pair<int, int>> possible_resolutions;
            int min_num = 1;
            int max_num = 12;
            for (int i = 0; i < 72 && params.image_grid_pinpoints[i] != 0; i+=2) {
                if (params.image_grid_pinpoints[i] * params.image_grid_pinpoints[i + 1] < min_num
                    || params.image_grid_pinpoints[i] * params.image_grid_pinpoints[i + 1] > max_num)
                    continue;
                possible_resolutions.push_back({params.image_grid_pinpoints[i] * params.image_size, params.image_grid_pinpoints[i+1] * params.image_size});
            }
            std::pair<int, int> best_resolution = select_best_resolution_python({img->nx, img->ny}, possible_resolutions);
            // std::pair<int, int> best_resolution = {4 * 448, 2 * 448};
            // clip_image_save_to_bmp(*img, "input.bmp");
            resize_and_pad_image(*img, *temp, best_resolution);  // we do not pad with mean-bg color anymore in llava-1.6
            // clip_image_save_to_bmp(*temp, "resized.bmp");
            // visually verify normalized image:
            // normalize_image_u8_to_f32(*temp, *res, ctx->image_mean, ctx->image_std);
            // {
            //     clip_image_u8 * temp2 = clip_image_u8_init();
            //     clip_image_convert_f32_to_u8(*res, *temp2);
            //     clip_image_save_to_bmp(*temp2, "resized_normalized_f32.bmp");
            //     clip_image_u8_free(temp2);
            // }

            std::vector<clip_image_u8 *> patches = divide_to_patches_u8(*temp, params.image_size); // prepare spatial sorted main patches of image_size each (336 in llava-1.6)

            clip_image_u8 *image_original_resize = clip_image_u8_init();
            // bilinear_resize(*img, *image_original_resize, params.image_size, params.image_size); // in python this is "shortest_edge", but all CLIP are square
            bicubic_resize(*img, *image_original_resize, params.image_size, params.image_size); // in python this is "shortest_edge", but all CLIP are square
            patches.push_back(image_original_resize);
            // clip_image_f32_batch_init(patches.size());
            res_imgs.size = patches.size();
            ctx->vision_model.hparams.batch_size = patches.size();
            res_imgs.data = new clip_image_f32[res_imgs.size];
            int num=0;
            for (auto& patch : patches) {
                normalize_image_u8_to_f32(patch, &res_imgs.data[num], ctx->image_mean, ctx->image_std);
                num++;
            }

            for (size_t i = 0; i < patches.size(); i++) {
                // LOG_TEE("patch %d: %d %d\n", i, patches[i]->nx, patches[i]->ny);
                clip_image_u8_free(patches[i]);
            }

            clip_image_u8_free(temp);

            return true;
        } else {
            temp->nx = img->nx;
            temp->ny = img->ny;
            temp->buf.resize(img->buf.size());
            memcpy(temp->buf.data(), img->buf.data(), temp->buf.size());
        }
    }

    const int nx = temp->nx;
    const int ny = temp->ny;
    // clip_image_save_to_bmp(*temp, "resized_vanilla.bmp");

    const int nx2 = ctx->vision_model.hparams.image_size;
    const int ny2 = ctx->vision_model.hparams.image_size;
    clip_image_f32 * res = clip_image_f32_init();
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

    // {
    //     clip_image_u8 * temp2 = clip_image_u8_init();
    //     clip_image_convert_f32_to_u8(*res, *temp2);
    //     clip_image_save_to_bmp(*temp2, "resized_normalized_f32_vanilla.bmp");
    //     clip_image_u8_free(temp2);
    // }
    // res_imgs.push_back(res);

    // res_imgs.size = 1;
    res_imgs.data = new clip_image_f32[res_imgs.size];
    res_imgs.data[0] = *res;
    clip_image_f32_free(res);

    return true;
}

void clip_free(clip_ctx * ctx) {
    ggml_free(ctx->ctx_data);
    gguf_free(ctx->ctx_gguf);

    ggml_backend_buffer_free(ctx->params_buffer);
    ggml_backend_free(ctx->backend);
    ggml_gallocr_free(ctx->compute_alloc);
    delete ctx;
}

size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    return clip_n_patches(ctx) * clip_n_mmproj_embd(ctx) * sizeof(float);
}

int32_t clip_image_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.image_size;
}

int32_t clip_patch_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.patch_size;
}

int32_t clip_hidden_size(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.hidden_size;
}

const char * clip_patch_merge_type(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.mm_patch_merge_type;
}

const int32_t * clip_image_grid(const struct clip_ctx * ctx) {
    return ctx->vision_model.hparams.image_grid_pinpoints;
}

bool clip_image_encode(struct clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    if (!ctx->has_vision_encoder) {
        LOG_TEE("This gguf file seems to have no vision encoder\n");
        return false;
    }

    clip_image_f32_batch imgs{};
    imgs.size = ctx->vision_model.hparams.batch_size;
    imgs.data = img;
    return clip_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool clip_image_batch_encode(clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs, float * vec) {
    if (!ctx->has_vision_encoder) {
        LOG_TEE("This gguf file seems to have no vision encoder\n");
        return false;
    }

    int batch_size = imgs->size;

    // build the inference graph
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs);
    ggml_gallocr_alloc_graph(ctx->compute_alloc, gf);

    // set inputs
    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size    = hparams.image_size;
    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size / patch_size) * (image_size / patch_size));
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

    // {
    //     struct ggml_tensor * embeddings = ggml_graph_get_tensor(gf, "embeddings");

    //     void* zero_mem = malloc(ggml_nbytes(embeddings));
    //     memset(zero_mem, 0, ggml_nbytes(embeddings));
    //     ggml_backend_tensor_set(embeddings, zero_mem, 0, ggml_nbytes(embeddings));
    //     free(zero_mem);
    // }

    // {
    //     struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "positions");

    //     int* positions_data = (int*)malloc(ggml_nbytes(positions));
    //     for (int i = 0; i < num_positions; i++) {
    //         positions_data[i] = i;
    //     }
    //     ggml_backend_tensor_set(positions, positions_data, 0, ggml_nbytes(positions));
    //     free(positions_data);
    // }

    // {
    //     struct ggml_tensor * patches = ggml_graph_get_tensor(gf, "patches");
    //     int* patches_data = (int*)malloc(ggml_nbytes(patches));
    //     for (int i = 0; i < num_patches; i++) {
    //         patches_data[i] = i + 1;
    //     }
    //     ggml_backend_tensor_set(patches, patches_data, 0, ggml_nbytes(patches));
    //     free(patches_data);
    // }

    std::vector<int32_t> slice_get_rows_ids(num_patches, 0);
    for (int32_t i = 0; i < num_patches; i++)
        slice_get_rows_ids[i] = i + 1;
    
    struct ggml_tensor * slice_get_rows = ggml_graph_get_tensor(gf, "slice_get_rows");
    ggml_backend_tensor_set(slice_get_rows, slice_get_rows_ids.data(), 0, ggml_nbytes(slice_get_rows));


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

// bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype) {
//     ggml_type type = GGML_TYPE_Q4_1;

//     assert(itype < GGML_TYPE_COUNT);
//     type = static_cast<ggml_type>(itype);

//     auto * ctx_clip = clip_model_load(fname_inp, 2);

//     const auto & ctx_src = ctx_clip->ctx_gguf;
//     const auto & ctx_data = ctx_clip->ctx_data;

//     auto * ctx_out = gguf_init_empty();
//     gguf_set_kv(ctx_out, ctx_src);
//     gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
//     gguf_set_val_u32(ctx_out, "general.file_type", itype);

//     auto fout = std::ofstream(fname_out, std::ios::binary);

//     const int n_tensors = gguf_get_n_tensors(ctx_src);

//     for (int i = 0; i < n_tensors; ++i) {
//         const char * name = gguf_get_tensor_name(ctx_src, i);
//         struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
//         gguf_add_tensor(ctx_out, cur);
//     }

//     const size_t meta_size = gguf_get_meta_size(ctx_out);
//     for (size_t i = 0; i < meta_size; ++i) {
//         fout.put(0);
//     }

//     // regexes of tensor names to be quantized
//     const std::vector<std::string> k_names = {
//         ".*weight",
//     };

//     std::vector<uint8_t> work(512);
//     std::vector<float> conv_buf(512);
//     std::vector<int64_t> hist_all(1 << 4, 0);
//     size_t total_size_org = 0;
//     size_t total_size_new = 0;

//     for (int i = 0; i < n_tensors; ++i) {
//         const std::string name = gguf_get_tensor_name(ctx_src, i);
//         struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name.c_str());

//         enum ggml_type new_type;
//         void * new_data;
//         size_t new_size;

//         bool quantize = false;
//         for (const auto & s : k_names) {
//             if (std::regex_match(name, std::regex(s))) {
//                 quantize = true;
//                 break;
//             }
//         }

//         // quantize only 2D tensors
//         quantize &= (ggml_n_dims(cur) == 2);

//         if (quantize) {
//             new_type = type;
//             if (new_type >= GGML_TYPE_Q2_K && name.find("embd") != std::string::npos) {
//                 new_type = GGML_TYPE_Q8_0; // ggml_get_rows needs non K type
//                 // fprintf(stderr, "%s: quantizing %s to %s\n", __func__, name.c_str(), ggml_type_name(new_type));
//             }
//             const size_t n_elms = ggml_nelements(cur);
//             float * f32_data;

//             switch (cur->type) {
//             case GGML_TYPE_F32:
//                 f32_data = (float *)cur->data;
//                 break;
//             case GGML_TYPE_F16:
//                 if (conv_buf.size() < n_elms) {
//                     conv_buf.resize(n_elms);
//                 }
//                 for (size_t j = 0; j < n_elms; ++j) {
//                     conv_buf[j] = ggml_fp16_to_fp32(((ggml_fp16_t *)cur->data)[j]);
//                 }
//                 f32_data = (float *)conv_buf.data();
//                 break;
//             default:
//                 LOG_TEE("Please use an input file in f32 or f16\n");
//                 return false;
//             }

//             if (work.size() < n_elms * 4) {
//                 work.resize(n_elms * 4);
//             }
//             new_data = work.data();

//             std::vector<int64_t> hist_cur(1 << 4, 0);

//             switch (new_type) {
//                 case GGML_TYPE_Q4_0: {
//                     new_size = ggml_quantize_q4_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
//                 } break;
//                 case GGML_TYPE_Q4_1: {
//                     new_size = ggml_quantize_q4_1(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
//                 } break;
//                 case GGML_TYPE_Q5_0: {
//                     new_size = ggml_quantize_q5_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
//                 } break;
//                 case GGML_TYPE_Q5_1: {
//                     new_size = ggml_quantize_q5_1(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
//                 } break;
//                 case GGML_TYPE_Q8_0: {
//                     new_size = ggml_quantize_q8_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
//                 } break;
//                 case GGML_TYPE_Q2_K: {
//                     new_size = ggml_quantize_q2_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
//                 } break;
//                 case GGML_TYPE_Q3_K: {
//                     new_size = ggml_quantize_q3_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
//                 } break;
//                 case GGML_TYPE_Q4_K: {
//                     new_size = ggml_quantize_q4_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
//                 } break;
//                 case GGML_TYPE_Q5_K: {
//                     new_size = ggml_quantize_q5_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
//                 } break;
//                 case GGML_TYPE_Q6_K: {
//                     new_size = ggml_quantize_q6_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
//                 } break;
//                 default: {
//                     fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, new_type);
//                     return false;
//                 }
//             }

//             for (size_t j = 0; j < hist_cur.size(); ++j) {
//                 hist_all[j] += hist_cur[j];
//             }
//         } else {
//             new_type = cur->type;
//             new_data = cur->data;
//             new_size = ggml_nbytes(cur);
//         }
//         const size_t orig_size = ggml_nbytes(cur);
//         total_size_org += orig_size;
//         total_size_new += new_size;
//         gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
//         gguf_set_tensor_data(ctx_out, name.c_str(), new_data, new_size);
//         fout.write((const char *)new_data, new_size);
//         size_t pad = GGML_PAD(new_size, gguf_get_alignment(ctx_out)) - new_size;
//         for (size_t j = 0; j < pad; ++j) {
//             fout.put(0);
//         }

//         LOG_TEE("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), ggml_n_dims(cur), quantize,
//                orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
//     }

//     // go back to beginning of file and write the updated metadata
//     fout.seekp(0, std::ios::beg);
//     std::vector<uint8_t> meta(meta_size);
//     gguf_get_meta_data(ctx_out, meta.data());
//     fout.write((const char *)meta.data(), meta_size);

//     fout.close();

//     clip_free(ctx_clip);
//     gguf_free(ctx_out);

//     {
//         LOG_TEE("%s: original  size = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
//         LOG_TEE("%s: quantized size = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);

//         int64_t sum_all = 0;
//         for (size_t i = 0; i < hist_all.size(); ++i) {
//             sum_all += hist_all[i];
//         }

//         LOG_TEE("%s: hist: ", __func__);
//         for (size_t i = 0; i < hist_all.size(); ++i) {
//             LOG_TEE("%5.3f ", hist_all[i] / (float)sum_all);
//         }
//         LOG_TEE("\n");
//     }

//     return true;
// }

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    return 2048;
}

int clip_n_patches(const struct clip_ctx * ctx) {
    auto & params = ctx->vision_model.hparams;
    int n_patches = (params.image_size / params.patch_size) * (params.image_size / params.patch_size);
    
    const int batch_size = ctx->vision_model.hparams.batch_size;
    if (batch_size > 0)
        return n_patches/4 * ctx->vision_model.hparams.batch_size;
    else if (strcmp(params.mm_patch_merge_type, "flat") == 0)
        return n_patches/4 * 1;
    else
        return n_patches/4 * 13;
}

