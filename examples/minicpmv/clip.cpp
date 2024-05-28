#include "clip.h"
#include "common.h"
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

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <cinttypes>
#include <limits>

//#define CLIP_DEBUG_FUNCTIONS

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

#define KEY_FTYPE          "general.file_type"
#define KEY_NAME           "general.name"
#define KEY_DESCRIPTION    "general.description"
#define KEY_HAS_TEXT_ENC   "clip.has_text_encoder"
#define KEY_HAS_VIS_ENC    "clip.has_vision_encoder"
#define KEY_HAS_LLAVA_PROJ "clip.has_minicpmv_projector"
#define KEY_USE_GELU       "clip.use_gelu"
#define KEY_N_EMBD         "clip.%s.embedding_length"
#define KEY_N_FF           "clip.%s.feed_forward_length"
#define KEY_N_BLOCK        "clip.%s.block_count"
#define KEY_N_HEAD         "clip.%s.attention.head_count"
#define KEY_LAYER_NORM_EPS "clip.%s.attention.layer_norm_epsilon"
#define KEY_PROJ_DIM       "clip.%s.projection_dim"
#define KEY_TOKENS         "tokenizer.ggml.tokens"
#define KEY_N_POSITIONS    "clip.text.context_length"
#define KEY_IMAGE_SIZE     "clip.vision.image_size"
#define KEY_PATCH_SIZE     "clip.vision.patch_size"
#define KEY_IMAGE_MEAN     "clip.vision.image_mean"
#define KEY_IMAGE_STD      "clip.vision.image_std"
#define KEY_PROJ_TYPE      "clip.projector_type"

#define KEY_MM_PATCH_MERGE_TYPE   "clip.vision.mm_patch_merge_type"
#define KEY_IMAGE_GRID_PINPOINTS  "clip.vision.image_grid_pinpoints"
#define KEY_IMAGE_CROP_RESOLUTION "clip.vision.image_crop_resolution"


//
// tensor name constants
//

#define TN_TOKEN_EMBD      "%s.token_embd.weight"
// #define TN_POS_EMBD        "%s.position_embd"
// // #define TN_CLASS_EMBD      "v.class_embd"
// #define TN_PATCH_EMBD      "v.patch_embd.proj.%s"
#define TN_POS_EMBD        "%s.position_embd.weight"
// #define TN_CLASS_EMBD      "v.class_embd"
#define TN_PATCH_EMBD      "v.patch_embd.%s"
#define TN_ATTN_K          "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q          "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V          "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT     "%s.blk.%d.attn_out.%s"
#define TN_FFN_DOWN        "%s.blk.%d.ffn_down.%s"
#define TN_FFN_UP          "%s.blk.%d.ffn_up.%s"
#define TN_LN_1            "%s.blk.%d.ln1.%s"
#define TN_LN_2            "%s.blk.%d.ln2.%s"
// #define TN_LN_PRE          "%s.pre_ln.%s"
#define TN_LN_POST         "%s.post_ln.%s"
#define TN_TEXT_PROJ       "text_projection.weight"
#define TN_VIS_PROJ        "visual_projection.weight"
#define TN_LLAVA_PROJ      "mm.%d.%s"
#define TN_MVLM_PROJ_MLP   "mm.model.mlp.%d.%s"
#define TN_MVLM_PROJ_BLOCK "mm.model.mb_block.%d.block.%d.%s"
#define TN_MVLM_PROJ_PEG   "mm.model.peg.%d.%s"
#define TN_IMAGE_NEWLINE   "model.image_newline"
// MINICPMV
// #define TN_MINICPMV_POS_EMBD "resampler.pos_embed"
#define TN_MINICPMV_POS_EMBD_K "resampler.pos_embed_k"
#define TN_MINICPMV_QUERY "resampler.query"
#define TN_MINICPMV_PROJ "resampler.proj.weight"
#define TN_MINICPMV_KV_PROJ "resampler.kv.weight"
#define TN_MINICPMV_ATTN "resampler.attn.%s.%s"
#define TN_MINICPMV_LN "resampler.ln_%s.%s"


enum projector_type {
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_MLP_NORM,
    PROJECTOR_TYPE_LDP,
    PROJECTOR_TYPE_LDPV2,
    PROJECTOR_TYPE_RESAMPLER,
    PROJECTOR_TYPE_UNKNOWN,
};

static std::map<projector_type, std::string> PROJECTOR_TYPE_NAMES = {
    { PROJECTOR_TYPE_MLP, "mlp" },
    { PROJECTOR_TYPE_LDP, "ldp" },
    { PROJECTOR_TYPE_LDPV2, "ldpv2"},
    { PROJECTOR_TYPE_RESAMPLER, "resampler"},
};


//
// utilities to get data from a gguf file
//

static int get_key_idx(const gguf_context * ctx, const char * key) {
    int i = gguf_find_key(ctx, key);
    if (i == -1) {
        LOG_TEE("key %s not found in file\n", key);
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

static void print_tensor_info(const ggml_tensor * tensor, const char * prefix = "") {
    size_t tensor_size = ggml_nbytes(tensor);
    LOG_TEE("%s: n_dims = %d, name = %s, tensor_size=%zu, shape:[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], type = %s\n",
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

#ifdef CLIP_DEBUG_FUNCTIONS
static void clip_image_write_image_to_ppm(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_TEE("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    // PPM header: P6 format, width, height, and max color value
    file << "P6\n" << img.nx << " " << img.ny << "\n255\n";

    // Write pixel data
    for (size_t i = 0; i < img.buf.size(); i += 3) {
        // PPM expects binary data in RGB format, which matches our image buffer
        file.write(reinterpret_cast<const char*>(&img.buf[i]), 3);
    }

    file.close();
}

static void clip_image_save_to_bmp(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_TEE("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    int fileSize = 54 + 3 * img.nx * img.ny; // File header + info header + pixel data
    int bytesPerPixel = 3;
    int widthInBytes = img.nx * bytesPerPixel;
    int paddingAmount = (4 - (widthInBytes % 4)) % 4;
    int stride = widthInBytes + paddingAmount;

    // Bitmap file header
    unsigned char fileHeader[14] = {
        'B','M',     // Signature
        0,0,0,0,    // Image file size in bytes
        0,0,0,0,    // Reserved
        54,0,0,0    // Start of pixel array
    };

    // Total file size
    fileSize = 54 + (stride * img.ny);
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);

    // Bitmap information header (BITMAPINFOHEADER)
    unsigned char infoHeader[40] = {
        40,0,0,0,   // Size of this header (40 bytes)
        0,0,0,0,    // Image width
        0,0,0,0,    // Image height
        1,0,        // Number of color planes
        24,0,       // Bits per pixel
        0,0,0,0,    // No compression
        0,0,0,0,    // Image size (can be 0 for no compression)
        0,0,0,0,    // X pixels per meter (not specified)
        0,0,0,0,    // Y pixels per meter (not specified)
        0,0,0,0,    // Total colors (color table not used)
        0,0,0,0     // Important colors (all are important)
    };

    // Width and height in the information header
    infoHeader[4] = (unsigned char)(img.nx);
    infoHeader[5] = (unsigned char)(img.nx >> 8);
    infoHeader[6] = (unsigned char)(img.nx >> 16);
    infoHeader[7] = (unsigned char)(img.nx >> 24);
    infoHeader[8] = (unsigned char)(img.ny);
    infoHeader[9] = (unsigned char)(img.ny >> 8);
    infoHeader[10] = (unsigned char)(img.ny >> 16);
    infoHeader[11] = (unsigned char)(img.ny >> 24);

    // Write file headers
    file.write(reinterpret_cast<char*>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char*>(infoHeader), sizeof(infoHeader));

    // Pixel data
    std::vector<unsigned char> padding(3, 0); // Max padding size to be added to each row
    for (int y = img.ny - 1; y >= 0; --y) { // BMP files are stored bottom-to-top
        for (int x = 0; x < img.nx; ++x) {
            // Each pixel
            size_t pixelIndex = (y * img.nx + x) * 3;
            unsigned char pixel[3] = {
                img.buf[pixelIndex + 2], // BMP stores pixels in BGR format
                img.buf[pixelIndex + 1],
                img.buf[pixelIndex]
            };
            file.write(reinterpret_cast<char*>(pixel), 3);
        }
        // Write padding for the row
        file.write(reinterpret_cast<char*>(padding.data()), paddingAmount);
    }

    file.close();
}

// debug function to convert f32 to u8
static void clip_image_convert_f32_to_u8(const clip_image_f32& src, clip_image_u8& dst) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(3 * src.nx * src.ny);
    for (size_t i = 0; i < src.buf.size(); ++i) {
        dst.buf[i] = static_cast<uint8_t>(std::min(std::max(int(src.buf[i] * 255.0f), 0), 255));
    }
}
#endif


//
// clip layers
//

struct clip_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;

    float eps;

    char mm_patch_merge_type[32] = "flat"; // spatial_unpad or flat (default)

    int32_t image_grid_pinpoints[32];
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
    struct clip_hparams hparams;

    // embeddings
    // struct ggml_tensor * class_embedding;
    struct ggml_tensor * patch_embeddings_w;
    struct ggml_tensor * patch_embeddings_b;
    struct ggml_tensor * position_embeddings;

    // struct ggml_tensor * pre_ln_w;
    // struct ggml_tensor * pre_ln_b;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;

    // LLaVA projection
    struct ggml_tensor * mm_0_w = NULL;
    struct ggml_tensor * mm_0_b = NULL;
    struct ggml_tensor * mm_2_w = NULL;
    struct ggml_tensor * mm_2_b = NULL;

    struct ggml_tensor * image_newline = NULL;

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

    // MobileVLM_V2 projection
    struct ggml_tensor * mm_model_mlp_0_w;
    struct ggml_tensor * mm_model_mlp_0_b;
    struct ggml_tensor * mm_model_mlp_2_w;
    struct ggml_tensor * mm_model_mlp_2_b;
    struct ggml_tensor * mm_model_peg_0_w;
    struct ggml_tensor * mm_model_peg_0_b;

    // MINICPMV projection
    // struct ggml_tensor * mm_model_pos_embed;
    struct ggml_tensor * mm_model_pos_embed_k;
    struct ggml_tensor * mm_model_query;
    struct ggml_tensor * mm_model_proj;
    struct ggml_tensor * mm_model_kv_proj;
    struct ggml_tensor * mm_model_attn_q_w;
    struct ggml_tensor * mm_model_attn_q_b;
    struct ggml_tensor * mm_model_attn_k_w;
    struct ggml_tensor * mm_model_attn_k_b;
    struct ggml_tensor * mm_model_attn_v_w;
    struct ggml_tensor * mm_model_attn_v_b;
    struct ggml_tensor * mm_model_attn_o_w;
    struct ggml_tensor * mm_model_attn_o_b;
    struct ggml_tensor * mm_model_ln_q_w;
    struct ggml_tensor * mm_model_ln_q_b;
    struct ggml_tensor * mm_model_ln_kv_w;
    struct ggml_tensor * mm_model_ln_kv_b;
    struct ggml_tensor * mm_model_ln_post_w;
    struct ggml_tensor * mm_model_ln_post_b;
};

struct clip_ctx {
    bool has_text_encoder    = false;
    bool has_vision_encoder  = false;
    bool has_minicpmv_projector = false;

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
    ggml_backend_buffer_t params_buffer  = NULL;

    ggml_backend_t backend       = NULL;
    ggml_gallocr_t compute_alloc = NULL;
};

static ggml_cgraph * clip_image_build_graph(clip_ctx * ctx, const clip_image_f32_batch * imgs, std::pair<int, int> load_image_size = {448, 448}) {
    if (!ctx->has_vision_encoder) {
        LOG_TEE("This gguf file seems to have no vision encoder\n");
        return nullptr;
    }
    
    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size_width     = load_image_size.first;
    const int image_size_height    = load_image_size.second;
    const int patch_size           = hparams.patch_size;
    const int num_patches          = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int num_positions        = num_patches;
    const int hidden_size          = hparams.hidden_size;
    const int n_head               = hparams.n_head;
    const int d_head               = hidden_size / n_head;
    const int n_layer              = hparams.n_layer;
    const float eps                = hparams.eps;

    const int batch_size = imgs->size;

    if (ctx->has_minicpmv_projector) {
        GGML_ASSERT(batch_size == 1);
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    
    LOG_TEE("%s: load_image_size: %d %d\n", __func__, load_image_size.first, load_image_size.second);
    struct ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size_width, image_size_height, 3, batch_size);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_w, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));
    inp = ggml_add(ctx0, inp, model.patch_embeddings_b);

    // concat class_embeddings and patch_embeddings
    // struct ggml_tensor * embeddings = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, batch_size);
    // ggml_set_name(embeddings, "embeddings");
    // ggml_set_input(embeddings);

    // embeddings = ggml_acc(ctx0, embeddings, model.class_embedding,
    //         embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], 0);

    // embeddings = ggml_acc(ctx0, embeddings, inp,
    //         embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], model.class_embedding->nb[1]);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_positions);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    struct ggml_tensor * embeddings =
        ggml_add(ctx0, inp, ggml_get_rows(ctx0, model.position_embeddings, positions));
    
    int pos_w = image_size_width/patch_size;
    int pos_h = image_size_height/patch_size;
    struct ggml_tensor * pos_embed = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 4096, pos_w * pos_h, 1);
    ggml_set_name(pos_embed, "pos_embed");
    ggml_set_input(pos_embed);

    // // pre-layernorm
    // {
        // embeddings = ggml_norm(ctx0, embeddings, eps);
        // ggml_set_name(embeddings, "pre_ln");

        // embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.pre_ln_w), model.pre_ln_b);
    // }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
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
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cont_3d(ctx0, KQV, hidden_size, num_positions, batch_size);
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

    { // post layernorm
        embeddings = ggml_norm(ctx0, embeddings, eps);
        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.post_ln_w), model.post_ln_b);
    }

    // llava projector
    {
        // embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);

        // struct ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 64);
        // ggml_set_name(patches, "patches");
        // ggml_set_input(patches);

        // shape [1, 576, 1024]
        // ne is whcn, ne = [1024, 576, 1, 1]
        // embeddings = ggml_get_rows(ctx0, embeddings, patches);

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
        else if (ctx->proj_type == PROJECTOR_TYPE_LDPV2)
        {
            int n_patch = 24;
            struct ggml_tensor * mlp_0 = ggml_mul_mat(ctx0, model.mm_model_mlp_0_w, embeddings);
            mlp_0 = ggml_add(ctx0, mlp_0, model.mm_model_mlp_0_b);
            mlp_0 = ggml_gelu(ctx0, mlp_0);
            struct ggml_tensor * mlp_2 = ggml_mul_mat(ctx0, model.mm_model_mlp_2_w, mlp_0);
            mlp_2 = ggml_add(ctx0, mlp_2, model.mm_model_mlp_2_b);
            // mlp_2 ne = [2048, 576, 1, 1]
            // // AVG Pool Layer 2*2, strides = 2
            mlp_2 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 0, 2, 3));
            // mlp_2 ne = [576, 2048, 1, 1]
            mlp_2 = ggml_reshape_4d(ctx0, mlp_2, n_patch, n_patch, mlp_2->ne[1], mlp_2->ne[2]);
            // mlp_2 ne [24, 24, 2048, 1]
            mlp_2 = ggml_pool_2d(ctx0, mlp_2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
            // weight ne = [3, 3, 2048, 1]
            struct ggml_tensor * peg_0 = ggml_conv_depthwise_2d(ctx0, model.mm_model_peg_0_w, mlp_2, 1, 1, 1, 1, 1, 1);
            peg_0 = ggml_cont(ctx0, ggml_permute(ctx0, peg_0, 1, 2, 0, 3));
            peg_0 = ggml_add(ctx0, peg_0, model.mm_model_peg_0_b);
            mlp_2 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 2, 0, 3));
            peg_0 = ggml_add(ctx0, peg_0, mlp_2);
            peg_0 = ggml_reshape_3d(ctx0, peg_0, peg_0->ne[0], peg_0->ne[1] * peg_0->ne[2], peg_0->ne[3]);
            embeddings = peg_0;
        }
        else if (ctx->proj_type == PROJECTOR_TYPE_RESAMPLER) {
            struct ggml_tensor * q = model.mm_model_query;
            { // layernorm
                q = ggml_norm(ctx0, q, eps);
                q = ggml_add(ctx0, ggml_mul(ctx0, q, model.mm_model_ln_q_w), model.mm_model_ln_q_b);
            }
            struct ggml_tensor *k, *v = ggml_mul_mat(ctx0, model.mm_model_kv_proj, embeddings);
            { // layernorm
                v = ggml_norm(ctx0, v, eps);
                v = ggml_add(ctx0, ggml_mul(ctx0, v, model.mm_model_ln_kv_w), model.mm_model_ln_kv_b);
            }
            { // position
                // q = ggml_add(ctx0, q, model.mm_model_pos_embed);
                k = ggml_add(ctx0, v, pos_embed);
            }

            { // attention
                const int hidden_size = 4096;
                const int d_head = 128;
                const int n_head = hidden_size/d_head;
                const int num_query = 96;

                struct ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_q_w, q), model.mm_model_attn_q_b);
                Q = ggml_scale_inplace(ctx0, Q, 1.0f / sqrt((float)d_head));
                struct ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_k_w, k), model.mm_model_attn_k_b);
                struct ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_v_w, v), model.mm_model_attn_v_b);
                // permute
                Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_query, batch_size);
                Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
                Q = ggml_reshape_3d(ctx0, Q, d_head, num_query, n_head * batch_size);
                K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
                K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
                K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);
                V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, batch_size);
                V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
                V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head * batch_size);
                struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
                KQ = ggml_soft_max_inplace(ctx0, KQ);
                struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
                KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_query, n_head, batch_size);
                KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
                KQV = ggml_cont_3d(ctx0, KQV, hidden_size, num_query, batch_size);

                embeddings = ggml_add(ctx0, ggml_mul_mat(ctx0, model.mm_model_attn_o_w, KQV), model.mm_model_attn_o_b);
            }
            { // layernorm
                embeddings = ggml_norm(ctx0, embeddings, eps);
                embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_model_ln_post_w), model.mm_model_ln_post_b);
            }
            embeddings = ggml_mul_mat(ctx0, model.mm_model_proj, embeddings);
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
struct clip_ctx * clip_model_load(const char * fname, const int verbosity = 1, std::pair<int, int> load_image_size = {448, 448}) {
    struct ggml_context * meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
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

    // update projector type
    {
        int idx = gguf_find_key(ctx, KEY_PROJ_TYPE);
        if (idx != -1) {
            const std::string proj_type = gguf_get_val_str(ctx, idx);
            new_clip->proj_type = clip_projector_type_from_string(proj_type);
        } else {
            new_clip->proj_type = PROJECTOR_TYPE_MLP;
        }

        if (new_clip->proj_type == PROJECTOR_TYPE_MLP) {
            if (gguf_find_tensor(ctx, format(TN_LLAVA_PROJ, 3, "weight").c_str()) != -1) {
                new_clip->proj_type = PROJECTOR_TYPE_MLP_NORM;
            }
        }
    }

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
        int idx = get_key_idx(ctx, KEY_HAS_TEXT_ENC);
        new_clip->has_text_encoder = gguf_get_val_bool(ctx, idx);

        idx = get_key_idx(ctx, KEY_HAS_VIS_ENC);
        new_clip->has_vision_encoder = gguf_get_val_bool(ctx, idx);

        idx = gguf_find_key(ctx, KEY_HAS_LLAVA_PROJ);
        if (idx != -1) {
            new_clip->has_minicpmv_projector = gguf_get_val_bool(ctx, idx);
        }

        GGML_ASSERT(new_clip->has_minicpmv_projector); // see monatis/clip.cpp for image and/or text encoding for semantic search
        GGML_ASSERT(new_clip->has_vision_encoder);
        GGML_ASSERT(!new_clip->has_text_encoder);

        idx = get_key_idx(ctx, KEY_USE_GELU);
        new_clip->use_gelu = gguf_get_val_bool(ctx, idx);

        if (verbosity >= 1) {
            LOG_TEE("%s: text_encoder:   %d\n", __func__, new_clip->has_text_encoder);
            LOG_TEE("%s: vision_encoder: %d\n", __func__, new_clip->has_vision_encoder);
            LOG_TEE("%s: llava_projector:  %d\n", __func__, new_clip->has_minicpmv_projector);
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
            LOG_TEE("%s: ggml_init() failed\n", __func__);
            clip_free(new_clip);
            gguf_free(ctx);
            return nullptr;
        }

        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            LOG_TEE("cannot open model file for loading tensors\n");
            clip_free(new_clip);
            gguf_free(ctx);
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
                gguf_free(ctx);
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

        try {
            int idx = get_key_idx(ctx, KEY_IMAGE_GRID_PINPOINTS);
            int n = gguf_get_arr_n(ctx, idx);
            const int32_t * pinpoints = (const int32_t *)gguf_get_arr_data(ctx, idx);
            for (int i = 0; i < 32 && i < n && pinpoints[i] != 0; ++i) {
                hparams.image_grid_pinpoints[i] = pinpoints[i];
            }
            if (n < 32)
                hparams.image_grid_pinpoints[n] = 0;
        } catch (std::runtime_error & e) {
            hparams.image_grid_pinpoints[0]=0;
        }

        try {
            int idx = get_key_idx(ctx, KEY_MM_PATCH_MERGE_TYPE);
            strcpy(hparams.mm_patch_merge_type, gguf_get_val_str(ctx, idx));
        } catch (std::runtime_error & e) {
            strcpy(hparams.mm_patch_merge_type, "flat");
        }

        try {
            hparams.image_crop_resolution = get_u32(ctx, KEY_IMAGE_CROP_RESOLUTION); // llava-1.6
        } catch(const std::exception& e) {
            hparams.image_crop_resolution = hparams.image_size;
        }

        int idx_mean = get_key_idx(ctx, KEY_IMAGE_MEAN);
        int idx_std  = get_key_idx(ctx, KEY_IMAGE_STD);

        const float * mean_data = (const float *)gguf_get_arr_data(ctx, idx_mean);
        const float * std_data  = (const float *)gguf_get_arr_data(ctx, idx_std);

        for (int i = 0; i < 3; ++i) {
            new_clip->image_mean[i] = mean_data[i];
            new_clip->image_std[i]  = std_data[i];
        }

        if (verbosity >= 2) {
            LOG_TEE("\n%s: vision model hparams\n", __func__);
            LOG_TEE("image_size         %d\n", hparams.image_size);
            LOG_TEE("patch_size         %d\n", hparams.patch_size);
            LOG_TEE("v_hidden_size      %d\n", hparams.hidden_size);
            LOG_TEE("v_n_intermediate   %d\n", hparams.n_intermediate);
            LOG_TEE("v_projection_dim   %d\n", hparams.projection_dim);
            LOG_TEE("v_n_head           %d\n", hparams.n_head);
            LOG_TEE("v_n_layer          %d\n", hparams.n_layer);
            LOG_TEE("v_eps              %f\n", hparams.eps);
            LOG_TEE("v_image_mean       %f %f %f\n", new_clip->image_mean[0], new_clip->image_mean[1], new_clip->image_mean[2]);
            LOG_TEE("v_image_std        %f %f %f\n", new_clip->image_std[0], new_clip->image_std[1], new_clip->image_std[2]);
            LOG_TEE("v_image_grid_pinpoints: ");
            for (int i = 0; i < 32 && (hparams.image_grid_pinpoints[i] != 0); ++i) {
                LOG_TEE("%d ", hparams.image_grid_pinpoints[i]);
            }
            LOG_TEE("\n");
            LOG_TEE("v_mm_patch_merge_type: %s\n", hparams.mm_patch_merge_type);

        }

        try {
            vision_model.patch_embeddings_w    = get_tensor(new_clip->ctx_data, format(TN_PATCH_EMBD, "weight"));
            vision_model.patch_embeddings_b    = get_tensor(new_clip->ctx_data, format(TN_PATCH_EMBD, "bias"));
            // vision_model.class_embedding     = get_tensor(new_clip->ctx_data, TN_CLASS_EMBD);
            vision_model.position_embeddings = get_tensor(new_clip->ctx_data, format(TN_POS_EMBD, "v"));
            // vision_model.pre_ln_w            = get_tensor(new_clip->ctx_data, format(TN_LN_PRE, "v", "weight"));
            // vision_model.pre_ln_b            = get_tensor(new_clip->ctx_data, format(TN_LN_PRE, "v", "bias"));
            vision_model.post_ln_w           = get_tensor(new_clip->ctx_data, format(TN_LN_POST, "v", "weight"));
            vision_model.post_ln_b           = get_tensor(new_clip->ctx_data, format(TN_LN_POST, "v", "bias"));
        } catch(const std::exception& e) {
            LOG_TEE("%s: failed to load vision model tensors\n", __func__);
        }

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
            try {
                vision_model.image_newline = get_tensor(new_clip->ctx_data, TN_IMAGE_NEWLINE);
                // LOG_TEE("%s: image_newline tensor (llava-1.6) found\n", __func__);
            } catch (std::runtime_error & e) {  }
        } else if (new_clip->proj_type == PROJECTOR_TYPE_LDP) {
            // MobileVLM projection
            vision_model.mm_model_mlp_1_w               = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 1, "weight"));
            vision_model.mm_model_mlp_1_b               = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 1, "bias"));
            vision_model.mm_model_mlp_3_w               = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 3, "weight"));
            vision_model.mm_model_mlp_3_b               = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 3, "bias"));
            vision_model.mm_model_block_1_block_0_0_w   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 0, "0.weight"));
            vision_model.mm_model_block_1_block_0_1_w   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.weight"));
            vision_model.mm_model_block_1_block_0_1_b   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.bias"));
            vision_model.mm_model_block_1_block_1_fc1_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.weight"));
            vision_model.mm_model_block_1_block_1_fc1_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.bias"));
            vision_model.mm_model_block_1_block_1_fc2_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.weight"));
            vision_model.mm_model_block_1_block_1_fc2_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.bias"));
            vision_model.mm_model_block_1_block_2_0_w   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 2, "0.weight"));
            vision_model.mm_model_block_1_block_2_1_w   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.weight"));
            vision_model.mm_model_block_1_block_2_1_b   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.bias"));
            vision_model.mm_model_block_2_block_0_0_w   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 0, "0.weight"));
            vision_model.mm_model_block_2_block_0_1_w   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.weight"));
            vision_model.mm_model_block_2_block_0_1_b   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.bias"));
            vision_model.mm_model_block_2_block_1_fc1_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.weight"));
            vision_model.mm_model_block_2_block_1_fc1_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.bias"));
            vision_model.mm_model_block_2_block_1_fc2_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.weight"));
            vision_model.mm_model_block_2_block_1_fc2_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.bias"));
            vision_model.mm_model_block_2_block_2_0_w   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 2, "0.weight"));
            vision_model.mm_model_block_2_block_2_1_w   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.weight"));
            vision_model.mm_model_block_2_block_2_1_b   = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.bias"));
        }
        else if (new_clip->proj_type == PROJECTOR_TYPE_LDPV2)
        {
            // MobilVLM_V2 projection
            vision_model.mm_model_mlp_0_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 0, "weight"));
            vision_model.mm_model_mlp_0_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 0, "bias"));
            vision_model.mm_model_mlp_2_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 2, "weight"));
            vision_model.mm_model_mlp_2_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_MLP, 2, "bias"));
            vision_model.mm_model_peg_0_w = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_PEG, 0, "weight"));
            vision_model.mm_model_peg_0_b = get_tensor(new_clip->ctx_data, format(TN_MVLM_PROJ_PEG, 0, "bias"));
        }
        else if (new_clip->proj_type == PROJECTOR_TYPE_RESAMPLER) {
            // vision_model.mm_model_pos_embed = get_tensor(new_clip->ctx_data, TN_MINICPMV_POS_EMBD);
            vision_model.mm_model_pos_embed_k = get_tensor(new_clip->ctx_data, TN_MINICPMV_POS_EMBD_K);
            vision_model.mm_model_query = get_tensor(new_clip->ctx_data, TN_MINICPMV_QUERY);
            vision_model.mm_model_proj = get_tensor(new_clip->ctx_data, TN_MINICPMV_PROJ);
            vision_model.mm_model_kv_proj = get_tensor(new_clip->ctx_data, TN_MINICPMV_KV_PROJ);
            vision_model.mm_model_attn_q_w = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_ATTN, "q", "weight"));
            vision_model.mm_model_attn_k_w = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_ATTN, "k", "weight"));
            vision_model.mm_model_attn_v_w = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_ATTN, "v", "weight"));
            vision_model.mm_model_attn_q_b = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_ATTN, "q", "bias"));
            vision_model.mm_model_attn_k_b = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_ATTN, "k", "bias"));
            vision_model.mm_model_attn_v_b = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_ATTN, "v", "bias"));
            vision_model.mm_model_attn_o_w = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_ATTN, "out", "weight"));
            vision_model.mm_model_attn_o_b = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_ATTN, "out", "bias"));
            vision_model.mm_model_ln_q_w = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_LN, "q", "weight"));
            vision_model.mm_model_ln_q_b = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_LN, "q", "bias"));
            vision_model.mm_model_ln_kv_w = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_LN, "kv", "weight"));
            vision_model.mm_model_ln_kv_b = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_LN, "kv", "bias"));
            vision_model.mm_model_ln_post_w = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_LN, "post", "weight"));
            vision_model.mm_model_ln_post_b = get_tensor(new_clip->ctx_data, format(TN_MINICPMV_LN, "post", "bias"));
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
        // todo
        new_clip->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
        new_clip->compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(new_clip->backend));
        clip_image_f32_batch batch;
        batch.size = 1;
        ggml_cgraph * gf = clip_image_build_graph(new_clip, &batch, load_image_size);
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
void clip_image_u8_batch_free(struct clip_image_u8_batch  * batch) {
    if (batch->size > 0) {
        delete[] batch->data;
        batch->size = 0;
    }
}
void clip_image_f32_batch_free(struct clip_image_f32_batch  * batch) {
    if (batch->size > 0) {
        delete[] batch->data;
        batch->size = 0;
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
        LOG_TEE("%s: failed to load image '%s'\n", __func__, fname);
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
        LOG_TEE("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

void normalize_image_u8_to_f32(struct clip_ctx * ctx, const clip_image_u8* src, clip_image_f32* dst) {
    dst->nx = src->nx;
    dst->ny = src->ny;
    dst->buf.resize(src->buf.size());
    const auto & m3 = ctx->image_mean;
    const auto & s3 = ctx->image_std;

    for (size_t i = 0; i < src->buf.size(); ++i) {
        int c = i % 3; // rgb
        dst->buf[i] = (static_cast<float>(src->buf[i]) / 255.0f - m3[c]) / s3[c];
    }
}

ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx) {
    return ctx->vision_model.image_newline;
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

int clip_n_patches(const struct clip_ctx * ctx) {
    const auto & params = ctx->vision_model.hparams;

    int n_patches = (params.image_size / params.patch_size) * (params.image_size / params.patch_size);

    if (ctx->proj_type == PROJECTOR_TYPE_LDP || ctx->proj_type == PROJECTOR_TYPE_LDPV2) {
        n_patches /= 4;
    } else if (ctx->proj_type == PROJECTOR_TYPE_RESAMPLER) {
        n_patches = 96;
    }

    return n_patches;
}

static std::vector<std::vector<std::vector<float>>> get_1d_sincos_pos_embed_from_grid_new(int embed_dim, const std::vector<std::vector<float>>& pos) {
    assert(embed_dim % 2 == 0);
    int H = pos.size();
    int W = pos[0].size();

    std::vector<float> omega(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; ++i) {
        omega[i] = 1.0 / pow(10000.0, static_cast<float>(i) / (embed_dim / 2));
    }

    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                float out_value = pos[h][w] * omega[d];
                emb[h][w][d] = sin(out_value);
                emb[h][w][d + embed_dim / 2] = cos(out_value);
            }
        }
    }

    return emb;
}

static std::vector<std::vector<std::vector<float>>> get_2d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<std::vector<std::vector<float>>>& grid) {
    assert(embed_dim % 2 == 0);
    std::vector<std::vector<std::vector<float>>> emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[0]); // (H, W, D/2)
    std::vector<std::vector<std::vector<float>>> emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[1]); // (H, W, D/2)

    int H = emb_h.size();
    int W = emb_h[0].size();
    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                emb[h][w][d] = emb_h[h][w][d];
                emb[h][w][d + embed_dim / 2] = emb_w[h][w][d];
            }
        }
    }
    return emb;
}

static std::vector<std::vector<float>> get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int> image_size) {
    int grid_h_size = image_size.first;
    int grid_w_size = image_size.second;

    std::vector<float> grid_h(grid_h_size);
    std::vector<float> grid_w(grid_w_size);

    for (int i = 0; i < grid_h_size; ++i) {
        grid_h[i] = static_cast<float>(i);
    }
    for (int i = 0; i < grid_w_size; ++i) {
        grid_w[i] = static_cast<float>(i);
    }

    std::vector<std::vector<float>> grid(grid_h_size, std::vector<float>(grid_w_size));
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid[h][w] = grid_w[w];
        }
    }
    std::vector<std::vector<std::vector<float>>> grid_2d = {grid, grid};
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid_2d[0][h][w] = grid_h[h];
            grid_2d[1][h][w] = grid_w[w];
        }
    }

    std::vector<std::vector<std::vector<float>>> pos_embed_3d = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_2d);

    int H = image_size.first;
    int W = image_size.second;
    std::vector<std::vector<float>> pos_embed_2d(H * W, std::vector<float>(embed_dim));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            pos_embed_2d[w * H + h] = pos_embed_3d[h][w];
        }
    }

    return pos_embed_2d;
}

bool clip_image_encode(struct clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec, std::pair<int, int> load_image_size = {448, 448}) {
    if (!ctx->has_vision_encoder) {
        LOG_TEE("This gguf file seems to have no vision encoder\n");
        return false;
    }

    clip_image_f32_batch imgs{};
    imgs.size = 1;
    imgs.data = img;
    return clip_image_batch_encode(ctx, n_threads, &imgs, vec, load_image_size);
}

bool clip_image_batch_encode(clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs, float * vec, std::pair<int, int> load_image_size = {448, 448}) {
    if (!ctx->has_vision_encoder) {
        LOG_TEE("This gguf file seems to have no vision encoder\n");
        return false;
    }

    int batch_size = imgs->size;
    if (ctx->has_minicpmv_projector) {
        GGML_ASSERT(batch_size == 1); // TODO: support multiple images
    }

    // build the inference graph
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs, load_image_size);
    ggml_gallocr_alloc_graph(ctx->compute_alloc, gf);

    // set inputs
    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size_width     = load_image_size.first;
    const int image_size_height    = load_image_size.second;
    const int patch_size           = hparams.patch_size;
    const int num_patches          = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int num_positions        = num_patches;

    {
        struct ggml_tensor * inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        float * data = (float *)malloc(ggml_nbytes(inp_raw));

        for (size_t i = 0; i < imgs->size; i++) {
            const int nx = imgs->data[i].nx;
            const int ny = imgs->data[i].ny;
            // GGML_ASSERT(nx == image_size && ny == image_size);

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
        // inspired from siglip:
        //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
        //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit/blob/d66538faeba44480d0bfaa42145eef26f9423199/modeling_siglip.py#L316
        struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "positions");

        int* positions_data = (int*)malloc(ggml_nbytes(positions));
        for (int i = 0; i < num_positions; i++) {
            positions_data[i] = std::floor(70.0*i/num_positions);
        }
        ggml_backend_tensor_set(positions, positions_data, 0, ggml_nbytes(positions));
        free(positions_data);
    }

    {
        // inspired from resampler of Qwen-VL:
        //    -> https://huggingface.co/Qwen/Qwen-VL/tree/main
        //    -> https://huggingface.co/Qwen/Qwen-VL/blob/0547ed36a86561e2e42fecec8fd0c4f6953e33c4/visual.py#L23
        struct ggml_tensor * pos_embed = ggml_graph_get_tensor(gf, "pos_embed");
        int pos_w = image_size_width/patch_size;
        int pos_h = image_size_height/patch_size;
        int embed_dim = 4096;
        auto pos_embed_t = get_2d_sincos_pos_embed(embed_dim, std::make_pair(pos_w, pos_h));

        float * pos_embed_data = (float *)malloc(ggml_nbytes(pos_embed));
        for(int i=0;i<pos_w * pos_h;++i){
            for(int j=0;j<embed_dim;++j){
                pos_embed_data[i*embed_dim+j]=pos_embed_t[i][j];
            }
        }

        ggml_backend_tensor_set(pos_embed, pos_embed_data, 0, ggml_nbytes(pos_embed));
        free(pos_embed_data);
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
                // LOG_TEE("%s: quantizing %s to %s\n", __func__, name.c_str(), ggml_type_name(new_type));
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
                LOG_TEE("Please use an input file in f32 or f16\n");
                gguf_free(ctx_out);
                return false;
            }

            if (work.size() < n_elms * 4) {
                work.resize(n_elms * 4);
            }
            new_data = work.data();

            new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, n_elms/cur->ne[0], cur->ne[0], nullptr);
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

        LOG_TEE("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), ggml_n_dims(cur), quantize,
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
        LOG_TEE("%s: original  size = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        LOG_TEE("%s: quantized size = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);
    }

    return true;
}

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    if (ctx->proj_type == PROJECTOR_TYPE_LDP) {
        return ctx->vision_model.mm_model_block_1_block_2_1_b->ne[0];
    }
    if (ctx->proj_type == PROJECTOR_TYPE_LDPV2) {
        return ctx->vision_model.mm_model_peg_0_b->ne[0];
    }
    if (ctx->proj_type == PROJECTOR_TYPE_MLP) {
        return ctx->vision_model.mm_2_b->ne[0];
    }
    if (ctx->proj_type == PROJECTOR_TYPE_MLP_NORM) {
        return ctx->vision_model.mm_3_b->ne[0];
    }
    if (ctx->proj_type == PROJECTOR_TYPE_RESAMPLER) {
        return 4096;
    }

    std::string proj_type = PROJECTOR_TYPE_NAMES[ctx->proj_type];
    throw std::runtime_error(format("%s: don't support projector with: %s currently\n", __func__, proj_type.c_str()));
}
