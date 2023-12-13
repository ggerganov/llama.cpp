// NOTE: This is modified from clip.cpp only for LLaVA,
// so there might be still unnecessary artifacts hanging around
// I'll gradually clean and extend it

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

#include "clip.h"
#include "ggml.h"
#include "ggml-alloc.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define CLIP_DEBUG

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
    switch (ftype) {
    case 0:
        return "f32";
    case 1:
        return "f16";
    case 2:
        return "q4_0";
    case 3:
        return "q4_1";
    case 6:
        return "q5_0";
    case 7:
        return "q5_1";
    case 8:
        return "q8_0";
    default:
        throw std::runtime_error(format("%s: Unrecognized file type: %d\n", __func__, ftype));
    }
}

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
    struct ggml_tensor * mm_0_w;
    struct ggml_tensor * mm_0_b;
    struct ggml_tensor * mm_2_w;
    struct ggml_tensor * mm_2_b;
};

// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct clip_buffer {
    uint8_t * data = NULL;
    size_t size = 0;

    void resize(size_t size) {
        delete[] data;
        data = new uint8_t[size];
        this->size = size;
    }

    ~clip_buffer() { delete[] data; }
};

struct clip_ctx {
    bool has_text_encoder = false;
    bool has_vision_encoder = false;
    bool has_llava_projector = false;
    struct clip_vision_model vision_model;
    float image_mean[3];
    float image_std[3];
    bool use_gelu = false;
    int32_t ftype = 1;
    struct ggml_context * ctx;
    struct gguf_context * ctx_gguf;

    // memory buffers to evaluate the model
    clip_buffer buf_compute;
    clip_buffer buf_alloc;
    ggml_allocr * alloc = NULL;
};

static ggml_cgraph * clip_image_build_graph(const clip_ctx * ctx, const clip_image_f32_batch * imgs) {
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
    if(ctx->has_llava_projector) {
        GGML_ASSERT(batch_size == 1);
    }

    const auto & buf_compute = ctx->buf_compute;

    struct ggml_init_params params = {
        /*.mem_size =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc =*/ false,
    };

    params.no_alloc = true;

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size, image_size, 3, batch_size);
    ggml_allocr_alloc(ctx->alloc, inp_raw);

    if (!ggml_allocr_is_measure(ctx->alloc)) {
        float * data = (float *)ggml_get_data(inp_raw);

        for (size_t i = 0; i < imgs->size; i++) {
            const int nx = imgs->data[i].nx;
            const int ny = imgs->data[i].ny;
            GGML_ASSERT(nx == image_size && ny == image_size);

            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < 3; k++) {
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            data[(b * 3 * n) + k * n + y * nx + x] = imgs->data[b].data[3 * (y * nx + x) + k];
                        }
                    }
                }
            }
        }
    }

    struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

    // concat class_embeddings and patch_embeddings
    struct ggml_tensor * embeddings = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, batch_size);
    ggml_allocr_alloc(ctx->alloc, embeddings);
    if (!ggml_allocr_is_measure(ctx->alloc)) {
        ggml_set_zero(embeddings);
    }

    struct ggml_tensor * temp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, 1, batch_size);
    ggml_allocr_alloc(ctx->alloc, temp);

    embeddings = ggml_acc(ctx0, embeddings, ggml_repeat(ctx0, model.class_embedding, temp), embeddings->nb[1],
                          embeddings->nb[2], embeddings->nb[3], 0);
    embeddings =
        ggml_acc(ctx0, embeddings, inp, embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], model.class_embedding->nb[1]);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_positions);
    ggml_allocr_alloc(ctx->alloc, positions);
    if (!ggml_allocr_is_measure(ctx->alloc)) {
        for (int i = 0; i < num_positions; i++) {
            ggml_set_i32_1d(positions, i, i);
        }
    }

    embeddings =
        ggml_add(ctx0, embeddings, ggml_repeat(ctx0, ggml_get_rows(ctx0, model.position_embeddings, positions), embeddings));

    // pre-layernorm
    {
        embeddings = ggml_norm(ctx0, embeddings, eps);

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.pre_ln_w, embeddings), embeddings),
                              ggml_repeat(ctx0, model.pre_ln_b, embeddings));
    }

    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_allocr_alloc(ctx->alloc, KQ_scale);
    if (!ggml_allocr_is_measure(ctx->alloc)) {
        ggml_set_f32(KQ_scale, 1.0f / sqrt((float)d_head));
    }

    // loop over layers
    for (int il = 0; il < n_layer - 1; il++) {
        struct ggml_tensor * cur = embeddings; // embeddings = residual, cur = hidden_states

        //const size_t nb_q_w = model.layers[il].q_w->nb[0];

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].ln_1_w, cur), cur),
                           ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        // self-attention
        {

            struct ggml_tensor * Q =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, cur), ggml_mul_mat(ctx0, model.layers[il].q_w, cur));

            Q = ggml_scale_inplace(ctx0, Q, KQ_scale);
            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_positions, batch_size);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * K =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, cur), ggml_mul_mat(ctx0, model.layers[il].k_w, cur));

            K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * V =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, cur), ggml_mul_mat(ctx0, model.layers[il].v_w, cur));

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
        cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].o_b, cur), ggml_mul_mat(ctx0, model.layers[il].o_w, cur));

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur, eps);

            cur = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].ln_2_w, cur), cur),
                           ggml_repeat(ctx0, model.layers[il].ln_2_b, cur));
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].ff_i_b, cur), cur);

        if (ctx->use_gelu) {
            cur = ggml_gelu_inplace(ctx0, cur);
        } else {
            cur = ggml_gelu_quick_inplace(ctx0, cur);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].ff_o_b, cur), cur);

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);

        embeddings = cur;
    }

    // llava projector
    {
        embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);

        struct ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_patches);
        ggml_allocr_alloc(ctx->alloc, patches);
        if (!ggml_allocr_is_measure(ctx->alloc)) {
            for (int i = 0; i < num_patches; ++i) {
                ggml_set_i32_1d(patches, i, i+1);
            }
        }

        embeddings = ggml_get_rows(ctx0, embeddings, patches);

        // mm projection 0
        embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
        embeddings = ggml_add(ctx0, ggml_repeat(ctx0, model.mm_0_b, embeddings), embeddings);

        embeddings = ggml_gelu(ctx0, embeddings);

        embeddings = ggml_mul_mat(ctx0, model.mm_2_w, embeddings);
        embeddings = ggml_add(ctx0, ggml_repeat(ctx0, model.mm_2_b, embeddings), embeddings);
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

    // kv
    if (verbosity >= 3) {
        const int n_kv = gguf_get_n_kv(ctx);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);

            printf("%s: kv[%d]: key = %s\n", __func__, i, key);
        }
        printf("\n");
    }

    // data
    size_t ctx_size = 0;
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);

            struct ggml_tensor * cur = ggml_get_tensor(meta, name);
            ctx_size += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            size_t tensor_size = ggml_nbytes(cur);
            size_t padded_size = ggml_nbytes_pad(cur);
            ctx_size += padded_size;
            if (verbosity >= 3) {
                printf("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, padded_size=%zu, offset=%zu\n", __func__, i,
                       cur->n_dims, cur->name, tensor_size, padded_size, offset);
            }
        }
    }

    clip_ctx * new_clip = new clip_ctx;

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
            printf("%s: model size:     %.2f MB\n", __func__, (ctx_size / 1024.0 / 1024.0));
            printf("%s: metadata size:  %.2f MB\n", __func__, ggml_get_mem_size(meta) / 1024.0 / 1024.0);
        }
    }

    // load tensors
    {
        struct ggml_init_params params = {
            /*.mem_size =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ false,
        };

        new_clip->ctx = ggml_init(params);
        if (!new_clip->ctx) {
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

        const int n_tensors = gguf_get_n_tensors(ctx);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx, i);
            struct ggml_tensor * t = ggml_get_tensor(meta, name);
            struct ggml_tensor * cur = ggml_dup_tensor(new_clip->ctx, t);
            ggml_set_name(cur, name);

            const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                printf("%s: failed to seek for tensor %s\n", __func__, name);
                clip_free(new_clip);
                return nullptr;
            }

            fin.read(reinterpret_cast<char *>(cur->data), ggml_nbytes(t));
        }

        fin.close();
    }

    // vision model
    if (new_clip->has_vision_encoder) {
        // load vision model
        auto & vision_model = new_clip->vision_model;
        auto & hparams = vision_model.hparams;
        hparams.hidden_size = get_u32(ctx, format(KEY_N_EMBD, "vision"));
        hparams.n_head = get_u32(ctx, format(KEY_N_HEAD, "vision"));
        hparams.n_intermediate = get_u32(ctx, format(KEY_N_FF, "vision"));
        hparams.n_layer = get_u32(ctx, format(KEY_N_BLOCK, "vision"));
        hparams.image_size = get_u32(ctx, KEY_IMAGE_SIZE);
        hparams.patch_size = get_u32(ctx, KEY_PATCH_SIZE);
        hparams.projection_dim = get_u32(ctx, format(KEY_PROJ_DIM, "vision"));
        hparams.eps = get_f32(ctx, format(KEY_LAYER_NORM_EPS, "vision"));

        int idx_mean = get_key_idx(ctx, KEY_IMAGE_MEAN);
        int idx_std = get_key_idx(ctx, KEY_IMAGE_STD);
        for (int i = 0; i < 3; ++i) {
            new_clip->image_mean[i] = *((const float *)gguf_get_arr_data(ctx, idx_mean));
            new_clip->image_std[i] = *((const float *)gguf_get_arr_data(ctx, idx_std));
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

        vision_model.patch_embeddings = get_tensor(new_clip->ctx, TN_PATCH_EMBD);
        vision_model.class_embedding = get_tensor(new_clip->ctx, TN_CLASS_EMBD);
        vision_model.position_embeddings = get_tensor(new_clip->ctx, format(TN_POS_EMBD, "v"));
        vision_model.pre_ln_w = get_tensor(new_clip->ctx, format(TN_LN_PRE, "v", "weight"));
        vision_model.pre_ln_b = get_tensor(new_clip->ctx, format(TN_LN_PRE, "v", "bias"));
        vision_model.mm_0_w = get_tensor(new_clip->ctx, format(TN_LLAVA_PROJ, 0, "weight"));
        vision_model.mm_0_b = get_tensor(new_clip->ctx, format(TN_LLAVA_PROJ, 0, "bias"));
        vision_model.mm_2_w = get_tensor(new_clip->ctx, format(TN_LLAVA_PROJ, 2, "weight"));
        vision_model.mm_2_b = get_tensor(new_clip->ctx, format(TN_LLAVA_PROJ, 2, "bias"));

        vision_model.layers.resize(hparams.n_layer);
        for (int il = 0; il < hparams.n_layer; ++il) {
            auto & layer = vision_model.layers[il];
            layer.k_w = get_tensor(new_clip->ctx, format(TN_ATTN_K, "v", il, "weight"));
            layer.q_w = get_tensor(new_clip->ctx, format(TN_ATTN_Q, "v", il, "weight"));
            layer.v_w = get_tensor(new_clip->ctx, format(TN_ATTN_V, "v", il, "weight"));
            layer.o_w = get_tensor(new_clip->ctx, format(TN_ATTN_OUTPUT, "v", il, "weight"));
            layer.ln_1_w = get_tensor(new_clip->ctx, format(TN_LN_1, "v", il, "weight"));
            layer.ln_2_w = get_tensor(new_clip->ctx, format(TN_LN_2, "v", il, "weight"));
            layer.ff_i_w = get_tensor(new_clip->ctx, format(TN_FFN_DOWN, "v", il, "weight"));
            layer.ff_o_w = get_tensor(new_clip->ctx, format(TN_FFN_UP, "v", il, "weight"));
            layer.k_b = get_tensor(new_clip->ctx, format(TN_ATTN_K, "v", il, "bias"));
            layer.q_b = get_tensor(new_clip->ctx, format(TN_ATTN_Q, "v", il, "bias"));
            layer.v_b = get_tensor(new_clip->ctx, format(TN_ATTN_V, "v", il, "bias"));
            layer.o_b = get_tensor(new_clip->ctx, format(TN_ATTN_OUTPUT, "v", il, "bias"));
            layer.ln_1_b = get_tensor(new_clip->ctx, format(TN_LN_1, "v", il, "bias"));
            layer.ln_2_b = get_tensor(new_clip->ctx, format(TN_LN_2, "v", il, "bias"));
            layer.ff_i_b = get_tensor(new_clip->ctx, format(TN_FFN_DOWN, "v", il, "bias"));
            layer.ff_o_b = get_tensor(new_clip->ctx, format(TN_FFN_UP, "v", il, "bias"));
        }
    }

    ggml_free(meta);

    new_clip->ctx_gguf = ctx;

// measure mem requirement and allocate
    {
        static const size_t tensor_alignment = 32;
        new_clip->buf_compute.resize(ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead());
        new_clip->alloc = ggml_allocr_new_measure(tensor_alignment);
        clip_image_f32_batch batch;
        batch.size = 1;
        ggml_cgraph * gf = clip_image_build_graph(new_clip, &batch);
        size_t alloc_size = ggml_allocr_alloc_graph(new_clip->alloc, gf) + tensor_alignment;
        ggml_allocr_free(new_clip->alloc);
        new_clip->buf_alloc.resize(alloc_size);
        new_clip->alloc = ggml_allocr_new(new_clip->buf_alloc.data, new_clip->buf_alloc.size, tensor_alignment);

        printf("%s: total allocated memory: %.2f MB\n", __func__, (new_clip->buf_compute.size + alloc_size)/1024.0/1024.0);
    }

    return new_clip;
}

clip_image_u8 * make_clip_image_u8() {
    auto img = new clip_image_u8();
    return img;
}
clip_image_f32 * make_clip_image_f32() { return new clip_image_f32(); }

void clip_image_u8_free(clip_image_u8 * img) { if (img->data) { delete[] img->data; } delete img; }
void clip_image_f32_free(clip_image_f32 * img) { if (img->data) { delete[] img->data; } delete img; }

static void build_clip_img_from_data(const stbi_uc * data, int nx, int ny, clip_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->size = nx * ny * 3;
    img->data = new uint8_t[img->size]();
    memcpy(img->data, data, img->size);
}

bool clip_image_load_from_file(const char * fname, clip_image_u8 * img) {
    int nx, ny, nc;
    auto data = stbi_load(fname, &nx, &ny, &nc, 3);
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
    auto data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
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
bool clip_image_preprocess(const clip_ctx * ctx, const clip_image_u8 * img, clip_image_f32 * res, const bool pad2square) {
    if (!ctx->has_vision_encoder) {
        printf("This gguf file seems to have no vision encoder\n");
        return false;
    }

    // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
    // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

    clip_image_u8 * temp = make_clip_image_u8(); // we will keep the input image data here temporarily
    if (pad2square && img->nx != img->ny) {
        int longer_side = std::max(img->nx, img->ny);
        temp->nx = longer_side;
        temp->ny = longer_side;
        temp->size = 3 * longer_side * longer_side;
        temp->data = new uint8_t[temp->size]();
        uint8_t bc[3] = {122, 116, 104}; // background color in RGB from LLaVA

        // fill with background color
        for (size_t i = 0; i < temp->size; i++) {
            temp->data[i] = bc[i % 3];
        }

        // copy from the input image
        for (int y = 0; y < img->ny; y++) {
            for (int x = 0; x < img->nx; x++) {
                const int i = 3 * (y * img->nx + x);
                const int j = 3 * (y * temp->nx + x);
                temp->data[j] = img->data[i];
                temp->data[j+1] = img->data[i+1];
                temp->data[j+2] = img->data[i+2];
            }
        }
    } else {
        temp->nx   = img->nx;
        temp->ny   = img->ny;
        temp->size = img->size;
        temp->data = new uint8_t[temp->size]();
        memcpy(&temp->data[0], &img->data[0], temp->size); // copy
    }

    const int nx = temp->nx;
    const int ny = temp->ny;

    const int nx2 = ctx->vision_model.hparams.image_size;
    const int ny2 = ctx->vision_model.hparams.image_size;

    res->nx = nx2;
    res->ny = ny2;
    res->size = 3 * nx2 * ny2;
    res->data = new float[res->size]();

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

                const float v00 = temp->data[j00];
                const float v01 = temp->data[j01];
                const float v10 = temp->data[j10];
                const float v11 = temp->data[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res->data[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }
    clip_image_u8_free(temp);

    return true;
}

void clip_free(clip_ctx * ctx) {
    ggml_free(ctx->ctx);
    gguf_free(ctx->ctx_gguf);
    delete ctx;
}

bool clip_image_encode(const clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    if (!ctx->has_vision_encoder) {
        printf("This gguf file seems to have no vision encoder\n");
        return false;
    }

    clip_image_f32_batch imgs{};
    imgs.size = 1;
    imgs.data = img;
    return clip_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool clip_image_batch_encode(const clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs, float * vec) {

    if (!ctx->has_vision_encoder) {
        printf("This gguf file seems to have no vision encoder\n");
        return false;
    }

    int batch_size = imgs->size;
    if(ctx->has_llava_projector) {
        GGML_ASSERT(batch_size == 1); // TODO: support multiple images
    }

    // reset alloc buffer to clean the memory from previous invocations
    ggml_allocr_reset(ctx->alloc);

    // build the inference graph
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs);
    ggml_allocr_alloc_graph(ctx->alloc, gf);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }

    ggml_graph_compute(gf, &plan);

    // the last node is the embedding tensor
struct ggml_tensor * embeddings = gf->nodes[gf->n_nodes - 1];

    // copy the embeddings to the location passed by the user
    memcpy(vec, ggml_get_data_f32(embeddings), ggml_nbytes(embeddings));

    if (plan.work_size > 0) {
        free(plan.work_data);
    }

    return true;
}

bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype) {

    ggml_type type = GGML_TYPE_Q4_1;

    switch (itype) {
    case 2:
        type = GGML_TYPE_Q4_0;
        break;
    case 3:
        type = GGML_TYPE_Q4_1;
        break;
    case 6:
        type = GGML_TYPE_Q5_0;
        break;
    case 7:
        type = GGML_TYPE_Q5_1;
        break;
    case 8:
        type = GGML_TYPE_Q8_0;
        break;
    default:
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype);
        return false;
    };

    auto ctx_clip = clip_model_load(fname_inp, 2);
    const auto & ctx_src = ctx_clip->ctx_gguf;
    const auto & ctx_data = ctx_clip->ctx;

    auto ctx_out = gguf_init_empty();
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

    std::vector<uint8_t> read_data(512);
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
        quantize &= (cur->n_dims == 2);

        if (quantize) {
            new_type = type;
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

        printf("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), cur->n_dims, quantize,
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
        printf("%s: original size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        printf("%s: quantized size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);

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
    return ctx->vision_model.mm_2_b->ne[0];
}

int clip_n_patches(const struct clip_ctx * ctx) {
    auto & params = ctx->vision_model.hparams;

    return (params.image_size / params.patch_size) * (params.image_size / params.patch_size);
}

size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    return clip_n_patches(ctx) * clip_n_mmproj_embd(ctx) * sizeof(float);
}
