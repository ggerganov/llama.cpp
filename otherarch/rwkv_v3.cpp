//adapted from RWKV.cpp repo under MIT license
// https://github.com/saharNooby/rwkv.cpp

#include "otherarch.h"

#include "rwkv_v3.h"
#include "ggml.h"

#include <string>
#include <vector>
#include <thread>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <memory>

#include <sys/stat.h> // fstat

#ifdef WIN32
#define stat64 _stat64
#define fstat64 _fstat64
#endif

// --- Error handling ---

enum rwkv_error_flags global_last_error = RWKV_ERROR_NONE;
bool global_print_errors = true;

enum rwkv_error_flags operator|(enum rwkv_error_flags a, enum rwkv_error_flags b) {
    return static_cast<enum rwkv_error_flags>(static_cast<int>(a) | static_cast<int>(b));
}

enum rwkv_error_flags operator|=(enum rwkv_error_flags & a, enum rwkv_error_flags b) {
    return a = a | b;
}

// If the condition x is false, adds ERR_VAL to the last error, prints a message to stderr, and returns RET_VAL.
#define RWKV_ASSERT_MSG(ERR_VAL, RET_VAL, x, ...) \
    if (!(x)) { \
        global_last_error |= ERR_VAL; \
        if (global_print_errors) { \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        } \
        return RET_VAL; \
    }

// If the condition x is false, adds ERR_VAL to the last error, and returns RET_VAL.
#define RWKV_ASSERT(ERR_VAL, RET_VAL, x) \
    if (!(x)) { \
        global_last_error |= ERR_VAL; \
        return RET_VAL; \
    }

// If the condition x is false, adds ERR_VAL to the ctx's last error, prints a message to stderr, and returns RET_VAL.
#define RWKV_CTX_ASSERT_MSG(ctx, ERR_VAL, RET_VAL, x, ...) \
    if (!(x)) { \
        ((struct rwkv_context *) ctx)->last_error |= ERR_VAL; \
        if (ctx->print_errors) { \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        } \
        return RET_VAL; \
    }

// If the condition x is false, adds ERR_VAL to the ctx's last error, and returns RET_VAL.
#define RWKV_CTX_ASSERT(ctx, ERR_VAL, RET_VAL, x) \
    if (!(x)) { \
        ctx->last_error |= ERR_VAL; \
        return RET_VAL; \
    }

#define RWKV_ASSERT_FALSE_MSG(ERR_VAL, x, ...) RWKV_ASSERT_MSG(ERR_VAL, false, x, __VA_ARGS__)
#define RWKV_ASSERT_NULL_MSG(ERR_VAL, x, ...) RWKV_ASSERT_MSG(ERR_VAL, NULL, x, __VA_ARGS__)
#define RWKV_CTX_ASSERT_FALSE_MSG(ctx, ERR_VAL, x, ...) RWKV_CTX_ASSERT_MSG(ctx, ERR_VAL, false, x, __VA_ARGS__)
#define RWKV_CTX_ASSERT_NULL_MSG(ctx, ERR_VAL, x, ...) RWKV_CTX_ASSERT_MSG(ctx, ERR_VAL, NULL, x, __VA_ARGS__)

#define RWKV_ASSERT_FALSE(ERR_VAL, x) RWKV_ASSERT(ERR_VAL, false, x)
#define RWKV_ASSERT_NULL(ERR_VAL, x) RWKV_ASSERT(ERR_VAL, NULL, x)
#define RWKV_CTX_ASSERT_FALSE(ctx, ERR_VAL, x) RWKV_CTX_ASSERT(ctx, ERR_VAL, false, x)
#define RWKV_CTX_ASSERT_NULL(ctx, ERR_VAL, x) RWKV_CTX_ASSERT(ctx, ERR_VAL, NULL, x)

// --- Utilities ---

// Reads single int32 value from a file.
bool read_int32(FILE * file, int32_t * dest, const char * name) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_READ, fread(dest, sizeof(int32_t), 1, file) == 1, "Failed to read an int32 value from a file (%s)", name);
    return true;
}

// Reads single uint32 value from a file.
bool read_uint32(FILE * file, uint32_t * dest, const char * name) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_READ, fread(dest, sizeof(uint32_t), 1, file) == 1, "Failed to read a uint32 value from a file (%s)", name);
    return true;
}

// Writes single int32 value to a file.
bool write_int32(FILE * file, int32_t value, const char * name) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_WRITE, fwrite((void *) &value, sizeof(int32_t), 1, file), "Failed to write an int32 value to a file (%s)", name);
    return true;
}

// Writes single uint32 value to a file.
bool write_uint32(FILE * file, uint32_t value, const char * name) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_WRITE, fwrite((void *) &value, sizeof(uint32_t), 1, file), "Failed to write a uint32 value to a file (%s)", name);
    return true;
}

#define GGML_TYPE_UNKNOWN GGML_TYPE_COUNT

#define FORMAT_TYPE_COUNT 10

static const ggml_type FORMAT_TYPE_TO_GGML_TYPE[FORMAT_TYPE_COUNT] = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_UNKNOWN, // Unused
    GGML_TYPE_UNKNOWN, // Unused
    GGML_TYPE_UNKNOWN, // Unused
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0
};

static bool is_non_quantized_format_type(int32_t format_type) {
    return format_type == 0 || format_type == 1;
}

static bool is_quantized_format_type(int32_t format_type) {
    return format_type == 2 ||
        format_type == 3 ||
        format_type == 7 ||
        format_type == 8 ||
        format_type == 9;
}

static int32_t format_name_to_format_type(const char * format_name) {
    if (strcmp(format_name, "Q4_0") == 0) return 2;
    if (strcmp(format_name, "Q4_1") == 0) return 3;
    if (strcmp(format_name, "Q5_0") == 0) return 7;
    if (strcmp(format_name, "Q5_1") == 0) return 8;
    if (strcmp(format_name, "Q8_0") == 0) return 9;

    return -1;
}

// --- Model definition and loading utilities ---

struct rwkv_layer {
    struct ggml_tensor * ln1_weight;
    struct ggml_tensor * ln1_bias;

    // RWKV, also called "attention" by the author.
    struct ggml_tensor * att_time_mix_k;
    struct ggml_tensor * att_time_mix_v;
    struct ggml_tensor * att_time_mix_r;
    struct ggml_tensor * att_time_first;
    struct ggml_tensor * att_time_decay;
    struct ggml_tensor * att_key;
    struct ggml_tensor * att_value;
    struct ggml_tensor * att_receptance;
    struct ggml_tensor * att_output;

    struct ggml_tensor * ln2_weight;
    struct ggml_tensor * ln2_bias;

    // FFN.
    struct ggml_tensor * ffn_time_mix_k;
    struct ggml_tensor * ffn_time_mix_r;
    struct ggml_tensor * ffn_key;
    struct ggml_tensor * ffn_value;
    struct ggml_tensor * ffn_receptance;
};

struct rwkv_model {
    uint32_t n_vocab;
    uint32_t n_layer;
    uint32_t n_embed;
    // 0 for float32, 1 for float16.
    int32_t data_type;

    struct ggml_tensor * emb;

    struct ggml_tensor * ln0_weight;
    struct ggml_tensor * ln0_bias;

    std::vector<rwkv_layer> layers;

    struct ggml_tensor * ln_out_weight;
    struct ggml_tensor * ln_out_bias;

    struct ggml_tensor * head;
};

// Finds model parameter by key and sets it into dest.
// If the parameter was not found, returns false.
bool set_parameter(std::unordered_map<std::string, struct ggml_tensor *> * parameters, std::string key, struct ggml_tensor ** dest) {
    struct ggml_tensor * parameter = (*parameters)[key];
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_PARAM_MISSING, parameter != NULL, "Parameter %s not found in model file", key.c_str());
    *dest = parameter;
    return true;
}

// Finds block parameter by block index and key and sets it into dest.
// If the parameter was not found, returns false.
bool set_block_parameter(std::unordered_map<std::string, struct ggml_tensor *> * parameters, int32_t block_index, std::string key, struct ggml_tensor ** dest) {
    char full_key[128];
    sprintf(full_key, "blocks.%d.%s", block_index, key.c_str());
    return set_parameter(parameters, full_key, dest);
}

// --- Operators ---

void rwkv_exp_impl(const int n_cols, float * dest, const float * src) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = expf(src[i]);
    }
}

void rwkv_1_minus_x_impl(const int n_cols, float * dest, const float * src) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = 1.0F - src[i];
    }
}

void rwkv_sigmoid_impl(const int n_cols, float * dest, const float * src) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = 1.0F / (1.0F + expf(-src[i]));
    }
}

void rwkv_max_impl(const int n_cols, float * dest, const float * src0, const float * src1) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = fmaxf(src0[i], src1[i]);
    }
}

struct ggml_tensor * rwkv_exp(ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_unary_f32(ctx, x, rwkv_exp_impl);
}

struct ggml_tensor * rwkv_1_minus_x(ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_unary_f32(ctx, x, rwkv_1_minus_x_impl);
}

struct ggml_tensor * rwkv_sigmoid(ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_unary_f32(ctx, x, rwkv_sigmoid_impl);
}

struct ggml_tensor * rwkv_max(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y) {
    return ggml_map_binary_f32(ctx, x, y, rwkv_max_impl);
}

struct ggml_tensor * rwkv_layer_norm(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // LayerNorm in RWKV is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // Looks like ggml_norm does the first part, we only need to apply weight & bias.
    return ggml_add_inplace(ctx, ggml_mul(ctx, ggml_norm(ctx, x), weight), bias);
}

// --- Implementation ---

struct rwkv_graph {
    struct ggml_tensor * state;
    std::unique_ptr<struct ggml_tensor * []> state_parts;
    struct ggml_tensor * token_index;
    struct ggml_tensor * logits;
    std::unique_ptr<struct ggml_cgraph> cgraph;
};

struct rwkv_context {
    std::unique_ptr<struct rwkv_model> model;
    struct ggml_context * ctx;
    struct rwkv_graph graph;
    enum rwkv_error_flags last_error;
    bool print_errors;

    float * state_in = 0; //stores input state, or use null for a new state
    float * state_out = 0; //stores address of output state buffer
    float * logits_out = 0; //stores address of output logit buffer
};

void rwkv_set_print_errors(struct rwkv_context * ctx, bool print_errors) {
    bool * ptr = ctx ? &ctx->print_errors : &global_print_errors;
    *ptr = print_errors;
}

bool rwkv_get_print_errors(struct rwkv_context * ctx) {
    return ctx ? ctx->print_errors : global_print_errors;
}

enum rwkv_error_flags rwkv_get_last_error(struct rwkv_context * ctx) {
    enum rwkv_error_flags * ptr = ctx ? &ctx->last_error : &global_last_error;
    enum rwkv_error_flags value = *ptr;
    *ptr = RWKV_ERROR_NONE;
    return value;
}

bool rwkv_build_graph(struct ggml_context * ctx, struct rwkv_model * model, const uint32_t n_threads, struct rwkv_graph * out) {
    std::unique_ptr<struct ggml_cgraph> cgraph(new(std::nothrow) struct ggml_cgraph());
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, cgraph.get(), "Failed to allocate graph");
    cgraph->n_threads = n_threads;

    size_t n_embed = model->n_embed, n_layer = model->n_layer;
    struct ggml_tensor * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_layer * 5 * n_embed);

    // We collect parts of new state here. Each part is (n_embed) vector.
    std::unique_ptr<struct ggml_tensor * []> state_parts(new(std::nothrow) ggml_tensor * [n_layer * 5]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, state_parts.get(), "Failed to allocate state parts");

    // x = self.w.emb.weight[token]
    struct ggml_tensor * token_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * x = ggml_get_rows(ctx, model->emb, token_index);

    // x = self.layer_norm(x, self.w.blocks[0].ln0)
    x = rwkv_layer_norm(ctx, x, model->ln0_weight, model->ln0_bias);

    for (size_t i = 0; i < n_layer; i++) {
        struct rwkv_layer layer = model->layers[i];
        size_t part_index = i * 5;
        size_t state_part_size = n_embed * sizeof(float);

        // RWKV/time mixing
        {
            // self.layer_norm(x, self.w.blocks[i].ln1)
            struct ggml_tensor * x0 = rwkv_layer_norm(ctx, x, layer.ln1_weight, layer.ln1_bias);

            // x0 = state[5 * i + 1]
            struct ggml_tensor * x_prev = ggml_view_1d(ctx, state, n_embed, (part_index + 1) * state_part_size);
            // aa = state[5 * i + 2]
            struct ggml_tensor * aa = ggml_view_1d(ctx, state, n_embed, (part_index + 2) * state_part_size);
            // bb = state[5 * i + 3]
            struct ggml_tensor * bb = ggml_view_1d(ctx, state, n_embed, (part_index + 3) * state_part_size);
            // pp = state[5 * i + 4]
            struct ggml_tensor * pp = ggml_view_1d(ctx, state, n_embed, (part_index + 4) * state_part_size);

            // xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
            struct ggml_tensor * xk = ggml_add_inplace(ctx,
                ggml_mul(ctx, x0, layer.att_time_mix_k),
                ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.att_time_mix_k))
            );

            // xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
            struct ggml_tensor * xv = ggml_add_inplace(ctx,
                ggml_mul(ctx, x0, layer.att_time_mix_v),
                ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.att_time_mix_v))
            );

            // xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
            struct ggml_tensor * xr = ggml_add_inplace(ctx,
                ggml_mul(ctx, x0, layer.att_time_mix_r),
                ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.att_time_mix_r))
            );

            // r = torch.sigmoid(rw @ xr)
            struct ggml_tensor * r = rwkv_sigmoid(ctx, ggml_mul_mat(ctx, layer.att_receptance, xr));
            // k = kw @ xk
            struct ggml_tensor * k = ggml_mul_mat(ctx, layer.att_key, xk);
            // v = vw @ xv
            struct ggml_tensor * v = ggml_mul_mat(ctx, layer.att_value, xv);

            // ww = time_first + k
            struct ggml_tensor * ww = ggml_add(ctx, layer.att_time_first, k);
            // qq = torch.maximum(pp, ww)
            struct ggml_tensor * qq = rwkv_max(ctx, pp, ww);
            // e1 = torch.exp(pp - qq)
            struct ggml_tensor * e1 = rwkv_exp(ctx, ggml_sub(ctx, pp, qq));
            // e2 = torch.exp(ww - qq)
            struct ggml_tensor * e2 = rwkv_exp(ctx, ggml_sub(ctx, ww, qq));
            
            // a = e1 * aa + e2 * v
            struct ggml_tensor * a = ggml_add_inplace(ctx, ggml_mul(ctx, e1, aa), ggml_mul(ctx, e2, v));
            // b = e1 * bb + e2
            struct ggml_tensor * b = ggml_add_inplace(ctx, ggml_mul(ctx, e1, bb), e2);

            // ww = pp + time_decay
            ww = ggml_add_inplace(ctx, pp, layer.att_time_decay);
            // qq = torch.maximum(ww, k)
            qq = rwkv_max(ctx, ww, k);
            // e1 = torch.exp(ww - qq)
            e1 = rwkv_exp(ctx, ggml_sub(ctx, ww, qq));
            // e2 = torch.exp(k - qq)
            e2 = rwkv_exp(ctx, ggml_sub(ctx, k, qq));

            // state[5 * i + 1] = x0
            // state[5 * i + 2] = e1 * aa + e2 * v
            // state[5 * i + 3] = e1 * bb + e2
            // state[5 * i + 4] = qq
            state_parts[part_index + 1] = x0;
            state_parts[part_index + 2] = ggml_add_inplace(ctx, ggml_mul(ctx, e1, aa), ggml_mul(ctx, e2, v));
            state_parts[part_index + 3] = ggml_add_inplace(ctx, ggml_mul(ctx, e1, bb), e2);
            state_parts[part_index + 4] = qq;

            // wkv = a / b
            struct ggml_tensor * wkv = ggml_div(ctx, a, b);

            // ow @ (r * wkv)
            x = ggml_add_inplace(ctx, x, ggml_mul_mat(ctx, layer.att_output, ggml_mul(ctx, r, wkv)));
        }

        // FFN/channel mixing
        {
            // self.layer_norm(x, self.w.blocks[i].ln2)
            struct ggml_tensor * x0 = rwkv_layer_norm(ctx, x, layer.ln2_weight, layer.ln2_bias);

            // x_prev = state[5 * i + 0]
            struct ggml_tensor * x_prev = ggml_view_1d(ctx, state, n_embed, part_index * state_part_size);

            // xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
            struct ggml_tensor * xk = ggml_add_inplace(
                ctx,
                ggml_mul(ctx, x0, layer.ffn_time_mix_k),
                ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.ffn_time_mix_k))
            );

            // xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
            struct ggml_tensor * xr = ggml_add_inplace(
                ctx,
                ggml_mul(ctx, x0, layer.ffn_time_mix_r),
                ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.ffn_time_mix_r))
            );

            // state[5 * i + 0] = x
            state_parts[part_index] = x0;

            // r = torch.sigmoid(rw @ xr)
            struct ggml_tensor * r = rwkv_sigmoid(ctx, ggml_mul_mat(ctx, layer.ffn_receptance, xr));

            // k = torch.square(torch.relu(kw @ xk))
            struct ggml_tensor * k = ggml_sqr(ctx, ggml_relu(ctx, ggml_mul_mat(ctx, layer.ffn_key, xk)));

            // r * (vw @ k)
            x = ggml_add_inplace(ctx, x, ggml_mul(ctx, r, ggml_mul_mat(ctx, layer.ffn_value, k)));
        }
    }

    // x = self.layer_norm(x, self.w.ln_out)
    x = rwkv_layer_norm(ctx, x, model->ln_out_weight, model->ln_out_bias);

    // x = (self.w.head.weight @ x).float()
    struct ggml_tensor * logits = ggml_mul_mat(ctx, model->head, x);

    ggml_build_forward_expand(cgraph.get(), logits);

    for (uint32_t i = 0; i < n_layer * 5; i++) {
       ggml_build_forward_expand(cgraph.get(), state_parts[i]);
    }

    out->state = state;
    out->state_parts = std::move(state_parts);
    out->token_index = token_index;
    out->logits = logits;
    out->cgraph = std::move(cgraph);
    return true;
}

struct rwkv_file_guard {
    FILE * file;
    ~rwkv_file_guard() { if (file) fclose(file); }
};

struct rwkv_ggml_guard {
    struct ggml_context * ctx;
    ~rwkv_ggml_guard() { if (ctx) ggml_free(ctx); }
};

struct rwkv_context * rwkv_init_from_file(const char * file_path, const uint32_t n_threads) {
    global_last_error = RWKV_ERROR_NONE;

    FILE * file = fopen(file_path, "rb");
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, file, "Failed to open file %s", file_path);
    rwkv_file_guard file_guard { file };

    // Be very careful when changing this code. It must support files larger than 2 GB by using 64-bit functions to the get file length.
    struct stat64 file_stat;
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_STAT, fstat64(fileno(file), &file_stat) == 0, "Failed to stat file %s", file_path);

    int32_t magic;
    RWKV_ASSERT_NULL(RWKV_ERROR_FILE, read_int32(file, &magic, "magic"));
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_MAGIC, magic == RWKV_FILE_MAGIC, "Unexpected magic value %d", magic);

    int32_t version;
    RWKV_ASSERT_NULL(RWKV_ERROR_FILE, read_int32(file, &version, "version"));
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_VERSION, version >= RWKV_FILE_VERSION_MIN && version <= RWKV_FILE_VERSION_MAX, "Unsupported file version %d", version);

    std::unique_ptr<rwkv_model> model(new(std::nothrow) struct rwkv_model());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL | RWKV_ERROR_ALLOC, model.get(), "Failed to allocate model");
 
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL, read_uint32(file, &model->n_vocab, "n_vocab"));
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL, read_uint32(file, &model->n_embed, "n_embed"));
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL, read_uint32(file, &model->n_layer, "n_layer"));
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL, read_int32(file, &model->data_type, "data_type"));

    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL | RWKV_ERROR_DATA_TYPE, model->data_type >= 0 && model->data_type < FORMAT_TYPE_COUNT, "Unsupported model data type %d", model->data_type);

    const char * unsupported_type_msg = "Models in %s format cannot be loaded anymore because the format was removed.\n"
        "You need to quantize the model into another format or use an older version of rwkv.cpp.\n"
        "See https://github.com/saharNooby/rwkv.cpp#compatibility for more info";
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL | RWKV_ERROR_UNSUPPORTED, model->data_type != 4, unsupported_type_msg, "Q4_1_O");
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL | RWKV_ERROR_UNSUPPORTED, model->data_type != 5, unsupported_type_msg, "Q4_2");
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL | RWKV_ERROR_UNSUPPORTED, model->data_type != 6, unsupported_type_msg, "Q4_3");

    RWKV_ASSERT_NULL_MSG(
        RWKV_ERROR_MODEL | RWKV_ERROR_UNSUPPORTED,
        !is_quantized_format_type(model->data_type) || version >= RWKV_FILE_VERSION_1,
        "The quantized model file was created with an old version of rwkv.cpp and can not be loaded anymore.\n"
            "You need to requantize the model or use an older version of rwkv.cpp.\n"
            "See https://github.com/saharNooby/rwkv.cpp#compatibility for more info"
    );

    size_t memory_required = file_stat.st_size +
        // Intermediary vectors for calculation; there are around 100 calls to ggml
        size_t(100) * model->n_embed * sizeof(float) +
        // State, in and out
        size_t(2) * 5 * model->n_layer * model->n_embed * sizeof(float) +
        // Logits
        size_t(model->n_vocab) * sizeof(float) +
        // +256 MB just for any overhead
        // TODO This is too much for smaller models; need a more proper and robust way of measuring required memory
        size_t(256) * 1024 * 1024;

    struct ggml_context * ctx = ggml_init({ memory_required, NULL, false });
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL | RWKV_ERROR_ALLOC, ctx, "Failed to allocate GGML context");
    rwkv_ggml_guard ggml_guard { ctx };

    std::unordered_map<std::string, struct ggml_tensor *> parameters;

    while (true) {
        int32_t dim_count, key_length, data_type;
        RWKV_ASSERT_NULL_MSG(
            RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_FILE_READ,
            fread(&dim_count, sizeof(int32_t), 1, file) == 1 || feof(file),
            "Failed to read an int32 value from a file (dim_count)"
        );

        if (feof(file)) {
            break;
        }

        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, read_int32(file, &key_length, "key_length"));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, read_int32(file, &data_type, "data_type"));

        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_SHAPE, dim_count == 1 || dim_count == 2, "Unsupported dimension count %d", dim_count);
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_KEY, key_length > 0, "Non-positive key length %d", key_length);
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_UNSUPPORTED, data_type >= 0 && data_type < FORMAT_TYPE_COUNT, "Unsupported parameter data type %d", data_type);

        ggml_type ggml_data_type = FORMAT_TYPE_TO_GGML_TYPE[data_type];
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_UNSUPPORTED, ggml_data_type != GGML_TYPE_UNKNOWN, "Unsupported parameter data type %d", data_type);

        struct ggml_tensor * tensor;

        if (dim_count == 1) {
            int32_t x;
            RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, read_int32(file, &x, "x"), "Failed to read parameter length");
            tensor = ggml_new_tensor_1d(ctx, ggml_data_type, x);
        } else {
            int32_t x, y;
            RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, read_int32(file, &x, "x"), "Failed to read parameter width");
            RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, read_int32(file, &y, "y"), "Failed to read parameter height");
            tensor = ggml_new_tensor_2d(ctx, ggml_data_type, x, y);
        }

        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_ALLOC, tensor, "Failed to allocate tensor");

        std::string key(key_length, 0);
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_KEY, fread(&key[0], key_length, 1, file) == 1, "Failed to read parameter key");

        size_t nbytes = ggml_nbytes(tensor);
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DATA, fread(tensor->data, nbytes, 1, file) == 1, "Failed to read parameter data");

        parameters[key] = tensor;
    }

    file_guard = { NULL }; // close file

    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_parameter(&parameters, "emb.weight", &model->emb));
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_parameter(&parameters, "blocks.0.ln0.weight", &model->ln0_weight));
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_parameter(&parameters, "blocks.0.ln0.bias", &model->ln0_bias));

    model->layers.resize(model->n_layer);

    for (uint32_t i = 0; i < model->n_layer; i++) {
        rwkv_layer * layer = &model->layers[i];
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "ln1.weight", &layer->ln1_weight));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "ln1.bias", &layer->ln1_bias));

        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "att.time_mix_k", &layer->att_time_mix_k));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "att.time_mix_v", &layer->att_time_mix_v));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "att.time_mix_r", &layer->att_time_mix_r));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "att.time_first", &layer->att_time_first));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "att.time_decay", &layer->att_time_decay));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "att.key.weight", &layer->att_key));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "att.value.weight", &layer->att_value));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "att.receptance.weight", &layer->att_receptance));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "att.output.weight", &layer->att_output));

        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "ln2.weight", &layer->ln2_weight));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "ln2.bias", &layer->ln2_bias));

        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "ffn.time_mix_k", &layer->ffn_time_mix_k));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "ffn.time_mix_r", &layer->ffn_time_mix_r));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "ffn.key.weight", &layer->ffn_key));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "ffn.value.weight", &layer->ffn_value));
        RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_block_parameter(&parameters, i, "ffn.receptance.weight", &layer->ffn_receptance));
    }

    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_parameter(&parameters, "ln_out.weight", &model->ln_out_weight));
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_parameter(&parameters, "ln_out.bias", &model->ln_out_bias));
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS, set_parameter(&parameters, "head.weight", &model->head));

    // Verify order of dimensions
    struct ggml_tensor * emb = model->emb;
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_SHAPE, emb->n_dims == 2, "Unexpected dimension count of embedding matrix %d", emb->n_dims);
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[0] == model->n_embed, "Unexpected dimension of embedding matrix %" PRId64, emb->ne[0]);
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[1] == model->n_vocab, "Unexpected dimension of embedding matrix %" PRId64, emb->ne[1]);

    // Build graph
    struct rwkv_graph graph;
    RWKV_ASSERT_NULL(RWKV_ERROR_GRAPH, rwkv_build_graph(ctx, model.get(), n_threads, &graph));

    std::unique_ptr<struct rwkv_context> rwkv_ctx(new(std::nothrow) struct rwkv_context());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, rwkv_ctx.get(), "Failed to allocate context");
    rwkv_ctx->model = std::move(model);
    rwkv_ctx->ctx = ctx;
    rwkv_ctx->graph = std::move(graph);
    rwkv_ctx->last_error = RWKV_ERROR_NONE;
    rwkv_ctx->print_errors = global_print_errors;
    // Don't free ggml context
    ggml_guard.ctx = NULL;
    return rwkv_ctx.release();
}

uint32_t rwkv_get_state_buffer_element_count(const struct rwkv_context * ctx) {
    return ctx->model->n_layer * 5 * ctx->model->n_embed;
}

uint32_t rwkv_get_logits_buffer_element_count(const struct rwkv_context * ctx) {
    return ctx->model->n_vocab;
}

bool rwkv_eval(const struct rwkv_context * ctx, const uint32_t token, const float * state_in, float * state_out, float * logits_out) {
    ((struct rwkv_context *) ctx)->last_error = RWKV_ERROR_NONE;

    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, state_out != NULL, "state_out is NULL");
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, logits_out != NULL, "logits_out is NULL");
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, token < ctx->model->n_vocab, "Token is out of range 0..%d", ctx->model->n_vocab - 1);

    const struct rwkv_graph * graph = &ctx->graph;
    size_t n_layer = ctx->model->n_layer;
    size_t n_embed = ctx->model->n_embed;

    ggml_set_i32_1d(graph->token_index, 0, token);

    if (state_in == NULL) {
        ggml_set_f32(graph->state, 0.0F);

        for (size_t i = 0; i < n_layer; i++) {
            // state[5 * i + 4] = -1e30
            ggml_set_f32(
                ggml_view_1d(ctx->ctx, graph->state, n_embed, (5 * i + 4) * n_embed * sizeof(float)),
                -1e30F
            );
        }
    } else {
        memcpy(graph->state->data, state_in, graph->state->ne[0] * sizeof(float));
    }

    ggml_graph_compute(ctx->ctx, graph->cgraph.get());

    for (size_t i = 0; i < n_layer * 5; i++) {
        struct ggml_tensor * part = graph->state_parts[i];
        memcpy(state_out + i * n_embed, part->data, part->ne[0] * sizeof(float));
    }

    memcpy(logits_out, graph->logits->data, graph->logits->ne[0] * sizeof(float));

    return true;
}

void rwkv_free(struct rwkv_context * ctx) {
    std::unique_ptr<struct rwkv_context> rwkv_ctx(ctx);
    ggml_free(ctx->ctx);
}

bool rwkv_quantize_model_file(const char * model_file_path_in, const char * model_file_path_out, const char * format_name) {
    global_last_error = RWKV_ERROR_NONE;

    int32_t format_data_type = format_name_to_format_type(format_name);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS | RWKV_ERROR_DATA_TYPE, format_data_type != -1, "Unsupported format \"%s\"", format_name);

    ggml_type format_ggml_type = FORMAT_TYPE_TO_GGML_TYPE[format_data_type];
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS | RWKV_ERROR_DATA_TYPE, format_ggml_type != GGML_TYPE_UNKNOWN, "Unsupported format \"%s\"", format_name);

    // Needed to initialize FP16 lookup table
    ggml_free(ggml_init({ 0, NULL, false }));

    printf("Loading model from '%s'\n", model_file_path_in);

    FILE * file_in = fopen(model_file_path_in, "rb");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, file_in, "Failed to open %s for reading", model_file_path_in);
    FILE * file_out = fopen(model_file_path_out, "wb");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, file_out, "Failed to open %s for writing", model_file_path_out);

    rwkv_file_guard file_in_guard { file_in };
    rwkv_file_guard file_out_guard { file_out };

    // Process header
    {
        uint32_t magic, version;
        int32_t n_vocab, n_embed, n_layer, data_type;

        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, read_uint32(file_in, &magic, "magic"));
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_MAGIC, magic == RWKV_FILE_MAGIC, "Unexpected magic value %d", magic);

        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, read_uint32(file_in, &version, "version"));
        RWKV_ASSERT_FALSE_MSG(
            RWKV_ERROR_FILE | RWKV_ERROR_FILE_VERSION,
            version >= RWKV_FILE_VERSION_MIN && version <= RWKV_FILE_VERSION_MAX,
            "Unsupported file version %d",
            version
        );

        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, read_int32(file_in, &n_vocab, "n_vocab"));
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, read_int32(file_in, &n_embed, "n_embed"));
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, read_int32(file_in, &n_layer, "n_layer"));

        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, read_int32(file_in, &data_type, "data_type"));
        RWKV_ASSERT_FALSE_MSG(
            RWKV_ERROR_FILE | RWKV_ERROR_DATA_TYPE,
            is_non_quantized_format_type(data_type),
            "Unsupported data type %d, only FP32 and FP16 can be quantized",
            data_type
        );

        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_uint32(file_out, magic, "magic"));
        // Always write latest version number when saving files
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_uint32(file_out, RWKV_FILE_VERSION_MAX, "version"));
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_int32(file_out, n_vocab, "n_vocab"));
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_int32(file_out, n_embed, "n_embed"));
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_int32(file_out, n_layer, "n_layer"));
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_int32(file_out, format_data_type, "data_type"));
    }

    // Process parameters
    size_t total_size_orig = 0;
    size_t total_size_new = 0;

    std::vector<float> work;

    std::vector<uint8_t>     data_u8;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<float>       data_f32;

    std::vector<int64_t> hist_all(1 << 4, 0);

    while (true) {
        int32_t n_dims, key_length, parameter_data_type;
        RWKV_ASSERT_FALSE_MSG(
            RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_FILE_READ,
            fread(&n_dims, sizeof(int32_t), 1, file_in) == 1 || feof(file_in),
            "Failed to read an int32 value from a file (n_dims)"
        );

        if (feof(file_in)) {
            break;
        }

        RWKV_ASSERT_FALSE(RWKV_ERROR_MODEL_PARAMS, read_int32(file_in, &key_length, "key_length"));
        RWKV_ASSERT_FALSE(RWKV_ERROR_MODEL_PARAMS, read_int32(file_in, &parameter_data_type, "parameter_data_type"));

        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_SHAPE, n_dims == 1 || n_dims == 2, "Unsupported dimension count %d", n_dims);
        RWKV_ASSERT_FALSE_MSG(
            RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_UNSUPPORTED,
            parameter_data_type >= 0 && parameter_data_type < FORMAT_TYPE_COUNT,
            "Unsupported parameter data type %d",
            parameter_data_type
        );

        ggml_type parameter_ggml_type = FORMAT_TYPE_TO_GGML_TYPE[parameter_data_type];
        RWKV_ASSERT_FALSE_MSG(
            RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_UNSUPPORTED,
            parameter_ggml_type != GGML_TYPE_UNKNOWN,
            "Unsupported parameter data type %d",
            parameter_data_type
        );

        int32_t nelements, x, y;

        if (n_dims == 1) {
            RWKV_ASSERT_FALSE(RWKV_ERROR_MODEL_PARAMS, read_int32(file_in, &x, "x"));
            y = 1;
            nelements = x;
        } else {
            RWKV_ASSERT_FALSE(RWKV_ERROR_MODEL_PARAMS, read_int32(file_in, &x, "x"));
            RWKV_ASSERT_FALSE(RWKV_ERROR_MODEL_PARAMS, read_int32(file_in, &y, "y"));
            nelements = x * y;
        }

        std::string name(key_length, 0);
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_KEY, fread(&name[0], key_length, 1, file_in) == 1, "Failed to read parameter key");

        printf("%48s - [%5d, %5d], type = %6s ", name.data(), x, y, ggml_type_name(parameter_ggml_type));
        total_size_orig += (size_t) (nelements * ggml_type_sizef(parameter_ggml_type));

        // Quantize only 2D tensors, except embedding and head matrices.
        // Embedding and head take not too much space, especially in bigger models;
        // but they significantly increase perplexity when quantized.
        bool quantize = n_dims == 2 && name != "emb.weight" && name != "head.weight";

        if (quantize) {
            RWKV_ASSERT_FALSE_MSG(
                RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DATA_TYPE,
                parameter_ggml_type == GGML_TYPE_F32 || parameter_data_type == GGML_TYPE_F16,
                "Unsupported parameter data type %d, only FP32 and FP16 can be quantized",
                parameter_ggml_type
            );

            data_f32.resize(nelements);

            if (parameter_data_type == GGML_TYPE_F16) {
                data_f16.resize(nelements);
                RWKV_ASSERT_FALSE_MSG(
                    RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DATA,
                    fread(data_f16.data(), nelements * sizeof(ggml_fp16_t), 1, file_in) == 1,
                    "Failed to read parameter data"
                );

                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                }
            } else {
                RWKV_ASSERT_FALSE_MSG(
                    RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DATA,
                    fread(data_f32.data(), nelements * sizeof(float), 1, file_in) == 1,
                    "Failed to read parameter data"
                );
            }

            parameter_data_type = format_data_type;
            parameter_ggml_type = format_ggml_type;
        } else {
            const size_t element_size = ggml_type_size(parameter_ggml_type);
            data_u8.resize(nelements * element_size);
            RWKV_ASSERT_FALSE_MSG(
                RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DATA,
                fread(data_u8.data(), nelements * element_size, 1, file_in) == 1,
                "Failed to read parameter data"
            );
        }

        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_int32(file_out, n_dims, "n_dims"));
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_int32(file_out, key_length, "key_length"));
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_int32(file_out, parameter_data_type, "parameter_data_type"));
            
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_int32(file_out, x, "x"));

        if (n_dims == 2) {
            RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, write_int32(file_out, y, "y"));
        }

        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_WRITE, fwrite(&name[0], key_length, 1, file_out) == 1, "Failed to write parameter key");

        if (quantize) {
            printf("quantizing... ");
            // For quantization
            work.resize(nelements);

            // This is a histogram of quantized values. If it shows single 1.0, then all 0.0, something went very wrong!
            std::vector<int64_t> hist_cur(1 << 4, 0);

            size_t (*f)(const float * src, void * dst, int n, int k, int64_t * hist) =
                format_ggml_type == GGML_TYPE_Q4_0 ? ggml_quantize_q4_0 :
                format_ggml_type == GGML_TYPE_Q4_1 ? ggml_quantize_q4_1 :
                format_ggml_type == GGML_TYPE_Q5_0 ? ggml_quantize_q5_0 :
                format_ggml_type == GGML_TYPE_Q5_1 ? ggml_quantize_q5_1 :
                format_ggml_type == GGML_TYPE_Q8_0 ? ggml_quantize_q8_0 :
                NULL;

            RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS | RWKV_ERROR_UNSUPPORTED, f, "Unsupported quantization type %d\n", format_ggml_type);

            size_t cur_size = (*f)(data_f32.data(), work.data(), nelements, x, hist_cur.data());
            total_size_new += cur_size;

            RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_WRITE, fwrite(work.data(), cur_size, 1, file_out) == 1, "Failed to write parameter data");

            printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float) / 1024.0 / 1024.0, cur_size / 1024.0 / 1024.0);

            for (int i = 0; i < (int) hist_cur.size(); ++i) {
                hist_all[i] += hist_cur[i];
            }

            for (int i = 0; i < (int) hist_cur.size(); ++i) {
                printf("%5.3f ", hist_cur[i] / float(nelements));
            }

            printf("\n");
        } else {
            printf("size = %8.3f MB\n", data_u8.size() / 1024.0 / 1024.0);
            RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_WRITE, fwrite(data_u8.data(), data_u8.size(), 1, file_out) == 1, "Failed to write parameter data");
            total_size_new += data_u8.size();
        }
    }

    printf("original size     = %8.2f MB\n", total_size_orig / 1024.0 / 1024.0);
    printf("quantized size    = %8.2f MB\n", total_size_new / 1024.0 / 1024.0);
    printf("compression ratio = %8.2f\n", 1.0 * total_size_orig / total_size_new);

    int64_t sum_all = 0;

    for (int i = 0; i < (int) hist_all.size(); ++i) {
        sum_all += hist_all[i];
    }

    printf("hist: ");

    for (int i = 0; i < (int) hist_all.size(); ++i) {
        printf("%5.3f ", hist_all[i] / float(sum_all));
    }

    printf("\n");

    return true;
}

const char * rwkv_get_system_info_string(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";

    return s.c_str();
}