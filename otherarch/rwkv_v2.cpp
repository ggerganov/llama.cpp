//adapted from RWKV.cpp repo under MIT license
// https://github.com/saharNooby/rwkv.cpp

#include "otherarch.h"

#include "rwkv_v2.h"
#include "ggml_v2.h"

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

#include "rwkv_vocab.cpp"

// --- Utilities ---

// Checks that x is not false. If x is false, prints fancy message to stderr and returns 0.
#define RWKV_V2_ASSERT_FALSE(x, ...) \
    do { \
        if (!(x)) { \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
            return false; \
        } \
    } while (0)

// Checks that x is not false. If x is false, prints fancy message to stderr and returns NULL.
#define RWKV_V2_ASSERT_NULL(x, ...) \
    do { \
        if (!(x)) { \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
            return NULL; \
        } \
    } while (0)

// Reads single int32 value from a file.
bool rwkv_v2_read_int32(FILE * file, int32_t * dest) {
    RWKV_V2_ASSERT_FALSE(fread(dest, 4, 1, file) == 1, "Failed to read an int32 value from a file");
    return true;
}

#define GGML_V2_TYPE_UNKNOWN GGML_V2_TYPE_COUNT

#define RWKV_V2_FORMAT_TYPE_COUNT 10

static const ggml_v2_type FORMAT_TYPE_TO_GGML_V2_TYPE[RWKV_V2_FORMAT_TYPE_COUNT] = {
    GGML_V2_TYPE_F32,
    GGML_V2_TYPE_F16,
    GGML_V2_TYPE_Q4_0,
    GGML_V2_TYPE_Q4_1,
    GGML_V2_TYPE_UNKNOWN, // Unused
    GGML_V2_TYPE_Q4_2,
    GGML_V2_TYPE_UNKNOWN, // Unused
    GGML_V2_TYPE_Q5_0,
    GGML_V2_TYPE_Q5_1,
    GGML_V2_TYPE_Q8_0
};

static int32_t rwkv_v2_format_name_to_format_type(const char * format_name) {
    if (strcmp(format_name, "Q4_0") == 0) return 2;
    if (strcmp(format_name, "Q4_1") == 0) return 3;
    if (strcmp(format_name, "Q4_2") == 0) return 5;
    if (strcmp(format_name, "Q5_0") == 0) return 7;
    if (strcmp(format_name, "Q5_1") == 0) return 8;
    if (strcmp(format_name, "Q8_0") == 0) return 9;

    return -1;
}

// --- Model definition and loading utilities ---

struct rwkv_v2_layer {
    struct ggml_v2_tensor * ln1_weight;
    struct ggml_v2_tensor * ln1_bias;

    // RWKV, also called "attention" by the author.
    struct ggml_v2_tensor * att_time_mix_k;
    struct ggml_v2_tensor * att_time_mix_v;
    struct ggml_v2_tensor * att_time_mix_r;
    struct ggml_v2_tensor * att_time_first;
    struct ggml_v2_tensor * att_time_decay;
    struct ggml_v2_tensor * att_key;
    struct ggml_v2_tensor * att_value;
    struct ggml_v2_tensor * att_receptance;
    struct ggml_v2_tensor * att_output;

    struct ggml_v2_tensor * ln2_weight;
    struct ggml_v2_tensor * ln2_bias;

    // FFN.
    struct ggml_v2_tensor * ffn_time_mix_k;
    struct ggml_v2_tensor * ffn_time_mix_r;
    struct ggml_v2_tensor * ffn_key;
    struct ggml_v2_tensor * ffn_value;
    struct ggml_v2_tensor * ffn_receptance;
};

struct rwkv_v2_model {
    int32_t n_vocab;
    int32_t n_layer;
    int32_t n_embed;
    // 0 for float32, 1 for float16.
    int32_t data_type;

    struct ggml_v2_tensor * emb;

    struct ggml_v2_tensor * ln0_weight;
    struct ggml_v2_tensor * ln0_bias;

    std::vector<rwkv_v2_layer> layers;

    struct ggml_v2_tensor * ln_out_weight;
    struct ggml_v2_tensor * ln_out_bias;

    struct ggml_v2_tensor * head;
};

// Finds model parameter by key and sets it into dest.
// If the parameter was not found, returns false.
bool rwkv_v2_set_parameter(std::unordered_map<std::string, struct ggml_v2_tensor *> * parameters, char * key, struct ggml_v2_tensor ** dest) {
    struct ggml_v2_tensor * parameter = (*parameters)[key];
    RWKV_V2_ASSERT_FALSE(parameter != NULL, "Parameter %s not found in model file", key);
    *dest = parameter;
    return true;
}

// Finds block parameter by block index and key and sets it into dest.
// If the parameter was not found, returns false.
bool rwkv_v2_set_block_parameter(std::unordered_map<std::string, struct ggml_v2_tensor *> * parameters, int32_t block_index, char * key, struct ggml_v2_tensor ** dest) {
    char full_key[128];
    sprintf(full_key, "blocks.%d.%s", block_index, key);
    return rwkv_v2_set_parameter(parameters, full_key, dest);
}

// --- Operators ---

void rwkv_v2_exp_impl(const int n_cols, float * dest, const float * src) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = expf(src[i]);
    }
}

void rwkv_v2_1_minus_x_impl(const int n_cols, float * dest, const float * src) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = 1.0F - src[i];
    }
}

void rwkv_v2_sigmoid_impl(const int n_cols, float * dest, const float * src) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = 1.0F / (1.0F + expf(-src[i]));
    }
}

void rwkv_v2_max_impl(const int n_cols, float * dest, const float * src0, const float * src1) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = fmaxf(src0[i], src1[i]);
    }
}

struct ggml_v2_tensor * rwkv_v2_exp(ggml_v2_context * ctx, struct ggml_v2_tensor * x) {
    return ggml_v2_map_unary_f32(ctx, x, rwkv_v2_exp_impl);
}

struct ggml_v2_tensor * rwkv_v2_1_minus_x(ggml_v2_context * ctx, struct ggml_v2_tensor * x) {
    return ggml_v2_map_unary_f32(ctx, x, rwkv_v2_1_minus_x_impl);
}

struct ggml_v2_tensor * rwkv_v2_sigmoid(ggml_v2_context * ctx, struct ggml_v2_tensor * x) {
    return ggml_v2_map_unary_f32(ctx, x, rwkv_v2_sigmoid_impl);
}

struct ggml_v2_tensor * rwkv_v2_max(ggml_v2_context * ctx, struct ggml_v2_tensor * x, struct ggml_v2_tensor * y) {
    return ggml_v2_map_binary_f32(ctx, x, y, rwkv_v2_max_impl);
}

struct ggml_v2_tensor * rwkv_v2_layer_norm(ggml_v2_context * ctx, struct ggml_v2_tensor * x, struct ggml_v2_tensor * weight, struct ggml_v2_tensor * bias) {
    // LayerNorm in RWKV is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // Looks like ggml_v2_norm does the first part, we only need to apply weight & bias.
    x = ggml_v2_norm(ctx, x);
    x = ggml_v2_mul(ctx, x, weight);
    x = ggml_v2_add(ctx, x, bias);
    return x;
}

// --- Implementation ---

struct rwkv_v2_context {
    struct rwkv_v2_model * model;
    struct ggml_v2_tensor * token_index;
    struct ggml_v2_tensor * state;
    struct ggml_v2_tensor ** state_parts;
    struct ggml_v2_tensor * logits;
    struct ggml_v2_context * ctx;
    struct ggml_v2_cgraph * graph;
    bool freed;
    float * state_in = 0; //stores input state, or use null for a new state
    float * state_out = 0; //stores address of output state buffer
    float * logits_out = 0; //stores address of output logit buffer
};

struct rwkv_v2_context * rwkv_v2_init_from_file(const char * file_path, uint32_t n_threads) {
    FILE * file = fopen(file_path, "rb");
    RWKV_V2_ASSERT_NULL(file != NULL, "Failed to open file %s", file_path);

    int32_t magic;
    rwkv_v2_read_int32(file, &magic);
    RWKV_V2_ASSERT_NULL(magic == RWKV_V2_FILE_MAGIC, "Unexpected magic value %d", magic);

    int32_t version;
    rwkv_v2_read_int32(file, &version);
    RWKV_V2_ASSERT_NULL(version == RWKV_V2_FILE_VERSION, "Unsupported file version %d", version);

    struct rwkv_v2_model * model = (struct rwkv_v2_model *) calloc(1, sizeof(struct rwkv_v2_model));

    rwkv_v2_read_int32(file, &(model->n_vocab));
    RWKV_V2_ASSERT_NULL(model->n_vocab > 0, "Non-positive n_vocab %d", model->n_vocab);

    rwkv_v2_read_int32(file, &(model->n_embed));
    RWKV_V2_ASSERT_NULL(model->n_embed > 0, "Non-positive n_embed %d", model->n_embed);

    rwkv_v2_read_int32(file, &(model->n_layer));
    RWKV_V2_ASSERT_NULL(model->n_layer > 0, "Non-positive n_layer %d", model->n_layer);

    rwkv_v2_read_int32(file, &(model->data_type));
    RWKV_V2_ASSERT_NULL(model->data_type >= 0 && model->data_type < RWKV_V2_FORMAT_TYPE_COUNT, "Unsupported model data type %d", model->data_type);

    RWKV_V2_ASSERT_NULL(
        model->data_type != 4,
        "Models in Q4_1_O format cannot be loaded anymore because the format was removed. You need to quantize the model into another format"
    );

    RWKV_V2_ASSERT_NULL(
        model->data_type != 6,
        "Models in Q4_3 format cannot be loaded anymore because the format was removed. You need to quantize the model into another format"
    );

    // Parameter tensors would take at least this amount in memory.
    size_t file_size;

    {
        auto fin = std::ifstream(file_path, std::ios::binary);
        RWKV_V2_ASSERT_NULL(fin, "Failed to open file %s", file_path);
        fin.seekg(0, fin.end);
        file_size = fin.tellg();
        fin.close();
    }

    size_t memory_required = file_size +
        // Intermediary vectors for calculation; there are around 100 calls to ggml
        size_t(100) * model->n_embed * sizeof(float) +
        // State, in and out
        size_t(2) * 5 * model->n_layer * model->n_embed * sizeof(float) +
        // Logits
        size_t(model->n_vocab) * sizeof(float) +
        // +256 MB just for any overhead
        // TODO This is too much for smaller models; need a more proper and robust way of measuring required memory
        size_t(256) * 1024 * 1024;

    // Initialize ggml
    struct ggml_v2_init_params params;
    params.mem_size = memory_required;
    params.mem_buffer = NULL;
    params.no_alloc = false;
    struct ggml_v2_context * ctx = ggml_v2_init(params);

    std::unordered_map<std::string, struct ggml_v2_tensor *> parameters;

    while (true) {
        int32_t dim_count;
        size_t elements_read = fread(&dim_count, 4, 1, file);

        if (feof(file)) {
            break;
        }

        RWKV_V2_ASSERT_NULL(elements_read == 1, "Failed to read dimension count");
        RWKV_V2_ASSERT_NULL(dim_count == 1 || dim_count == 2, "Unsupported dimension count %d", dim_count);

        int32_t key_length;
        rwkv_v2_read_int32(file, &key_length);
        RWKV_V2_ASSERT_NULL(key_length > 0, "Non-positive key length %d", key_length);

        int32_t data_type;
        rwkv_v2_read_int32(file, &data_type);
        RWKV_V2_ASSERT_NULL(data_type >= 0 && data_type < RWKV_V2_FORMAT_TYPE_COUNT, "Unsupported parameter data type %d", data_type);

        ggml_v2_type ggml_v2_data_type = FORMAT_TYPE_TO_GGML_V2_TYPE[data_type];

        RWKV_V2_ASSERT_NULL(ggml_v2_data_type != GGML_V2_TYPE_UNKNOWN, "Unsupported parameter data type %d", data_type);

        struct ggml_v2_tensor * tensor;

        int32_t x = -1;
        int32_t y = -1;

        if (dim_count == 1) {
            rwkv_v2_read_int32(file, &x);
            tensor = ggml_v2_new_tensor_1d(ctx, ggml_v2_data_type, x);
        } else if (dim_count == 2) {
            rwkv_v2_read_int32(file, &x);
            rwkv_v2_read_int32(file, &y);
            tensor = ggml_v2_new_tensor_2d(ctx, ggml_v2_data_type, x, y);
        } else {
            abort();
        }

        std::string key(key_length, 0);
        RWKV_V2_ASSERT_NULL(fread(&key[0], 1, key_length, file) == uint32_t(key_length), "Failed to read parameter key");

        RWKV_V2_ASSERT_NULL(fread(tensor->data, 1, ggml_v2_nbytes(tensor), file) == ggml_v2_nbytes(tensor), "Failed to read parameter data");

        parameters[key] = tensor;
    }

    fclose(file);

    model->layers.resize(model->n_layer);

    rwkv_v2_set_parameter(&parameters, "emb.weight", &(model->emb));

    rwkv_v2_set_parameter(&parameters, "blocks.0.ln0.weight", &(model->ln0_weight));
    rwkv_v2_set_parameter(&parameters, "blocks.0.ln0.bias", &(model->ln0_bias));

    for (int i = 0; i < model->n_layer; i++) {
        rwkv_v2_layer layer = model->layers[i];

        rwkv_v2_set_block_parameter(&parameters, i, "ln1.weight", &(layer.ln1_weight));
        rwkv_v2_set_block_parameter(&parameters, i, "ln1.bias", &(layer.ln1_bias));

        rwkv_v2_set_block_parameter(&parameters, i, "att.time_mix_k", &(layer.att_time_mix_k));
        rwkv_v2_set_block_parameter(&parameters, i, "att.time_mix_v", &(layer.att_time_mix_v));
        rwkv_v2_set_block_parameter(&parameters, i, "att.time_mix_r", &(layer.att_time_mix_r));
        rwkv_v2_set_block_parameter(&parameters, i, "att.time_first", &(layer.att_time_first));
        rwkv_v2_set_block_parameter(&parameters, i, "att.time_decay", &(layer.att_time_decay));
        rwkv_v2_set_block_parameter(&parameters, i, "att.key.weight", &(layer.att_key));
        rwkv_v2_set_block_parameter(&parameters, i, "att.value.weight", &(layer.att_value));
        rwkv_v2_set_block_parameter(&parameters, i, "att.receptance.weight", &(layer.att_receptance));
        rwkv_v2_set_block_parameter(&parameters, i, "att.output.weight", &(layer.att_output));

        rwkv_v2_set_block_parameter(&parameters, i, "ln2.weight", &(layer.ln2_weight));
        rwkv_v2_set_block_parameter(&parameters, i, "ln2.bias", &(layer.ln2_bias));

        rwkv_v2_set_block_parameter(&parameters, i, "ffn.time_mix_k", &(layer.ffn_time_mix_k));
        rwkv_v2_set_block_parameter(&parameters, i, "ffn.time_mix_r", &(layer.ffn_time_mix_r));
        rwkv_v2_set_block_parameter(&parameters, i, "ffn.key.weight", &(layer.ffn_key));
        rwkv_v2_set_block_parameter(&parameters, i, "ffn.value.weight", &(layer.ffn_value));
        rwkv_v2_set_block_parameter(&parameters, i, "ffn.receptance.weight", &(layer.ffn_receptance));

        model->layers[i] = layer;
    }

    rwkv_v2_set_parameter(&parameters, "ln_out.weight", &(model->ln_out_weight));
    rwkv_v2_set_parameter(&parameters, "ln_out.bias", &(model->ln_out_bias));

    rwkv_v2_set_parameter(&parameters, "head.weight", &(model->head));

    // Verify order of dimensions
    struct ggml_v2_tensor * emb = model->emb;
    RWKV_V2_ASSERT_NULL(emb->n_dims == 2, "Unexpected dimension count of embedding matrix %d", emb->n_dims);
    RWKV_V2_ASSERT_NULL(emb->ne[0] == model->n_embed, "Unexpected dimension of embedding matrix %lld", emb->ne[0]);
    RWKV_V2_ASSERT_NULL(emb->ne[1] == model->n_vocab, "Unexpected dimension of embedding matrix %lld", emb->ne[1]);

    int32_t n_embed = model->n_embed;
    int32_t n_layer = model->n_layer;

    // Build graph
    struct ggml_v2_tensor * state = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32, n_layer * 5 * n_embed);

    // x = self.w.emb.weight[token]
    struct ggml_v2_tensor * token_index = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_I32, 1);
    struct ggml_v2_tensor * x = ggml_v2_get_rows(ctx, model->emb, token_index);

    // x = self.layer_norm(x, self.w.blocks[0].ln0)
    x = rwkv_v2_layer_norm(ctx, x, model->ln0_weight, model->ln0_bias);

    // We collect parts of new state here. Each part is (n_embed) vector.
    struct ggml_v2_tensor ** state_parts = new ggml_v2_tensor * [n_layer * 5];

    for (int i = 0; i < n_layer; i++) {
        auto layer = model->layers[i];

        // RWKV/time mixing
        {
            // self.layer_norm(x, self.w.blocks[i].ln1)
            struct ggml_v2_tensor * x0 = rwkv_v2_layer_norm(ctx, x, layer.ln1_weight, layer.ln1_bias);
            // state[5 * i + 1]
            struct ggml_v2_tensor * x_prev = ggml_v2_view_1d(ctx, state, n_embed, (5 * i + 1) * n_embed * sizeof(float));
            // xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
            // xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
            // xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
            struct ggml_v2_tensor * xk = ggml_v2_add(
                ctx,
                ggml_v2_mul(ctx, x0, layer.att_time_mix_k),
                ggml_v2_mul(ctx, x_prev, rwkv_v2_1_minus_x(ctx, layer.att_time_mix_k))
            );
            struct ggml_v2_tensor * xv = ggml_v2_add(
                ctx,
                ggml_v2_mul(ctx, x0, layer.att_time_mix_v),
                ggml_v2_mul(ctx, x_prev, rwkv_v2_1_minus_x(ctx, layer.att_time_mix_v))
            );
            struct ggml_v2_tensor * xr = ggml_v2_add(
                ctx,
                ggml_v2_mul(ctx, x0, layer.att_time_mix_r),
                ggml_v2_mul(ctx, x_prev, rwkv_v2_1_minus_x(ctx, layer.att_time_mix_r))
            );
            // state[5 * i + 1] = x
            state_parts[5 * i + 1] = x0;

            // r = torch.sigmoid(rw @ xr)
            struct ggml_v2_tensor * r = rwkv_v2_sigmoid(
                ctx,
                ggml_v2_mul_mat(ctx, layer.att_receptance, xr)
            );
            // k = kw @ xk
            struct ggml_v2_tensor * k = ggml_v2_mul_mat(ctx, layer.att_key, xk);
            // v = vw @ xv
            struct ggml_v2_tensor * v = ggml_v2_mul_mat(ctx, layer.att_value, xv);

            // aa = state[5 * i + 2]
            // bb = state[5 * i + 3]
            // pp = state[5 * i + 4]
            struct ggml_v2_tensor * aa = ggml_v2_view_1d(ctx, state, n_embed, (5 * i + 2) * n_embed * sizeof(float));
            struct ggml_v2_tensor * bb = ggml_v2_view_1d(ctx, state, n_embed, (5 * i + 3) * n_embed * sizeof(float));
            struct ggml_v2_tensor * pp = ggml_v2_view_1d(ctx, state, n_embed, (5 * i + 4) * n_embed * sizeof(float));

            // ww = time_first + k
            struct ggml_v2_tensor * ww = ggml_v2_add(ctx, layer.att_time_first, k);
            // qq = torch.maximum(pp, ww)
            struct ggml_v2_tensor * qq = rwkv_v2_max(ctx, pp, ww);
            // e1 = torch.exp(pp - qq)
            struct ggml_v2_tensor * e1 = rwkv_v2_exp(ctx, ggml_v2_sub(ctx, pp, qq));
            // e2 = torch.exp(ww - qq)
            struct ggml_v2_tensor * e2 = rwkv_v2_exp(ctx, ggml_v2_sub(ctx, ww, qq));
            // a = e1 * aa + e2 * v
            struct ggml_v2_tensor * a = ggml_v2_add(
                ctx,
                ggml_v2_mul(ctx, e1, aa),
                ggml_v2_mul(ctx, e2, v)
            );
            // b = e1 * bb + e2
            struct ggml_v2_tensor * b = ggml_v2_add(
                ctx,
                ggml_v2_mul(ctx, e1, bb),
                e2
            );
            // wkv = a / b
            struct ggml_v2_tensor * wkv = ggml_v2_div(ctx, a, b);
            // ww = pp + time_decay
            ww = ggml_v2_add(ctx, pp, layer.att_time_decay);
            // qq = torch.maximum(ww, k)
            qq = rwkv_v2_max(ctx, ww, k);
            // e1 = torch.exp(ww - qq)
            e1 = rwkv_v2_exp(ctx, ggml_v2_sub(ctx, ww, qq));
            // e2 = torch.exp(k - qq)
            e2 = rwkv_v2_exp(ctx, ggml_v2_sub(ctx, k, qq));
            // state[5 * i + 2] = e1 * aa + e2 * v
            state_parts[5 * i + 2] = ggml_v2_add(
                ctx,
                ggml_v2_mul(ctx, e1, aa),
                ggml_v2_mul(ctx, e2, v)
            );
            // state[5 * i + 3] = e1 * bb + e2
            state_parts[5 * i + 3] = ggml_v2_add(
                ctx,
                ggml_v2_mul(ctx, e1, bb),
                e2
            );
            // state[5 * i + 4] = qq
            state_parts[5 * i + 4] = qq;
            // ow @ (r * wkv)
            x = ggml_v2_add(
                ctx,
                x,
                ggml_v2_mul_mat(
                    ctx,
                    layer.att_output,
                    ggml_v2_mul(ctx, r, wkv)
                )
            );
        }

        // FFN/channel mixing
        {
            // self.layer_norm(x, self.w.blocks[i].ln2)
            struct ggml_v2_tensor * x0 = rwkv_v2_layer_norm(ctx, x, layer.ln2_weight, layer.ln2_bias);
            // state[5 * i + 0]
            struct ggml_v2_tensor * x_prev = ggml_v2_view_1d(ctx, state, n_embed, (5 * i + 0) * n_embed * sizeof(float));
            // xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
            // xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
            struct ggml_v2_tensor * xk = ggml_v2_add(
                ctx,
                ggml_v2_mul(ctx, x0, layer.ffn_time_mix_k),
                ggml_v2_mul(ctx, x_prev, rwkv_v2_1_minus_x(ctx, layer.ffn_time_mix_k))
            );
            struct ggml_v2_tensor * xr = ggml_v2_add(
                ctx,
                ggml_v2_mul(ctx, x0, layer.ffn_time_mix_r),
                ggml_v2_mul(ctx, x_prev, rwkv_v2_1_minus_x(ctx, layer.ffn_time_mix_r))
            );
            // state[5 * i + 0] = x
            state_parts[5 * i + 0] = x0;

            // r = torch.sigmoid(rw @ xr)
            struct ggml_v2_tensor * r = rwkv_v2_sigmoid(
                ctx,
                ggml_v2_mul_mat(ctx, layer.ffn_receptance, xr)
            );
            // k = torch.square(torch.relu(kw @ xk))
            struct ggml_v2_tensor * k = ggml_v2_sqr(ctx, ggml_v2_relu(
                ctx,
                ggml_v2_mul_mat(ctx, layer.ffn_key, xk)
            ));
            // r * (vw @ k)
            x = ggml_v2_add(
                ctx,
                x,
                ggml_v2_mul(
                    ctx,
                    r,
                    ggml_v2_mul_mat(ctx, layer.ffn_value, k)
                )
            );
        }
    }

    // x = self.layer_norm(x, self.w.ln_out)
    x = rwkv_v2_layer_norm(ctx, x, model->ln_out_weight, model->ln_out_bias);

    // x = (self.w.head.weight @ x).float()
    struct ggml_v2_tensor * logits = ggml_v2_mul_mat(ctx, model->head, x);

    struct ggml_v2_cgraph * graph = (struct ggml_v2_cgraph *) calloc(1, sizeof(struct ggml_v2_cgraph));

    *graph = ggml_v2_build_forward(logits);

    for (int i = 0; i < n_layer * 5; i++) {
       ggml_v2_build_forward_expand(graph, state_parts[i]);
    }

    graph->n_threads = n_threads;

    struct rwkv_v2_context * rwkv_ctx = (struct rwkv_v2_context *) calloc(1, sizeof(struct rwkv_v2_context));
    rwkv_ctx->model = model;
    rwkv_ctx->token_index = token_index;
    rwkv_ctx->state = state;
    rwkv_ctx->state_parts = state_parts;
    rwkv_ctx->logits = logits;
    rwkv_ctx->ctx = ctx;
    rwkv_ctx->graph = graph;
    return rwkv_ctx;
}

uint32_t rwkv_v2_get_state_buffer_element_count(struct rwkv_v2_context * ctx) {
    return ctx->model->n_layer * 5 * ctx->model->n_embed;
}

uint32_t rwkv_v2_get_logits_buffer_element_count(struct rwkv_v2_context * ctx) {
    return ctx->model->n_vocab;
}

bool rwkv_v2_eval(struct rwkv_v2_context * ctx, int32_t token, float * state_in, float * state_out, float * logits_out) {
    RWKV_V2_ASSERT_FALSE(state_out != NULL, "state_out is NULL");
    RWKV_V2_ASSERT_FALSE(logits_out != NULL, "logits_out is NULL");

    int32_t n_layer = ctx->model->n_layer;
    int32_t n_embed = ctx->model->n_embed;
    int32_t n_vocab = ctx->model->n_vocab;

    RWKV_V2_ASSERT_FALSE(token >= 0 && token < n_vocab, "Token is out of range 0..%d", n_vocab - 1);

    ggml_v2_set_i32_1d(ctx->token_index, 0, token);

    if (state_in == NULL) {
        ggml_v2_set_f32(ctx->state, 0.0F);

        for (int i = 0; i < n_layer; i++) {
            // state[5 * i + 4] = -1e30
            ggml_v2_set_f32(
                ggml_v2_view_1d(ctx->ctx, ctx->state, n_embed, (5 * i + 4) * n_embed * sizeof(float)),
                -1e30F
            );
        }
    } else {
        memcpy(ctx->state->data, state_in, ctx->state->ne[0] * sizeof(float));
    }

    ggml_v2_graph_compute(ctx->ctx, ctx->graph);

    for (size_t i = 0; i < size_t(n_layer * 5); i++) {
        struct ggml_v2_tensor * part = ctx->state_parts[i];

        memcpy(state_out + i * n_embed, part->data, part->ne[0] * sizeof(float));
    }

    memcpy(logits_out, ctx->logits->data, ctx->logits->ne[0] * sizeof(float));

    return true;
}

void rwkv_v2_free(struct rwkv_v2_context * ctx) {
    ctx->model->layers.~vector();
    free(ctx->model);
    delete[] ctx->state_parts;
    ggml_v2_free(ctx->ctx);
    free(ctx->graph);
    free(ctx);
}

bool rwkv_v2_quantize_model_file(const char * model_file_path_in, const char * model_file_path_out, const char * format_name) {
    int32_t format_type = rwkv_v2_format_name_to_format_type(format_name);

    RWKV_V2_ASSERT_FALSE(format_type != -1, "Unsupported format \"%s\"", format_name);

    ggml_v2_type type = FORMAT_TYPE_TO_GGML_V2_TYPE[format_type];

    RWKV_V2_ASSERT_FALSE(type != GGML_V2_TYPE_UNKNOWN, "Unsupported format \"%s\"", format_name);

    // Needed to initialize FP16 lookup table
    {
        struct ggml_v2_init_params params = { 0, NULL, false };
        struct ggml_v2_context * ctx = ggml_v2_init(params);
        ggml_v2_free(ctx);
    }

    printf("Loading model from '%s'\n", model_file_path_in);

    auto finp = std::ifstream(model_file_path_in, std::ios::binary);
    RWKV_V2_ASSERT_FALSE(finp, "Failed to open %s for reading", model_file_path_in);

    auto fout = std::ofstream(model_file_path_out, std::ios::binary);
    RWKV_V2_ASSERT_FALSE(fout, "Failed to open %s for writing", model_file_path_out);

    // Process header
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        RWKV_V2_ASSERT_FALSE(magic == RWKV_V2_FILE_MAGIC, "Unexpected magic value %d", magic);
        fout.write((char *) &magic, sizeof(magic));

        uint32_t format_version;
        finp.read((char *) &format_version, sizeof(format_version));
        RWKV_V2_ASSERT_FALSE(format_version == RWKV_V2_FILE_VERSION, "Unsupported file version %d", format_version);
        fout.write((char *) &format_version, sizeof(format_version));

        int32_t n_vocab;
        int32_t n_embed;
        int32_t n_layer;
        int32_t data_type;

        finp.read((char *) &n_vocab, sizeof(n_vocab));
        finp.read((char *) &n_embed, sizeof(n_embed));
        finp.read((char *) &n_layer, sizeof(n_layer));
        finp.read((char *) &data_type, sizeof(data_type));

        RWKV_V2_ASSERT_FALSE(data_type == 0 || data_type == 1, "Unsupported data type %d, only FP32 and FP16 can be quantized", data_type);

        data_type = format_type;

        fout.write((char *) &n_vocab, sizeof(n_vocab));
        fout.write((char *) &n_embed, sizeof(n_embed));
        fout.write((char *) &n_layer, sizeof(n_layer));
        fout.write((char *) &data_type, sizeof(data_type));
    }

    // Process parameters
    {
        size_t total_size_orig = 0;
        size_t total_size_new = 0;

        std::vector<float> work;

        std::vector<uint8_t>     data_u8;
        std::vector<ggml_v2_fp16_t> data_f16;
        std::vector<float>       data_f32;

        std::vector<int64_t> hist_all(1 << 4, 0);

        while (true) {
            int32_t n_dims;
            int32_t key_length;
            int32_t parameter_data_type;

            finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            finp.read(reinterpret_cast<char *>(&key_length), sizeof(key_length));
            finp.read(reinterpret_cast<char *>(&parameter_data_type),  sizeof(parameter_data_type));

            if (finp.eof()) {
                break;
            }

            RWKV_V2_ASSERT_FALSE(parameter_data_type >= 0 && parameter_data_type < RWKV_V2_FORMAT_TYPE_COUNT, "Invalid parameter data type %d", parameter_data_type);

            ggml_v2_type parameter_ggml_v2_type = FORMAT_TYPE_TO_GGML_V2_TYPE[parameter_data_type];

            RWKV_V2_ASSERT_FALSE(parameter_ggml_v2_type != GGML_V2_TYPE_UNKNOWN, "Invalid parameter data type %d", parameter_data_type);

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(key_length, 0);
            finp.read(&name[0], key_length);

            {
                printf("%48s - [%5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ggml_v2_type_name(parameter_ggml_v2_type));

                total_size_orig += (size_t) (nelements * ggml_v2_type_sizef(parameter_ggml_v2_type));
            }

            // Quantize only 2D tensors, except embedding and head matrices.
            // Embedding and head take not too much space, especially in bigger models;
            // but they significantly increase perplexity when quantized.
            bool quantize = n_dims == 2 &&
                    name != std::string("emb.weight") &&
                    name != std::string("head.weight");

            if (quantize) {
                RWKV_V2_ASSERT_FALSE(
                    parameter_data_type == 0 || parameter_data_type == 1,
                    "Unsupported parameter data type %d, only FP32 and FP16 can be quantized",
                    parameter_data_type
                );

                if (parameter_data_type == 1) {
                    data_f16.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_v2_fp16_t));
                    data_f32.resize(nelements);
                    for (int i = 0; i < nelements; ++i) {
                        data_f32[i] = ggml_v2_fp16_to_fp32(data_f16[i]);
                    }
                } else {
                    data_f32.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
                }

                parameter_data_type = format_type;
            } else {
                const int bytes_per_element = (parameter_data_type == 0) ? sizeof(float) : sizeof(uint16_t);
                data_u8.resize(nelements * bytes_per_element);
                finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bytes_per_element);
            }

            fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fout.write(reinterpret_cast<char *>(&key_length), sizeof(key_length));
            fout.write(reinterpret_cast<char *>(&parameter_data_type),  sizeof(parameter_data_type));

            for (int i = 0; i < n_dims; ++i) {
                fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            }

            fout.write(&name[0], key_length);

            if (quantize) {
                printf("quantizing... ");
                work.resize(nelements); // for quantization

                size_t cur_size = 0;
                // This is a histogramm of some values. If it shows single 1.0, then all 0.0, something went very wrong!
                std::vector<int64_t> hist_cur(1 << 4, 0);

                switch (type) {
                    case GGML_V2_TYPE_Q4_0:
                        cur_size = ggml_v2_quantize_q4_0_v2(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                        break;
                    case GGML_V2_TYPE_Q4_1:
                        cur_size = ggml_v2_quantize_q4_1_v2(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                        break;
                    case GGML_V2_TYPE_Q4_2:
                        cur_size = ggml_v2_quantize_q4_2_v2(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                        break;
                    case GGML_V2_TYPE_Q5_0:
                        cur_size = ggml_v2_quantize_q5_0_v2(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                        break;
                    case GGML_V2_TYPE_Q5_1:
                        cur_size = ggml_v2_quantize_q5_1_v2(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                        break;
                    case GGML_V2_TYPE_Q8_0:
                        cur_size = ggml_v2_quantize_q8_0_v2(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                        break;
                    default: {
                        fprintf(stderr, "unsupported quantization type %d\n", type);
                        return false;
                    }
                }

                fout.write(reinterpret_cast<char *>(work.data()), cur_size);
                total_size_new += cur_size;

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
                fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
                total_size_new += data_u8.size();
            }
        }

        printf("original size     = %8.2f MB\n", total_size_orig / 1024.0 / 1024.0);
        printf("quantized size    = %8.2f MB\n", total_size_new / 1024.0 / 1024.0);
        printf("compression ratio = %8.2f\n", 1.0 * total_size_orig / total_size_new);

        {
            int64_t sum_all = 0;

            for (int i = 0; i < (int) hist_all.size(); ++i) {
                sum_all += hist_all[i];
            }

            printf("hist: ");

            for (int i = 0; i < (int) hist_all.size(); ++i) {
                printf("%5.3f ", hist_all[i] / float(sum_all));
            }

            printf("\n");
        }
    }

    finp.close();
    fout.close();

    return true;
}

const char * rwkv_v2_get_system_info_string(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_v2_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_v2_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_v2_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_v2_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_v2_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_v2_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_v2_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_v2_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_v2_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_v2_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_v2_cpu_has_sse3())      + " | ";
    s += "VSX = "       + std::to_string(ggml_v2_cpu_has_vsx())       + " | ";

    return s.c_str();
}