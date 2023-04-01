#include "rwkv.h"
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

// --- Utilities ---

#define FP32_SIZE 4

// Checks that x is not false. If x is false, prints fancy message to stderr and returns 0.
#define RWKV_ASSERT_FALSE(x, ...) \
    do { \
        if (!(x)) { \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
            return false; \
        } \
    } while (0)

// Checks that x is not false. If x is false, prints fancy message to stderr and returns NULL.
#define RWKV_ASSERT_NULL(x, ...) \
    do { \
        if (!(x)) { \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
            return NULL; \
        } \
    } while (0)

// Reads single int32 value from a file.
bool read_int32(FILE * file, int32_t * dest) {
    // TODO Will not read correct values on machine with different endianness
    RWKV_ASSERT_FALSE(fread(dest, 4, 1, file) == 1, "Failed to read an int32 value from a file");
    return true;
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
    int32_t n_vocab;
    int32_t n_embed;
    int32_t n_layer;
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
bool set_parameter(std::unordered_map<std::string, struct ggml_tensor *> * parameters, char * key, struct ggml_tensor ** dest) {
    struct ggml_tensor * parameter = (*parameters)[key];
    RWKV_ASSERT_FALSE(parameter != NULL, "Parameter %s not found in model file", key);
    *dest = parameter;
    return true;
}

// Finds block parameter by block index and key and sets it into dest.
// If the parameter was not found, returns false.
bool set_block_parameter(std::unordered_map<std::string, struct ggml_tensor *> * parameters, int32_t block_index, char * key, struct ggml_tensor ** dest) {
    char full_key[128];
    sprintf(full_key, "blocks.%d.%s", block_index, key);
    return set_parameter(parameters, full_key, dest);
}

// --- Operators ---

struct ggml_tensor * rwkv_layer_norm(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // LayerNorm in RWKV is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // Looks like ggml_norm does the first part, we only need to apply weight & bias.
    x = ggml_norm(ctx, x);
    x = ggml_mul(ctx, x, weight);
    x = ggml_add(ctx, x, bias);
    return x;
}

// --- Implementation ---

struct rwkv_context {
    struct rwkv_model * model;
    struct ggml_tensor * token_index;
    struct ggml_tensor * state;
    struct ggml_tensor ** state_parts;
    struct ggml_tensor * logits;
    struct ggml_context * ctx;
    struct ggml_cgraph * graph;
    bool freed;
};

struct rwkv_context * rwkv_init_from_file(const char * file_path, int n_threads) {
    FILE * file = fopen(file_path, "rb");
    RWKV_ASSERT_NULL(file != NULL, "Failed to open file %s", file_path);

    int32_t magic;
    read_int32(file, &magic);
    RWKV_ASSERT_NULL(magic == RWKV_FILE_MAGIC, "Unexpected magic value %d", magic);

    int32_t version;
    read_int32(file, &version);
    RWKV_ASSERT_NULL(version == RWKV_FILE_VERSION, "Unsupported file version %d", version);

    struct rwkv_model * model = (struct rwkv_model *) calloc(1, sizeof(struct rwkv_model));

    read_int32(file, &(model->n_vocab));
    RWKV_ASSERT_NULL(model->n_vocab > 0, "Non-positive n_vocab %d", model->n_vocab);

    read_int32(file, &(model->n_embed));
    RWKV_ASSERT_NULL(model->n_embed > 0, "Non-positive n_embed %d", model->n_embed);

    read_int32(file, &(model->n_layer));
    RWKV_ASSERT_NULL(model->n_layer > 0, "Non-positive n_layer %d", model->n_layer);

    read_int32(file, &(model->data_type));
    RWKV_ASSERT_NULL(model->data_type == 0 || model->data_type == 1, "Unsupported model data type %d", model->data_type);

    // Initialize ggml
    struct ggml_init_params params;
    // TODO Calculate required memory (automatically or manually)
    params.mem_size = 1024 * 1024 * 1024;
    params.mem_buffer = NULL;
    struct ggml_context * ctx = ggml_init(params);

    std::unordered_map<std::string, struct ggml_tensor *> parameters;

    while (true) {
        int32_t dim_count;
        fread(&dim_count, 4, 1, file);

        if (feof(file)) {
            break;
        }

        RWKV_ASSERT_NULL(dim_count == 1 || dim_count == 2, "Unsupported dimension count %d", dim_count);

        int32_t key_length;
        read_int32(file, &key_length);
        RWKV_ASSERT_NULL(key_length > 0, "Non-positive key length %d", key_length);

        int32_t data_type;
        read_int32(file, &data_type);
        RWKV_ASSERT_NULL(data_type == 0 || data_type == 1, "Unsupported parameter data type %d", data_type);

        ggml_type ggml_data_type = data_type == 0 ? GGML_TYPE_F32 : GGML_TYPE_F16;

        struct ggml_tensor * tensor;

        int32_t x = -1;
        int32_t y = -1;
        int32_t z = -1;
        int32_t element_count;

        if (dim_count == 1) {
            read_int32(file, &x);
            element_count = x;
            tensor = ggml_new_tensor_1d(ctx, ggml_data_type, x);
        } else if (dim_count == 2) {
            read_int32(file, &x);
            read_int32(file, &y);
            element_count = x * y;
            // Dimension order is reversed here:
            // * PyTorch shape is (x rows, y columns)
            // * ggml shape is (y elements in a row, x elements in a column)
            // Both shapes represent the same tensor.
            tensor = ggml_new_tensor_2d(ctx, ggml_data_type, y, x);
        } else {
            abort();
        }

        std::string key(key_length, 0);
        RWKV_ASSERT_NULL(fread(&key[0], 1, key_length, file) == key_length, "Failed to read parameter key");

        size_t byte_count = element_count * ggml_type_size(ggml_data_type);
        RWKV_ASSERT_NULL(fread(tensor->data, 1, byte_count, file) == byte_count, "Failed to read parameter data");

        parameters[key] = tensor;
    }

    fclose(file);

    model->layers.resize(model->n_layer);

    set_parameter(&parameters, "emb.weight", &(model->emb));

    set_parameter(&parameters, "blocks.0.ln0.weight", &(model->ln0_weight));
    set_parameter(&parameters, "blocks.0.ln0.bias", &(model->ln0_bias));

    for (int i = 0; i < model->n_layer; i++) {
        rwkv_layer layer = model->layers[i];

        set_block_parameter(&parameters, i, "ln1.weight", &(layer.ln1_weight));
        set_block_parameter(&parameters, i, "ln1.bias", &(layer.ln1_bias));

        set_block_parameter(&parameters, i, "att.time_mix_k", &(layer.att_time_mix_k));
        set_block_parameter(&parameters, i, "att.time_mix_v", &(layer.att_time_mix_v));
        set_block_parameter(&parameters, i, "att.time_mix_r", &(layer.att_time_mix_r));
        set_block_parameter(&parameters, i, "att.time_first", &(layer.att_time_first));
        set_block_parameter(&parameters, i, "att.time_decay", &(layer.att_time_decay));
        set_block_parameter(&parameters, i, "att.key.weight", &(layer.att_key));
        set_block_parameter(&parameters, i, "att.value.weight", &(layer.att_value));
        set_block_parameter(&parameters, i, "att.receptance.weight", &(layer.att_receptance));
        set_block_parameter(&parameters, i, "att.output.weight", &(layer.att_output));

        set_block_parameter(&parameters, i, "ln2.weight", &(layer.ln2_weight));
        set_block_parameter(&parameters, i, "ln2.bias", &(layer.ln2_bias));

        set_block_parameter(&parameters, i, "ffn.time_mix_k", &(layer.ffn_time_mix_k));
        set_block_parameter(&parameters, i, "ffn.time_mix_r", &(layer.ffn_time_mix_r));
        set_block_parameter(&parameters, i, "ffn.key.weight", &(layer.ffn_key));
        set_block_parameter(&parameters, i, "ffn.value.weight", &(layer.ffn_value));
        set_block_parameter(&parameters, i, "ffn.receptance.weight", &(layer.ffn_receptance));

        model->layers[i] = layer;
    }

    set_parameter(&parameters, "ln_out.weight", &(model->ln_out_weight));
    set_parameter(&parameters, "ln_out.bias", &(model->ln_out_bias));

    set_parameter(&parameters, "head.weight", &(model->head));

    // Verify order of dimensions
    struct ggml_tensor * emb = model->emb;
    RWKV_ASSERT_NULL(emb->n_dims == 2, "Unexpected dimension count of embedding matrix %d", emb->n_dims);
    RWKV_ASSERT_NULL(emb->ne[0] == model->n_embed, "Unexpected dimension of embedding matrix %d", emb->ne[0]);
    RWKV_ASSERT_NULL(emb->ne[1] == model->n_vocab, "Unexpected dimension of embedding matrix %d", emb->ne[1]);

    int32_t n_vocab = model->n_vocab;
    int32_t n_embed = model->n_embed;
    int32_t n_layer = model->n_layer;

    // Build graph
    struct ggml_tensor * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_layer * 5 * n_embed);

    // x = self.w.emb.weight[token]
    struct ggml_tensor * token_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * x = ggml_get_rows(ctx, model->emb, token_index);

    // x = self.layer_norm(x, self.w.blocks[0].ln0)
    x = rwkv_layer_norm(ctx, x, model->ln0_weight, model->ln0_bias);

    // We collect parts of new state here. Each part is (n_embed) vector.
    struct ggml_tensor ** state_parts = new ggml_tensor * [n_layer * 5];

    for (int i = 0; i < n_layer; i++) {
        auto layer = model->layers[i];

        // RWKV/time mixing
        {
            // self.layer_norm(x, self.w.blocks[i].ln1)
            struct ggml_tensor * x0 = rwkv_layer_norm(ctx, x, layer.ln1_weight, layer.ln1_bias);
            // state[5 * i + 1]
            struct ggml_tensor * x_prev = ggml_view_1d(ctx, state, n_embed, (5 * i + 1) * n_embed * FP32_SIZE);
            // xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
            // xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
            // xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
            struct ggml_tensor * xk = ggml_add(
                ctx,
                ggml_mul(ctx, x0, layer.att_time_mix_k),
                ggml_mul(ctx, x_prev, ggml_1_minus_x(ctx, layer.att_time_mix_k))
            );
            struct ggml_tensor * xv = ggml_add(
                ctx,
                ggml_mul(ctx, x0, layer.att_time_mix_v),
                ggml_mul(ctx, x_prev, ggml_1_minus_x(ctx, layer.att_time_mix_v))
            );
            struct ggml_tensor * xr = ggml_add(
                ctx,
                ggml_mul(ctx, x0, layer.att_time_mix_r),
                ggml_mul(ctx, x_prev, ggml_1_minus_x(ctx, layer.att_time_mix_r))
            );
            // state[5 * i + 1] = x
            state_parts[5 * i + 1] = x0;

            // r = torch.sigmoid(rw @ xr)
            struct ggml_tensor * r = ggml_sigmoid(
                ctx,
                ggml_mul_mat(ctx, layer.att_receptance, xr)
            );
            // k = kw @ xk
            struct ggml_tensor * k = ggml_mul_mat(ctx, layer.att_key, xk);
            // v = vw @ xv
            struct ggml_tensor * v = ggml_mul_mat(ctx, layer.att_value, xv);

            // aa = state[5 * i + 2]
            // bb = state[5 * i + 3]
            // pp = state[5 * i + 4]
            struct ggml_tensor * aa = ggml_view_1d(ctx, state, n_embed, (5 * i + 2) * n_embed * FP32_SIZE);
            struct ggml_tensor * bb = ggml_view_1d(ctx, state, n_embed, (5 * i + 3) * n_embed * FP32_SIZE);
            struct ggml_tensor * pp = ggml_view_1d(ctx, state, n_embed, (5 * i + 4) * n_embed * FP32_SIZE);

            // ww = time_first + k
            struct ggml_tensor * ww = ggml_add(ctx, layer.att_time_first, k);
            // qq = torch.maximum(pp, ww)
            struct ggml_tensor * qq = ggml_max(ctx, pp, ww);
            // e1 = torch.exp(pp - qq)
            struct ggml_tensor * e1 = ggml_exp(ctx, ggml_sub(ctx, pp, qq));
            // e2 = torch.exp(ww - qq)
            struct ggml_tensor * e2 = ggml_exp(ctx, ggml_sub(ctx, ww, qq));
            // a = e1 * aa + e2 * v
            struct ggml_tensor * a = ggml_add(
                ctx,
                ggml_mul(ctx, e1, aa),
                ggml_mul(ctx, e2, v)
            );
            // b = e1 * bb + e2
            struct ggml_tensor * b = ggml_add(
                ctx,
                ggml_mul(ctx, e1, bb),
                e2
            );
            // wkv = a / b
            struct ggml_tensor * wkv = ggml_div(ctx, a, b);
            // ww = pp + time_decay
            ww = ggml_add(ctx, pp, layer.att_time_decay);
            // qq = torch.maximum(ww, k)
            qq = ggml_max(ctx, ww, k);
            // e1 = torch.exp(ww - qq)
            e1 = ggml_exp(ctx, ggml_sub(ctx, ww, qq));
            // e2 = torch.exp(k - qq)
            e2 = ggml_exp(ctx, ggml_sub(ctx, k, qq));
            // state[5 * i + 2] = e1 * aa + e2 * v
            state_parts[5 * i + 2] = ggml_add(
                ctx,
                ggml_mul(ctx, e1, aa),
                ggml_mul(ctx, e2, v)
            );
            // state[5 * i + 3] = e1 * bb + e2
            state_parts[5 * i + 3] = ggml_add(
                ctx,
                ggml_mul(ctx, e1, bb),
                e2
            );
            // state[5 * i + 4] = qq
            state_parts[5 * i + 4] = qq;
            // ow @ (r * wkv)
            x = ggml_add(
                ctx,
                x,
                ggml_mul_mat(
                    ctx,
                    layer.att_output,
                    ggml_mul(ctx, r, wkv)
                )
            );
        }

        // FFN/channel mixing
        {
            // self.layer_norm(x, self.w.blocks[i].ln2)
            struct ggml_tensor * x0 = rwkv_layer_norm(ctx, x, layer.ln2_weight, layer.ln2_bias);
            // state[5 * i + 0]
            struct ggml_tensor * x_prev = ggml_view_1d(ctx, state, n_embed, (5 * i + 0) * n_embed * FP32_SIZE);
            // xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
            // xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
            struct ggml_tensor * xk = ggml_add(
                ctx,
                ggml_mul(ctx, x0, layer.ffn_time_mix_k),
                ggml_mul(ctx, x_prev, ggml_1_minus_x(ctx, layer.ffn_time_mix_k))
            );
            struct ggml_tensor * xr = ggml_add(
                ctx,
                ggml_mul(ctx, x0, layer.ffn_time_mix_r),
                ggml_mul(ctx, x_prev, ggml_1_minus_x(ctx, layer.ffn_time_mix_r))
            );
            // state[5 * i + 0] = x
            state_parts[5 * i + 0] = x0;

            // r = torch.sigmoid(rw @ xr)
            struct ggml_tensor * r = ggml_sigmoid(
                ctx,
                ggml_mul_mat(ctx, layer.ffn_receptance, xr)
            );
            // k = torch.square(torch.relu(kw @ xk))
            struct ggml_tensor * k = ggml_sqr(ctx, ggml_relu(
                ctx,
                ggml_mul_mat(ctx, layer.ffn_key, xk)
            ));
            // r * (vw @ k)
            x = ggml_add(
                ctx,
                x,
                ggml_mul(
                    ctx,
                    r,
                    ggml_mul_mat(ctx, layer.ffn_value, k)
                )
            );
        }
    }

    // x = self.layer_norm(x, self.w.ln_out)
    x = rwkv_layer_norm(ctx, x, model->ln_out_weight, model->ln_out_bias);

    // x = (self.w.head.weight @ x).float()
    struct ggml_tensor * logits = ggml_mul_mat(ctx, model->head, x);

    struct ggml_cgraph * graph = (struct ggml_cgraph *) calloc(1, sizeof(struct ggml_cgraph));

    *graph = ggml_build_forward(logits);

    for (int i = 0; i < n_layer * 5; i++) {
       ggml_build_forward_expand(graph, state_parts[i]);
    }

    graph->n_threads = n_threads;

    struct rwkv_context * rwkv_ctx = (struct rwkv_context *) calloc(1, sizeof(struct rwkv_context));
    rwkv_ctx->model = model;
    rwkv_ctx->token_index = token_index;
    rwkv_ctx->state = state;
    rwkv_ctx->state_parts = state_parts;
    rwkv_ctx->logits = logits;
    rwkv_ctx->ctx = ctx;
    rwkv_ctx->graph = graph;
    return rwkv_ctx;
}

size_t rwkv_get_state_buffer_element_count(struct rwkv_context * ctx) {
    return ctx->model->n_layer * 5 * ctx->model->n_embed;
}

size_t rwkv_get_logits_buffer_element_count(struct rwkv_context * ctx) {
    return ctx->model->n_vocab;
}

bool rwkv_eval(struct rwkv_context * ctx, long int token, float * state_in, float * state_out, float * logits_out) {
    RWKV_ASSERT_FALSE(state_out != NULL, "state_out is NULL");
    RWKV_ASSERT_FALSE(logits_out != NULL, "logits_out is NULL");

    int32_t n_layer = ctx->model->n_layer;
    int32_t n_embed = ctx->model->n_embed;
    int32_t n_vocab = ctx->model->n_vocab;

    RWKV_ASSERT_FALSE(token >= 0 && token < n_vocab, "Token is out of range 0..%d", n_vocab - 1);

    ggml_set_i32(ctx->token_index, 0);
    ggml_set_i32_1d(ctx->token_index, 0, token);

    if (state_in == NULL) {
        ggml_set_f32(ctx->state, 0.0F);

        for (int i = 0; i < n_layer; i++) {
            // state[5 * i + 4] = -1e30
            ggml_set_f32(
                ggml_view_1d(ctx->ctx, ctx->state, n_embed, (5 * i + 4) * n_embed * FP32_SIZE),
                -1e30F
            );
        }
    } else {
        memcpy(ctx->state->data, state_in, ctx->state->ne[0] * FP32_SIZE);
    }

    ggml_graph_compute(ctx->ctx, ctx->graph);

    for (size_t i = 0; i < n_layer * 5; i++) {
        struct ggml_tensor * part = ctx->state_parts[i];

        memcpy(state_out + i * n_embed, part->data, part->ne[0] * FP32_SIZE);
    }

    memcpy(logits_out, ctx->logits->data, ctx->logits->ne[0] * FP32_SIZE);

    return true;
}

void rwkv_free(struct rwkv_context * ctx) {
    ggml_free(ctx->ctx);

    delete ctx->model;
    delete ctx->state_parts;
    delete ctx;
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
