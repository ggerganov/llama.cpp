#include "common.h"
#include "ggml.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

// --- Utilities ---

// Checks that x is not false. If it is false, prints fancy message to stderr and aborts the execution.
#define RWKV_ASSERT(x, ...) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "*** Assertion failed ***\n"); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

// Formats and prints a message to stderr. Trailing newline is added automatically.
#define RWKV_LOG(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while (0)

// TODO Move to ggml, if correct
float ggml_get_f32_2d(struct ggml_tensor * tensor, int i, int j) {
    RWKV_ASSERT(tensor->n_dims == 2, "Not a 2D tensor");
    RWKV_ASSERT(tensor->type == GGML_TYPE_F32, "Unsupported data type");
    return *(float *) ((char *) tensor->data + j * tensor->nb[1] + i * tensor->nb[0]);
}

// TODO Move to ggml, if correct
float ggml_get_f32_3d(struct ggml_tensor * tensor, int i, int j, int k) {
    RWKV_ASSERT(tensor->n_dims == 3, "Not a 3D tensor");
    RWKV_ASSERT(tensor->type == GGML_TYPE_F32, "Unsupported data type");
    return *(float *) ((char *) tensor->data + k * tensor->nb[2] + j * tensor->nb[1] + i * tensor->nb[0]);
}

void print_tensor(struct ggml_tensor * tensor, char * name) {
    int n_dims = tensor->n_dims;

    if (n_dims == 1) {
        int x = tensor->ne[0];

        RWKV_ASSERT(x >= 6, "Too small tensor");

        RWKV_LOG(
            "1D tensor %s, shape (%d), [%f %f %f ... %f %f %f]",
            name,
            x,
            ggml_get_f32_1d(tensor, 0),
            ggml_get_f32_1d(tensor, 1),
            ggml_get_f32_1d(tensor, 2),
            ggml_get_f32_1d(tensor, x - 3),
            ggml_get_f32_1d(tensor, x - 2),
            ggml_get_f32_1d(tensor, x - 1)
        );
    } else if (n_dims == 2) {
        int x = tensor->ne[0];
        int y = tensor->ne[1];

        if (y < 6) {
            RWKV_LOG(
                "2D tensor %s, shape (%d, %d), [[%f %f %f ... %f %f %f]]",
                name,
                x,
                y,
                ggml_get_f32_2d(tensor, 0, 0),
                ggml_get_f32_2d(tensor, 1, 0),
                ggml_get_f32_2d(tensor, 2, 0),
                ggml_get_f32_2d(tensor, x - 3, y - 1),
                ggml_get_f32_2d(tensor, x - 2, y - 1),
                ggml_get_f32_2d(tensor, x - 1, y - 1)
            );
        } else {
            RWKV_LOG(
                "2D tensor %s, shape (%d, %d), [[%f %f %f ... ] ... [ ... %f %f %f]]",
                name,
                x,
                y,
                ggml_get_f32_2d(tensor, 0, 0),
                ggml_get_f32_2d(tensor, 0, 1),
                ggml_get_f32_2d(tensor, 0, 2),
                ggml_get_f32_2d(tensor, x - 1, y - 3),
                ggml_get_f32_2d(tensor, x - 1, y - 2),
                ggml_get_f32_2d(tensor, x - 1, y - 1)
            );
        }
    } else {
        RWKV_ASSERT(false, "Unsupported dimension count %d", n_dims);
    }
}

// Prints tensor name, dimensionality, shape and part of its contents.
#define PRINT_TENSOR(x) print_tensor(x, #x)

// Same as PRINT_TENSOR, but additionally computes tensor graph before printing the tensor.
#define COMPUTE_AND_PRINT_TENSOR(ctx, x) do { compute_graph(ctx, x); print_tensor(x, #x); } while (0)

// Computes value of the tensor and all tensors it depends on.
void compute_graph(struct ggml_context * ctx, struct ggml_tensor * tensor) {
    struct ggml_cgraph graph = ggml_build_forward(tensor);

    graph.n_threads = 1;

    ggml_graph_compute(ctx, &graph);
}

// --- Model definition and loading code ---

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

// Reads single int32 value from a file.
void read_int32(FILE * file, int32_t * dest) {
    // TODO Will not read correct values on machine with different endianness
    RWKV_ASSERT(fread(dest, 4, 1, file) == 1, "Failed to read an int32 value from a file");
}

// Finds model parameter by key and sets it into dest.
// If the parameter was not found, aborts the execution.
void set_parameter(std::unordered_map<std::string, struct ggml_tensor *> * parameters, char * key, struct ggml_tensor ** dest) {
    struct ggml_tensor * parameter = (*parameters)[key];
    RWKV_ASSERT(parameter != NULL, "Parameter %s not found in model file", key);
    *dest = parameter;
}

// Finds block parameter by block index and key and sets it into dest.
// If the parameter was not found, aborts the execution.
void set_block_parameter(std::unordered_map<std::string, struct ggml_tensor *> * parameters, int32_t block_index, char * key, struct ggml_tensor ** dest) {
    char full_key[128];
    sprintf(full_key, "blocks.%d.%s", block_index, key);
    set_parameter(parameters, full_key, dest);
}

// Loads RWKV model metadata and parameters from a file.
void load_rwkv_model(ggml_context * ctx, char * file_path, struct rwkv_model * model) {
    RWKV_LOG("Loading model from %s", file_path);
    FILE * file = fopen(file_path, "rb");
    RWKV_ASSERT(file != NULL, "Failed to open file %s", file_path);

    int32_t magic;
    read_int32(file, &magic);
    RWKV_ASSERT(magic == 0x67676d66, "Unexpected magic value %d", magic);

    int32_t version;
    read_int32(file, &version);
    RWKV_ASSERT(version == 100, "Unsupported file version %d", version);

    read_int32(file, &(model->n_vocab));
    RWKV_ASSERT(model->n_vocab > 0, "Non-positive n_vocab %d", model->n_vocab);

    read_int32(file, &(model->n_embed));
    RWKV_ASSERT(model->n_embed > 0, "Non-positive n_embed %d", model->n_embed);

    read_int32(file, &(model->n_layer));
    RWKV_ASSERT(model->n_layer > 0, "Non-positive n_layer %d", model->n_layer);

    read_int32(file, &(model->data_type));
    RWKV_ASSERT(model->data_type == 0 || model->data_type == 1, "Unsupported model data type %d", model->data_type);

    RWKV_ASSERT(model->data_type == 0, "Data types other than float32 are not yet supported"); // TODO

    RWKV_LOG("n_vocab = %d", model->n_vocab);
    RWKV_LOG("n_embed = %d", model->n_embed);
    RWKV_LOG("n_layer = %d", model->n_layer);

    std::unordered_map<std::string, struct ggml_tensor *> parameters;

    while (true) {
        int32_t dim_count;
        fread(&dim_count, 4, 1, file);

        if (feof(file)) {
            break;
        }

        RWKV_ASSERT(dim_count == 1 || dim_count == 2, "Unsupported dimension count %d", dim_count);

        int32_t key_length;
        read_int32(file, &key_length);
        RWKV_ASSERT(key_length > 0, "Non-positive key length %d", key_length);

        int32_t data_type;
        read_int32(file, &data_type);
        RWKV_ASSERT(data_type == 0 || data_type == 1, "Unsupported parameter data type %d", data_type);

        RWKV_ASSERT(data_type == 0, "Data types other than float32 are not yet supported"); // TODO

        struct ggml_tensor * tensor;

        int32_t x = -1;
        int32_t y = -1;
        int32_t z = -1;
        int32_t element_count;

        if (dim_count == 1) {
            read_int32(file, &x);
            element_count = x;
            tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, x);
        } else if (dim_count == 2) {
            read_int32(file, &x);
            read_int32(file, &y);
            element_count = x * y;
            // Dimension order is reversed here:
            // * PyTorch shape is (x rows, y columns)
            // * ggml shape is (y elements in a row, x elements in a column)
            // Both shapes represent the same tensor.
            tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, y, x);
        } else {
            abort();
        }

        std::string key(key_length, 0);
        RWKV_ASSERT(fread(&key[0], 1, key_length, file) == key_length, "Failed to read parameter key");

        // TODO Use ggml_type_size
        size_t element_size = data_type == 0 ? 4 : 2;
        size_t byte_count = element_count * element_size;

        RWKV_ASSERT(fread(tensor->data, 1, byte_count, file) == byte_count, "Failed to read parameter data");

        parameters[key] = tensor;
    }

    fclose(file);

    RWKV_LOG("Initializing model parameters");

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
    RWKV_ASSERT(emb->n_dims == 2, "Unexpected dimension count of embedding matrix %d", emb->n_dims);
    RWKV_ASSERT(emb->ne[0] == model->n_vocab, "Unexpected dimension of embedding matrix %d", emb->ne[0]);
    RWKV_ASSERT(emb->ne[1] == model->n_embed, "Unexpected dimension of embedding matrix %d", emb->ne[1]);
}

// --- Operators ---

struct ggml_tensor * ggml_layer_norm(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // LayerNorm in RWKV is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // Looks like ggml_norm does the first part, we only need to apply weight & bias.
    x = ggml_norm(ctx, x);
    x = ggml_mul(ctx, x, weight);
    x = ggml_add(ctx, x, bias);
    return x;
}

// --- Script ---

// Usage: main_rwkv.exe "C:\model.bin" <token index> "C:\state_in.bin" "C:\state_out.bin" "C:\logits_out.bin"
// Token index is 0-based.
// To start from new state, pass empty string instead of input state file path.
int main(int argc, char ** argv) {
    ggml_run_test_suite();

    RWKV_ASSERT(argc - 1 == 5, "Expected 5 arguments, got %d", argc - 1);
    char * model_path = argv[1];
    char * token_s = argv[2];
    char * state_in_path = argv[3];
    char * state_out_path = argv[4];
    char * logits_out_path = argv[5];

    int32_t token = strtol(token_s, (char **) NULL, 10);
    RWKV_LOG("Token index is %d", token);

    bool create_new_state = strcmp(state_in_path, "") == 0;

    // Initialize ggml
    struct ggml_init_params params;
    // TODO Calculate required memory (automatically or manually)
    params.mem_size = 1024 * 1024 * 1024;
    params.mem_buffer = NULL;

    struct ggml_context * ctx = ggml_init(params);

    // Load model
    struct rwkv_model model;
    load_rwkv_model(ctx, model_path, &model);

    int32_t n_vocab = model.n_vocab;
    int32_t n_embed = model.n_embed;
    int32_t n_layer = model.n_layer;

    // Load input state
    int32_t state_element_count = n_layer * 5 * n_embed;
    struct ggml_tensor * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, state_element_count);

    if (create_new_state) {
        RWKV_LOG("Creating new state");
        ggml_set_f32(state, 0.0F);

        for (int i = 0; i < n_layer; i++) {
            // state[5 * i + 4] = -1e30
            int32_t offset_in_bytes = (5 * i + 4) * n_embed * 4;
            struct ggml_tensor * state_part = ggml_view_1d(ctx, state, n_embed, offset_in_bytes);
            ggml_set_f32(state_part, -1e30F);
        }
    } else {
        RWKV_LOG("Loading state from %s", state_in_path);
        int32_t state_file_size = state_element_count * 4;

        FILE * state_in_file = fopen(state_in_path, "rb");
        RWKV_ASSERT(state_in_file != NULL, "Failed to open file %s", state_in_path);

        // TODO Saving/loading raw data makes state cache machine-dependent
        RWKV_ASSERT(fread(state->data, 1, state_file_size, state_in_file) == state_file_size, "Failed to read state from a file");

        fclose(state_in_file);
    }

    // --- Evaluate model ---

    // x = self.w.emb.weight[token]
    // TODO Replace with ggml_get_rows or similar
    struct ggml_tensor * one_hot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, 1);
    ggml_set_f32_1d(one_hot, token, 1.0F);
    struct ggml_tensor * x = ggml_mul_mat(ctx, model.emb, one_hot);

    // x = self.layer_norm(x, self.w.blocks[0].ln0)
    x = ggml_layer_norm(ctx, x, model.ln0_weight, model.ln0_bias);

    // We collect parts of new state here. Each part is (n_embed) vector.
    struct ggml_tensor ** state_parts = new ggml_tensor * [5 * n_layer];

    for (int i = 0; i < n_layer; i++) {
        auto layer = model.layers[i];

        // RWKV/time mixing
        {
            // self.layer_norm(x, self.w.blocks[i].ln1)
            struct ggml_tensor * x0 = ggml_layer_norm(ctx, x, layer.ln1_weight, layer.ln1_bias);
            // state[5 * i + 1]
            struct ggml_tensor * x_prev = ggml_view_1d(ctx, state, n_embed, (5 * i + 1) * n_embed * 4);
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
            struct ggml_tensor * aa = ggml_view_1d(ctx, state, n_embed, (5 * i + 2) * n_embed * 4);
            struct ggml_tensor * bb = ggml_view_1d(ctx, state, n_embed, (5 * i + 3) * n_embed * 4);
            struct ggml_tensor * pp = ggml_view_1d(ctx, state, n_embed, (5 * i + 4) * n_embed * 4);

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
            struct ggml_tensor * x0 = ggml_layer_norm(ctx, x, layer.ln2_weight, layer.ln2_bias);
            // state[5 * i + 0]
            struct ggml_tensor * x_prev = ggml_view_1d(ctx, state, n_embed, (5 * i + 0) * n_embed * 4);
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
    x = ggml_layer_norm(ctx, x, model.ln_out_weight, model.ln_out_bias);

    // x = (self.w.head.weight @ x).float()
    struct ggml_tensor * logits = ggml_mul_mat(ctx, model.head, x);

    struct ggml_cgraph graph = ggml_build_forward(logits);

    for (int i = 0; i < n_layer * 5; i++) {
        ggml_build_forward_expand(&graph, state_parts[i]);
    }

    // TODO Move to script arguments
    graph.n_threads = std::max(1, (int32_t) std::thread::hardware_concurrency() / 2);

    ggml_graph_compute(ctx, &graph);

    // Update state
    for (int i = 0; i < n_layer * 5; i++) {
        struct ggml_tensor * state_part_src = state_parts[i];
        struct ggml_tensor * state_part_dest = ggml_view_1d(ctx, state, n_embed, i * n_embed * 4);

        for (int j = 0; j < n_embed; j++) {
            ggml_set_f32_1d(state_part_dest, j, ggml_get_f32_1d(state_part_src, j));
        }
    }

    {
        RWKV_LOG("Saving state to %s", state_out_path);
        int32_t state_file_size = state_element_count * 4;

        FILE * state_out_file = fopen(state_out_path, "wb");
        RWKV_ASSERT(state_out_file != NULL, "Failed to open file %s", state_out_path);

        RWKV_ASSERT(fwrite(state->data, 1, state_file_size, state_out_file) == state_file_size, "Failed to write state to a file");

        fclose(state_out_file);
    }

    {
        RWKV_LOG("Saving logits to %s", logits_out_path);
        int32_t logits_file_size = n_vocab * 4;

        FILE * logits_out_file = fopen(logits_out_path, "wb");
        RWKV_ASSERT(logits_out_file != NULL, "Failed to open file %s", logits_out_path);

        RWKV_ASSERT(fwrite(logits->data, 1, logits_file_size, logits_out_file) == logits_file_size, "Failed to write logits to a file");

        fclose(logits_out_file);
    }

    ggml_free(ctx);

    RWKV_LOG("OK");

    return 0;
}
