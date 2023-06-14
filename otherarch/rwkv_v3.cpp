//adapted from RWKV.cpp repo under MIT license
// https://github.com/saharNooby/rwkv.cpp

#include "otherarch.h"

#include "rwkv_v3.h"
#include "ggml.h"

#include <string>
#include <vector>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <utility>

#define _FILE_OFFSET_BITS 64
#define RWKV_MAYBE_BREAK

#include <sys/stat.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define stat _stat64
#define fstat _fstat64
#define ftell _ftelli64
#define fseek _fseeki64

#ifndef NDEBUG
#include <intrin.h>
#define RWKV_MAYBE_BREAK __debugbreak()
#endif
#else
#if !defined(__APPLE__)
#define ftell ftello
#define fseek fseeko
#endif
#endif

// static_assert(sizeof(stat::st_size) >= 8, "File offsets should be 64-bit or else rwkv.cpp will not be able to load model files over 2GB");
// static_assert(sizeof(decltype(ftell(NULL))) >= 8, "File offsets should be 64-bit or else rwkv.cpp will not be able to load model files over 2GB");

// --- Error handling ---

thread_local enum rwkv_error_flags global_last_error = RWKV_ERROR_NONE;
thread_local bool global_print_errors = true;

inline enum rwkv_error_flags operator|(enum rwkv_error_flags a, enum rwkv_error_flags b) {
    return static_cast<enum rwkv_error_flags>(static_cast<int>(a) | static_cast<int>(b));
}

inline enum rwkv_error_flags operator|=(enum rwkv_error_flags & a, enum rwkv_error_flags b) {
    return a = a | b;
}

#define RWKV_MSG(...) do { if (global_print_errors) fprintf(stderr, __VA_ARGS__); } while (0)
#define RWKV_CTX_MSG(ctx, ...) do { if (ctx->print_errors) fprintf(stderr, __VA_ARGS__); } while (0)

// If the condition x is false, adds ERR_VAL to the last error, and returns RET_VAL.
#define RWKV_ASSERT(ERR_VAL, RET_VAL, x) do { \
    if (!(x)) { \
        global_last_error |= ERR_VAL; \
        RWKV_MSG("\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

// If the condition x is false, adds ERR_VAL to the last error, prints a message to stderr, and returns RET_VAL.
#define RWKV_ASSERT_MSG(ERR_VAL, RET_VAL, x, ...) do { \
    if (!(x)) { \
        global_last_error |= ERR_VAL; \
        RWKV_MSG(__VA_ARGS__); \
        RWKV_MSG("\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

// If the condition x is false, adds ERR_VAL to the ctx's last error, prints a message to stderr, and returns RET_VAL.
#define RWKV_CTX_ASSERT_MSG(ctx, ERR_VAL, RET_VAL, x, ...) do { \
    if (!(x)) { \
        ((struct rwkv_context *) ctx)->last_error |= ERR_VAL; \
        RWKV_CTX_MSG(ctx, __VA_ARGS__); \
        RWKV_CTX_MSG(ctx, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

// If the condition x is false, adds ERR_VAL to the ctx's last error, and returns RET_VAL.
#define RWKV_CTX_ASSERT(ctx, ERR_VAL, RET_VAL, x) do { \
    if (!(x)) { \
        ((struct rwkv_context *) ctx)->last_error |= ERR_VAL; \
        RWKV_CTX_MSG(ctx, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

// If the condition x is false, returns RET_VAL.
#define RWKV_ENSURE(RET_VAL, x) do { \
    if (!(x)) { \
        RWKV_MSG("\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

// If the condition x is false, prints a message to stderr, and returns RET_VAL.
#define RWKV_ENSURE_MSG(RET_VAL, x, ...) do { \
    if (!(x)) { \
        RWKV_MSG(__VA_ARGS__); \
        RWKV_MSG("\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

// If the condition x is false, prints a message to stderr, and returns RET_VAL.
#define RWKV_CTX_ENSURE_MSG(ctx, RET_VAL, x, ...) do { \
    if (!(x)) { \
        ((struct rwkv_context *) ctx)->last_error |= ERR_VAL; \
        RWKV_CTX_MSG(ctx, __VA_ARGS__); \
        RWKV_CTX_MSG(ctx, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

#define RWKV_ASSERT_FALSE_MSG(ERR_VAL, x, ...) RWKV_ASSERT_MSG(ERR_VAL, false, x, __VA_ARGS__)
#define RWKV_ASSERT_NULL_MSG(ERR_VAL, x, ...) RWKV_ASSERT_MSG(ERR_VAL, NULL, x, __VA_ARGS__)
#define RWKV_CTX_ASSERT_FALSE_MSG(ctx, ERR_VAL, x, ...) RWKV_CTX_ASSERT_MSG(ctx, ERR_VAL, false, x, __VA_ARGS__)
#define RWKV_CTX_ASSERT_NULL_MSG(ctx, ERR_VAL, x, ...) RWKV_CTX_ASSERT_MSG(ctx, ERR_VAL, NULL, x, __VA_ARGS__)

#define RWKV_ASSERT_FALSE(ERR_VAL, x) RWKV_ASSERT(ERR_VAL, false, x)
#define RWKV_ASSERT_NULL(ERR_VAL, x) RWKV_ASSERT(ERR_VAL, NULL, x)
#define RWKV_CTX_ASSERT_FALSE(ctx, ERR_VAL, x) RWKV_CTX_ASSERT(ctx, ERR_VAL, false, x)
#define RWKV_CTX_ASSERT_NULL(ctx, ERR_VAL, x) RWKV_CTX_ASSERT(ctx, ERR_VAL, NULL, x)

#define RWKV_ENSURE_OR_FALSE(x) RWKV_ENSURE(false, x)
#define RWKV_ENSURE_OR_NULL(x) RWKV_ENSURE(NULL, x)
#define RWKV_ENSURE_OR_FALSE_MSG(x, ...) RWKV_ENSURE_MSG(false, x, __VA_ARGS__)
#define RWKV_ENSURE_OR_NULL_MSG(x, ...) RWKV_ENSURE_MSG(NULL, x, __VA_ARGS__)
#define RWKV_CTX_ENSURE_OR_FALSE_MSG(ctx, x, ...) RWKV_CTX_ENSURE_MSG(ctx, false, x, __VA_ARGS__)
#define RWKV_CTX_ENSURE_OR_NULL_MSG(ctx, x, ...) RWKV_CTX_ENSURE_MSG(ctx, NULL, x, __VA_ARGS__)

// --- Utilities ---

// Reads a single uint32 value from a file.
bool rwkv_fread_uint32(FILE * file, uint32_t & dest) {
    return fread((void *) &dest, sizeof(uint32_t), 1, file) == 1;
}

// Reads a single string value from a file.
bool rwkv_fread_string(FILE * file, size_t length, std::string & dest) {
    dest.resize(length);
    return fread((void *) dest.data(), length, 1, file) == 1;
}

// Reads a single data buffer from a file.
bool rwkv_fread_data(FILE * file, size_t length, void * dest) {
    return fread(dest, length, 1, file) == 1;
}

// Writes a single uint32 value to a file.
bool rwkv_fwrite_uint32(FILE * file, const uint32_t value) {
    return fwrite((const void *) &value, sizeof(uint32_t), 1, file);
}

// Writes a single string value to a file.
bool rwkv_fwrite_string(FILE * file, const std::string & value) {
    return fwrite((const void *) value.data(), value.length(), 1, file) == 1;
}

// Writes a single data buffer to a file.
bool rwkv_fwrite_data(FILE * file, const void * data, const size_t length) {
    return fwrite(data, length, 1, file) == 1;
}

// --- File data structures ---

#define TYPE_UNKNOWN TYPE_COUNT

enum rwkv_type {
    TYPE_F32,
    TYPE_F16,
    TYPE_Q4_0,
    TYPE_Q4_1,
    TYPE_Q4_1_O, // Unsupported
    TYPE_Q4_2, // Unsupported
    TYPE_Q4_3, // Unsupported
    TYPE_Q5_0,
    TYPE_Q5_1,
    TYPE_Q8_0,
    TYPE_COUNT
};

#define GGML_TYPE_UNKNOWN GGML_TYPE_COUNT

extern const enum ggml_type rwkv_type_to_ggml[TYPE_COUNT + 1] = {
    GGML_TYPE_F32,     /* F32    */
    GGML_TYPE_F16,     /* F16    */
    GGML_TYPE_Q4_0,    /* Q4_0   */
    GGML_TYPE_Q4_1,    /* Q4_1   */
    GGML_TYPE_UNKNOWN, /* Q4_1_O */
    GGML_TYPE_UNKNOWN, /* Q4_2   */
    GGML_TYPE_UNKNOWN, /* Q4_3   */
    GGML_TYPE_Q5_0,    /* Q5_0   */
    GGML_TYPE_Q5_1,    /* Q5_1   */
    GGML_TYPE_Q8_0,    /* Q8_0   */
    GGML_TYPE_COUNT    /* COUNT  */
};

extern const enum rwkv_type rwkv_type_from_ggml[GGML_TYPE_COUNT + 1] = {
    TYPE_F32,    /* F32   */
    TYPE_F16,    /* F16   */
    TYPE_Q4_0,   /* Q4_0  */
    TYPE_Q4_1,   /* Q4_1  */
    TYPE_Q4_2,   /* Q4_2  */
    TYPE_Q4_3,   /* Q4_3  */
    TYPE_Q5_0,   /* Q5_0  */
    TYPE_Q5_1,   /* Q5_1  */
    TYPE_Q8_0,   /* Q8_0  */
    TYPE_COUNT,  /* Q8_1  */
    TYPE_COUNT,  /* I8    */
    TYPE_COUNT,  /* I16   */
    TYPE_COUNT,  /* I32   */
    TYPE_COUNT,  /* COUNT */
};

extern const char * rwkv_type_to_string[TYPE_COUNT + 1] = {"float32", "float16", "Q4_0", "Q4_1", "Q4_1_O", "Q4_2", "Q4_3", "Q5_0", "Q5_1", "Q8_0", "unknown"};

enum rwkv_type rwkv_type_from_string(const char * str) {
    for (int ord = 0; ord < TYPE_COUNT; ord++) {
        if (strcmp(str, rwkv_type_to_string[ord]) == 0) {
            return (enum rwkv_type) ord;
        }
    }

    return TYPE_UNKNOWN;
}

struct rwkv_file_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_vocab;
    uint32_t n_embed;
    uint32_t n_layer;
    uint32_t data_type;
};

bool rwkv_is_file_version_in_range(uint32_t version) {
    return version >= RWKV_FILE_VERSION_MIN && version <= RWKV_FILE_VERSION_MAX;
}

bool rwkv_fread_file_header(FILE * file, struct rwkv_file_header & header, bool verify_data_type = true) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fread_data(file, sizeof(struct rwkv_file_header), &header));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_MAGIC, header.magic == RWKV_FILE_MAGIC);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_VERSION, rwkv_is_file_version_in_range(header.version), "Unsupported file version %" PRId32, header.version);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA_TYPE, header.data_type < TYPE_COUNT, "Model data type out of range (%" PRId32 " > %" PRId32 ")", header.data_type, TYPE_COUNT - 1);

    if (verify_data_type) {
        enum ggml_type ggml_type = rwkv_type_to_ggml[header.data_type];

        RWKV_ASSERT_FALSE_MSG(
            RWKV_ERROR_DATA_TYPE,
            ggml_type != GGML_TYPE_UNKNOWN,
            "Models in %s format cannot be loaded anymore because the format was removed.\n"
            "You need to quantize the model into another format or use an older version of rwkv.cpp.\n"
            "See https://github.com/saharNooby/rwkv.cpp#compatibility for more info",
            rwkv_type_to_string[header.data_type]
        );

        RWKV_ASSERT_FALSE_MSG(
            RWKV_ERROR_DATA_TYPE,
            (!ggml_is_quantized(ggml_type) || header.version == RWKV_FILE_VERSION_1),
            "The quantized model file in %s format was created with an old version of rwkv.cpp and can not be loaded anymore.\n"
            "You need to requantize the model or use an older version of rwkv.cpp.\n"
            "See https://github.com/saharNooby/rwkv.cpp#compatibility for more info",
            rwkv_type_to_string[header.data_type]
        );
    }

    return true;
}

bool rwkv_fwrite_file_header(FILE * file, const struct rwkv_file_header & header) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, rwkv_fwrite_data(file, &header, sizeof(struct rwkv_file_header)));
    return true;
}

struct rwkv_tensor_header {
    uint32_t dim_count;
    uint32_t key_length;
    uint32_t data_type;
    uint32_t width;
    uint32_t height;
};

struct rwkv_tensor {
    struct rwkv_tensor_header header;
    std::string name;
    uint8_t * data;
};

bool rwkv_fread_tensor_header(FILE * file, struct rwkv_tensor_header & header) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fread_data(file, sizeof(struct rwkv_tensor_header) - sizeof(uint32_t), &header));
    header.height = 1;
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_SHAPE, header.dim_count == 1 || header.dim_count == 2, "Tensor has an invalid shape (%" PRId32 " dimensions)", header.dim_count);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA_TYPE, header.data_type < TYPE_COUNT, "Tensor data type out of range (%" PRId32 " > %" PRId32 ")", header.data_type, TYPE_COUNT - 1);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA_TYPE, rwkv_type_to_ggml[header.data_type] != GGML_TYPE_UNKNOWN, "Tensor data type (%s) is no longer supported", rwkv_type_to_string[header.data_type]);

    if (header.dim_count == 2) {
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fread_uint32(file, header.height));
    }

    return true;
}

bool rwkv_fwrite_tensor_header(FILE * file, const struct rwkv_tensor_header & header) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, rwkv_fwrite_data(file, &header, sizeof(struct rwkv_tensor_header) - (header.dim_count == 1 ? sizeof(uint32_t) : 0)));
    return true;
}

size_t rwkv_tensor_size(enum ggml_type type, const int64_t width, const int64_t height = 1) {
    struct ggml_tensor decoy {};
    decoy.type = type;
    decoy.ne[0] = width;
    decoy.ne[1] = height;
    decoy.ne[2] = 1;
    decoy.ne[3] = 1;
    return ggml_nbytes(&decoy);
}

size_t rwkv_tensor_size(const struct rwkv_tensor_header & header) {
    return rwkv_tensor_size(rwkv_type_to_ggml[header.data_type], header.width, header.height);
}

bool rwkv_fskip_tensor_data(FILE * file, const struct rwkv_tensor_header & header) {
    return fseek(file, header.key_length + rwkv_tensor_size(header), SEEK_CUR) == 0;
}

bool rwkv_fread_tensor_header_and_skip(FILE * file, struct rwkv_tensor_header & header) {
    RWKV_ENSURE_OR_FALSE(rwkv_fread_tensor_header(file, header));
    RWKV_ASSERT_FALSE(RWKV_ERROR_DATA, rwkv_fskip_tensor_data(file, header));
    return true;
}

bool rwkv_fread_tensor_data(FILE * file, struct rwkv_tensor & output, void * buffer = NULL) {
    size_t data_size = rwkv_tensor_size(output.header);
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fread_string(file, output.header.key_length, output.name));

    if (buffer) {
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fread_data(file, data_size, buffer));
    } else {
        output.data = NULL;
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fskip_tensor_data(file, output.header));
    }

    return true;
}

bool rwkv_fread_tensor(FILE * file, struct rwkv_tensor & output, void * buffer = NULL) {
    RWKV_ENSURE_OR_FALSE(rwkv_fread_tensor_header(file, output.header));
    RWKV_ENSURE_OR_FALSE(rwkv_fread_tensor_data(file, output, buffer));
    return true;
}

bool rwkv_fwrite_tensor(FILE * file, const struct rwkv_tensor & tensor) {
    RWKV_ENSURE_OR_FALSE(rwkv_fwrite_tensor_header(file, tensor.header));
    RWKV_ENSURE_OR_FALSE(rwkv_fwrite_string(file, tensor.name));
    RWKV_ENSURE_OR_FALSE(rwkv_fwrite_data(file, tensor.data, rwkv_tensor_size(tensor.header)));
    return true;
}

// --- Model definition ---

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
    struct rwkv_file_header header;

    struct ggml_tensor * emb;

    struct ggml_tensor * ln0_weight;
    struct ggml_tensor * ln0_bias;

    std::unique_ptr<struct rwkv_layer []> layers;

    struct ggml_tensor * ln_out_weight;
    struct ggml_tensor * ln_out_bias;

    struct ggml_tensor * head;
};

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

// Used to calculate the memory usage of GGML contexts before allocating them.
// Since GGML uses an internal bump allocator that can't be grown at runtime, we need to ensure we have enough space,
// while at the same time, not using more memory than necessary.
struct rwkv_ctx_size {
    size_t objects_count = 0;
    size_t objects_size = 0;
    size_t scratch_size = 0;
};

struct rwkv_ggml_context {
    std::unique_ptr<uint8_t []> scratch;
    struct ggml_context * ctx;

    rwkv_ggml_context(): ctx(NULL) {}

    rwkv_ggml_context(struct rwkv_ctx_size size): ctx(NULL) {
        scratch.reset(new(std::nothrow) uint8_t [size.scratch_size]);

        if (!scratch) {
            return;
        }

        const size_t memory_required_overhead = size_t(128) * 1024 * 1024;
        const size_t memory_required_overhead_sc = size_t(64) * 1024 * 1024;

        ctx = ggml_init({ size.objects_count * GGML_OBJECT_SIZE + size.objects_size + memory_required_overhead, NULL, false});

        if (!ctx) {
            return;
        }

        ggml_set_scratch(ctx, { 0, memory_required_overhead_sc + size.scratch_size, scratch.get() });
    }

    struct rwkv_ggml_context & operator=(struct rwkv_ggml_context && source) {
        scratch.reset(source.scratch.release());
        std::swap(ctx, source.ctx);
        return *this;
    }

    ~rwkv_ggml_context() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

// An instance of an RWKV model loaded into memory.
// Contains all the model weights.
// Shared by one or more contexts.
struct rwkv_instance {
    struct rwkv_ggml_context ctx;
    struct rwkv_model model;

    // TODO come up with a better solution to estimate "work tensor" size.
    // The ggml_cgraph allocates a "work tensor" the first time it is used.
    // Currently, the height of blocks.0.ffn.key.weight is the bottleneck in our implementation of RWKV.
    // Since it is the largest dimension used in any matrix multiply, it is the size used for the "work tensor".
    // However, if ggml changes its implementation, or rwkv.cpp changes its own implementation, at any point,
    // this may become outdated. We need to find a way not to hardcode a specific tensor, but to calculate accurately.
    // This may come out of a ggml issue: https://github.com/ggerganov/ggml/issues/214
    size_t ffn_key_size;
};

// The hidden state of a single RWKV layer.
// These are mostly used for dividing up the input state, and writing portions of the output state.
// But they're also used in building the computation graphs, to represent the operations used from input->output
// (operating "in place" on a rwkv_layer_state).
struct rwkv_layer_state {
    struct ggml_tensor * ffn_xx;
    struct ggml_tensor * att_xx;
    struct ggml_tensor * att_aa;
    struct ggml_tensor * att_bb;
    struct ggml_tensor * att_pp;
};

// Holds a single computation graph and its GGML context.
// Graphs each have their own context so that they can be individually freed and rebuilt.
// Graphs read hidden state from the rwkv_context and then write it back to the rwkv_context.
// (see rwkv_context.input_layers and rwkv_context.output_layers)
struct rwkv_graph {
    struct rwkv_ggml_context ctx;
    struct ggml_tensor * tokens;

    // ggml_cgraph is so large that it can cause stack overflows if not stored on the heap
    std::unique_ptr<struct ggml_cgraph> cgraph;
};

// RWKV context for a specific instance.
// Contains computation graphs and is used for inference.
struct rwkv_context {
    std::shared_ptr<struct rwkv_instance> instance;

    // Reused by all graphs.
    struct rwkv_ggml_context ctx;
    struct ggml_tensor * input_state;
    std::unique_ptr<struct rwkv_layer_state []> input_layers;
    struct ggml_tensor * output_state;
    std::unique_ptr<struct rwkv_layer_state []> output_layers;
    struct ggml_tensor * logits;

    uint32_t n_threads;

    // The serial graph implements the traditional RNN mode that processes only one token at a time (serial mode).
    struct rwkv_graph serial_graph;

    // The sequence graph implements the "sequence mode" (or transformer/GPT mode) that processes multiple tokens at a time.
    // This can be an order of magnitude or so faster than serial execution if used properly.
    size_t sequence_len;
    struct rwkv_graph sequence_graph;

    enum rwkv_error_flags last_error;
    bool print_errors;

    float * state_in = 0; //stores input state, or use null for a new state
    float * state_out = 0; //stores address of output state buffer
    float * logits_out = 0; //stores address of output logit buffer

    size_t gpu_layers;
    size_t vram_total;
};

bool rwkv_fread_ggml_tensor_data(FILE * file, const struct rwkv_tensor_header & header, struct ggml_context * ctx, std::string & name, struct ggml_tensor *& tensor) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_READ, rwkv_fread_string(file, header.key_length, name), "Failed to read tensor name");

    enum ggml_type ggml_type = rwkv_type_to_ggml[header.data_type];
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_UNSUPPORTED, ggml_type != GGML_TYPE_UNKNOWN, "Unsupported tensor data type %s from %s", rwkv_type_to_string[header.data_type], name.c_str());

    tensor = header.dim_count == 1
        ? ggml_new_tensor_1d(ctx, ggml_type, header.width)
        : ggml_new_tensor_2d(ctx, ggml_type, header.width, header.height);

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, tensor, "Failed to allocate tensor");
    ggml_set_name(tensor, name.c_str());

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_READ, rwkv_fread_data(file, ggml_nbytes(tensor), tensor->data), "Failed to read tensor data from %s", name.c_str());
    return true;
}

bool rwkv_fread_ggml_tensor(FILE * file, struct ggml_context * ctx, std::string & name, struct ggml_tensor *& tensor) {
    struct rwkv_tensor_header header;
    RWKV_ENSURE_OR_FALSE_MSG(rwkv_fread_tensor_header(file, header), "Invalid tensor header");
    return rwkv_fread_ggml_tensor_data(file, header, ctx, name, tensor);
}

template<typename F> // https://stackoverflow.com/a/6458689
bool rwkv_set_params(struct rwkv_model & model, F callback) {
    RWKV_ENSURE_OR_FALSE(callback("emb.weight", model.emb));
    RWKV_ENSURE_OR_FALSE(callback("blocks.0.ln0.weight", model.ln0_weight));
    RWKV_ENSURE_OR_FALSE(callback("blocks.0.ln0.bias", model.ln0_bias));

    uint32_t n_layer = model.header.n_layer;
    std::unique_ptr<struct rwkv_layer []> layers(new(std::nothrow) struct rwkv_layer [n_layer]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, layers.get(), "Failed to allocate model layers");
    model.layers = std::move(layers);

    for (uint32_t i = 0; i < n_layer; i++) {
        char buffer[128];
        size_t offset = sprintf(buffer, "blocks.%" PRId32 ".", i);

        rwkv_layer & layer = model.layers[i];
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln1.weight"), buffer), layer.ln1_weight));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln1.bias"), buffer), layer.ln1_bias));

        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_mix_k"), buffer), layer.att_time_mix_k));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_mix_v"), buffer), layer.att_time_mix_v));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_mix_r"), buffer), layer.att_time_mix_r));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_first"), buffer), layer.att_time_first));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_decay"), buffer), layer.att_time_decay));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.key.weight"), buffer), layer.att_key));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.value.weight"), buffer), layer.att_value));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.receptance.weight"), buffer), layer.att_receptance));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.output.weight"), buffer), layer.att_output));

        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln2.weight"), buffer), layer.ln2_weight));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln2.bias"), buffer), layer.ln2_bias));

        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.time_mix_k"), buffer), layer.ffn_time_mix_k));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.time_mix_r"), buffer), layer.ffn_time_mix_r));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.key.weight"), buffer), layer.ffn_key));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.value.weight"), buffer), layer.ffn_value));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.receptance.weight"), buffer), layer.ffn_receptance));
    }

    RWKV_ENSURE_OR_FALSE(callback("ln_out.weight", model.ln_out_weight));
    RWKV_ENSURE_OR_FALSE(callback("ln_out.bias", model.ln_out_bias));
    RWKV_ENSURE_OR_FALSE(callback("head.weight", model.head));
    return true;
}

void rwkv_ctx_size_add_objects(struct rwkv_ctx_size & ctx_size, size_t objects, size_t object_size = sizeof(struct ggml_tensor)) {
    ctx_size.objects_count += objects;
    ctx_size.objects_size += ((object_size + 15) & ~15) * objects;
}

void rwkv_ctx_size_add_scratch(struct rwkv_ctx_size & ctx_size, size_t length, size_t count = 1) {
    ctx_size.scratch_size += ((length + 15) & ~15) * count;
}

void rwkv_ctx_size_add(struct rwkv_ctx_size & ctx_size, size_t objects, size_t scratch = 0, size_t scratches = 1) {
    rwkv_ctx_size_add_objects(ctx_size, objects);
    rwkv_ctx_size_add_scratch(ctx_size, scratch, scratches);
}

void rwkv_ctx_size_add(struct rwkv_ctx_size & ctx_size, size_t count, const struct rwkv_ctx_size & other) {
    ctx_size.objects_count += other.objects_count * count;
    ctx_size.objects_size += other.objects_size * count;
    ctx_size.scratch_size += other.scratch_size * count;
}

void rwkv_ctx_size_add_tensor(struct rwkv_ctx_size & ctx_size, const uint64_t tensors, const uint64_t views, const enum ggml_type type, const uint64_t width, const uint64_t height = 1) {
    rwkv_ctx_size_add_objects(ctx_size, tensors + views);
    rwkv_ctx_size_add_scratch(ctx_size, rwkv_tensor_size(type, width, height), tensors);
}

void rwkv_ctx_size_add_tensor(struct rwkv_ctx_size & size, const uint64_t tensors, const uint64_t views, const struct rwkv_tensor_header & header) {
    rwkv_ctx_size_add_tensor(size, tensors, views, rwkv_type_to_ggml[header.data_type], header.width, header.height);
}

struct rwkv_ctx_size rwkv_xx_size(const size_t n_embed = 0, const size_t sequence_len = 1) {
    struct rwkv_ctx_size ctx_size;

    if (sequence_len == 1) {
        /*  x0 */ rwkv_ctx_size_add_tensor(ctx_size, 2, 1, GGML_TYPE_F32, n_embed);
    } else {
        /*  x0 */ rwkv_ctx_size_add_tensor(ctx_size, 4, 1, GGML_TYPE_F32, n_embed, sequence_len);

        /*  xx */ rwkv_ctx_size_add_tensor(ctx_size, 1, 2, GGML_TYPE_F32, n_embed, sequence_len);
        /*  xx */ rwkv_ctx_size_add_objects(ctx_size, 2, sizeof(struct ggml_tensor) + rwkv_tensor_size(GGML_TYPE_I32, 5));
        /*  xx */ rwkv_ctx_size_add_tensor(ctx_size, 0, 1, GGML_TYPE_F32, n_embed * sequence_len - 1);

        /*  xx */ rwkv_ctx_size_add_tensor(ctx_size, 0, 1, GGML_TYPE_F32, n_embed);
    }

    return ctx_size;
}

void rwkv_xx(struct ggml_context * ctx, struct ggml_tensor * weight, struct ggml_tensor * bias, struct ggml_tensor *& x, struct ggml_tensor *& xx, struct ggml_tensor *& state) {
    size_t n_embed = x->ne[0];
    size_t sequence_len = x->ne[1];

    if (sequence_len == 1) {
        // self.layer_norm(x, self.w.blocks[i].ln2)
        x = rwkv_layer_norm(ctx, x, weight, bias);

        // xx = state[5*i+0]
        xx = state;

        // state[5*i+0] = x
        state = x;
    } else {
        // self.layer_norm(x, self.w.blocks[i].ln2)
        x = rwkv_layer_norm(ctx, x, ggml_repeat(ctx, weight, x), ggml_repeat(ctx, bias, x));

        // xx = torch.cat((state[5*i+0].to(dtype=self.FLOAT_MODE).unsqueeze(0), x[:-1,:]))
        xx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embed, sequence_len);
        xx = ggml_set_1d_inplace(ctx, xx, state, 0);
        xx = ggml_set_1d_inplace(ctx, xx, ggml_view_1d(ctx, x, n_embed * (sequence_len - 1), 0), n_embed * sizeof(float));

        // state[5*i+0] = x[-1,:]
        state = ggml_view_1d(ctx, x, n_embed, n_embed * (sequence_len - 1) * sizeof(float));
    }
}

struct rwkv_ctx_size rwkv_att_rkv_size(const size_t n_embed = 0, const size_t sequence_len = 1) {
    size_t ptr_nelem = sizeof(void *) / sizeof(uint32_t);

    struct rwkv_ctx_size ctx_size;
    /*  xk */ rwkv_ctx_size_add_tensor(ctx_size, 2, 1, GGML_TYPE_F32, n_embed, sequence_len);
    /*  xk */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);
    /*  xk */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);

    /*  xv */ rwkv_ctx_size_add_tensor(ctx_size, 2, 1, GGML_TYPE_F32, n_embed, sequence_len);
    /*  xv */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);
    /*  xv */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);

    /*  xr */ rwkv_ctx_size_add_tensor(ctx_size, 2, 1, GGML_TYPE_F32, n_embed, sequence_len);
    /*  xr */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);
    /*  xr */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);

    /*   r */ rwkv_ctx_size_add_tensor(ctx_size, 2, 0, GGML_TYPE_F32, n_embed, sequence_len);
    /*   r */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);
    /*   k */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed, sequence_len);
    /*   v */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed, sequence_len);

    return ctx_size;
}

void rwkv_att_rkv(struct ggml_context * ctx, struct rwkv_layer layer, struct ggml_tensor * x0, struct ggml_tensor * xx, struct ggml_tensor *& r, struct ggml_tensor *& k, struct ggml_tensor *& v) {
    // xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
    struct ggml_tensor * xk = ggml_add_inplace(ctx,
        ggml_mul(ctx, x0, layer.att_time_mix_k),
        ggml_mul(ctx, xx, rwkv_1_minus_x(ctx, layer.att_time_mix_k))
    );

    // xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
    struct ggml_tensor * xv = ggml_add_inplace(ctx,
        ggml_mul(ctx, x0, layer.att_time_mix_v),
        ggml_mul(ctx, xx, rwkv_1_minus_x(ctx, layer.att_time_mix_v))
    );

    // xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
    struct ggml_tensor * xr = ggml_add_inplace(ctx,
        ggml_mul(ctx, x0, layer.att_time_mix_r),
        ggml_mul(ctx, xx, rwkv_1_minus_x(ctx, layer.att_time_mix_r))
    );

    // r = torch.sigmoid(rw @ xr)
    r = rwkv_sigmoid(ctx, ggml_mul_mat(ctx, layer.att_receptance, xr));
    // k = kw @ xk
    k = ggml_mul_mat(ctx, layer.att_key, xk);
    // v = vw @ xv
    v = ggml_mul_mat(ctx, layer.att_value, xv);
}

struct rwkv_ctx_size rwkv_att_wkv_size(const size_t n_embed = 0) {
    size_t ptr_nelem = sizeof(void *) / sizeof(uint32_t);

    struct rwkv_ctx_size ctx_size;
    /*  ww */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);
    /*  qq */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);
    /*  qq */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);
    /*  e1 */ rwkv_ctx_size_add_tensor(ctx_size, 2, 0, GGML_TYPE_F32, n_embed);
    /*  e1 */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);
    /*  e2 */ rwkv_ctx_size_add_tensor(ctx_size, 2, 0, GGML_TYPE_F32, n_embed);
    /*  e2 */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);

    /*   a */ rwkv_ctx_size_add_tensor(ctx_size, 2, 1, GGML_TYPE_F32, n_embed);
    /*   b */ rwkv_ctx_size_add_tensor(ctx_size, 1, 1, GGML_TYPE_F32, n_embed);

    /*  ww */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);
    /*  qq */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);
    /*  qq */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);
    /*  e1 */ rwkv_ctx_size_add_tensor(ctx_size, 2, 0, GGML_TYPE_F32, n_embed);
    /*  e1 */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);
    /*  e2 */ rwkv_ctx_size_add_tensor(ctx_size, 2, 0, GGML_TYPE_F32, n_embed);
    /*  e2 */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);

    /*  aa */ rwkv_ctx_size_add_tensor(ctx_size, 2, 1, GGML_TYPE_F32, n_embed);
    /*  bb */ rwkv_ctx_size_add_tensor(ctx_size, 1, 1, GGML_TYPE_F32, n_embed);
    /*  pp */ rwkv_ctx_size_add_tensor(ctx_size, 0, 0, GGML_TYPE_F32, n_embed);

    /* wkv */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);

    return ctx_size;
}

struct ggml_tensor * rwkv_att_wkv(struct ggml_context * ctx, struct ggml_tensor * att_time_first, struct ggml_tensor * att_time_decay, struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor *& aa, struct ggml_tensor *& bb, struct ggml_tensor *& pp) {
    // ww = time_first + k
    struct ggml_tensor * ww = ggml_add(ctx, att_time_first, k);
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
    ww = ggml_add(ctx, pp, att_time_decay);
    // qq = torch.maximum(ww, k)
    qq = rwkv_max(ctx, ww, k);
    // e1 = torch.exp(ww - qq)
    e1 = rwkv_exp(ctx, ggml_sub(ctx, ww, qq));
    // e2 = torch.exp(k[t] - qq)
    e2 = rwkv_exp(ctx, ggml_sub(ctx, k, qq));

    // state[5 * i + 2] = e1 * aa + e2 * v
    // state[5 * i + 3] = e1 * bb + e2
    // state[5 * i + 4] = qq
    aa = ggml_add_inplace(ctx, ggml_mul(ctx, e1, aa), ggml_mul(ctx, e2, v));
    bb = ggml_add_inplace(ctx, ggml_mul(ctx, e1, bb), e2);
    pp = qq;

    // wkv = a / b
    return ggml_div(ctx, a, b);
}

struct rwkv_ctx_size rwkv_att_size(const size_t n_embed = 0) {
    size_t ptr_nelem = sizeof(void *) / sizeof(uint32_t);

    struct rwkv_ctx_size ctx_size;
    /*  xx */ rwkv_ctx_size_add(ctx_size, 1, rwkv_xx_size(n_embed));
    /* rkv */ rwkv_ctx_size_add(ctx_size, 1, rwkv_att_rkv_size(n_embed));
    /* wkv */ rwkv_ctx_size_add(ctx_size, 1, rwkv_att_wkv_size(n_embed));
    /*   x */ rwkv_ctx_size_add_tensor(ctx_size, 2, 0, GGML_TYPE_F32, n_embed);

    return ctx_size;
}

struct ggml_tensor * rwkv_att(struct ggml_context * ctx, struct ggml_tensor * x, struct rwkv_layer layer, struct rwkv_layer_state & state) {
    struct ggml_tensor * x0 = x, * xx;
    rwkv_xx(ctx, layer.ln1_weight, layer.ln1_bias, x0, xx, state.att_xx);

    struct ggml_tensor * r, * k, * v;
    rwkv_att_rkv(ctx, layer, x0, xx, r, k, v);

    struct ggml_tensor * wkv = rwkv_att_wkv(ctx, layer.att_time_first, layer.att_time_decay, k, v, state.att_aa, state.att_bb, state.att_pp);

    // ow @ (r * xx)
    return ggml_mul_mat(ctx, layer.att_output, ggml_mul(ctx, r, wkv));
}

struct rwkv_ctx_size rwkv_ffn_size(const size_t n_embed = 0, const size_t ffn_key = 0, const size_t sequence_len = 1) {
    size_t ptr_nelem = sizeof(void *) / sizeof(uint32_t);

    struct rwkv_ctx_size ctx_size;
    /* xx */ rwkv_ctx_size_add(ctx_size, 1, rwkv_xx_size(n_embed, sequence_len));

    /* xk */ rwkv_ctx_size_add_tensor(ctx_size, 2, 1, GGML_TYPE_F32, n_embed, sequence_len);
    /* xk */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);
    /* xk */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);

    /* xr */ rwkv_ctx_size_add_tensor(ctx_size, 2, 1, GGML_TYPE_F32, n_embed, sequence_len);
    /* xr */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);
    /* xr */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);

    /*  r */ rwkv_ctx_size_add_tensor(ctx_size, 2, 0, GGML_TYPE_F32, n_embed, sequence_len);
    /*  r */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, ptr_nelem);
    /*  k */ rwkv_ctx_size_add_tensor(ctx_size, 3, 0, GGML_TYPE_F32, ffn_key, sequence_len);

    /*  x */ rwkv_ctx_size_add_tensor(ctx_size, 2, 0, GGML_TYPE_F32, n_embed, sequence_len);

    return ctx_size;
}

struct ggml_tensor * rwkv_ffn(struct ggml_context * ctx, struct ggml_tensor * x, struct rwkv_layer layer, struct rwkv_layer_state & state) {
    struct ggml_tensor * x0 = x, * xx;
    rwkv_xx(ctx, layer.ln2_weight, layer.ln2_bias, x0, xx, state.ffn_xx);

    // xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
    // xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
    struct ggml_tensor * xk = ggml_add_inplace(
        ctx,
        ggml_mul(ctx, x0, layer.ffn_time_mix_k),
        ggml_mul(ctx, xx, rwkv_1_minus_x(ctx, layer.ffn_time_mix_k))
    );

    // xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
    struct ggml_tensor * xr = ggml_add_inplace(
        ctx,
        ggml_mul(ctx, x0, layer.ffn_time_mix_r),
        ggml_mul(ctx, xx, rwkv_1_minus_x(ctx, layer.ffn_time_mix_r))
    );

    // r = torch.sigmoid(rw @ xr)
    struct ggml_tensor * r = rwkv_sigmoid(ctx, ggml_mul_mat(ctx, layer.ffn_receptance, xr));

    // k = torch.square(torch.relu(kw @ xk))
    struct ggml_tensor * k = ggml_sqr(ctx, ggml_relu(ctx, ggml_mul_mat(ctx, layer.ffn_key, xk)));

    // r * (vw @ k)
    return ggml_mul(ctx, r, ggml_mul_mat(ctx, layer.ffn_value, k));
}

struct rwkv_ctx_size rwkv_serial_graph_size(const size_t n_vocab, const size_t n_embed, const size_t n_layer, const size_t ffn_key_size) {
    struct rwkv_ctx_size ctx_size;
    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed);
    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, 2, 1, GGML_TYPE_F32, n_embed);

    /*    att */ rwkv_ctx_size_add(ctx_size, n_layer, rwkv_att_size(n_embed));
    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, 0, n_layer, GGML_TYPE_F32, n_embed);
    /*    ffn */ rwkv_ctx_size_add(ctx_size, n_layer, rwkv_ffn_size(n_embed, ffn_key_size));
    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, 0, n_layer, GGML_TYPE_F32, n_embed);

    /* output */ rwkv_ctx_size_add_tensor(ctx_size, 0, n_layer * 5, GGML_TYPE_F32, n_embed);

    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, 2, 1, GGML_TYPE_F32, n_embed);
    /* logits */ rwkv_ctx_size_add_tensor(ctx_size, 1, 1, GGML_TYPE_F32, n_vocab);

    return ctx_size;
}

bool rwkv_build_serial_graph(
    struct ggml_context * ctx,
    struct rwkv_model & model,
    struct ggml_tensor * tokens,
    struct rwkv_layer_state * inputs,
    struct rwkv_layer_state * outputs,
    struct ggml_tensor * logits,
    struct ggml_cgraph * cgraph
) {
    size_t n_embed = model.header.n_embed;

    // x = self.w.emb.weight[token]
    struct ggml_tensor * x = ggml_get_rows(ctx, model.emb, tokens);

    // x = self.layer_norm(x, self.w.blocks[0].ln0)
    x = rwkv_layer_norm(ctx, x, model.ln0_weight, model.ln0_bias);

    for (size_t i = 0; i < model.header.n_layer; i++) {
        struct rwkv_layer & layer = model.layers[i];

        struct rwkv_layer_state state = inputs[i];
        x = ggml_add_inplace(ctx, x, rwkv_att(ctx, x, layer, state));
        x = ggml_add_inplace(ctx, x, rwkv_ffn(ctx, x, layer, state));

        struct rwkv_layer_state & output = outputs[i];
        ggml_build_forward_expand(cgraph, ggml_cpy(ctx, state.ffn_xx, output.ffn_xx));
        ggml_build_forward_expand(cgraph, ggml_cpy(ctx, state.att_xx, output.att_xx));
        ggml_build_forward_expand(cgraph, ggml_cpy(ctx, state.att_aa, output.att_aa));
        ggml_build_forward_expand(cgraph, ggml_cpy(ctx, state.att_bb, output.att_bb));
        ggml_build_forward_expand(cgraph, ggml_cpy(ctx, state.att_pp, output.att_pp));
    }

    // x = self.layer_norm(x[-1,:], self.w.ln_out)
    x = rwkv_layer_norm(ctx, x, model.ln_out_weight, model.ln_out_bias);

    // x = (self.w.head.weight @ x).float()
    ggml_build_forward_expand(cgraph, ggml_cpy(ctx, ggml_mul_mat(ctx, model.head, x), logits));

    return true;
}

struct rwkv_ctx_size rwkv_sequence_graph_size(const size_t n_vocab, const size_t n_embed, const size_t n_layer, const size_t ffn_key_size, const size_t sequence_len) {
    struct rwkv_ctx_size ctx_size;
    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed, sequence_len);
    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, 4, 1, GGML_TYPE_F32, n_embed, sequence_len);

    /*     xx */ rwkv_ctx_size_add(ctx_size, n_layer, rwkv_xx_size(n_embed, sequence_len));
    /*    rkv */ rwkv_ctx_size_add(ctx_size, n_layer, rwkv_att_rkv_size(n_embed, sequence_len));

    /*     kt */ rwkv_ctx_size_add_tensor(ctx_size, 0, n_layer * sequence_len, GGML_TYPE_F32, n_embed);
    /*     vt */ rwkv_ctx_size_add_tensor(ctx_size, 0, n_layer * sequence_len, GGML_TYPE_F32, n_embed);
    /*     xt */ rwkv_ctx_size_add_tensor(ctx_size, 0, n_layer * sequence_len, GGML_TYPE_F32, n_embed);
    /*    wkv */ rwkv_ctx_size_add(ctx_size, n_layer * sequence_len, rwkv_att_wkv_size(n_embed));
    /*     xt */ rwkv_ctx_size_add_tensor(ctx_size, 0, n_layer * sequence_len, GGML_TYPE_F32, n_embed);
    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, n_layer * 2, 0, GGML_TYPE_F32, n_embed, sequence_len);

    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, 0, n_layer, GGML_TYPE_F32, n_embed, sequence_len);
    /*    ffn */ rwkv_ctx_size_add(ctx_size, n_layer, rwkv_ffn_size(n_embed, ffn_key_size, sequence_len));
    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, 0, n_layer, GGML_TYPE_F32, n_embed, sequence_len);

    /* output */ rwkv_ctx_size_add_tensor(ctx_size, 0, n_layer * 5, GGML_TYPE_F32, n_embed);

    /*      x */ rwkv_ctx_size_add_tensor(ctx_size, 2, 2, GGML_TYPE_F32, n_embed);
    /* logits */ rwkv_ctx_size_add_tensor(ctx_size, 1, 1, GGML_TYPE_F32, n_vocab);

    return ctx_size;
}

bool rwkv_build_sequence_graph(
    struct ggml_context * ctx,
    struct rwkv_model & model,
    struct ggml_tensor * tokens,
    struct rwkv_layer_state * inputs,
    struct rwkv_layer_state * outputs,
    struct ggml_tensor * logits,
    struct ggml_cgraph * cgraph
) {
    const uint32_t n_embed = model.header.n_embed;
    const size_t sequence_len = tokens->ne[0];

    struct ggml_tensor * x = ggml_get_rows(ctx, model.emb, tokens);
    x = rwkv_layer_norm(ctx, x, ggml_repeat(ctx, model.ln0_weight, x), ggml_repeat(ctx, model.ln0_bias, x));

    for (size_t i = 0; i < model.header.n_layer; i++) {
        struct rwkv_layer & layer = model.layers[i];
        struct rwkv_layer_state state = inputs[i];

        struct ggml_tensor * x0 = x, * xx;
        rwkv_xx(ctx, layer.ln1_weight, layer.ln1_bias, x0, xx, state.att_xx);

        struct ggml_tensor * r, * k, * v;
        rwkv_att_rkv(ctx, layer, x0, xx, r, k, v);

        ggml_build_forward_expand(cgraph, r);

        for (uint32_t t = 0; t < sequence_len; t++) {
            struct ggml_tensor * kt = ggml_view_1d(ctx, k, n_embed, n_embed * sizeof(float) * t);
            struct ggml_tensor * vt = ggml_view_1d(ctx, v, n_embed, n_embed * sizeof(float) * t);
            struct ggml_tensor * xt = ggml_view_1d(ctx, xx, n_embed, n_embed * sizeof(float) * t);
            struct ggml_tensor * wkv = rwkv_att_wkv(ctx, layer.att_time_first, layer.att_time_decay, kt, vt, state.att_aa, state.att_bb, state.att_pp);
            ggml_build_forward_expand(cgraph, ggml_cpy(ctx, wkv, xt));
        }

        x = ggml_add_inplace(ctx, x, ggml_mul_mat(ctx, layer.att_output, ggml_mul(ctx, r, xx)));
        x = ggml_add_inplace(ctx, x, rwkv_ffn(ctx, x, layer, state));

        struct rwkv_layer_state & output = outputs[i];
        ggml_build_forward_expand(cgraph, ggml_cpy(ctx, state.ffn_xx, output.ffn_xx));
        ggml_build_forward_expand(cgraph, ggml_cpy(ctx, state.att_xx, output.att_xx));
        ggml_build_forward_expand(cgraph, ggml_cpy(ctx, state.att_aa, output.att_aa));
        ggml_build_forward_expand(cgraph, ggml_cpy(ctx, state.att_bb, output.att_bb));
        ggml_build_forward_expand(cgraph, ggml_cpy(ctx, state.att_pp, output.att_pp));
    }

    // x = self.layer_norm(x[-1,:], self.w.ln_out)
    x = rwkv_layer_norm(ctx, ggml_view_1d(ctx, x, n_embed, n_embed * sizeof(float) * (sequence_len - 1)), model.ln_out_weight, model.ln_out_bias);

    // x = (self.w.head.weight @ x).float()
    ggml_build_forward_expand(cgraph, ggml_cpy(ctx, ggml_mul_mat(ctx, model.head, x), logits));

    return true;
}

size_t rwkv_estimate_graph_work(const enum ggml_type type, const size_t ffn_key_size, const uint32_t n_threads, const size_t sequence_len = 1) {

    enum ggml_type mul_mat_type = ggml_is_quantized(type) ? GGML_TYPE_Q8_1 : type;
    return rwkv_tensor_size(GGML_TYPE_I8, rwkv_tensor_size(mul_mat_type, ffn_key_size, sequence_len) * n_threads + 64 * (n_threads - 1));
}

struct rwkv_file {
    FILE * file;

    rwkv_file(FILE * file): file(file) {}

    ~rwkv_file() {
        if (file) {
            fclose(file);
        }
    }
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

bool rwkv_instance_from_file(const char * file_path, struct rwkv_instance & instance) {
    struct stat file_stat;
    struct rwkv_model model;
    struct rwkv_ggml_context ctx;
    size_t ffn_key_size = 0;

    std::unordered_map<std::string, struct ggml_tensor *> parameters;

    {
        rwkv_file file(fopen(file_path, "rb"));

        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, file.file, "Failed to open file %s", file_path);
        // Be very careful when changing this code. It must support files larger than 2 GB by using 64-bit functions to get the file length.
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_STAT, fstat(fileno(file.file), &file_stat) == 0, "Failed to stat file %s", file_path);
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE, rwkv_fread_file_header(file.file, model.header), "Invalid file header");

        struct rwkv_tensor_header tensor_header;
        std::string name;
        struct rwkv_ctx_size ctx_size;

        while ((size_t) ftell(file.file) < (size_t) file_stat.st_size) {
            RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS, rwkv_fread_tensor_header(file.file, tensor_header), "Invalid tensor header");
            RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS, rwkv_fread_string(file.file, tensor_header.key_length, name), "Failed to read tensor name");
            RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_READ, fseek(file.file, rwkv_tensor_size(tensor_header), SEEK_CUR) == 0, "Failed to read tensor data");

            rwkv_ctx_size_add_tensor(ctx_size, 1, 0, tensor_header);

            if (ffn_key_size == 0 && name == "blocks.0.ffn.key.weight") {
                ffn_key_size = tensor_header.height;
            }
        }

        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_PARAM_MISSING, ffn_key_size, "Model is missing parameter blocks.0.ffn.key.weight");
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_READ, fseek(file.file, sizeof(struct rwkv_file_header), SEEK_SET) == 0, "Failed to seek in file");

        ctx = ctx_size;
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, ctx.ctx, "Failed to allocate model context");

        struct ggml_tensor * tensor;

        while ((size_t) ftell(file.file) < (size_t) file_stat.st_size) {
            RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS, rwkv_fread_ggml_tensor(file.file, ctx.ctx, name, tensor), "Failed to read model params");
            parameters[std::move(name)] = tensor;
        }
    }

    std::unordered_map<std::string, struct ggml_tensor *> & parameters_ref = parameters;
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_PARAM_MISSING, rwkv_set_params(model, [&](const char * key, struct ggml_tensor *& dest) {
        struct ggml_tensor * tensor = parameters_ref[key];
        RWKV_ENSURE_OR_FALSE_MSG(tensor, "Model parameter %s not found", key);
        dest = tensor;
        return true;
    }));

    // Verify order of dimensions
    struct ggml_tensor * emb = model.emb;
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_SHAPE, emb->n_dims == 2, "Unexpected dimension count of embedding matrix %d", emb->n_dims);
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[0] == model.header.n_embed, "Unexpected dimension of embedding matrix %" PRId64, emb->ne[0]);
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[1] == model.header.n_vocab, "Unexpected dimension of embedding matrix %" PRId64, emb->ne[1]);

    instance.ctx = std::move(ctx);
    instance.model = std::move(model);
    instance.ffn_key_size = ffn_key_size;
    return true;
}

struct rwkv_context * rwkv_new_context_impl(std::shared_ptr<struct rwkv_instance> instance, const uint32_t n_threads) {
    global_last_error = RWKV_ERROR_NONE;

    struct rwkv_file_header & header = instance->model.header;
    const size_t n_vocab = header.n_vocab;
    const size_t n_embed = header.n_embed;
    const size_t n_layer = header.n_layer;

    struct rwkv_ctx_size ctx_size;
    /*   input */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed * 5 * n_layer);
    /*  output */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_embed * 5 * n_layer);
    /*  inputs */ rwkv_ctx_size_add_tensor(ctx_size, 0, 5 * n_layer, GGML_TYPE_F32, n_embed);
    /* outputs */ rwkv_ctx_size_add_tensor(ctx_size, 0, 5 * n_layer, GGML_TYPE_F32, n_embed);
    /*  logits */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_F32, n_vocab);

    struct rwkv_ggml_context ctx(ctx_size);
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, ctx.ctx, "Failed to allocate model context");

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx.ctx, GGML_TYPE_F32, n_embed * 5 * n_layer);
    struct ggml_tensor * output = ggml_new_tensor_1d(ctx.ctx, GGML_TYPE_F32, n_embed * 5 * n_layer);

    // We collect parts of input state here. Each part is (n_embed) vector.
    std::unique_ptr<struct rwkv_layer_state []> inputs(new(std::nothrow) struct rwkv_layer_state [n_layer]);
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_ALLOC, inputs.get(), "Failed to allocate input state parts");

    // We collect parts of output state here. Each part is (n_embed) vector.
    std::unique_ptr<struct rwkv_layer_state []> outputs(new(std::nothrow) struct rwkv_layer_state [n_layer]);
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_ALLOC, outputs.get(), "Failed to allocate output state parts");

    for (size_t i = 0; i < n_layer; i++) {
        struct rwkv_layer_state & input_state = inputs[i];
        input_state.ffn_xx = ggml_view_1d(ctx.ctx, input, n_embed, n_embed * (i * 5 + 0) * sizeof(float));
        input_state.att_xx = ggml_view_1d(ctx.ctx, input, n_embed, n_embed * (i * 5 + 1) * sizeof(float));
        input_state.att_aa = ggml_view_1d(ctx.ctx, input, n_embed, n_embed * (i * 5 + 2) * sizeof(float));
        input_state.att_bb = ggml_view_1d(ctx.ctx, input, n_embed, n_embed * (i * 5 + 3) * sizeof(float));
        input_state.att_pp = ggml_view_1d(ctx.ctx, input, n_embed, n_embed * (i * 5 + 4) * sizeof(float));

        struct rwkv_layer_state & output_state = outputs[i];
        output_state.ffn_xx = ggml_view_1d(ctx.ctx, output, n_embed, n_embed * (i * 5 + 0) * sizeof(float));
        output_state.att_xx = ggml_view_1d(ctx.ctx, output, n_embed, n_embed * (i * 5 + 1) * sizeof(float));
        output_state.att_aa = ggml_view_1d(ctx.ctx, output, n_embed, n_embed * (i * 5 + 2) * sizeof(float));
        output_state.att_bb = ggml_view_1d(ctx.ctx, output, n_embed, n_embed * (i * 5 + 3) * sizeof(float));
        output_state.att_pp = ggml_view_1d(ctx.ctx, output, n_embed, n_embed * (i * 5 + 4) * sizeof(float));
    }

    struct ggml_tensor * logits = ggml_new_tensor_1d(ctx.ctx, GGML_TYPE_F32, n_vocab);

    struct rwkv_ctx_size graph_ctx_size;
    /* token */ rwkv_ctx_size_add_objects(graph_ctx_size, 1, sizeof(struct ggml_tensor) + sizeof(uint32_t));
    /* graph */ rwkv_ctx_size_add(graph_ctx_size, 1, rwkv_serial_graph_size(n_vocab, n_embed, n_layer, instance->ffn_key_size));
    /*  work */ rwkv_ctx_size_add(graph_ctx_size, 1, rwkv_estimate_graph_work(rwkv_type_to_ggml[header.data_type], instance->ffn_key_size, n_threads));

    struct rwkv_graph serial_graph;
    serial_graph.ctx = graph_ctx_size;
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, serial_graph.ctx.ctx, "Failed to allocate serial graph context");
    serial_graph.tokens = ggml_new_i32(serial_graph.ctx.ctx, 0);
    serial_graph.cgraph.reset(new(std::nothrow) struct ggml_cgraph());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_ALLOC, serial_graph.cgraph, "Failed to allocate serial graph");
    serial_graph.cgraph->n_threads = n_threads;
    RWKV_ASSERT_NULL(RWKV_ERROR_GRAPH, rwkv_build_serial_graph(serial_graph.ctx.ctx, instance->model, serial_graph.tokens, inputs.get(), outputs.get(), logits, serial_graph.cgraph.get()));

    std::unique_ptr<struct rwkv_context> rwkv_ctx(new(std::nothrow) struct rwkv_context());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, rwkv_ctx, "Failed to allocate rwkv_context");
    rwkv_ctx->instance = std::move(instance);
    rwkv_ctx->ctx = std::move(ctx);
    rwkv_ctx->input_state = input;
    rwkv_ctx->input_layers = std::move(inputs);
    rwkv_ctx->output_state = output;
    rwkv_ctx->output_layers = std::move(outputs);
    rwkv_ctx->logits = logits;
    rwkv_ctx->n_threads = n_threads;
    rwkv_ctx->serial_graph = std::move(serial_graph);
    rwkv_ctx->last_error = RWKV_ERROR_NONE;
    rwkv_ctx->print_errors = global_print_errors;
    return rwkv_ctx.release();
}

struct rwkv_context * rwkv_init_from_file(const char * file_path, const uint32_t n_threads) {
    global_last_error = RWKV_ERROR_NONE;

    std::shared_ptr<struct rwkv_instance> instance(new(std::nothrow) struct rwkv_instance());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, instance, "Failed to allocate instance");
    RWKV_ENSURE_OR_NULL(rwkv_instance_from_file(file_path, *instance.get()));
    return rwkv_new_context_impl(instance, n_threads);
}

struct rwkv_context * rwkv_clone_context(struct rwkv_context * ctx, const uint32_t n_threads) {
    struct rwkv_context * clone = rwkv_new_context_impl(ctx->instance, n_threads);

    if (clone) {
        clone->print_errors = ctx->print_errors;
    }

    return clone;
}

bool rwkv_gpu_offload_layers(const struct rwkv_context * ctx, const uint32_t n_gpu_layers) {

    return true;
}

void rwkv_set_inputs(const struct rwkv_context * ctx, const float * state_in) {
    if (state_in) {
        memcpy(ctx->input_state->data, state_in, ggml_nbytes(ctx->input_state));
    } else {
        ggml_set_f32(ctx->input_state, 0.0F);

        for (size_t i = 0; i < ctx->instance->model.header.n_layer; i++) {
            ggml_set_f32(ctx->input_layers[i].att_pp, -1e30F);
        }
    }
}

void rwkv_get_outputs(const struct rwkv_context * ctx, float * state_out, float * logits_out) {
    if (state_out) {
        memcpy(state_out, ctx->output_state->data, ggml_nbytes(ctx->output_state));
    }

    if (logits_out) {
        memcpy(logits_out, ctx->logits->data, ggml_nbytes(ctx->logits));
    }
}

bool rwkv_eval(const struct rwkv_context * ctx, const uint32_t token, const float * state_in, float * state_out, float * logits_out) {
    ((struct rwkv_context *) ctx)->last_error = RWKV_ERROR_NONE;

    const struct rwkv_file_header & header = ctx->instance->model.header;
    const size_t n_vocab = header.n_vocab;
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, token < n_vocab, "Token (%" PRId32 ") is out of range (0 ..= %zu)", token, n_vocab - 1);

    rwkv_set_inputs(ctx, state_in);
    ggml_set_i32(ctx->serial_graph.tokens, token);
    ggml_graph_compute(ctx->serial_graph.ctx.ctx, ctx->serial_graph.cgraph.get());
    rwkv_get_outputs(ctx, state_out, logits_out);

    return true;
}

bool rwkv_eval_sequence(const struct rwkv_context * ctx, const uint32_t * sequence, const size_t sequence_len, const float * state_in, float * state_out, float * logits_out) {
    ((struct rwkv_context *) ctx)->last_error = RWKV_ERROR_NONE;

    const struct rwkv_file_header & header = ctx->instance->model.header;
    const size_t n_vocab = header.n_vocab;
    const size_t n_embed = header.n_embed;
    const size_t n_layer = header.n_layer;

    if (sequence) {
        for (size_t i = 0; i < sequence_len; i++) {
            const uint32_t token = sequence[i];
            RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, token < n_vocab, "Tokens[%zu] (%" PRId32 ") is out of range (0 ..= %zu)", i, token, n_vocab - 1);
        }
    }

    if (ctx->sequence_len != sequence_len) {
        // Build new sequence graph
        struct rwkv_ctx_size ctx_size;
        /* tokens */ rwkv_ctx_size_add_tensor(ctx_size, 1, 0, GGML_TYPE_I32, sequence_len);
        /*  graph */ rwkv_ctx_size_add(ctx_size, 1, rwkv_sequence_graph_size(n_vocab, n_embed, n_layer, ctx->instance->ffn_key_size, sequence_len));
        /*   work */ rwkv_ctx_size_add(ctx_size, 1, rwkv_estimate_graph_work(rwkv_type_to_ggml[header.data_type], ctx->instance->ffn_key_size, 1, sequence_len));

        struct rwkv_graph graph;
        graph.ctx = ctx_size;
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, graph.ctx.ctx, "Failed to allocate sequence graph context");
        graph.tokens = ggml_new_tensor_1d(graph.ctx.ctx, GGML_TYPE_I32, sequence_len);
        graph.cgraph.reset(new(std::nothrow) struct ggml_cgraph());
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, graph.cgraph, "Failed to allocate sequence graph");
        graph.cgraph->n_threads = 1;
        RWKV_ASSERT_FALSE(RWKV_ERROR_GRAPH, rwkv_build_sequence_graph(graph.ctx.ctx, ctx->instance->model, graph.tokens, ctx->input_layers.get(), ctx->output_layers.get(), ctx->logits, graph.cgraph.get()));

        ((struct rwkv_context *) ctx)->sequence_len = sequence_len;
        ((struct rwkv_context *) ctx)->sequence_graph = std::move(graph);
    }

    // Allow building the sequence graph without actually evaluating, by specifying sequence = NULL.
    if (sequence) {
        rwkv_set_inputs(ctx, state_in);
        memcpy(ctx->sequence_graph.tokens->data, sequence, sequence_len * sizeof(uint32_t));
        ggml_graph_compute(ctx->sequence_graph.ctx.ctx, ctx->sequence_graph.cgraph.get());
        rwkv_get_outputs(ctx, state_out, logits_out);
    }

    return true;
}

uint32_t rwkv_get_state_buffer_element_count(const struct rwkv_context * ctx) {
    return ctx->instance->model.header.n_layer * 5 * ctx->instance->model.header.n_embed;
}

uint32_t rwkv_get_logits_buffer_element_count(const struct rwkv_context * ctx) {
    return ctx->instance->model.header.n_vocab;
}

void rwkv_free(struct rwkv_context * ctx) {
    std::unique_ptr<struct rwkv_context> rwkv_ctx(ctx);
}

bool rwkv_quantize_model_file(const char * in_path, const char * out_path, const char * type_name) {
    global_last_error = RWKV_ERROR_NONE;

    enum ggml_type out_type = rwkv_type_to_ggml[rwkv_type_from_string(type_name)];
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS | RWKV_ERROR_DATA_TYPE, ggml_is_quantized(out_type), "Unsupported output data type (%s)", rwkv_type_to_string[rwkv_type_from_ggml[out_type]]);

    RWKV_MSG("Loading model from '%s'\n", in_path);

    struct stat in_stat;

    struct rwkv_file in_file(fopen(in_path, "rb"));
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, in_file.file, "Failed to open %s for reading", in_path);

    // Be very careful when changing this code. It must support files larger than 2 GB by using 64-bit functions to the get file length.
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_STAT, fstat(fileno(in_file.file), &in_stat) == 0, "failed to stat file %s", in_path);

    struct rwkv_file out_file(fopen(out_path, "wb"));
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, out_file.file, "Failed to open %s for writing", out_path);

    struct rwkv_file_header in_header;
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE, rwkv_fread_file_header(in_file.file, in_header), "Invalid file header");

    enum ggml_type in_type = rwkv_type_to_ggml[in_header.data_type];
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE, in_type == GGML_TYPE_F32 || in_type == GGML_TYPE_F16, "Unsupported input data type (%s); needs to be f32 or f16", rwkv_type_to_string[rwkv_type_from_ggml[in_type]]);

    struct rwkv_file_header out_header = in_header;
    out_header.version = RWKV_FILE_VERSION;
    out_header.data_type = rwkv_type_from_ggml[out_type];
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE, rwkv_fwrite_file_header(out_file.file, out_header), "Failed to write file header");

    // Process parameters
    size_t orig_total_size = 0;
    size_t new_total_size = 0;

    // Required to init the fp16 tables
    // Doesn't crash if ggml_init fails
    ggml_free(ggml_init({ 0, NULL, true }));

    size_t max_in_size = 0;
    size_t max_out_size = 0;
    size_t max_key_length = 0;

    while (ftell(in_file.file) < in_stat.st_size) {
        struct rwkv_tensor_header header;
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, rwkv_fread_tensor_header_and_skip(in_file.file, header));

        size_t in_size = rwkv_tensor_size(header);

        if (in_size > max_in_size) {
            max_in_size = in_size;
        }

        // f16 type tensors get relocated to out and then converted into f32 at in
        if (header.data_type == TYPE_F16) {
            if (in_size > max_out_size) {
                max_out_size = in_size;
            }

            size_t f32_size = rwkv_tensor_size(GGML_TYPE_F32, header.width, header.height);

            if (f32_size > max_in_size) {
                max_in_size = f32_size;
            }
        }

        size_t out_size = rwkv_tensor_size(out_type, header.width, header.height);

        if (out_size > max_out_size) {
            max_out_size = out_size;
        }

        if (header.key_length > max_key_length) {
            max_key_length = header.key_length;
        }
    }

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_READ, fseek(in_file.file, sizeof(struct rwkv_file_header), SEEK_SET) == 0, "Failed to seek in file");

    // This is a histogram of quantized values. If it shows single 1.0, then all 0.0, something went very wrong!
    int64_t hist_all[16] {};

    std::unique_ptr<uint8_t []> scratch(new(std::nothrow) uint8_t [max_in_size + max_out_size]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, scratch.get(), "Failed to allocate buffer");

    uint8_t * in_buf = scratch.get();
    uint8_t * out_buf = in_buf + max_in_size;

    struct rwkv_tensor tensor;
    struct rwkv_tensor_header & header = tensor.header;
    std::string & name = tensor.name;
    uint8_t *& data = tensor.data;

    while (ftell(in_file.file) < in_stat.st_size) {
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS, rwkv_fread_tensor_header(in_file.file, header), "Failed to read tensor header");
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS, rwkv_fread_string(in_file.file, header.key_length, name), "Failed to read tensor name");

        const char * name_str = name.c_str();
        RWKV_MSG("%*s - [%5" PRId32 ", %5" PRId32 "], type = %6s ", (int) max_key_length, name_str, header.width, header.height, rwkv_type_to_string[header.data_type]);

        data = header.data_type == TYPE_F16 ? out_buf : in_buf;
        size_t orig_size = rwkv_tensor_size(header), new_size = orig_size;
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS, rwkv_fread_data(in_file.file, orig_size, data), "\nFailed to read tensor data of %s", name_str);

        // Quantize only 2D tensors, except embedding and head matrices.
        // Embedding and head take not too much space, especially in bigger models;
        // but they significantly increase perplexity when quantized.
        if ((header.data_type == TYPE_F32 || header.data_type == TYPE_F16) && header.dim_count == 2 && name != "emb.weight" && name != "head.weight") {
            RWKV_MSG("quantizing... ");

            size_t nelements = (size_t) header.width * (size_t) header.height;

            if (header.data_type == TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) out_buf, (float *) in_buf, nelements);
            }

            int64_t hist_cur[16] {};
            new_size = ggml_quantize_chunk(out_type, (const float *) in_buf, out_buf, 0, nelements, hist_cur);
            header.data_type = rwkv_type_from_ggml[out_type];
            data = out_buf;

            RWKV_MSG("size = %8.2f MB -> %8.2f MB | hist: ", orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);

            for (int i = 0; i < 16; i++) {
                RWKV_MSG("%5.3f ", hist_cur[i] / (float) nelements);
                hist_all[i] += hist_cur[i];
            }

            RWKV_MSG("\n");
        } else {
            RWKV_MSG("size = %8.3f MB\n", orig_size / 1024.0 / 1024.0);
        }

        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_WRITE, rwkv_fwrite_tensor(out_file.file, tensor), "Failed to write tensor %s", name_str);
        orig_total_size += orig_size;
        new_total_size += new_size;
    }

    RWKV_MSG("original size     = %8.2f MB\n", orig_total_size / 1024.0 / 1024.0);
    RWKV_MSG("quantized size    = %8.2f MB\n", new_total_size / 1024.0 / 1024.0);
    RWKV_MSG("compression ratio = %8.2f\n", orig_total_size / float(new_total_size));

    int64_t sum_all = 0;

    for (int i = 0; i < 16; i++) {
        sum_all += hist_all[i];
    }

    RWKV_MSG("hist: ");

    for (int i = 0; i < 16; ++i) {
        printf("%5.3f ", hist_all[i] / float(sum_all));
    }

    RWKV_MSG("\n");

    return true;
}

const char * rwkv_get_system_info_string(void) {
    static std::string s;

    s  = "";
    s += "AVX="       + std::to_string(ggml_cpu_has_avx())       + " ";
    s += "AVX2="      + std::to_string(ggml_cpu_has_avx2())      + " ";
    s += "AVX512="    + std::to_string(ggml_cpu_has_avx512())    + " ";
    s += "FMA="       + std::to_string(ggml_cpu_has_fma())       + " ";
    s += "NEON="      + std::to_string(ggml_cpu_has_neon())      + " ";
    s += "ARM_FMA="   + std::to_string(ggml_cpu_has_arm_fma())   + " ";
    s += "F16C="      + std::to_string(ggml_cpu_has_f16c())      + " ";
    s += "FP16_VA="   + std::to_string(ggml_cpu_has_fp16_va())   + " ";
    s += "WASM_SIMD=" + std::to_string(ggml_cpu_has_wasm_simd()) + " ";
    s += "BLAS="      + std::to_string(ggml_cpu_has_blas())      + " ";
    s += "SSE3="      + std::to_string(ggml_cpu_has_sse3())      + " ";
    s += "VSX="       + std::to_string(ggml_cpu_has_vsx());

    return s.c_str();
}