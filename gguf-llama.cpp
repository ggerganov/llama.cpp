// Defines fileno on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#include "gguf-util.h"
#include "gguf-llama.h"

#include "ggml.h"
#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#include "ggml-opencl.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_MPI
#include "ggml-mpi.h"
#endif
#ifdef GGML_USE_K_QUANTS
#ifndef QK_K
#ifdef GGML_QKK_64
#define QK_K 64
#else
#define QK_K 256
#endif
#endif
#endif

#include <array>
#include <ctime>
#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cstring>
#include <climits>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <thread>
#include <mutex>
#include <sstream>
#include <numeric>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void llama_log_internal(llama_log_level level, const char* format, ...);
static void llama_log_callback_default(llama_log_level level, const char * text, void * user_data);
#define LLAMA_LOG_INFO(...)  llama_log_internal(LLAMA_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(LLAMA_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(LLAMA_LOG_LEVEL_ERROR, __VA_ARGS__)

template<typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

#if !defined(GGML_USE_CUBLAS) && !defined(GGML_USE_METAL)
#include "ggml-alloc.h"
#define LLAMA_USE_ALLOCATOR
#else
#define LLAMA_USE_SCRATCH
#define LLAMA_MAX_SCRATCH_BUFFERS 16
#endif

typedef void (*offload_func_t)(struct ggml_tensor * tensor);

#ifdef GGML_USE_CUBLAS
#define llama_host_malloc(n)  ggml_cuda_host_malloc(n)
#define llama_host_free(data) ggml_cuda_host_free(data)
#elif GGML_USE_METAL
#define llama_host_malloc(n)  ggml_metal_host_malloc(n)
#define llama_host_free(data) ggml_metal_host_free(data)
#else
#define llama_host_malloc(n)  malloc(n)
#define llama_host_free(data) free(data)
#endif

struct llama_buffer {
    void * data = NULL;
    size_t size = 0;

    // fallback to malloc / free
    // useful in cases where CUDA can try to allocate PINNED memory
    bool fallback = false;

    void resize(size_t n) {
        llama_host_free(data);

        data = llama_host_malloc(n);
        if (!data) {
            fallback = true;
            data = malloc(n);
        } else {
            fallback = false;
        }

        GGML_ASSERT(data);
        size = n;
    }

    ~llama_buffer() {
        if (data) {
            if (fallback) { // NOLINT
                free(data);
            } else {
                llama_host_free(data);
            }
        }

        data = NULL;
    }
};

void llama_nop(struct ggml_tensor * tensor) { // don't offload by default
    (void) tensor;
}

//
// ggml helpers
//

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

//
// memory sizes (calculated for n_batch == 512)
//

// computed for n_ctx == 2048
// TODO: dynamically determine these sizes
//       needs modifications in ggml

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_3B,
    MODEL_7B,
    MODEL_13B,
    MODEL_30B,
    MODEL_65B,
    MODEL_70B,
};

static const size_t kB = 1024;
static const size_t MB = 1024*1024;

static const std::map<e_model, size_t> & MEM_REQ_SCRATCH0(int n_ctx)
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,   ((size_t) n_ctx / 16ull +  92ull) * MB },
        { MODEL_7B,   ((size_t) n_ctx / 16ull + 100ull) * MB },
        { MODEL_13B,  ((size_t) n_ctx / 12ull + 120ull) * MB },
        { MODEL_30B,  ((size_t) n_ctx /  9ull + 160ull) * MB },
        { MODEL_65B,  ((size_t) n_ctx /  6ull + 256ull) * MB }, // guess
        { MODEL_70B,  ((size_t) n_ctx /  7ull + 164ull) * MB },
    };
    return k_sizes;
}

static const std::map<e_model, size_t> & MEM_REQ_SCRATCH1()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,  128ull * MB },
        { MODEL_7B,  160ull * MB },
        { MODEL_13B, 192ull * MB },
        { MODEL_30B, 256ull * MB },
        { MODEL_65B, 384ull * MB }, // guess
        { MODEL_70B, 304ull * MB },
    };
    return k_sizes;
}

// used to store the compute graph tensors + non-scratch data
static const std::map<e_model, size_t> & MEM_REQ_EVAL()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,   8ull * MB },
        { MODEL_7B,  10ull * MB },
        { MODEL_13B, 12ull * MB },
        { MODEL_30B, 16ull * MB },
        { MODEL_65B, 24ull * MB }, // guess
        { MODEL_70B, 24ull * MB },
    };
    return k_sizes;
}

// amount of VRAM needed per batch size to hold temporary results
// the values for 3b are not derived from testing but instead chosen conservatively
static const std::map<e_model, size_t> & VRAM_REQ_SCRATCH_BASE()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,   512ull * kB },
        { MODEL_7B,   512ull * kB },
        { MODEL_13B,  640ull * kB },
        { MODEL_30B,  768ull * kB },
        { MODEL_65B, 1280ull * kB },
        { MODEL_70B, 1280ull * kB },
    };
    return k_sizes;
}

// amount of VRAM needed per batch size and context to hold temporary results
// the values for 3b are not derived from testing but instead chosen conservatively
static const std::map<e_model, size_t> & VRAM_REQ_SCRATCH_PER_CONTEXT()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,  128ull },
        { MODEL_7B,  128ull },
        { MODEL_13B, 160ull },
        { MODEL_30B, 208ull },
        { MODEL_65B, 256ull },
        { MODEL_70B, 256ull },
    };
    return k_sizes;
}

// default hparams (LLaMA 7B)
struct llama_hparams {
    uint32_t n_vocab   = 32000;
    uint32_t n_ctx     = 512;
    uint32_t n_embd    = 4096;
    uint32_t n_head    = 32;
    uint32_t n_head_kv = 32;
    uint32_t n_layer   = 32;
    uint32_t n_rot     = 64;
    uint32_t n_ff      = 11008;

    float f_rms_norm_eps = 1e-5;

    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;

    enum llama_ftype ftype = LLAMA_FTYPE_MOSTLY_F16;

    bool operator!=(const llama_hparams & other) const {
        return static_cast<bool>(memcmp(this, &other, sizeof(llama_hparams))); // NOLINT
    }

    uint32_t n_gqa() const {
        return n_head/n_head_kv;
    }

    uint32_t n_embd_head() const {
        return n_embd/n_head;
    }

    uint32_t n_embd_gqa() const {
        return n_embd/n_gqa();
    }

    size_t kv_size() const {
        size_t result = 2ull;
        result *= (size_t) n_embd_gqa();
        result *= (size_t) n_ctx;
        result *= (size_t) n_layer;
        result *= sizeof(ggml_fp16_t);
        return result;
    }
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct llama_kv_cache {
    struct ggml_tensor * k = NULL;
    struct ggml_tensor * v = NULL;

    struct ggml_context * ctx = NULL;

    llama_buffer buf;

    int n; // number of tokens currently in the cache

    ~llama_kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        ggml_cuda_free_data(k);
        ggml_cuda_free_data(v);
#endif // GGML_USE_CUBLAS
    }
};

struct llama_vocab {
    // TODO: convert to this gguf_vocab
    // add a vector of merges
    // add members for bos/eos/pad/sep tokens
    // so that we can pass it to different types of tokenizers with a common interface

    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

struct llama_model {
    e_model type = MODEL_UNKNOWN;

    llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;
    int n_gpu_layers;

    // context
    struct ggml_context * ctx = NULL;

    // the model memory buffer
    llama_buffer buf;

    // model memory mapped file
    std::unique_ptr<gguf_mmap> mapping;

    // objects representing data potentially being locked in memory
    gguf_mlock mlock_buf;
    gguf_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    llama_vocab vocab;

    ~llama_model() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cuda_free_data(tensors_by_name[i].second);
        }
        ggml_cuda_free_scratch();
#elif defined(GGML_USE_CLBLAST)
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cl_free_data(tensors_by_name[i].second);
        }
#endif
    }
};

struct llama_context {
    llama_context(const llama_model & model) : model(model), t_load_us(model.t_load_us), t_start_us(model.t_start_us) {}
    ~llama_context() {
        if (model_owner) {
            delete &model;
        }
#ifdef GGML_USE_METAL
        if (ctx_metal) {
            ggml_metal_free(ctx_metal);
        }
#endif
#ifdef LLAMA_USE_ALLOCATOR
        if (alloc) {
            ggml_allocr_free(alloc);
        }
#endif
    }

    std::mt19937 rng;

    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    const llama_model & model;

    bool model_owner = false;

    int64_t t_load_us;
    int64_t t_start_us;

    // key + value cache for the self attention
    struct llama_kv_cache kv_self;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // memory buffers used to evaluate the model
    // TODO: move in llama_state
    llama_buffer buf_compute;

#ifdef LLAMA_USE_ALLOCATOR
    llama_buffer buf_alloc;
    ggml_allocr * alloc = NULL;
#endif

#ifdef LLAMA_USE_SCRATCH
    llama_buffer buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = { 0 };
#endif

#ifdef GGML_USE_METAL
    ggml_metal_context * ctx_metal = NULL;
#endif

#ifdef GGML_USE_MPI
    ggml_mpi_context * ctx_mpi = NULL;
#endif

    static void use_buf(struct ggml_context * ctx, int i) {
#if defined(LLAMA_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size, buf.data, });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    static size_t get_buf_max_mem(int i) {
#if defined(LLAMA_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};

struct llama_state {
    // We save the log callback globally
    llama_log_callback log_callback = llama_log_callback_default;
    void * log_callback_user_data = nullptr;
};
// global state
static llama_state g_state;

template <typename T>
static T checked_mul(T a, T b) {
    T ret = a * b;
    if (a != 0 && ret / a != b) {
        throw std::runtime_error(format("overflow multiplying %llu * %llu",
                     (unsigned long long) a, (unsigned long long) b));
    }
    return ret;
}

static size_t checked_div(size_t a, size_t b) {
    if (b == 0 || a % b != 0) {
        throw std::runtime_error(format("error dividing %zu / %zu", a, b));
    }
    return a / b;
}

static std::string llama_format_tensor_shape(const std::vector<uint32_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5u", ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " x %5u", ne.at(i));
    }
    return buf;
}

static size_t llama_calc_tensor_size(const std::vector<uint32_t> & ne, enum ggml_type type) {
    size_t size = ggml_type_size(type);
    for (uint32_t dim : ne) {
        size = checked_mul<size_t>(size, dim);
    }
    return size / ggml_blck_size(type);
}

struct gguf_load_tensor {
    std::string name;
    enum ggml_type type = GGML_TYPE_F32;
    std::vector<uint32_t> ne;
    size_t file_off;
    size_t size;
    struct ggml_tensor * ggml_tensor = NULL;
    uint8_t * data;
};

struct gguf_load_tensors_map {
    // tensors is kept in a separate vector to preserve file order
    std::vector<gguf_load_tensor> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;
};

enum gguf_file_version {
    GGUF_FILE_VERSION_V1 = 1,
};

struct gguf_file_loader {
    gguf_file file;
    gguf_context * gguf_ctx;
    gguf_file_version file_version;
    llama_hparams hparams;
    llama_vocab vocab;

    struct ggml_context * ctx_data = NULL;

    gguf_file_loader(const char * fname, gguf_load_tensors_map & tensors_map)
        : file(fname, "rb") {
        fprintf(stderr, "llama.cpp: loading model from %s\n", fname);

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };

    gguf_ctx = gguf_init_from_file(fname, params);
    file_version = (enum gguf_file_version) gguf_get_version(gguf_ctx);

    read_hparams();
    read_vocab();
        read_tensor_metadata(tensors_map);
    }

    uint32_t read_u32(const char * key) const {
        int i = gguf_find_key(gguf_ctx, key);
        if (i == -1) {
            throw std::runtime_error(format("cannot find param with key %s\n", key));
        }

        return gguf_get_val_u32(gguf_ctx, i);
    }

    float read_f32(const char * key) const {
        int i = gguf_find_key(gguf_ctx, key);
        if (i == -1) {
            throw std::runtime_error(format("cannot find param with key %s\n", key));
        }

        return gguf_get_val_f32(gguf_ctx, i);
    }

    int read_n_vocab() const {
        int i = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
        if (i == -1) {
            throw std::runtime_error("cannot find token list in GGUF file\n");
        }

        return gguf_get_arr_n(gguf_ctx, i);
    }

    void read_hparams() {
        // TODO define keys as constants in header
        // TODO: read all hparams from file

        hparams.n_vocab        = read_n_vocab();
        hparams.n_ctx          = read_u32("llama.context_length");
        hparams.n_embd         = read_u32("llama.embedding_length");
        hparams.n_ff           = read_u32("llama.feed_forward_length");
        hparams.n_head         = read_u32("llama.attention.head_count");
        hparams.n_layer        = read_u32("llama.layer_count");
        hparams.n_rot          = read_u32("llama.rope.dimension_count");
        hparams.f_rms_norm_eps = read_f32("llama.attention.layer_norm_rms_epsilon");

        // LLaMAv2
        // hparams.n_head_kv = read_u32("llama.attention.head_count_kv");
    }

    void read_vocab() {
        vocab.id_to_token.resize(hparams.n_vocab);
        int token_idx = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
        if (token_idx == -1) {
            throw std::runtime_error("cannot find token list in GGUF file\n");
        }

        int score_idx = gguf_find_key(gguf_ctx, "tokenizer.ggml.scores");
        if (score_idx == -1) {
            throw std::runtime_error("cannot find token scores list in GGUF file\n");
        }

        for (uint32_t i = 0; i < hparams.n_vocab; i++) {

            std::string word = gguf_get_arr_str(gguf_ctx, token_idx, i);

            vocab.token_to_id[word] = i;

            auto & tok_score = vocab.id_to_token[i];
            tok_score.tok = std::move(word);
            tok_score.score = gguf_get_arr_f32(gguf_ctx, score_idx, i);
        }
    }

    void read_tensor_metadata(gguf_load_tensors_map & tensors_map) const {
        const int n_tensors = gguf_get_n_tensors(gguf_ctx);

        for (int i = 0; i < n_tensors; ++i) {
            gguf_load_tensor tensor;
            const char * name = gguf_get_tensor_name(gguf_ctx, i);

            struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);

            const uint32_t n_dims = cur->n_dims;
            tensor.type = cur->type;
            tensor.ne.resize(n_dims);

            for (uint32_t j = 0; j < n_dims; ++j) {
                tensor.ne[j] = cur->ne[j];
            }

            if (n_dims < 1 || n_dims > 2) {
                throw std::runtime_error(format("llama.cpp: tensor '%s' should not be %u-dimensional", name, n_dims));
            }

            switch (tensor.type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    break;
                default: {
                    throw std::runtime_error(format("unrecognized tensor type %u\n", tensor.type));
                }
            }

            tensor.file_off = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);

            tensor.name = name;
            tensor.size = llama_calc_tensor_size(tensor.ne, tensor.type);

            tensors_map.tensors.push_back(tensor);
            tensors_map.name_to_idx[name] = tensors_map.tensors.size() - 1;
        }
    }
};

struct gguf_file_saver {
    // TODO
    // this implementation now assumes that the data section is of the same length as the unquantized model.
    // this is needed to write tensor metadata and weights in a single pass by seeking to appropriate positions in the file.
    // this may not be true when we add quantization version and change ftype description (currently it's string according to the specs,
    // but better to have it as uint32).
    // we need to calculate the delta in number of bytes written with a counter as a struct member.

    gguf_file file;
    gguf_file_loader * fl;
    size_t info_offset;
    size_t tensor_offset = 0;

    gguf_file_saver(const char * fname, gguf_file_loader * fl, enum llama_ftype new_ftype)
        : file(fname, "wb"), fl(fl) {
            fprintf(stderr, "llama.cpp: saving model to %s\n", fname);
            write_header();
            write_hparams(new_ftype);
        }

    void write_header() {
        const int32_t magic = GGUF_MAGIC;
        file.write_i32(magic);

        const int32_t version = GGUF_VERSION;
        file.write_i32(version);

        const int32_t n_tensors = gguf_get_n_tensors(fl->gguf_ctx);
        file.write_i32(n_tensors);

        const int32_t n_kv = gguf_get_n_kv(fl->gguf_ctx);
        file.write_i32(n_kv);
    }

    void write_hparam_arr_str(const std::string & key, enum gguf_type type, int i, int n_arr) {
        std::vector<std::string> data(n_arr);

        for (int j = 0; j < n_arr; ++j) {
            std::string val = gguf_get_arr_str(fl->gguf_ctx, i, j);
            data[j] = val;
        }

        file.write_arr<std::string>(key, type, data);
    }

    void write_hparam_arr_f32(const std::string & key, enum gguf_type type, int i, int n_arr) {
        std::vector<float> data(n_arr);

        for (int j = 0; j < n_arr; ++j) {
            float val = gguf_get_arr_f32(fl->gguf_ctx, i, j);
            data[j] = val;
        }

        file.write_arr<float>(key, type, data);
    }

    void write_hparams(enum llama_ftype new_ftype) {
        const int32_t n_kv = gguf_get_n_kv(fl->gguf_ctx);
        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(fl->gguf_ctx, i);
            if (strcmp(key, "general.quantization_version") == 0) {
                file.write_val<uint32_t>("general.quantization_version", GGUF_TYPE_UINT32, new_ftype);
            } else {
                const gguf_type vtype = gguf_get_kv_type(fl->gguf_ctx, i);

                bool bool_val;
                float f32_val;
                int16_t i16_val;
                int32_t i32_val;
                int8_t i8_val;
                std::string str_val;
                uint16_t u16_val;
                uint32_t u32_val;
                uint8_t u8_val;
                gguf_type arr_type;
                int n_arr;

                switch(vtype) {
                    case GGUF_TYPE_BOOL:
                        bool_val = gguf_get_val_bool(fl->gguf_ctx, i);
                        file.write_val<bool>(key, GGUF_TYPE_BOOL, bool_val);
                        break;
                    case GGUF_TYPE_FLOAT32:
                        f32_val = gguf_get_val_f32(fl->gguf_ctx, i);
                        file.write_val<float>(key, GGUF_TYPE_FLOAT32, f32_val);
                        break;
                    case GGUF_TYPE_INT16:
                        i16_val = gguf_get_val_i16(fl->gguf_ctx, i);
                        file.write_val<int16_t>(key, GGUF_TYPE_INT16, i16_val);
                        break;
                    case GGUF_TYPE_INT32:
                        i32_val = gguf_get_val_i32(fl->gguf_ctx, i);
                        file.write_val<int32_t>(key, GGUF_TYPE_INT32, i32_val);
                        break;
                    case GGUF_TYPE_INT8:
                        i8_val = gguf_get_val_i8(fl->gguf_ctx, i);
                        file.write_val<int8_t>(key, GGUF_TYPE_INT8, i8_val);
                        break;
                    case GGUF_TYPE_STRING:
                        str_val = gguf_get_val_str(fl->gguf_ctx, i);
                        file.write_val<std::string>(key, GGUF_TYPE_STRING, str_val);
                        break;
                    case GGUF_TYPE_UINT16:
                        u16_val = gguf_get_val_u16(fl->gguf_ctx, i);
                        file.write_val<uint16_t>(key, GGUF_TYPE_UINT16, u16_val);
                        break;
                    case GGUF_TYPE_UINT32:
                        u32_val = gguf_get_val_u32(fl->gguf_ctx, i);
                        file.write_val<uint32_t>(key, GGUF_TYPE_UINT32, u32_val);
                        break;
                    case GGUF_TYPE_UINT8:
                        u8_val = gguf_get_val_u8(fl->gguf_ctx, i);
                        file.write_val<uint8_t>(key, GGUF_TYPE_UINT8, u8_val);
                        break;
                    case GGUF_TYPE_ARRAY:
                        arr_type = gguf_get_arr_type(fl->gguf_ctx, i);
                        n_arr    = gguf_get_arr_n(fl->gguf_ctx, i);
                        if (arr_type == GGUF_TYPE_FLOAT32) {
                            write_hparam_arr_f32(key, arr_type, i, n_arr);
                        } else if (arr_type == GGUF_TYPE_STRING) {
                            write_hparam_arr_str(key, GGUF_TYPE_STRING, i, n_arr);
                        } else {
                            throw std::runtime_error("not implemented");
                        }
                        break;
                    default:
                        throw std::runtime_error(format("cannot recognize value type for key %s\n", key));
                }
            }
        }

        info_offset = file.tell();

        GGML_ASSERT(gguf_get_data_offset(fl->gguf_ctx) >= info_offset);

        size_t count = gguf_get_data_offset(fl->gguf_ctx) - info_offset;
        file.write_zeros(count);
        file.seek(info_offset, SEEK_SET);
        GGML_ASSERT(info_offset == file.tell());
    }

    size_t write_tensor_info(gguf_load_tensor & tensor, enum ggml_type type) {
        size_t total_written = 0;
        file.seek(info_offset, SEEK_SET);
        GGML_ASSERT(info_offset == file.tell());
        total_written += file.write_str(tensor.name);

        int32_t n_dims = tensor.ne.size();
        total_written += file.write_i32(n_dims);
        for (int32_t i = 0; i < n_dims; ++i) {
            total_written += file.write_i32(tensor.ne[i]);
        }

        total_written += file.write_i32(type);
        total_written += file.write_u64(tensor_offset);
        info_offset   += total_written; // position to write info of the next tensor

        file.seek(0, SEEK_END);

        return total_written;
    }

    void write_tensor(gguf_load_tensor & tensor, enum ggml_type new_type, const void * new_data, size_t new_size) {
        switch (new_type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                break;
            default: GGML_ASSERT(false);
        }

        write_tensor_info(tensor, new_type);
        file.write_raw(new_data, new_size);
        size_t padded_size = GGML_PAD(new_size, GGUF_DEFAULT_ALIGNMENT); // TODO: handle custom alignment
        size_t pad = padded_size - new_size;
        file.write_zeros(pad);
        tensor_offset += padded_size; // offset of the next tensor
    }
};

struct llama_model_loader {
    std::unique_ptr<gguf_file_loader> file_loader;
    gguf_load_tensors_map tensors_map;
    bool use_mmap;
    size_t num_ggml_tensors_created = 0;
    struct ggml_context * ggml_ctx = NULL;
    std::unique_ptr<gguf_mmap> mapping;

    llama_model_loader(const std::string & fname_base, bool use_mmap) {
        file_loader = std::unique_ptr<gguf_file_loader>(new gguf_file_loader(fname_base.c_str(), tensors_map));
        if (!gguf_mmap::SUPPORTED) {
            use_mmap = false;
        }
        this->use_mmap = use_mmap;
    }

    void calc_sizes(size_t * ctx_size_p, size_t * mmapped_size_p) const {
        *ctx_size_p = *mmapped_size_p = 0;
        for (const gguf_load_tensor & lt : tensors_map.tensors) {
            *ctx_size_p += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            *(use_mmap ? mmapped_size_p : ctx_size_p) += lt.size + 16;
        }
    }

    struct ggml_tensor * get_tensor(const std::string & name, const std::vector<uint32_t> & ne, ggml_backend backend) {
        auto it = tensors_map.name_to_idx.find(name);
        if (it == tensors_map.name_to_idx.end()) {
            throw std::runtime_error(std::runtime_error(format("llama.cpp: tensor '%s' is missing from model", name.c_str())));
        }
        gguf_load_tensor & lt = tensors_map.tensors.at(it->second);
        if (lt.ne != ne) {
            throw std::runtime_error(format("llama.cpp: tensor '%s' has wrong shape; expected %s, got %s",
                         name.c_str(), llama_format_tensor_shape(ne).c_str(), llama_format_tensor_shape(lt.ne).c_str()));
        }

        return get_tensor_for(lt, backend);
    }

    struct ggml_tensor * get_tensor_for(gguf_load_tensor & lt, ggml_backend backend) {
        struct ggml_tensor * tensor;
        if (backend != GGML_BACKEND_CPU) {
            ggml_set_no_alloc(ggml_ctx, true);
        }
        if (lt.ne.size() == 2) {
            tensor = ggml_new_tensor_2d(ggml_ctx, lt.type, lt.ne.at(0), lt.ne.at(1));
        } else {
            GGML_ASSERT(lt.ne.size() == 1);
            tensor = ggml_new_tensor_1d(ggml_ctx, lt.type, lt.ne.at(0));
        }
        ggml_set_name(tensor, lt.name.c_str());
        GGML_ASSERT(lt.ggml_tensor == NULL); // if this fails, we called get_tensor twice on the same tensor

        if (backend != GGML_BACKEND_CPU) {
            ggml_set_no_alloc(ggml_ctx, use_mmap);
        }
        tensor->backend = backend;
        lt.ggml_tensor = tensor;
        num_ggml_tensors_created++;
        return tensor;
    }

    void done_getting_tensors() const {
        if (num_ggml_tensors_created != tensors_map.tensors.size()) {
            throw std::runtime_error(std::string("llama.cpp: file contained more tensors than expected"));
        }
    }

    void load_all_data(llama_progress_callback progress_callback, void *  progress_callback_user_data, gguf_mlock * lmlock) {
        size_t data_size = 0;
        size_t prefetch_size = 0;
        size_t lock_size = 0;
        for (const gguf_load_tensor & lt : tensors_map.tensors) {
            data_size += lt.size;
            if (lt.ggml_tensor->backend == GGML_BACKEND_CPU) {
                prefetch_size += lt.size;
            }
        }

        if (use_mmap) {
            mapping.reset(new gguf_mmap(&file_loader->file, prefetch_size, ggml_is_numa()));
            if (lmlock) {
                lmlock->init(mapping->addr);
            }
        }

        size_t done_size = 0;
        for (gguf_load_tensor & lt : tensors_map.tensors) {
            if (progress_callback) {
                progress_callback((float) done_size / data_size, progress_callback_user_data);
            }
            GGML_ASSERT(lt.ggml_tensor); // unused tensors should have been caught by load_data already
            lt.data = (uint8_t *) lt.ggml_tensor->data;

            // allocate temp buffer if not using mmap
            if (!use_mmap && lt.data == NULL) {
                GGML_ASSERT(lt.ggml_tensor->backend != GGML_BACKEND_CPU);
                lt.data = (uint8_t*)malloc(ggml_nbytes(lt.ggml_tensor));
            }

            load_data_for(lt);

            switch(lt.ggml_tensor->backend) {
                case GGML_BACKEND_CPU:
                    lt.ggml_tensor->data = lt.data;
                    if (use_mmap && lmlock) {
                        lock_size += lt.size;
                        lmlock->grow_to(lock_size);
                    }
                    break;
#if defined(GGML_USE_CUBLAS)
                case GGML_BACKEND_GPU:
                case GGML_BACKEND_GPU_SPLIT:
                    ggml_cuda_transform_tensor(lt.data, lt.ggml_tensor);
                    if (!use_mmap) {
                        free(lt.data);
                    }
                    break;
#elif defined(GGML_USE_CLBLAST)
                case GGML_BACKEND_GPU:
                    ggml_cl_transform_tensor(lt.data, lt.ggml_tensor);
                    if (!use_mmap) {
                        free(lt.data);
                    }
                    break;
#endif
                default:
                    continue;
            }

            done_size += lt.size;
        }
    }

    void load_data_for(gguf_load_tensor & lt) {
        if (use_mmap) {
            lt.data = (uint8_t *) mapping->addr + lt.file_off;
        } else {
            gguf_file & file = file_loader->file;
            file.seek(lt.file_off, SEEK_SET);
            file.read_raw(lt.data, lt.size);
        }

        if (0) {
            print_checksum(lt);
        }
    }

    static void print_checksum(gguf_load_tensor & lt) {
        uint32_t sum = 0;
        for (size_t i = 0; i < lt.size; i++) {
            uint8_t byte = lt.data[i];
            sum = byte + (sum << 6) + (sum << 16) - sum; // sdbm hash
        }
        fprintf(stderr, "%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
                llama_format_tensor_shape(lt.ne).c_str(), lt.size);
    }

};

//
// kv cache
//

static bool kv_cache_init(
        const struct llama_hparams & hparams,
             struct llama_kv_cache & cache,
                         ggml_type   wtype,
                               int   n_ctx,
                               int   n_gpu_layers) {
    const int n_embd  = hparams.n_embd_gqa();
    const int n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx;
    const int64_t n_elements = n_embd*n_mem;

    cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);
    cache.n = 0;

    struct ggml_init_params params;
    params.mem_size   = cache.buf.size;
    params.mem_buffer = cache.buf.data;
    params.no_alloc   = false;

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    ggml_set_name(cache.k, "cache_k");
    ggml_set_name(cache.v, "cache_v");

    (void) n_gpu_layers;
#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer + 1) {
        ggml_cuda_assign_buffers_no_scratch(cache.v);
    }
    if (n_gpu_layers > n_layer + 2) {
        ggml_cuda_assign_buffers_no_scratch(cache.k);
    }
#endif // GGML_USE_CUBLAS

    return true;
}

struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        /*.seed                        =*/ LLAMA_DEFAULT_SEED,
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 512,
        /*.gpu_layers                  =*/ 0,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.rope_freq_base              =*/ 10000.0f,
        /*.rope_freq_scale             =*/ 1.0f,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.low_vram                    =*/ false,
        /*.mul_mat_q                   =*/ false,
        /*.f16_kv                      =*/ true,
        /*.logits_all                  =*/ false,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.embedding                   =*/ false,
    };

    return result;
}

struct llama_model_quantize_params llama_model_quantize_default_params() {
    struct llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q5_1,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
    };

    return result;
}

int llama_max_devices() {
    return LLAMA_MAX_DEVICES;
}

bool llama_mmap_supported() {
    return gguf_mmap::SUPPORTED;
}

bool llama_mlock_supported() {
    return gguf_mlock::SUPPORTED;
}

void llama_backend_init(bool numa) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if (numa) {
        ggml_numa_init();
    }

#ifdef GGML_USE_MPI
    ggml_mpi_backend_init();
#endif
}

void llama_backend_free() {
#ifdef GGML_USE_MPI
    ggml_mpi_backend_free();
#endif
}

int64_t llama_time_us() {
    return ggml_time_us();
}

//
// model loading
//

static const char *gguf_file_version_name(gguf_file_version version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1: return "GGUF V1 (latest)";
        }

    return "unknown";
}

static const char *llama_ftype_name(enum llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:     return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:  return "mostly F16";
        case LLAMA_FTYPE_MOSTLY_Q4_0: return "mostly Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1: return "mostly Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16:
                                      return "mostly Q4_1, some F16";
        case LLAMA_FTYPE_MOSTLY_Q5_0: return "mostly Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1: return "mostly Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0: return "mostly Q8_0";
        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K: return "mostly Q2_K";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S: return "mostly Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M: return "mostly Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L: return "mostly Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S: return "mostly Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: return "mostly Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S: return "mostly Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M: return "mostly Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K: return "mostly Q6_K";
        default:                      return "unknown, may not work";
    }
}

static const char *llama_model_type_name(e_model type) {
    switch (type) {
        case MODEL_3B: return "3B";
        case MODEL_7B: return "7B";
        case MODEL_13B: return "13B";
        case MODEL_30B: return "30B";
        case MODEL_65B: return "65B";
        case MODEL_70B: return "70B";
        default: GGML_ASSERT(false);
    }
}

static void llama_model_load_internal(
        const std::string & fname,
        llama_model & model,
        llama_vocab & vocab,
        int n_ctx,
        int n_batch,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        const bool mul_mat_q,
        float rope_freq_base,
        float rope_freq_scale,
        bool low_vram,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    model.t_start_us = ggml_time_us();

    std::unique_ptr<llama_model_loader> ml(new llama_model_loader(fname, use_mmap));

    vocab = std::move(ml->file_loader->vocab);
    model.hparams = ml->file_loader->hparams;
    model.n_gpu_layers = n_gpu_layers;
    gguf_file_version file_version = ml->file_loader->file_version;

    auto & hparams = model.hparams;

    {
        switch (hparams.n_layer) {
            case 26: model.type = e_model::MODEL_3B; break;
            case 32: model.type = e_model::MODEL_7B; break;
            case 40: model.type = e_model::MODEL_13B; break;
            case 60: model.type = e_model::MODEL_30B; break;
            case 80: model.type = e_model::MODEL_65B; break;
            default:
                {
                    if (hparams.n_layer < 32) {
                        model.type = e_model::MODEL_7B;
                    }
                } break;
        }

        hparams.n_ctx = n_ctx;

        // LLaMAv2
        // TODO: probably not needed
        {
            const auto n_gqa = hparams.n_gqa();

            if (model.type == e_model::MODEL_65B && n_gqa == 8) {
                fprintf(stderr, "%s: warning: assuming 70B model based on GQA == %d\n", __func__, n_gqa);
                model.type = e_model::MODEL_70B;
            }
        }

        hparams.rope_freq_base  = rope_freq_base;
        hparams.rope_freq_scale = rope_freq_scale;
    }

    const uint32_t n_ff = hparams.n_ff;

    {
        LLAMA_LOG_INFO("%s: format     = %s\n",   __func__, gguf_file_version_name(file_version));
        LLAMA_LOG_INFO("%s: n_vocab    = %u\n",   __func__, hparams.n_vocab);
        LLAMA_LOG_INFO("%s: n_ctx      = %u\n",   __func__, hparams.n_ctx);
        LLAMA_LOG_INFO("%s: n_embd     = %u\n",   __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_head     = %u\n",   __func__, hparams.n_head);
        LLAMA_LOG_INFO("%s: n_head_kv  = %u\n",   __func__, hparams.n_head_kv);
        LLAMA_LOG_INFO("%s: n_layer    = %u\n",   __func__, hparams.n_layer);
        LLAMA_LOG_INFO("%s: n_rot      = %u\n",   __func__, hparams.n_rot); // a.k.a. n_embd_head, n_head_dim
        LLAMA_LOG_INFO("%s: n_gqa      = %u\n",   __func__, hparams.n_gqa());
        LLAMA_LOG_INFO("%s: rnorm_eps  = %.1e\n", __func__, hparams.f_rms_norm_eps);
        LLAMA_LOG_INFO("%s: n_ff       = %u\n",   __func__, n_ff);
        LLAMA_LOG_INFO("%s: freq_base  = %.1f\n", __func__, hparams.rope_freq_base);
        LLAMA_LOG_INFO("%s: freq_scale = %g\n",   __func__, hparams.rope_freq_scale);
        LLAMA_LOG_INFO("%s: ftype      = %u (%s)\n", __func__, hparams.ftype, llama_ftype_name(hparams.ftype));
        LLAMA_LOG_INFO("%s: model size = %s\n",   __func__, llama_model_type_name(model.type));
    }

    if (vocab_only) {
        return;
    }

    auto & ctx = model.ctx;

    size_t ctx_size;
    size_t mmapped_size;
    ml->calc_sizes(&ctx_size, &mmapped_size);
    LLAMA_LOG_INFO("%s: ggml ctx size = %7.2f MB\n", __func__, ctx_size/1024.0/1024.0);

    // create the ggml context
    {
        model.buf.resize(ctx_size);
        if (use_mlock) {
            model.mlock_buf.init   (model.buf.data);
            model.mlock_buf.grow_to(model.buf.size);
        }

        struct ggml_init_params params = {
            /*.mem_size   =*/ model.buf.size,
            /*.mem_buffer =*/ model.buf.data,
            /*.no_alloc   =*/ ml->use_mmap,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            throw std::runtime_error(format("ggml_init() failed"));
        }
    }

    (void) main_gpu;
    (void) mul_mat_q;
#if defined(GGML_USE_CUBLAS)
    LLAMA_LOG_INFO("%s: using CUDA for GPU acceleration\n", __func__);
    ggml_cuda_set_main_device(main_gpu);
    ggml_cuda_set_mul_mat_q(mul_mat_q);
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU_SPLIT
#elif defined(GGML_USE_CLBLAST)
    LLAMA_LOG_INFO("%s: using OpenCL for GPU acceleration\n", __func__);
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU
#else
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_CPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_CPU
#endif

    // prepare memory for the weights
    size_t vram_weights = 0;
    size_t vram_scratch = 0;
    {
        const uint32_t n_embd     = hparams.n_embd;
        const uint32_t n_embd_gqa = hparams.n_embd_gqa();
        const uint32_t n_layer    = hparams.n_layer;
        const uint32_t n_vocab    = hparams.n_vocab;

        ml->ggml_ctx = ctx;

        model.tok_embeddings = ml->get_tensor("tok_embeddings.weight", {n_embd, n_vocab}, GGML_BACKEND_CPU);

        // "output" tensor
        {
            ggml_backend backend_norm;
            ggml_backend backend_output;
            if (n_gpu_layers > int(n_layer)) { // NOLINT
                // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                backend_norm = low_vram ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#else
                backend_norm = low_vram || n_gpu_layers <= (int) n_layer + 2 ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#endif // _WIN32

                backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
            } else {
                backend_norm = GGML_BACKEND_CPU;
                backend_output = GGML_BACKEND_CPU;
            }

            model.norm   = ml->get_tensor("norm.weight",   {n_embd},          backend_norm);
            model.output = ml->get_tensor("output.weight", {n_embd, n_vocab}, backend_output);
            if (backend_norm == GGML_BACKEND_GPU) {
                vram_weights += ggml_nbytes(model.norm);
            }
            if (backend_output == GGML_BACKEND_GPU_SPLIT) {
                vram_weights += ggml_nbytes(model.output);
            }
        }

        const int i_gpu_start = n_layer - n_gpu_layers;

        model.layers.resize(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            const ggml_backend backend = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD; // NOLINT
            const ggml_backend backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT; // NOLINT

            auto & layer = model.layers[i];

            std::string layers_i = "layers." + std::to_string(i);

            layer.attention_norm = ml->get_tensor(layers_i + ".attention_norm.weight", {n_embd}, backend);

            layer.wq = ml->get_tensor(layers_i + ".attention.wq.weight", {n_embd, n_embd},     backend_split);
            layer.wk = ml->get_tensor(layers_i + ".attention.wk.weight", {n_embd, n_embd_gqa}, backend_split);
            layer.wv = ml->get_tensor(layers_i + ".attention.wv.weight", {n_embd, n_embd_gqa}, backend_split);
            layer.wo = ml->get_tensor(layers_i + ".attention.wo.weight", {n_embd, n_embd},     backend_split);

            layer.ffn_norm = ml->get_tensor(layers_i + ".ffn_norm.weight", {n_embd}, backend);

            layer.w1 = ml->get_tensor(layers_i + ".feed_forward.w1.weight", {n_embd,   n_ff}, backend_split);
            layer.w2 = ml->get_tensor(layers_i + ".feed_forward.w2.weight", {  n_ff, n_embd}, backend_split);
            layer.w3 = ml->get_tensor(layers_i + ".feed_forward.w3.weight", {n_embd,   n_ff}, backend_split);

            if (backend == GGML_BACKEND_GPU) {
                vram_weights +=
                    ggml_nbytes(layer.attention_norm) + ggml_nbytes(layer.wq) + ggml_nbytes(layer.wk)             +
                    ggml_nbytes(layer.wv)             + ggml_nbytes(layer.wo) + ggml_nbytes(layer.ffn_norm) +
                    ggml_nbytes(layer.w1)             + ggml_nbytes(layer.w2) + ggml_nbytes(layer.w3);
            }
        }
    }

    ml->done_getting_tensors();

    // print memory requirements
    {
        const size_t scale = memory_type == GGML_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        size_t mem_required =
            ctx_size +
            mmapped_size - vram_weights; // weights in VRAM not in memory

#ifndef LLAMA_USE_ALLOCATOR
        mem_required +=
            MEM_REQ_SCRATCH0(hparams.n_ctx).at(model.type) +
            MEM_REQ_SCRATCH1().at(model.type) +
            MEM_REQ_EVAL().at(model.type);
#endif

        // this is the memory required by one llama_state
        const size_t mem_required_state =
            scale*hparams.kv_size();

        LLAMA_LOG_INFO("%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);

        (void) vram_scratch;
        (void) n_batch;
#ifdef GGML_USE_CUBLAS
        if (low_vram) {
            LLAMA_LOG_INFO("%s: not allocating a VRAM scratch buffer due to low VRAM option\n", __func__);
            ggml_cuda_set_scratch_size(0); // disable scratch
        } else {
            const size_t vram_scratch_base = VRAM_REQ_SCRATCH_BASE().at(model.type);
            const size_t vram_scratch_per_context = VRAM_REQ_SCRATCH_PER_CONTEXT().at(model.type);
            vram_scratch = n_batch * (vram_scratch_base + n_ctx * vram_scratch_per_context);
            ggml_cuda_set_scratch_size(vram_scratch);
            if (n_gpu_layers > 0) {
                LLAMA_LOG_INFO("%s: allocating batch_size x (%zd kB + n_ctx x %zd B) = %zd MB VRAM for the scratch buffer\n",
                        __func__, vram_scratch_base / kB, vram_scratch_per_context,
                        (vram_scratch + MB - 1) / MB); // round up
            }
        }
#endif // GGML_USE_CUBLAS

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading non-repeating layers to GPU\n", __func__);
        }
        size_t vram_kv_cache = 0;

#ifdef GGML_USE_CUBLAS
        const int max_backend_supported_layers = hparams.n_layer + 3;
        const int max_offloadable_layers = low_vram ? hparams.n_layer + 1 : hparams.n_layer + 3;
        if (n_gpu_layers > (int) hparams.n_layer + 1) {
            if (low_vram) {
                LLAMA_LOG_INFO("%s: cannot offload v cache to GPU due to low VRAM option\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: offloading v cache to GPU\n", __func__);
                vram_kv_cache += hparams.kv_size() / 2;
            }
        }
        if (n_gpu_layers > (int) hparams.n_layer + 2) {
            if (low_vram) {
                LLAMA_LOG_WARN("%s: cannot offload k cache to GPU due to low VRAM option\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: offloading k cache to GPU\n", __func__);
                vram_kv_cache += hparams.kv_size() / 2;
            }
        }
#elif defined(GGML_USE_CLBLAST)
        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers = hparams.n_layer + 1;
#endif // GGML_USE_CUBLAS

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n",
                __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
        LLAMA_LOG_INFO("%s: total VRAM used: %zu MB\n",
                __func__, (vram_weights + vram_scratch + vram_kv_cache + MB - 1) / MB); // round up
#else
        (void) n_gpu_layers;
#endif // defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
    }

    // populate `tensors_by_name`
    for (gguf_load_tensor & lt : ml->tensors_map.tensors) {
        model.tensors_by_name.emplace_back(lt.name, lt.ggml_tensor);
    }

    (void) tensor_split;
#if defined(GGML_USE_CUBLAS)
    {
        ggml_cuda_set_tensor_split(tensor_split);
    }
#endif

    ml->load_all_data(progress_callback, progress_callback_user_data, use_mlock ? &model.mlock_mmap : NULL);

    if (progress_callback) {
        progress_callback(1.0f, progress_callback_user_data);
    }

    model.mapping = std::move(ml->mapping);

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = ggml_time_us() - model.t_start_us;
}

static bool llama_model_load(
        const std::string & fname,
        llama_model & model,
        llama_vocab & vocab,
        int n_ctx,
        int n_batch,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        const bool mul_mat_q,
        float rope_freq_base,
        float rope_freq_scale,
        bool low_vram,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_progress_callback progress_callback,
        void *progress_callback_user_data) {
    try {
        llama_model_load_internal(fname, model, vocab, n_ctx, n_batch, n_gpu_layers,
                                  main_gpu, tensor_split, mul_mat_q, rope_freq_base, rope_freq_scale, low_vram, memory_type,
                                  use_mmap, use_mlock, vocab_only, progress_callback, progress_callback_user_data);
        return true;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("error loading model: %s\n", err.what());
        return false;
    }
}

static struct ggml_cgraph * llama_build_graph(
         llama_context & lctx,
     const llama_token * tokens,
           const float * embd,
                   int   n_tokens,
                   int   n_past) {

    GGML_ASSERT((!tokens && embd) || (tokens && !embd));

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = hparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();


    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const float freq_base  = hparams.rope_freq_base;
    const float freq_scale = hparams.rope_freq_scale;
    const float rms_norm_eps = hparams.f_rms_norm_eps;

    const int n_gpu_layers = model.n_gpu_layers;

    auto & mem_per_token = lctx.mem_per_token;
    auto & buf_compute   = lctx.buf_compute;


    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ false,
    };

#ifdef LLAMA_USE_ALLOCATOR
    params.no_alloc = true;
#endif

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    if (tokens) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);

#ifdef LLAMA_USE_ALLOCATOR
        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, tokens, N*ggml_element_size(inp_tokens));
        }
#else
        memcpy(inp_tokens->data, tokens, N*ggml_element_size(inp_tokens));
#endif
        ggml_set_name(inp_tokens, "inp_tokens");

        inpL = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N);

#ifdef LLAMA_USE_ALLOCATOR
        ggml_allocr_alloc(lctx.alloc, inpL);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inpL->data, embd, N * n_embd * ggml_element_size(inpL));
        }
#else
        memcpy(inpL->data, embd, N * n_embd * ggml_element_size(inpL));
#endif
    }

    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;

    // offload functions set the tensor output backend to GPU
    // tensors are GPU-accelerated if any input or the output has been offloaded
    //
    // with the low VRAM option VRAM scratch is disabled in llama_load_model_internal
    // in that case ggml_cuda_assign_buffers has no effect
    offload_func_t offload_func_nr = llama_nop; // nr = non-repeating
    offload_func_t offload_func_kq = llama_nop;
    offload_func_t offload_func_v  = llama_nop;

#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer) {
        offload_func_nr = ggml_cuda_assign_buffers;
    }
    if (n_gpu_layers > n_layer + 1) {
        offload_func_v  = ggml_cuda_assign_buffers;
    }
    if (n_gpu_layers > n_layer + 2) {
        offload_func_kq = ggml_cuda_assign_buffers;
    }
#endif // GGML_USE_CUBLAS

    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
#ifdef LLAMA_USE_ALLOCATOR
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
    }
#else
    ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
#endif
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");

    for (int il = 0; il < n_layer; ++il) {
        ggml_format_name(inpL, "layer_inp_%d", il);

        offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
        if (il >= i_gpu_start) {
            offload_func = ggml_cuda_assign_buffers;
        }
#endif // GGML_USE_CUBLAS

        struct ggml_tensor * inpSA = inpL;

        lctx.use_buf(ctx0, 0);

        // norm
        {
            cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
            offload_func(cur);
            ggml_set_name(cur, "rms_norm_0");

            // cur = cur*attention_norm(broadcasted)
            cur = ggml_mul(ctx0, cur, model.layers[il].attention_norm);
            offload_func(cur);
            ggml_set_name(cur, "attention_norm_0");
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_tensor * tmpk = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            offload_func_kq(tmpk);
            ggml_set_name(tmpk, "tmpk");

            struct ggml_tensor * tmpq = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            offload_func_kq(tmpq);
            ggml_set_name(tmpq, "tmpq");

            struct ggml_tensor * Kcur = ggml_rope_custom_inplace(ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, N), n_past, n_embd_head, 0, 0, freq_base, freq_scale);
            offload_func_kq(Kcur);
            ggml_set_name(Kcur, "Kcur");

            struct ggml_tensor * Qcur = ggml_rope_custom_inplace(ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head, N),    n_past, n_embd_head, 0, 0, freq_base, freq_scale);
            offload_func_kq(Qcur);
            ggml_set_name(Qcur, "Qcur");

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix

                struct ggml_tensor * tmpv = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                offload_func_v(tmpv);
                ggml_set_name(tmpv, "tmpv");

                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, tmpv, n_embd_gqa, N));
                offload_func_v(Vcur);
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd_gqa, (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + n_past));
                offload_func_kq(k);
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + n_past*ggml_element_size(kv_self.v));
                offload_func_v(v);
                ggml_set_name(v, "v");

                // important: storing RoPE-ed version of K in the KV cache!
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);
            offload_func_kq(Q);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, kv_self.k, (n_past + N)*n_embd_gqa, il*n_ctx*ggml_element_size(kv_self.k)*n_embd_gqa),
                            n_embd_head, n_head_kv, n_past + N),
                        0, 2, 1, 3);
            offload_func_kq(K);
            ggml_set_name(K, "K");

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd_head)
            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0, KQ, KQ_scale);
            offload_func_kq(KQ_scaled);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
            offload_func_kq(KQ_masked);
            ggml_set_name(KQ_masked, "KQ_masked");

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
            offload_func_v(KQ_soft_max);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            // split cached V into n_head heads
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_past + N, n_embd_head, n_head_kv,
                        n_ctx*ggml_element_size(kv_self.v),
                        n_ctx*ggml_element_size(kv_self.v)*n_embd_head,
                        n_ctx*ggml_element_size(kv_self.v)*n_embd_gqa*il);
            offload_func_v(V);
            ggml_set_name(V, "V");

#if 1
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_set_name(KQV, "KQV");
#else
            // make V contiguous in memory to speed up the matmul, however we waste time on the copy
            // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
            // is there a better way?
            struct ggml_tensor * V_cont = ggml_cpy(ctx0, V, ggml_new_tensor_3d(ctx0, kv_self.v->type, n_past + N, n_embd_head, n_head));
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
            offload_func_v(cur);
            ggml_set_name(cur, "KQV_merged_contiguous");

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_wo");
        }

        lctx.use_buf(ctx0, 1);

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
        offload_func(inpFF);
        ggml_set_name(inpFF, "inpFF");

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF, rms_norm_eps);
                offload_func(cur);
                ggml_set_name(cur, "rms_norm_1");

                // cur = cur*ffn_norm(broadcasted)
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);
                offload_func(cur);
                ggml_set_name(cur, "ffn_norm");
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);
            offload_func(tmp);
            ggml_set_name(tmp, "result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w1");

            // SILU activation
            cur = ggml_silu(ctx0, cur);
            offload_func(cur);
            ggml_set_name(cur, "silu");

            cur = ggml_mul(ctx0, cur, tmp);
            offload_func(cur);
            ggml_set_name(cur, "silu_x_result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w2");
        }

        cur = ggml_add(ctx0, cur, inpFF);
        offload_func(cur);
        ggml_set_name(cur, "inpFF_+_result_w2");

        // input for next layer
        inpL = cur;
    }

    lctx.use_buf(ctx0, 0);

    // norm
    {
        cur = ggml_rms_norm(ctx0, inpL, rms_norm_eps);
        offload_func_nr(cur);
        ggml_set_name(cur, "rms_norm_2");

        // cur = cur*norm(broadcasted)
        cur = ggml_mul(ctx0, cur, model.norm);
        // offload_func_nr(cur); // TODO CPU + GPU mirrored backend
        ggml_set_name(cur, "result_norm");
    }

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");

    lctx.use_buf(ctx0, -1);

    // logits -> probs
    //cur = ggml_soft_max_inplace(ctx0, cur);

    ggml_build_forward_expand(gf, cur);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }

#if 0
    LLAMA_LOG_INFO("\n%s: used_mem: eval ctx %.3f MB, scratch %.3f MB %.3f MB, work buf %.3f MB, n_past = %d, N = %d\n", __func__,
            ggml_used_mem(ctx0)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0,
            lctx.work_buffer.size()/1024.0/1024.0,
            n_past, N);
#endif

    ggml_free(ctx0);

    return gf;
}

// evaluate the transformer
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - embd       embeddings input
//   - n_tokens   number of tokens
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool llama_eval_internal(
         llama_context & lctx,
     const llama_token * tokens,
           const float * embd,
                   int   n_tokens,
                   int   n_past,
                   int   n_threads,
            const char * cgraph_fname) {

    GGML_ASSERT((!tokens && embd) || (tokens && !embd));

    const int64_t t_start_us = ggml_time_us();

#ifdef GGML_USE_MPI
    ggml_mpi_eval_init(lctx.ctx_mpi, &n_tokens, &n_past, &n_threads);
#endif

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_vocab     = hparams.n_vocab;

#ifdef LLAMA_USE_ALLOCATOR
    ggml_allocr_reset(lctx.alloc);
#endif

    ggml_cgraph * gf = llama_build_graph(lctx, tokens, embd, n_tokens, n_past);

#ifdef LLAMA_USE_ALLOCATOR
    ggml_allocr_alloc_graph(lctx.alloc, gf);
#endif

    // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    n_threads = N >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas() ? 1 : n_threads;

    struct ggml_tensor * res = gf->nodes[gf->n_nodes - 1];
    struct ggml_tensor * embeddings = gf->nodes[gf->n_nodes - 2];

    GGML_ASSERT(strcmp(res->name, "result_output") == 0);
    GGML_ASSERT(strcmp(embeddings->name, "result_norm") == 0);

#if GGML_USE_MPI
    const int64_t n_layer = hparams.n_layer;
    ggml_mpi_graph_compute_pre(lctx.ctx_mpi, gf, n_layer);
#endif

#ifdef GGML_USE_METAL
    if (lctx.ctx_metal && N == 1) {
        // TODO: disabled until #2413 is resolved
        //if (!ggml_metal_if_optimized(lctx.ctx_metal)) {
        //    ggml_metal_graph_find_concurrency(lctx.ctx_metal, gf);
        //}
        ggml_metal_set_n_cb     (lctx.ctx_metal, n_threads);
        ggml_metal_graph_compute(lctx.ctx_metal, gf);
        ggml_metal_get_tensor   (lctx.ctx_metal, res);
        if (!lctx.embedding.empty()) {
            ggml_metal_get_tensor(lctx.ctx_metal, embeddings);
        }
    } else {
        // IMPORTANT:
        // Since we don't have efficient Matrix x Matrix Metal multiplication yet, we fallback to vanilla
        // ggml_graph_compute(). It uses Apple's Accelerate CBLAS API which takes advantage of the ANE or the AMX
        // coprocessor.
        //
        // When we implement Matrix x Matrix Metal multiplication, we can avoid this branch.
        // But for now, we have focused only on Matrix x Vector Metal multiplication.
        //
        // TODO: avoid these syncs via shared memory (ref #1696)
        //
        if (lctx.ctx_metal) {
            // We need to sync the GPU KV cache with the CPU KV cache
            ggml_metal_get_tensor(lctx.ctx_metal, kv_self.k);
            ggml_metal_get_tensor(lctx.ctx_metal, kv_self.v);
        }

        ggml_graph_compute_helper(lctx.work_buffer, gf, n_threads);
    }
#else
    ggml_graph_compute_helper(lctx.work_buffer, gf, n_threads);
#endif

#if GGML_USE_MPI
    ggml_mpi_graph_compute_post(lctx.ctx_mpi, gf, n_layer);
#endif

    // update kv token count
    lctx.kv_self.n = n_past + N;

    if (cgraph_fname) {
        ggml_graph_export(gf, cgraph_fname);
    }

#ifdef GGML_PERF
    // print timing information per ggml operation (for debugging purposes)
    // requires GGML_PERF to be defined
    ggml_graph_print(gf);
#endif

    // plot the computation graph in dot format (for debugging purposes)
    //if (n_past%100 == 0) {
    //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
    //}

    // extract logits
    {
        auto & logits_out = lctx.logits;

        if (lctx.logits_all) {
            logits_out.resize(n_vocab * N);
            memcpy(logits_out.data(), (float *) ggml_get_data(res), sizeof(float)*n_vocab*N);
        } else {
            // return result for just the last token
            logits_out.resize(n_vocab);
            memcpy(logits_out.data(), (float *) ggml_get_data(res) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
        }
    }

    // extract embeddings
    if (!lctx.embedding.empty()) {
        auto & embedding_out = lctx.embedding;

        embedding_out.resize(n_embd);
        memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
    }

    // measure the performance only for the single-token evals
    if (N == 1) {
        lctx.t_eval_us += ggml_time_us() - t_start_us;
        lctx.n_eval++;
    }
    else if (N > 1) {
        lctx.t_p_eval_us += ggml_time_us() - t_start_us;
        lctx.n_p_eval += N;
    }

    return true;
}

//
// tokenizer
//

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llama_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llama_sp_symbol>::value, "llama_sp_symbol is not trivially copyable");

struct llama_sp_bigram {
    struct comparator {
        bool operator()(llama_sp_bigram & l, llama_sp_bigram & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_sp_bigram>;
    using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
    llama_sp_symbol::index left;
    llama_sp_symbol::index right;
    float score;
    size_t size;
};

// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct llama_tokenizer {
    llama_tokenizer(const llama_vocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llama_sp_symbol sym;
            size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int) symbol.n; ++j) {
                    // NOTE: old version, before #2420 - not sure what are the implications of this
                    //llama_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    llama_vocab::id token_id = vocab_.token_to_id.at(std::string(1, symbol.text[j]));
                    output.push_back(token_id);
                }
            } else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        llama_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const llama_vocab & vocab_;
    std::vector<llama_sp_symbol> symbols_;
    llama_sp_bigram::queue work_queue_;
};

static std::vector<llama_vocab::id> llama_tokenize(const llama_vocab & vocab, const std::string & text, bool bos) {
    llama_tokenizer tokenizer(vocab);
    std::vector<llama_vocab::id> output;

    if (text.empty()) {
        return output;
    }

    if (bos) {
        output.push_back(llama_token_bos());
    }

    tokenizer.tokenize(text, output);
    return output;
}

//
// grammar - internal
//

struct llama_grammar {
    const std::vector<std::vector<llama_grammar_element>>   rules;
    std::vector<std::vector<const llama_grammar_element *>> stacks;
};

struct llama_grammar_candidate {
    size_t           index;
    const uint32_t * code_points;
};

// NOTE: assumes valid utf8 (but checks for overrun)
// adds a terminating 0 for use as pointer
std::vector<uint32_t> decode_utf8(const char * src) {
    static const int      lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    const char          * pos      = src;
    std::vector<uint32_t> code_points;
    while (*pos != 0) {
        uint8_t  first_byte = static_cast<uint8_t>(*pos);
        uint8_t  highbits   = first_byte >> 4;
        int      len        = lookup[highbits];
        uint8_t  mask       = (1 << (8 - len)) - 1;
        uint32_t value      = first_byte & mask;
        const char * end    = pos + len; // may overrun!
        ++pos;
        for ( ; pos < end && *pos != 0; ++pos) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
        }
        code_points.push_back(value);
    }
    code_points.push_back(0);
    return code_points;
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool llama_grammar_is_end_of_sequence(const llama_grammar_element * pos) {
    switch (pos->type) {
        case LLAMA_GRETYPE_END: return true;
        case LLAMA_GRETYPE_ALT: return true;
        default:                return false;
    }
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static std::pair<bool, const llama_grammar_element *> llama_grammar_match_char(
        const llama_grammar_element * pos,
        const uint32_t                chr) {

    bool found            = false;
    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR;
    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT);

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return std::make_pair(found == is_positive_char, pos);
}

// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static void llama_grammar_advance_stack(
        const std::vector<std::vector<llama_grammar_element>>   & rules,
        const std::vector<const llama_grammar_element *>        & stack,
        std::vector<std::vector<const llama_grammar_element *>> & new_stacks) {

    if (stack.empty()) {
        new_stacks.push_back(stack);
        return;
    }

    const llama_grammar_element * pos = stack.back();

    switch (pos->type) {
        case LLAMA_GRETYPE_RULE_REF: {
            const size_t                  rule_id = static_cast<size_t>(pos->value);
            const llama_grammar_element * subpos  = rules[rule_id].data();
            do {
                // init new stack without the top (pos)
                std::vector<const llama_grammar_element *> new_stack(stack.begin(), stack.end() - 1);
                if (!llama_grammar_is_end_of_sequence(pos + 1)) {
                    // if this rule ref is followed by another element, add that to stack
                    new_stack.push_back(pos + 1);
                }
                if (!llama_grammar_is_end_of_sequence(subpos)) {
                    // if alternate is nonempty, add to stack
                    new_stack.push_back(subpos);
                }
                llama_grammar_advance_stack(rules, new_stack, new_stacks);
                while (!llama_grammar_is_end_of_sequence(subpos)) {
                    // scan to end of alternate def
                    subpos++;
                }
                if (subpos->type == LLAMA_GRETYPE_ALT) {
                    // there's another alternate def of this rule to process
                    subpos++;
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case LLAMA_GRETYPE_CHAR:
        case LLAMA_GRETYPE_CHAR_NOT:
            new_stacks.push_back(stack);
            break;
        default:
            // end of alternate (LLAMA_GRETYPE_END, LLAMA_GRETYPE_ALT) or middle of char range
            // (LLAMA_GRETYPE_CHAR_ALT, LLAMA_GRETYPE_CHAR_RNG_UPPER); stack should never be left on
            // those
            GGML_ASSERT(false);
    }
}

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llama_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
static std::vector<std::vector<const llama_grammar_element *>> llama_grammar_accept(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const uint32_t                                                  chr) {

    std::vector<std::vector<const llama_grammar_element *>> new_stacks;

    for (const auto & stack : stacks) {
        if (stack.empty()) {
            continue;
        }

        auto match = llama_grammar_match_char(stack.back(), chr);
        if (match.first) {
            const llama_grammar_element * pos = match.second;

            // update top of stack to next element, if any
            std::vector<const llama_grammar_element *> new_stack(stack.begin(), stack.end() - 1);
            if (!llama_grammar_is_end_of_sequence(pos)) {
                new_stack.push_back(pos);
            }
            llama_grammar_advance_stack(rules, new_stack, new_stacks);
        }
    }

    return new_stacks;
}

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const std::vector<llama_grammar_candidate>                    & candidates);

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates_for_stack(
        const std::vector<std::vector<llama_grammar_element>> & rules,
        const std::vector<const llama_grammar_element *>      & stack,
        const std::vector<llama_grammar_candidate>            & candidates) {

    std::vector<llama_grammar_candidate> rejects;

    if (stack.empty()) {
        // accept nothing; EOS is handled elsewhere
        rejects.insert(rejects.end(), candidates.begin(), candidates.end());
        return rejects;
    }

    const llama_grammar_element * stack_pos = stack.back();

    std::vector<llama_grammar_candidate> next_candidates;
    for (auto tok : candidates) {
        if (llama_grammar_match_char(stack_pos, tok.code_points[0]).first) {
            if (tok.code_points[1] != 0) {
                next_candidates.push_back({ tok.index, tok.code_points + 1 });
            }
        } else {
            rejects.push_back(tok);
        }
    }

    auto stack_pos_after = llama_grammar_match_char(stack_pos, 0).second;

    // update top of stack to next element, if any
    std::vector<const llama_grammar_element *> stack_after(stack.begin(), stack.end() - 1);
    if (!llama_grammar_is_end_of_sequence(stack_pos_after)) {
        stack_after.push_back(stack_pos_after);
    }
    std::vector<std::vector<const llama_grammar_element *>> next_stacks;
    llama_grammar_advance_stack(rules, stack_after, next_stacks);

    auto next_rejects = llama_grammar_reject_candidates(rules, next_stacks, next_candidates);
    for (auto tok : next_rejects) {
        rejects.push_back({ tok.index, tok.code_points - 1 });
    }

    return rejects;
}

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const std::vector<llama_grammar_candidate>                    & candidates) {
    GGML_ASSERT(!stacks.empty()); // REVIEW

    if (candidates.empty()) {
        return std::vector<llama_grammar_candidate>();
    }

    auto rejects = llama_grammar_reject_candidates_for_stack(rules, stacks.front(), candidates);

    for (size_t i = 1, size = stacks.size(); i < size; ++i) {
        rejects = llama_grammar_reject_candidates_for_stack(rules, stacks[i], rejects);
    }
    return rejects;
}

//
// grammar - external
//

struct llama_grammar * llama_grammar_init(
            const llama_grammar_element ** rules,
                                 size_t    n_rules,
                                 size_t    start_rule_index) {
    const llama_grammar_element * pos;

    // copy rule definitions into vectors
    std::vector<std::vector<llama_grammar_element>> vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
    }

    // loop over alternates of start rule to build initial stacks
    std::vector<std::vector<const llama_grammar_element *>> stacks;
    pos = rules[start_rule_index];
    do {
        std::vector<const llama_grammar_element *> stack;
        if (!llama_grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        llama_grammar_advance_stack(vec_rules, stack, stacks);
        while (!llama_grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == LLAMA_GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    return new llama_grammar{ std::move(vec_rules), std::move(stacks) };
}

void llama_grammar_free(struct llama_grammar * grammar) {
    delete grammar;
}

//
// sampling
//

void llama_sample_softmax(struct llama_context * ctx, llama_token_data_array * candidates) {
    assert(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_time_us();

    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_k(struct llama_context * ctx, llama_token_data_array * candidates, int k, size_t min_keep) {
    const int64_t t_start_sample_us = ggml_time_us();

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        };
        if (k == (int) candidates->size) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
        candidates->sorted = true;
    }
    candidates->size = k;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    llama_sample_softmax(ctx, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_tail_free(struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    llama_sample_softmax(nullptr, candidates);
    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    {
        const float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);

        if (second_derivatives_sum > 1e-6f) {
            for (float & value : second_derivatives) {
                value /= second_derivatives_sum;
            }
        } else {
            for (float & value : second_derivatives) {
                value = 1.0f / second_derivatives.size();
            }
        }
    }

    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}


void llama_sample_typical(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_sample_softmax(nullptr, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < candidates->size; ++i) {
        float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_token_data> new_candidates;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates.push_back(candidates->data[idx]);
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_temperature(struct llama_context * ctx, llama_token_data_array * candidates_p, float temp) {
    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= temp;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_repetition_penalty(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float penalty) {
    if (last_tokens_size == 0 || penalty == 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates->size; ++i) {
        const auto * token_iter = std::find(last_tokens, last_tokens + last_tokens_size, candidates->data[i].id);
        if (token_iter == last_tokens + last_tokens_size) {
            continue;
        }

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty;
        } else {
            candidates->data[i].logit /= penalty;
        }
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_frequency_and_presence_penalties(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens_p, size_t last_tokens_size, float alpha_frequency, float alpha_presence) {
    if (last_tokens_size == 0 || (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<llama_token, int> token_count;
    for (size_t i = 0; i < last_tokens_size; ++i) {
        token_count[last_tokens_p[i]]++;
    }

    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        int count = token_iter->second;
        candidates->data[i].logit -= float(count) * alpha_frequency + float(count > 0) * alpha_presence;
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_grammar(struct llama_context * ctx, llama_token_data_array * candidates, const struct llama_grammar * grammar) {
    assert(ctx);
    const int64_t t_start_sample_us = ggml_time_us();

    bool allow_eos = false;
    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            allow_eos = true;
            break;
        }
    }

    const llama_token eos = llama_token_eos();

    std::vector<std::vector<uint32_t>>   candidates_decoded;
    std::vector<llama_grammar_candidate> candidates_grammar;

    for (size_t i = 0; i < candidates->size; ++i) {
        const llama_token id  = candidates->data[i].id;
        const char *      str = llama_token_to_str(ctx, id);
        if (id == eos) {
            if (!allow_eos) {
                candidates->data[i].logit = -INFINITY;
            }
        } else if (*str == 0) {
            candidates->data[i].logit = -INFINITY;
        } else {
            candidates_decoded.push_back(decode_utf8(str));
            candidates_grammar.push_back({ i, candidates_decoded.back().data() });
        }
    }

    const auto rejects =
        llama_grammar_reject_candidates(grammar->rules, grammar->stacks, candidates_grammar);
    for (auto & reject : rejects) {
        candidates->data[reject.index].logit = -INFINITY;
    }

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
}

static void llama_log_softmax(float * array, size_t size) {
    float max_l = *std::max_element(array, array + size);
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float p = expf(array[i] - max_l);
        sum += p;
        array[i] = p;
    }

    for (size_t i = 0; i < size; ++i) {
        array[i] = logf(array[i] / sum);
    }
}

void llama_sample_classifier_free_guidance(
          struct llama_context * ctx,
        llama_token_data_array * candidates,
          struct llama_context * guidance_ctx,
                         float   scale) {
    int64_t t_start_sample_us = ggml_time_us();

    assert(ctx);
    auto n_vocab = llama_n_vocab(ctx);
    assert(n_vocab == (int)candidates->size);
    assert(!candidates->sorted);

    std::vector<float> logits_base;
    logits_base.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        logits_base.push_back(candidates->data[i].logit);
    }
    llama_log_softmax(logits_base.data(), candidates->size);

    float* logits_guidance = llama_get_logits(guidance_ctx);
    llama_log_softmax(logits_guidance, n_vocab);

    for (int i = 0; i < n_vocab; ++i) {
        float logit_guidance = logits_guidance[i];
        float logit_base = logits_base[i];
        candidates->data[i].logit = scale * (logit_base - logit_guidance) + logit_guidance;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

llama_token llama_sample_token_mirostat(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int m, float * mu) {
    assert(ctx);
    auto N = float(llama_n_vocab(ctx));
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax(nullptr, candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    llama_sample_top_k(nullptr, candidates, int(k), 1);
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    llama_token X = llama_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_mirostat_v2(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu) {
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax(ctx, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    if (candidates->size == 0) {
        candidates->size = 1;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }

    // Normalize the probabilities of the remaining words
    llama_sample_softmax(ctx, candidates);

    // Sample the next word X from the remaining words
    llama_token X = llama_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_greedy(struct llama_context * ctx, llama_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_time_us();

    // Find max element
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit < b.logit;
    });

    llama_token result = max_iter->id;
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return result;
}

llama_token llama_sample_token(struct llama_context * ctx, llama_token_data_array * candidates) {
    assert(ctx);
    const int64_t t_start_sample_us = ggml_time_us();
    llama_sample_softmax(nullptr, candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    auto & rng = ctx->rng;
    int idx = dist(rng);

    llama_token result = candidates->data[idx].id;

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
    return result;
}

void llama_grammar_accept_token(struct llama_context * ctx, struct llama_grammar * grammar, llama_token token) {
    const int64_t t_start_sample_us = ggml_time_us();

    if (token == llama_token_eos()) {
        for (const auto & stack : grammar->stacks) {
            if (stack.empty()) {
                return;
            }
        }
        GGML_ASSERT(false);
    }

    const char * str = llama_token_to_str(ctx, token);
    // Note terminating 0 in decoded string
    auto code_points = decode_utf8(str);
    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        grammar->stacks = llama_grammar_accept(grammar->rules, grammar->stacks, *it);
    }
    GGML_ASSERT(!grammar->stacks.empty());

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
}

//
// quantization
//

static void llama_convert_tensor_internal(const gguf_load_tensor & tensor, std::vector<float> & output, const size_t nelements, const int nthread) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    ggml_type_traits_t qtype;
    if (ggml_is_quantized(tensor.type)) {
        qtype = ggml_internal_get_type_traits(tensor.type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor.type)));
        }
    } else if (tensor.type != GGML_TYPE_F16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor.type)));
    }

    if (nthread < 2) {
        if (tensor.type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor.data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor.type)) {
            qtype.to_float(tensor.data, f32_output, nelements);
        } else {
            GGML_ASSERT(false); // unreachable
        }
        return;
    }

    auto block_size = tensor.type == GGML_TYPE_F16 ? 1 : (size_t)ggml_blck_size(tensor.type);
    auto block_size_bytes = ggml_type_size(tensor.type);

    GGML_ASSERT(nelements % block_size == 0);
    auto nblocks = nelements / block_size;
    auto blocks_per_thread = nblocks / nthread;
    auto spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    std::vector<std::thread> workers;
    for (auto tnum = 0, in_buff_offs = 0, out_buff_offs = 0; tnum < nthread; tnum++) {
        auto thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        auto thr_elems = thr_blocks * block_size; // number of elements for this thread
        auto thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else {
                qtype.to_float(inbuf, outbuf, nels);
            }
        };
        workers.push_back(std::thread(compute, tensor.type, tensor.data + in_buff_offs, f32_output + out_buff_offs, thr_elems));
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & worker : workers) {
        worker.join();
    }

}

static void llama_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type quantized_type;
    llama_ftype ftype = params->ftype;
    int nthread = params->nthread;

    switch (params->ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: quantized_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: quantized_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: quantized_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: quantized_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: quantized_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_F16:  quantized_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_ALL_F32:     quantized_type = GGML_TYPE_F32;  break;

#ifdef GGML_USE_K_QUANTS
        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K:   quantized_type = GGML_TYPE_Q2_K; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L: quantized_type = GGML_TYPE_Q3_K; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: quantized_type = GGML_TYPE_Q4_K; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M: quantized_type = GGML_TYPE_Q5_K; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:   quantized_type = GGML_TYPE_Q6_K; break;
#endif
        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    std::unique_ptr<llama_model_loader> model_loader(new llama_model_loader(fname_inp, /*use_mmap*/ false));
    gguf_file_saver file_saver(fname_out.c_str(), model_loader->file_loader.get(), params->ftype);

#ifdef GGML_USE_K_QUANTS
    int n_attention_wv    = 0;
    int n_feed_forward_w2 = 0;
    for (auto& tensor : model_loader->tensors_map.tensors) {
        if (tensor.name.find("attention.wv.weight") != std::string::npos) {
            ++n_attention_wv;
        }
        else if (tensor.name.find("feed_forward.w2.weight") != std::string::npos) {
            ++n_feed_forward_w2;
        }
    }

    int i_attention_wv = 0;
    int i_feed_forward_w2 = 0;
#endif

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<int64_t> hist_all(1 << 4, 0);

    std::vector<std::thread> workers;
    std::mutex mutex;

    auto use_more_bits = [] (int i_layer, int num_layers) -> bool {
        return i_layer < num_layers/8 || i_layer >= 7*num_layers/8 || (i_layer - num_layers/8)%3 == 2;
    };

    size_t idx = 0;

    std::vector<uint8_t> read_data;
    std::vector<uint8_t> work;

    for (gguf_load_tensor & tensor : model_loader->tensors_map.tensors) {
        read_data.resize(tensor.size);
        tensor.data = read_data.data();
        model_loader->load_data_for(tensor);

        LLAMA_LOG_INFO("[%4zu/%4zu] %36s - %16s, type = %6s, ",
               ++idx, model_loader->tensors_map.tensors.size(),
               tensor.name.c_str(), llama_format_tensor_shape(tensor.ne).c_str(),
               ggml_type_name(tensor.type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = tensor.name.rfind("weight") == tensor.name.size() - 6; // ends with 'weight'?

        // quantize only 2D tensors
        quantize &= (tensor.ne.size() == 2);
        quantize &= params->quantize_output_tensor || tensor.name != "output.weight";
        quantize &= quantized_type != tensor.type;

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;

        if (!quantize) {
            new_type = tensor.type;
            new_data = tensor.data;
            new_size = tensor.size;
            LLAMA_LOG_INFO("size = %8.3f MB\n", tensor.size/1024.0/1024.0);
        } else {
            new_type = quantized_type;
#ifdef GGML_USE_K_QUANTS
            if (tensor.name == "output.weight") {
                int nx = tensor.ne.at(0);
                int ny = tensor.ne.at(1);
                if (nx % QK_K == 0 && ny % QK_K == 0) {
                    new_type = GGML_TYPE_Q6_K;
                }
            } else if (tensor.name.find("attention.wv.weight") != std::string::npos) {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
                else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                        use_more_bits(i_attention_wv, n_attention_wv)) new_type = GGML_TYPE_Q6_K;
                else if (QK_K == 64 && (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S) &&
                        (i_attention_wv < n_attention_wv/8 || i_attention_wv >= 7*n_attention_wv/8)) new_type = GGML_TYPE_Q6_K;
                ++i_attention_wv;
            } else if (tensor.name.find("feed_forward.w2.weight") != std::string::npos) {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
                else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                         use_more_bits(i_feed_forward_w2, n_feed_forward_w2)) new_type = GGML_TYPE_Q6_K;
                //else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && i_feed_forward_w2 < n_feed_forward_w2/8) new_type = GGML_TYPE_Q6_K;
                ++i_feed_forward_w2;
            } else if (tensor.name.find("attention.wo.weight") != std::string::npos) {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
            }
            bool convert_incompatible_tensor = false;
            if (new_type == GGML_TYPE_Q2_K || new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K ||
                new_type == GGML_TYPE_Q5_K || new_type == GGML_TYPE_Q6_K) {
                int nx = tensor.ne.at(0);
                int ny = tensor.ne.at(1);
                if (nx % QK_K != 0 || ny % QK_K != 0) {
                    LLAMA_LOG_INFO("\n\nTensor sizes %d x %d are not divisible by %d, required for k-quants.\n",nx,ny,QK_K);
                    convert_incompatible_tensor = true;
                }
            }
            if (convert_incompatible_tensor) {
                if (tensor.name == "output.weight") {
                    new_type = GGML_TYPE_F16; //fall back to F16 instead of just failing.
                    LLAMA_LOG_WARN("F16 will be used for this tensor instead.\n");
                } else if (tensor.name == "tok_embeddings.weight") {
                    new_type = GGML_TYPE_Q4_0; //fall back to Q4_0 instead of just failing.
                    LLAMA_LOG_WARN("Q4_0 will be used for this tensor instead.\n");
                } else {
                    throw std::runtime_error("Unsupported tensor size encountered\n");
                }
            }
#endif

            const size_t nelements = tensor.ne.at(0) * tensor.ne.at(1);

            float * f32_data;
            std::vector<float> f32_conv_buf;

            if (tensor.type == GGML_TYPE_F32) {
                f32_data = (float *) tensor.data;
            } else if (ggml_is_quantized(tensor.type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor.type)));
            } else {
                llama_convert_tensor_internal(tensor, f32_conv_buf, nelements, nthread);
                f32_data = (float *) f32_conv_buf.data();
            }

            LLAMA_LOG_INFO("quantizing to %s .. ", ggml_type_name(new_type));
            fflush(stdout);

            work.resize(nelements * 4); // upper bound on size
            new_data = work.data();
            std::vector<int64_t> hist_cur(1 << 4, 0);

            const int chunk_size = 32 * 512;
            const int nchunk = (nelements + chunk_size - 1)/chunk_size;
            const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
            if (nthread_use < 2) {
                new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nelements, hist_cur.data());
            } else {
                size_t counter = 0;
                new_size = 0;
                auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, nelements] () {
                    std::vector<int64_t> local_hist;
                    size_t local_size = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        size_t first = counter; counter += chunk_size;
                        if (first >= nelements) {
                            if (!local_hist.empty()) {
                                for (int j=0; j<int(local_hist.size()); ++j) {
                                    hist_cur[j] += local_hist[j];
                                }
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        size_t last = std::min(nelements, first + chunk_size);
                        if (local_hist.empty()) {
                            local_hist.resize(hist_cur.size(), 0);
                        }
                        local_size += ggml_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
                    }
                };
                if ((int) workers.size() < nthread_use - 1) {
                    workers.resize(nthread_use - 1);
                }
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers[it] = std::thread(compute);
                }
                compute();
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers[it].join();
                }
            }

            LLAMA_LOG_INFO("size = %8.2f MB -> %8.2f MB | hist: ", tensor.size/1024.0/1024.0, new_size/1024.0/1024.0);
            int64_t tot_count = 0;
            for (size_t i = 0; i < hist_cur.size(); i++) {
                hist_all[i] += hist_cur[i];
                tot_count += hist_cur[i];
            }

            if (tot_count > 0) {
                for (size_t i = 0; i < hist_cur.size(); i++) {
                    LLAMA_LOG_INFO("%5.3f ", hist_cur[i] / float(nelements));
                }
            }
            LLAMA_LOG_INFO("\n");
        }
        total_size_org += tensor.size;
        total_size_new += new_size;
        file_saver.write_tensor(tensor, new_type, new_data, new_size);
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

    {
        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); i++) {
            sum_all += hist_all[i];
        }

        if (sum_all > 0) {
            LLAMA_LOG_INFO("%s: hist: ", __func__);
            for (size_t i = 0; i < hist_all.size(); i++) {
                LLAMA_LOG_INFO("%5.3f ", hist_all[i] / float(sum_all));
            }
            LLAMA_LOG_INFO("\n");
        }
    }
}

//
// interface implementation
//

struct llama_model * llama_load_model_from_file(
                             const char * path_model,
            struct llama_context_params   params) {
    ggml_time_init();

    llama_model * model = new llama_model;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!llama_model_load(path_model, *model, model->vocab, params.n_ctx, params.n_batch, params.n_gpu_layers,
                params.main_gpu, params.tensor_split, params.mul_mat_q, params.rope_freq_base, params.rope_freq_scale,
                params.low_vram, memory_type, params.use_mmap, params.use_mlock, params.vocab_only,
                params.progress_callback, params.progress_callback_user_data)) {
        LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        delete model;
        return nullptr;
    }

    return model;
}

void llama_free_model(struct llama_model * model) {
    delete model;
}

struct llama_context * llama_new_context_with_model(
                 struct llama_model * model,
        struct llama_context_params   params) {

    if (!model) {
        return nullptr;
    }

    llama_context * ctx = new llama_context(*model);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_INFO(".");
                if (percentage >= 100) {
                    LLAMA_LOG_INFO("\n");
                }
            }
        };
    }

    ctx->rng = std::mt19937(params.seed);
    ctx->logits_all = params.logits_all;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // reserve memory for context buffers
    if (!params.vocab_only) {
        if (!kv_cache_init(ctx->model.hparams, ctx->kv_self, memory_type, ctx->model.hparams.n_ctx, params.n_gpu_layers)) {
            LLAMA_LOG_ERROR("%s: kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_nbytes(ctx->kv_self.k) + ggml_nbytes(ctx->kv_self.v);
            LLAMA_LOG_INFO("%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
        }

        const auto & hparams = ctx->model.hparams;

        // resized during inference
        if (params.logits_all) {
            ctx->logits.reserve(hparams.n_ctx*hparams.n_vocab);
        } else {
            ctx->logits.reserve(hparams.n_vocab);
        }

        if (params.embedding){
            ctx->embedding.resize(hparams.n_embd);
        }

#ifdef LLAMA_USE_ALLOCATOR
        {
            static const size_t tensor_alignment = 32;
            // the compute buffer is used to store the tensor and graph structs, while the allocator buffer is used for the tensor data
            ctx->buf_compute.resize(ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead());

            // create measure allocator
            ctx->alloc = ggml_allocr_new_measure(tensor_alignment);

            // build worst-case graph
            int n_tokens = std::min((int)hparams.n_ctx, params.n_batch);
            int n_past = hparams.n_ctx - n_tokens;
            llama_token token = llama_token_bos(); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
            ggml_cgraph * gf = llama_build_graph(*ctx, &token, NULL, n_tokens, n_past);

            // measure memory requirements for the graph
            size_t alloc_size = ggml_allocr_alloc_graph(ctx->alloc, gf) + tensor_alignment;

            LLAMA_LOG_INFO("%s: compute buffer total size = %7.2f MB\n", __func__, (ctx->buf_compute.size + alloc_size) / 1024.0 / 1024.0);

            // debug - for comparison with scratch buffer
            //size_t prev_req =
            //    MEM_REQ_SCRATCH0(hparams.n_ctx).at(ctx->model.type) +
            //    MEM_REQ_SCRATCH1().at(ctx->model.type) +
            //    MEM_REQ_EVAL().at(ctx->model.type);
            //LLAMA_LOG_INFO("%s: (debug) equivalent with scratch buffer = %7.2f MB\n", __func__, prev_req / 1024.0 / 1024.0);

            // recreate allocator with exact memory requirements
            ggml_allocr_free(ctx->alloc);

            ctx->buf_alloc.resize(alloc_size);
            ctx->alloc = ggml_allocr_new(ctx->buf_alloc.data, ctx->buf_alloc.size, tensor_alignment);
        }
#else
        ctx->buf_compute.resize(MEM_REQ_EVAL().at(ctx->model.type) + ggml_graph_overhead());
#endif

#ifdef LLAMA_USE_SCRATCH
        ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0(hparams.n_ctx).at(ctx->model.type));
        ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1().at(ctx->model.type));
#endif
    }

#ifdef GGML_USE_METAL
    if (params.n_gpu_layers > 0) {
        // this allocates all Metal resources and memory buffers
        ctx->ctx_metal = ggml_metal_init(1);

        void * data_ptr  = NULL;
        size_t data_size = 0;

        if (params.use_mmap) {
            data_ptr  = ctx->model.mapping->addr;
            data_size = ctx->model.mapping->size;
        } else {
            data_ptr  = ggml_get_mem_buffer(ctx->model.ctx);
            data_size = ggml_get_mem_size  (ctx->model.ctx);
        }

        const size_t max_size = ggml_get_max_tensor_size(ctx->model.ctx);

        LLAMA_LOG_INFO("%s: max tensor size = %8.2f MB\n", __func__, max_size/1024.0/1024.0);

#define LLAMA_METAL_CHECK_BUF(result)                            \
    if (!(result)) {                                             \
        LLAMA_LOG_ERROR("%s: failed to add buffer\n", __func__); \
        llama_free(ctx);                                         \
        return NULL;                                             \
    }

        LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "data", data_ptr, data_size, max_size));

        LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "eval", ctx->buf_compute.data, ctx->buf_compute.size, 0));
        LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "kv",   ctx->kv_self.buf.data, ctx->kv_self.buf.size, 0));

        LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "scr0", ctx->buf_scratch[0].data, ctx->buf_scratch[0].size, 0));
        LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "scr1", ctx->buf_scratch[1].data, ctx->buf_scratch[1].size, 0));
#undef LLAMA_METAL_CHECK_BUF
    }
#endif

#ifdef GGML_USE_MPI
    ctx->ctx_mpi = ggml_mpi_init();

    if (ggml_mpi_rank(ctx->ctx_mpi) > 0) {
        // Enter a blocking eval loop with dummy input, letting rank=0 drive the process
        const std::vector<llama_token> tmp(ctx->model.hparams.n_ctx, llama_token_bos());
        while (!llama_eval(ctx, tmp.data(), tmp.size(), 0, 0)) {};
        llama_backend_free();
        exit(1);
    }
#endif

    return ctx;
}

struct llama_context * llama_init_from_file(
                             const char * path_model,
            struct llama_context_params   params) {

    struct llama_model * model = llama_load_model_from_file(path_model, params);
    if (!model) {
        return nullptr;
    }
    struct llama_context * ctx = llama_new_context_with_model(model, params);
    ctx->model_owner = true;
    return ctx;
}

void llama_free(struct llama_context * ctx) {
    delete ctx;
}

int llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params *params) {
    try {
        llama_model_quantize_internal(fname_inp, fname_out, params);
        return 0;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_apply_lora_from_file_internal(const struct llama_model & model, const char * path_lora, const char * path_base_model, int n_threads) {
    LLAMA_LOG_INFO("%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

    const int64_t t_start_lora_us = ggml_time_us();

    auto fin = std::ifstream(path_lora, std::ios::binary);
    if (!fin) {
        LLAMA_LOG_ERROR("%s: failed to open '%s'\n", __func__, path_lora);
        return 1;
    }

    // verify magic and version
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != 1) {
            LLAMA_LOG_ERROR("%s: unsupported file version\n", __func__ );
            return 1;
        }
    }

    int32_t lora_r;
    int32_t lora_alpha;
    fin.read((char *) &lora_r, sizeof(lora_r));
    fin.read((char *) &lora_alpha, sizeof(lora_alpha));
    float scaling = (float)lora_alpha / (float)lora_r;

    LLAMA_LOG_INFO("%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);

    // create a temporary ggml context to store the lora tensors
    // todo: calculate size from biggest possible tensor
    std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
    struct ggml_init_params params;
    params.mem_size   = lora_buf.size();
    params.mem_buffer = lora_buf.data();
    params.no_alloc   = false;

    ggml_context * lora_ctx = ggml_init(params);
    std::unordered_map<std::string, struct ggml_tensor *> lora_tensors;

    // create a name -> tensor map of the model to accelerate lookups
    std::unordered_map<std::string, struct ggml_tensor*> model_tensors;
    for (const auto & kv: model.tensors_by_name) {
        model_tensors.insert(kv);
    }

    // load base model
    std::unique_ptr<llama_model_loader> model_loader;
    ggml_context * base_ctx = NULL;
    std::vector<uint8_t> base_buf;
    if (path_base_model) {
        LLAMA_LOG_INFO("%s: loading base model from '%s'\n", __func__, path_base_model);
        model_loader.reset(new llama_model_loader(path_base_model, /*use_mmap*/ true));

        size_t ctx_size;
        size_t mmapped_size;
        model_loader->calc_sizes(&ctx_size, &mmapped_size);
        base_buf.resize(ctx_size);

        ggml_init_params base_params;
        base_params.mem_size   = base_buf.size();
        base_params.mem_buffer = base_buf.data();
        base_params.no_alloc   = model_loader->use_mmap;

        base_ctx = ggml_init(base_params);

        model_loader->ggml_ctx = base_ctx;

        // maybe this should in llama_model_loader
        if (model_loader->use_mmap) {
            model_loader->mapping.reset(new gguf_mmap(&model_loader->file_loader->file, /* prefetch */ 0, ggml_is_numa()));
        }
    }

    // read tensors and apply
    bool warned = false;
    int n_tensors = 0;

    std::vector<uint8_t> work_buffer;

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ftype;

        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fin.read(reinterpret_cast<char *>(&length), sizeof(length));
        fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
        if (fin.eof()) {
            break;
        }

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }

        std::string name;
        {
            char buf[1024];
            fin.read(buf, length);
            name = std::string(buf, length);
        }

        // check for lora suffix and get the type of tensor
        const std::string lora_suffix = ".lora";
        size_t pos = name.rfind(lora_suffix);
        if (pos == std::string::npos) {
            LLAMA_LOG_ERROR("%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
            return 1;
        }

        std::string lora_type = name.substr(pos + lora_suffix.length());
        std::string base_name = name;
        base_name.erase(pos);
        // LLAMA_LOG_INFO("%s: %s => %s (lora type %s) \n", __func__, name.c_str(),base_name.c_str(), lora_type.c_str());

        if (model_tensors.find(base_name) == model_tensors.end()) {
            LLAMA_LOG_ERROR("%s: unknown tensor '%s' in lora adapter\n", __func__, name.data());
            return 1;
        }

        // create ggml tensor
        ggml_type wtype;
        switch (ftype) {
            case 0: wtype = GGML_TYPE_F32;  break;
            case 1: wtype = GGML_TYPE_F16;  break;
            default:
                    {
                        LLAMA_LOG_ERROR("%s: invalid tensor data type '%d'\n",
                                __func__, ftype);
                        return false;
                    }
        }
        ggml_tensor * lora_tensor;
        if (n_dims == 2) {
            lora_tensor = ggml_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1]);
        }
        else {
            LLAMA_LOG_ERROR("%s: unsupported tensor dimension %d\n", __func__, n_dims);
            return 1;
        }
        ggml_set_name(lora_tensor, "lora_tensor");

        // load tensor data
        size_t offset = fin.tellg();
        size_t tensor_data_size = ggml_nbytes(lora_tensor);
        offset = (offset + 31) & -32;
        fin.seekg(offset);
        fin.read((char*)lora_tensor->data, tensor_data_size);

        lora_tensors[name] = lora_tensor;

        // check if we have both A and B tensors and apply
        if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
            lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {

            ggml_tensor * dest_t = model_tensors[base_name];

            offload_func_t offload_func = llama_nop;
            offload_func_t offload_func_force_inplace = llama_nop;

#ifdef GGML_USE_CUBLAS
            if (dest_t->backend == GGML_BACKEND_GPU || dest_t->backend == GGML_BACKEND_GPU_SPLIT) {
                if (dest_t->type != GGML_TYPE_F16) {
                    throw std::runtime_error(format(
                        "%s: error: the simultaneous use of LoRAs and GPU acceleration is only supported for f16 models", __func__));
                }
                offload_func = ggml_cuda_assign_buffers;
                offload_func_force_inplace = ggml_cuda_assign_buffers_force_inplace;
            }
#endif // GGML_USE_CUBLAS

            ggml_tensor * base_t;
            if (model_loader) {
                // load from base model
                if (model_loader->tensors_map.name_to_idx.find(base_name) == model_loader->tensors_map.name_to_idx.end()) {
                    LLAMA_LOG_ERROR("%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
                    return 1;
                }
                size_t idx = model_loader->tensors_map.name_to_idx[base_name];
                gguf_load_tensor & lt = model_loader->tensors_map.tensors[idx];
                base_t = model_loader->get_tensor(base_name, { (uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1] }, GGML_BACKEND_CPU);
                lt.data = (uint8_t *) lt.ggml_tensor->data;
                model_loader->load_data_for(lt);
                lt.ggml_tensor->data = lt.data;
            }
            else {
                base_t = dest_t;
            }

            if (ggml_is_quantized(base_t->type)) {
                if (!warned) {
                    LLAMA_LOG_WARN("%s: warning: using a lora adapter with a quantized model may result in poor quality, "
                                   "use a f16 or f32 base model with --lora-base\n", __func__);
                    warned = true;
                }
            }

            ggml_tensor * loraA = lora_tensors[base_name + ".loraA"];
            GGML_ASSERT(loraA->type == GGML_TYPE_F32);
            ggml_set_name(loraA, "loraA");

            ggml_tensor * loraB = lora_tensors[base_name + ".loraB"];
            GGML_ASSERT(loraB->type == GGML_TYPE_F32);
            ggml_set_name(loraB, "loraB");

            if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
                LLAMA_LOG_ERROR("%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64 ");"
                                " are you sure that this adapter is for this model?\n", __func__, base_t->ne[0], loraA->ne[1]);
                return 1;
            }

            // w = w + BA*s
            ggml_tensor * BA = ggml_mul_mat(lora_ctx, loraA, loraB);
            offload_func(BA);
            ggml_set_name(BA, "BA");

            if (scaling != 1.0f) {
                ggml_tensor * scale_tensor = ggml_new_f32(lora_ctx, scaling);
                ggml_set_name(scale_tensor, "scale_tensor");

                BA = ggml_scale_inplace(lora_ctx, BA, scale_tensor);
                offload_func(BA);
                ggml_set_name(BA, "BA_scaled");
            }

            ggml_tensor * r;
            if (base_t == dest_t) {
                r = ggml_add_inplace(lora_ctx, dest_t, BA);
                offload_func_force_inplace(r);
                ggml_set_name(r, "r_add_inplace");
            }
            else {
                r = ggml_add(lora_ctx, base_t, BA);
                offload_func(r);
                ggml_set_name(r, "r_add");

                r = ggml_cpy(lora_ctx, r, dest_t);
                offload_func(r);
                ggml_set_name(r, "r_cpy");
            }

            struct ggml_cgraph gf = ggml_build_forward(r);

            ggml_graph_compute_helper(work_buffer, &gf, n_threads);

            // we won't need these tensors again, reset the context to save memory
            ggml_free(lora_ctx);
            lora_ctx = ggml_init(params);
            lora_tensors.clear();

            n_tensors++;
            if (n_tensors % 4 == 0) {
                LLAMA_LOG_INFO(".");
            }
        }
    }

    // TODO: this should be in a destructor, it will leak on failure
    ggml_free(lora_ctx);
    if (base_ctx) {
        ggml_free(base_ctx);
    }

    const int64_t t_lora_us = ggml_time_us() - t_start_lora_us;
    LLAMA_LOG_INFO(" done (%.2f ms)\n", t_lora_us / 1000.0);

    return 0;
}

int llama_apply_lora_from_file(struct llama_context * ctx, const char * path_lora, const char * path_base_model, int n_threads) {
    try {
        return llama_apply_lora_from_file_internal(ctx->model, path_lora, path_base_model, n_threads);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_model_apply_lora_from_file(const struct llama_model * model, const char * path_lora, const char * path_base_model, int n_threads) {
    try {
        return llama_apply_lora_from_file_internal(*model, path_lora, path_base_model, n_threads);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_get_kv_cache_token_count(const struct llama_context * ctx) {
    return ctx->kv_self.n;
}

#define LLAMA_MAX_RNG_STATE (64*1024)

void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = time(NULL);
    }
    ctx->rng.seed(seed);
}

// Returns the *maximum* size of the state
size_t llama_get_state_size(const struct llama_context * ctx) {
    // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized state.
    // for reference, std::mt19937(1337) serializes to 6701 bytes.
    const size_t s_rng_size        = sizeof(size_t);
    const size_t s_rng             = LLAMA_MAX_RNG_STATE;
    const size_t s_logits_capacity = sizeof(size_t);
    const size_t s_logits_size     = sizeof(size_t);
    const size_t s_logits          = ctx->logits.capacity() * sizeof(float);
    const size_t s_embedding_size  = sizeof(size_t);
    const size_t s_embedding       = ctx->embedding.size() * sizeof(float);
    const size_t s_kv_size         = sizeof(size_t);
    const size_t s_kv_ntok         = sizeof(int);
    const size_t s_kv              = ctx->kv_self.buf.size;

    const size_t s_total = (
        + s_rng_size
        + s_rng
        + s_logits_capacity
        + s_logits_size
        + s_logits
        + s_embedding_size
        + s_embedding
        + s_kv_size
        + s_kv_ntok
        + s_kv
    );

    return s_total;
}

// llama_context_data
struct llama_data_context {
    virtual void write(const void * src, size_t size) = 0;
    virtual size_t get_size_written() = 0;
    virtual ~llama_data_context() = default;
};

struct llama_data_buffer_context : llama_data_context {
    uint8_t * ptr;
    size_t size_written = 0;

    llama_data_buffer_context(uint8_t * p) : ptr(p) {}

    void write(const void * src, size_t size) override {
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_file_context : llama_data_context {
    FILE * file;
    size_t size_written = 0;

    llama_data_file_context(FILE * f) : file(f) {}

    void write(const void * src, size_t size) override {
        fwrite(src, size, 1, file);
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

/** copy state data into either a buffer or file depending on the passed in context
 *
 * file context:
 * llama_file file("/path", "wb");
 * llama_data_file_context data_ctx(&file);
 * llama_copy_state_data(ctx, &data_ctx);
 *
 * buffer context:
 * std::vector<uint8_t> buf(max_size, 0);
 * llama_data_buffer_context data_ctx(&buf.data());
 * llama_copy_state_data(ctx, &data_ctx);
 *
*/
void llama_copy_state_data_internal(struct llama_context * ctx, llama_data_context * data_ctx) {
    // copy rng
    {
        std::stringstream rng_ss;
        rng_ss << ctx->rng;

        const size_t rng_size = rng_ss.str().size();
        char rng_buf[LLAMA_MAX_RNG_STATE];

        memset(&rng_buf[0], 0, LLAMA_MAX_RNG_STATE);
        memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

        data_ctx->write(&rng_size,   sizeof(rng_size));
        data_ctx->write(&rng_buf[0], LLAMA_MAX_RNG_STATE);
    }

    // copy logits
    {
        const size_t logits_cap  = ctx->logits.capacity();
        const size_t logits_size = ctx->logits.size();

        data_ctx->write(&logits_cap,  sizeof(logits_cap));
        data_ctx->write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            data_ctx->write(ctx->logits.data(), logits_size * sizeof(float));
        }

        // If there is a gap between the size and the capacity, write padding
        size_t padding_size = (logits_cap - logits_size) * sizeof(float);
        if (padding_size > 0) {
            std::vector<uint8_t> padding(padding_size, 0); // Create a buffer filled with zeros
            data_ctx->write(padding.data(), padding_size);
        }
    }

    // copy embeddings
    {
        const size_t embedding_size = ctx->embedding.size();

        data_ctx->write(&embedding_size, sizeof(embedding_size));

        if (embedding_size) {
            data_ctx->write(ctx->embedding.data(), embedding_size * sizeof(float));
        }
    }

    // copy kv cache
    {
        const auto & kv_self = ctx->kv_self;
        const auto & hparams = ctx->model.hparams;
        const int    n_layer = hparams.n_layer;
        const int    n_embd  = hparams.n_embd_gqa();
        const int    n_ctx   = hparams.n_ctx;

        const size_t kv_size = kv_self.buf.size;
        const int    kv_ntok = llama_get_kv_cache_token_count(ctx);

        data_ctx->write(&kv_size, sizeof(kv_size));
        data_ctx->write(&kv_ntok, sizeof(kv_ntok));

        if (kv_size) {
            const size_t elt_size = ggml_element_size(kv_self.k);

            ggml_context * cpy_ctx = ggml_init({ 4096, NULL, /* no_alloc */ true });
            ggml_cgraph gf{};

            ggml_tensor * kout3d = ggml_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer);
            std::vector<uint8_t> kout3d_data(ggml_nbytes(kout3d), 0);
            kout3d->data = kout3d_data.data();

            ggml_tensor * vout3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer);
            std::vector<uint8_t> vout3d_data(ggml_nbytes(vout3d), 0);
            vout3d->data = vout3d_data.data();

            ggml_tensor * k3d = ggml_view_3d(cpy_ctx, kv_self.k,
                n_embd, kv_ntok, n_layer,
                elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

            ggml_tensor * v3d = ggml_view_3d(cpy_ctx, kv_self.v,
                kv_ntok, n_embd, n_layer,
                elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);

            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, k3d, kout3d));
            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, v3d, vout3d));
            ggml_graph_compute_helper(ctx->work_buffer, &gf, /*n_threads*/ 1);

            ggml_free(cpy_ctx);

            // our data is now in the kout3d_data and vout3d_data buffers
            // write them to file
            data_ctx->write(kout3d_data.data(), kout3d_data.size());
            data_ctx->write(vout3d_data.data(), vout3d_data.size());
        }
    }
}

size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst) {
    llama_data_buffer_context data_ctx(dst);
    llama_copy_state_data_internal(ctx, &data_ctx);

    return data_ctx.get_size_written();
}

// Sets the state reading from the specified source address
size_t llama_set_state_data(struct llama_context * ctx, uint8_t * src) {
    uint8_t * inp = src;

    // set rng
    {
        size_t rng_size;
        char   rng_buf[LLAMA_MAX_RNG_STATE];

        memcpy(&rng_size,   inp, sizeof(rng_size));    inp += sizeof(rng_size);
        memcpy(&rng_buf[0], inp, LLAMA_MAX_RNG_STATE); inp += LLAMA_MAX_RNG_STATE;

        std::stringstream rng_ss;
        rng_ss.str(std::string(&rng_buf[0], rng_size));
        rng_ss >> ctx->rng;

        GGML_ASSERT(rng_ss.fail() == false);
    }

    // set logits
    {
        size_t logits_cap;
        size_t logits_size;

        memcpy(&logits_cap,  inp, sizeof(logits_cap));  inp += sizeof(logits_cap);
        memcpy(&logits_size, inp, sizeof(logits_size)); inp += sizeof(logits_size);

        GGML_ASSERT(ctx->logits.capacity() == logits_cap);

        if (logits_size) {
            ctx->logits.resize(logits_size);
            memcpy(ctx->logits.data(), inp, logits_size * sizeof(float));
        }

        inp += logits_cap * sizeof(float);
    }

    // set embeddings
    {
        size_t embedding_size;

        memcpy(&embedding_size, inp, sizeof(embedding_size)); inp += sizeof(embedding_size);

        GGML_ASSERT(ctx->embedding.capacity() == embedding_size);

        if (embedding_size) {
            memcpy(ctx->embedding.data(), inp, embedding_size * sizeof(float));
            inp += embedding_size * sizeof(float);
        }
    }

    // set kv cache
    {
        const auto & kv_self = ctx->kv_self;
        const auto & hparams = ctx->model.hparams;
        const int    n_layer = hparams.n_layer;
        const int    n_embd  = hparams.n_embd_gqa();
        const int    n_ctx   = hparams.n_ctx;

        size_t kv_size;
        int kv_ntok;

        memcpy(&kv_size, inp, sizeof(kv_size)); inp += sizeof(kv_size);
        memcpy(&kv_ntok, inp, sizeof(kv_ntok)); inp += sizeof(kv_ntok);

        if (kv_size) {
            GGML_ASSERT(kv_self.buf.size == kv_size);

            const size_t elt_size = ggml_element_size(kv_self.k);

            ggml_context * cpy_ctx = ggml_init({ 4096, NULL, /* no_alloc */ true });
            ggml_cgraph gf{};

            ggml_tensor * kin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer);
            kin3d->data = (void *) inp;
            inp += ggml_nbytes(kin3d);

            ggml_tensor * vin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer);
            vin3d->data = (void *) inp;
            inp += ggml_nbytes(vin3d);

            ggml_tensor * k3d = ggml_view_3d(cpy_ctx, kv_self.k,
                n_embd, kv_ntok, n_layer,
                elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

            ggml_tensor * v3d = ggml_view_3d(cpy_ctx, kv_self.v,
                kv_ntok, n_embd, n_layer,
                elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);

            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, kin3d, k3d));
            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, vin3d, v3d));
            ggml_graph_compute_helper(ctx->work_buffer, &gf, /*n_threads*/ 1);

            ggml_free(cpy_ctx);
        }

        ctx->kv_self.n = kv_ntok;
    }

    const size_t nread    = inp - src;
    const size_t max_size = llama_get_state_size(ctx);

    GGML_ASSERT(nread <= max_size);

    return nread;
}

static bool llama_load_session_file_internal(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    gguf_file file(path_session, "rb");
    GGML_UNUSED(ctx);
    GGML_UNUSED(path_session);
    GGML_UNUSED(tokens_out);
    GGML_UNUSED(n_token_capacity);
    GGML_UNUSED(n_token_count_out);


// TODO: implement with GGUF format
    return true;
}

bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_load_session_file_internal(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("error loading session file: %s\n", err.what());
        return false;
    }
}

bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    gguf_file file(path_session, "wb");
    GGML_UNUSED(ctx);
    GGML_UNUSED(tokens);
    GGML_UNUSED(n_token_count);

    // TODO: implement with GGUF format

    return true;
}

int llama_eval(
        struct llama_context * ctx,
           const llama_token * tokens,
                         int   n_tokens,
                         int   n_past,
                         int   n_threads) {
    if (!llama_eval_internal(*ctx, tokens, nullptr, n_tokens, n_past, n_threads, nullptr)) {
        LLAMA_LOG_ERROR("%s: failed to eval\n", __func__);
        return 1;
    }

    // get a more accurate load time, upon first eval
    // TODO: fix this
    if (!ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    return 0;
}


int llama_eval_embd(
            struct llama_context * ctx,
                     const float * embd,
                             int   n_tokens,
                             int   n_past,
                             int   n_threads) {
    if (!llama_eval_internal(*ctx, nullptr, embd, n_tokens, n_past, n_threads, nullptr)) {
        LLAMA_LOG_ERROR("%s: failed to eval\n", __func__);
        return 1;
    }

    // get a more accurate load time, upon first eval
    // TODO: fix this
    if (!ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    return 0;
}

int llama_eval_export(struct llama_context * ctx, const char * fname) {
    const int n_batch = 1;
    const int n_ctx   = 512 - n_batch;

    const std::vector<llama_token> tmp(n_batch, llama_token_bos());

    if (!llama_eval_internal(*ctx, tmp.data(), nullptr, tmp.size(), n_ctx, 1, fname)) {
        LLAMA_LOG_ERROR("%s: failed to eval\n", __func__);
        return 1;
    }

    return 0;
}

int llama_tokenize_with_model(
    const struct llama_model * model,
                  const char * text,
                 llama_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto res = llama_tokenize(model->vocab, text, add_bos);

    if (n_max_tokens < (int) res.size()) {
        LLAMA_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int llama_tokenize(
        struct llama_context * ctx,
                  const char * text,
                 llama_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    return llama_tokenize_with_model(&ctx->model, text, tokens, n_max_tokens, add_bos);
}

int llama_n_vocab_from_model(const struct llama_model * model) {
    return model->vocab.id_to_token.size();
}

int llama_n_ctx_from_model(const struct llama_model * model) {
    return model->hparams.n_ctx;
}

int llama_n_embd_from_model(const struct llama_model * model) {
    return model->hparams.n_embd;
}

int llama_n_vocab(const struct llama_context * ctx) {
    return ctx->model.vocab.id_to_token.size();
}

int llama_n_ctx(const struct llama_context * ctx) {
    return ctx->model.hparams.n_ctx;
}

int llama_n_embd(const struct llama_context * ctx) {
    return ctx->model.hparams.n_embd;
}

int llama_get_vocab_from_model(
        const struct llama_model * model,
        const char * * strings,
        float  * scores,
        int capacity) {
    int n = std::min(capacity, (int) model->vocab.id_to_token.size());
    for (int i = 0; i<n; ++i) {
        strings[i] = model->vocab.id_to_token[i].tok.c_str();
        scores[i]  = model->vocab.id_to_token[i].score;
    }
    return n;
}

int llama_get_vocab(
        const struct llama_context * ctx,
        const char * * strings,
        float  * scores,
        int capacity) {
    return llama_get_vocab_from_model(&ctx->model, strings, scores, capacity);
}

float * llama_get_logits(struct llama_context * ctx) {
    return ctx->logits.data();
}

float * llama_get_embeddings(struct llama_context * ctx) {
    return ctx->embedding.data();
}

const char * llama_token_to_str_with_model(const struct llama_model * model, llama_token token) {
    if (token >= llama_n_vocab_from_model(model)) {
        return nullptr;
    }

    return model->vocab.id_to_token[token].tok.c_str();
}

const char * llama_token_to_str(const struct llama_context * ctx, llama_token token) {
    return llama_token_to_str_with_model(&ctx->model, token);
}

llama_token llama_token_bos() {
    return 1;
}

llama_token llama_token_eos() {
    return 2;
}

llama_token llama_token_nl() {
    return 13;
}

struct llama_timings llama_get_timings(struct llama_context * ctx) {
    struct llama_timings result = {
        /*.t_start_ms  =*/ 1e-3 * ctx->t_start_us,
        /*.t_end_ms    =*/ 1.00 * ggml_time_ms(),
        /*.t_load_ms   =*/ 1e-3 * ctx->t_load_us,
        /*.t_sample_ms =*/ 1e-3 * ctx->t_sample_us,
        /*.t_p_eval_ms =*/ 1e-3 * ctx->t_p_eval_us,
        /*.t_eval_ms   =*/ 1e-3 * ctx->t_eval_us,

        /*.n_sample =*/ std::max(1, ctx->n_sample),
        /*.n_p_eval =*/ std::max(1, ctx->n_p_eval),
        /*.n_eval   =*/ std::max(1, ctx->n_eval),
    };

    return result;
}

void llama_print_timings(struct llama_context * ctx) {
    const llama_timings timings = llama_get_timings(ctx);

    LLAMA_LOG_INFO("\n");
    LLAMA_LOG_INFO("%s:        load time = %8.2f ms\n", __func__, timings.t_load_ms);
    LLAMA_LOG_INFO("%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_sample_ms, timings.n_sample, timings.t_sample_ms / timings.n_sample, 1e3 / timings.t_sample_ms * timings.n_sample);
    LLAMA_LOG_INFO("%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_p_eval_ms, timings.n_p_eval, timings.t_p_eval_ms / timings.n_p_eval, 1e3 / timings.t_p_eval_ms * timings.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_eval_ms, timings.n_eval, timings.t_eval_ms / timings.n_eval, 1e3 / timings.t_eval_ms * timings.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %8.2f ms\n", __func__, (timings.t_end_ms - timings.t_start_ms));
}

void llama_reset_timings(struct llama_context * ctx) {
    ctx->t_start_us = ggml_time_us();
    ctx->t_sample_us = ctx->n_sample = 0;
    ctx->t_eval_us   = ctx->n_eval   = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "         + std::to_string(ggml_cpu_has_avx())         + " | ";
    s += "AVX2 = "        + std::to_string(ggml_cpu_has_avx2())        + " | ";
    s += "AVX512 = "      + std::to_string(ggml_cpu_has_avx512())      + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
    s += "FMA = "         + std::to_string(ggml_cpu_has_fma())         + " | ";
    s += "NEON = "        + std::to_string(ggml_cpu_has_neon())        + " | ";
    s += "ARM_FMA = "     + std::to_string(ggml_cpu_has_arm_fma())     + " | ";
    s += "F16C = "        + std::to_string(ggml_cpu_has_f16c())        + " | ";
    s += "FP16_VA = "     + std::to_string(ggml_cpu_has_fp16_va())     + " | ";
    s += "WASM_SIMD = "   + std::to_string(ggml_cpu_has_wasm_simd())   + " | ";
    s += "BLAS = "        + std::to_string(ggml_cpu_has_blas())        + " | ";
    s += "SSE3 = "        + std::to_string(ggml_cpu_has_sse3())        + " | ";
    s += "VSX = "         + std::to_string(ggml_cpu_has_vsx())         + " | ";

    return s.c_str();
}

// For internal test use
const std::vector<std::pair<std::string, struct ggml_tensor *>>& llama_internal_get_tensor_map(struct llama_context * ctx) {
    return ctx->model.tensors_by_name;
}


void llama_log_set(llama_log_callback log_callback, void * user_data) {
    g_state.log_callback = log_callback ? log_callback : llama_log_callback_default;
    g_state.log_callback_user_data = user_data;
}

#if defined(_MSC_VER) && !defined(vsnprintf)
#define vsnprintf _vsnprintf
#endif

static void llama_log_internal_v(llama_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args_copy);
        buffer2[len] = 0;
        g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}

static void llama_log_internal(llama_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, args);
    va_end(args);
}

static void llama_log_callback_default(llama_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}
