#include "llama.h"

#include "ggml.h"

#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <regex>
#include <cassert>
#include <cstring>

#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#endif

#define Min(X, Y) ((Y) > (X) ? (X) : (Y))
#define Max(X, Y) ((Y) < (X) ? (X) : (Y))

#define LLAMA_USE_SCRATCH
#define LLAMA_MAX_SCRATCH_BUFFERS 16

#define LLAMA_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "LLAMA_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)


// determine number of model parts based on the dimension
static const std::unordered_map<int, int> LLAMA_N_PARTS = {
    { 4096, 1 },
    { 5120, 2 },
    { 6656, 4 },
    { 8192, 8 },
};

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_7B,
    MODEL_13B,
    MODEL_30B,
    MODEL_65B,
};

static const size_t MB = 1024*1024;

// computed for n_ctx == 2048
// TODO: dynamically determine these sizes
//       needs modifications in ggml

static const std::map<e_model, size_t> MEM_REQ_SCRATCH0 = {
    { MODEL_7B,    512ull*MB },
    { MODEL_13B,   512ull*MB },
    { MODEL_30B,   512ull*MB },
    { MODEL_65B,   512ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_SCRATCH1 = {
    { MODEL_7B,    512ull*MB },
    { MODEL_13B,   512ull*MB },
    { MODEL_30B,   512ull*MB },
    { MODEL_65B,   512ull*MB },
};

// 2*n_embd*n_ctx*n_layer*sizeof(float16)
static const std::map<e_model, size_t> MEM_REQ_KV_SELF = {
    { MODEL_7B,   1026ull*MB },
    { MODEL_13B,  1608ull*MB },
    { MODEL_30B,  3124ull*MB },
    { MODEL_65B,  5120ull*MB },
};

// this is mostly needed for temporary mul_mat buffers to dequantize the data
// not actually needed if BLAS is disabled
static const std::map<e_model, size_t> MEM_REQ_EVAL = {
    { MODEL_7B,   768ull*MB },
    { MODEL_13B, 1024ull*MB },
    { MODEL_30B, 1280ull*MB },
    { MODEL_65B, 1536ull*MB },
};

// default hparams (LLaMA 7B)
struct llama_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
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
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx;

    std::vector<uint8_t> buf;

    int n; // number of tokens currently in the cache
};

struct llama_model {
    e_model type = MODEL_UNKNOWN;

    llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    // context
    struct ggml_context * ctx;

    // key + value cache for the self attention
    // TODO: move to llama_state
    struct llama_kv_cache kv_self;

    // the model memory buffer
    std::vector<uint8_t> buf;

    // model memory mapped file
    void * mm_addr = NULL;
    uint64_t mm_length = 0;

    // tensors
    int n_loaded;
    std::unordered_map<std::string, struct ggml_tensor *> tensors;
};

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

struct llama_context {
    std::mt19937 rng;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;
    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    llama_model model;
    llama_vocab vocab;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // memory buffers used to evaluate the model
    // TODO: move in llama_state
    std::vector<uint8_t> buf_compute;
    std::vector<uint8_t> buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = { 0 };

    void use_buf(struct ggml_context * ctx, int i) {
#if defined(LLAMA_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size(), buf.data(), });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = Max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(LLAMA_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};

//
// kv cache
//

static bool kv_cache_init(
        const struct llama_hparams & hparams,
             struct llama_kv_cache & cache,
                         ggml_type   wtype,
                               int   n_ctx) {
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;

    const int64_t n_mem      = (int64_t)n_layer*n_ctx;
    const int64_t n_elements = n_embd*n_mem;

    cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);

    struct ggml_init_params params;
    params.mem_size   = cache.buf.size();
    params.mem_buffer = cache.buf.data();
    params.no_alloc   = false;

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);

    return true;
}

static void kv_cache_free(struct llama_kv_cache & cache) {
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
}

struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_parts                     =*/ -1,
        /*.seed                        =*/ 0,
        /*.f16_kv                      =*/ false,
        /*.logits_all                  =*/ false,
        /*.vocab_only                  =*/ false,
        /*.use_mlock                   =*/ false,
        /*.embedding                   =*/ false,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
    };

    return result;
}

//
// model loading
//

static void *mmap_file(const char *fname, uint64_t *mm_length) {
#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
    HANDLE hFile = CreateFileA(fname,
                               GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED,
                               NULL);
    if (hFile == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER fileSize;
    fileSize.QuadPart = -1;
    GetFileSizeEx(hFile, &fileSize);
    int64_t length = fileSize.QuadPart;
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);
    if (!hMapping) return 0;
    void *addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    if (!addr) return 0;
#else
    int fd = open(fname, O_RDONLY);
    if (fd == -1) return 0;
    int64_t length = lseek(fd, 0, SEEK_END);
    void *addr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) return 0;
#endif
    *mm_length = length;
    return addr;
}

static void munmap_file(void * addr, size_t length) {
#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
    UnmapViewOfFile(addr);
#else
    munmap(addr, length);
#endif
}

static bool report_bad_magic(const char *path, uint32_t got, uint32_t want) {
    fprintf(stderr,
            "%s: invalid model file (bad magic [got %#x want %#x])\n"
            "\tyou most likely need to regenerate your ggml files\n"
            "\tthe benefit is you'll get 10-100x faster load times\n"
            "\tsee https://github.com/ggerganov/llama.cpp/issues/91\n"
            "\tuse convert-pth-to-ggml.py to regenerate from original pth\n"
            "\tuse migrate-ggml-2023-03-30-pr613.py if you deleted originals\n",
            path, got, want);
    return false;
}

static bool llama_model_load(
        const std::string & fname,
        llama_context & lctx,
        int n_ctx,
        int n_parts,
        ggml_type memory_type,
        bool vocab_only,
        llama_progress_callback progress_callback,
        void *progress_callback_user_data) {
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    lctx.t_start_us = ggml_time_us();

    auto & model = lctx.model;
    auto & vocab = lctx.vocab;

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    std::vector<char> f_buf(1024*1024);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());

    fin.seekg(0, fin.end);
    const size_t file_size = fin.tellg();
    fin.seekg(0);

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic == LLAMA_FILE_MAGIC_UNVERSIONED) {
            fprintf(stderr, "%s: invalid model file '%s' (too old, regenerate your model files or convert them with convert-unversioned-ggml-to-ggml.py!)\n",
                    __func__, fname.c_str());
            return false;
        }
        if (magic != LLAMA_FILE_MAGIC) {
            return report_bad_magic(fname.c_str(), magic, LLAMA_FILE_MAGIC);
        }

        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != LLAMA_FILE_VERSION) {
            fprintf(stderr, "%s: invalid model file '%s' (unsupported format version %" PRIu32 ", expected %d)\n",
                    __func__, fname.c_str(), format_version, LLAMA_FILE_VERSION);
            return false;
        }
    }

    int n_ff = 0;

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        hparams.n_ctx = n_ctx;

        n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;

        if (n_parts < 1) {
            n_parts = LLAMA_N_PARTS.at(hparams.n_embd);
        }

        // temp warning to tell the user to use "--n_parts"
        if (hparams.f16 == 4 && n_parts != 1) {
            fprintf(stderr, "%s: GPTQ model detected - are you sure n_parts should be %d? we normally expect it to be 1\n", __func__, n_parts);
            fprintf(stderr, "%s: use '--n_parts 1' if necessary\n", __func__);
        }

        if (hparams.n_layer == 32) {
            model.type = e_model::MODEL_7B;
        }

        if (hparams.n_layer == 40) {
            model.type = e_model::MODEL_13B;
        }

        if (hparams.n_layer == 60) {
            model.type = e_model::MODEL_30B;
        }

        if (hparams.n_layer == 80) {
            model.type = e_model::MODEL_65B;
        }

        fprintf(stderr, "%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        fprintf(stderr, "%s: n_embd  = %d\n", __func__, hparams.n_embd);
        fprintf(stderr, "%s: n_mult  = %d\n", __func__, hparams.n_mult);
        fprintf(stderr, "%s: n_head  = %d\n", __func__, hparams.n_head);
        fprintf(stderr, "%s: n_layer = %d\n", __func__, hparams.n_layer);
        fprintf(stderr, "%s: n_rot   = %d\n", __func__, hparams.n_rot);
        fprintf(stderr, "%s: f16     = %d\n", __func__, hparams.f16);
        fprintf(stderr, "%s: n_ff    = %d\n", __func__, n_ff);
        fprintf(stderr, "%s: n_parts = %d\n", __func__, n_parts);
        fprintf(stderr, "%s: type    = %d\n", __func__, model.type);
    }

    // load vocab
    {
        std::string word;
        vocab.id_to_token.resize(model.hparams.n_vocab);
        std::vector<char> tmp(64);

        for (int i = 0; i < model.hparams.n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            if (len > 0) {
                tmp.resize(len);
                fin.read(tmp.data(), len);
                word.assign(tmp.data(), len);
            } else {
                word.clear();
            }

            float score;
            fin.read((char *) &score, sizeof(score));

            vocab.token_to_id[word] = i;

            auto &tok_score = vocab.id_to_token[i];
            tok_score.tok = word;
            tok_score.score = score;
        }
    }

    if (vocab_only) {
        return true;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    // wtype is for per-layer weights, while vtype is for other weights
    ggml_type wtype, vtype;
    switch (model.hparams.f16) {
        case 0: wtype = vtype = GGML_TYPE_F32;  break;
        case 1: wtype = vtype = GGML_TYPE_F16;  break;
        case 2: wtype = vtype = GGML_TYPE_Q4_0; break;
        case 3: wtype = vtype = GGML_TYPE_Q4_1; break;
        case 4: wtype = GGML_TYPE_Q4_1; vtype = GGML_TYPE_F16; break;
        default:
                {
                    fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                            __func__, fname.c_str(), model.hparams.f16);
                    return false;
                }
    }

    // map model into memory
    char *mm_addr = NULL;
    model.mm_addr = mmap_file(fname.c_str(), &model.mm_length);
    if (model.mm_addr == NULL) {
        fprintf(stderr, "%s: failed to mmap '%s'\n", __func__, fname.c_str());
        return false;
    }
    mm_addr = (char *)model.mm_addr;
    fprintf(stderr, "%s: ggml map size = %6.2f MB\n", __func__, model.mm_length/(1024.0*1024.0));

    auto & ctx = model.ctx;

    size_t ctx_size = 0;
    {
        const auto &hparams = model.hparams;
        const int n_layer = hparams.n_layer;
        ctx_size += (5 + 10*n_layer)*256; // object overhead
        fprintf(stderr, "%s: ggml ctx size = %6.2f KB\n", __func__, ctx_size/1024.0);
    }

    // print memory requirements
    {
        const size_t scale = memory_type == GGML_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        const size_t mem_required =
            ctx_size +
            model.mm_length +
            MEM_REQ_SCRATCH0.at(model.type) +
            MEM_REQ_SCRATCH1.at(model.type) +
            MEM_REQ_EVAL.at    (model.type);

        // this is the memory required by one llama_state
        const size_t mem_required_state =
            scale*MEM_REQ_KV_SELF.at(model.type);

        fprintf(stderr, "%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);
    }

    // create the ggml context
    {
        lctx.model.buf.resize(ctx_size);

        struct ggml_init_params params = {
            /*.mem_size   =*/ lctx.model.buf.size(),
            /*.mem_buffer =*/ lctx.model.buf.data(),
            /*.no_alloc   =*/ true,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.tok_embeddings = ggml_new_tensor_2d(ctx, vtype, n_embd, n_vocab);

        model.norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.output = ggml_new_tensor_2d(ctx, vtype,         n_embd, n_vocab);

        // map by name
        model.tensors["tok_embeddings.weight"] = model.tok_embeddings;

        model.tensors["norm.weight"]   = model.norm;
        model.tensors["output.weight"] = model.output;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.wq = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wk = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wv = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wo = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.w1 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);
            layer.w2 = ggml_new_tensor_2d(ctx, wtype,   n_ff, n_embd);
            layer.w3 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);

            // map by name
            model.tensors["layers." + std::to_string(i) + ".attention_norm.weight"] = layer.attention_norm;

            model.tensors["layers." + std::to_string(i) + ".attention.wq.weight"] = layer.wq;
            model.tensors["layers." + std::to_string(i) + ".attention.wk.weight"] = layer.wk;
            model.tensors["layers." + std::to_string(i) + ".attention.wv.weight"] = layer.wv;
            model.tensors["layers." + std::to_string(i) + ".attention.wo.weight"] = layer.wo;

            model.tensors["layers." + std::to_string(i) + ".ffn_norm.weight"] = layer.ffn_norm;

            model.tensors["layers." + std::to_string(i) + ".feed_forward.w1.weight"] = layer.w1;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w2.weight"] = layer.w2;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w3.weight"] = layer.w3;
        }
    }

    std::vector<uint8_t> tmp;

    if (progress_callback) {
        progress_callback(0.0, progress_callback_user_data);
    }

    fprintf(stderr, "%s: loading tensors from '%s'\n", __func__, fname.c_str());

    // load weights
    {
        size_t total_size = 0;
        model.n_loaded = 0;

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

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];

            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }
            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%" PRId64 ", %" PRId64 "], expected [%d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }
            if (0) {
                static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                fprintf(stderr, "%24s - [%5d, %5d], type = %6s\n", name.data(), ne[0], ne[1], ftype_str[ftype]);
            }

            switch (ftype) {
                case 0:  // f32
                case 1:  // f16
                    break;
                case 2:  // q4_0
                case 3:  // q4_1
                    assert(ne[0] % 64 == 0);
                    break;
                default:
                    fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                    return false;
            };

            // load the tensor data into memory without copying or reading it
            size_t offset = fin.tellg();
            size_t tensor_data_size = ggml_nbytes(tensor);
            offset = (offset + 31) & -32;
            tensor->data = mm_addr + offset;
            fin.seekg(offset + tensor_data_size);
            total_size += tensor_data_size;
            model.n_loaded++;

            // progress
            if (progress_callback) {
                double current_progress = size_t(fin.tellg()) / double(file_size);
                progress_callback(current_progress, progress_callback_user_data);
            }
        }

        fin.close();

        fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, model.n_loaded);
        if (model.n_loaded == 0) {
            fprintf(stderr, "%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
        } else if (model.n_loaded != (int) model.tensors.size()) {
            fprintf(stderr, "%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
            return false;
        }
    }

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    lctx.t_load_us = ggml_time_us() - lctx.t_start_us;

    if (progress_callback) {
        progress_callback(1.0, progress_callback_user_data);
    }

    return true;
}

// evaluate the transformer
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool llama_eval_internal(
        llama_context & lctx,
    const llama_token * tokens,
            const int   n_tokens,
            const int   n_past,
            const int   n_threads) {
    const int64_t t_start_us = ggml_time_us();

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    auto & kv_self = model.kv_self;

    LLAMA_ASSERT(!!kv_self.ctx);

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_embd/hparams.n_head;

    auto & mem_per_token = lctx.mem_per_token;
    auto & buf_compute   = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size(),
        /*.mem_buffer =*/ buf_compute.data(),
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    ggml_cgraph gf = {};
    gf.n_threads = N >= 32 && ggml_cpu_has_blas() ? 1 : n_threads;

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, tokens, N*ggml_element_size(embd));

    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.tok_embeddings, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        lctx.use_buf(ctx0, 0);

        // norm
        {
            cur = ggml_rms_norm(ctx0, inpL);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].attention_norm, cur),
                        cur);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, kv_self.v, N*n_embd, (ggml_element_size(kv_self.v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_rope(ctx0,
                            ggml_cpy(ctx0,
                                Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                            n_past, n_rot, 0),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_rope(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, kv_self.k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kv_self.k)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            n_past, n_rot, 1),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head)));

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans =
                ggml_cpy(ctx0,
                    ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, kv_self.v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kv_self.v)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            1, 2, 0, 3),
                    ggml_new_tensor_3d(ctx0, kv_self.v->type, n_past + N, n_embd/n_head, n_head));

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
        }

        lctx.use_buf(ctx0, 1);

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF);

                // cur = ffn_norm*cur
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ffn_norm, cur),
                        cur);
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);

            // SILU activation
            cur = ggml_silu(ctx0, cur);

            cur = ggml_mul(ctx0, cur, tmp);

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
        }

        cur = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    lctx.use_buf(ctx0, 0);

    // used at the end to optionally extract the embeddings
    struct ggml_tensor * embeddings = NULL;

    // norm
    {

        inpL = ggml_rms_norm(ctx0, inpL);

        // inpL = norm*inpL
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.norm, inpL),
                    inpL);

        embeddings = inpL;
    }

    // lm_head
    inpL = ggml_mul_mat(ctx0, model.output, inpL);

    lctx.use_buf(ctx0, -1);

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // extract logits
    {
        auto & logits_out = lctx.logits;

        if (lctx.logits_all) {
            logits_out.resize(n_vocab * N);
            memcpy(logits_out.data(), (float *) ggml_get_data(inpL), sizeof(float)*n_vocab*N);
        } else {
            // return result for just the last token
            logits_out.resize(n_vocab);
            memcpy(logits_out.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
        }
    }

    // extract embeddings
    if (lctx.embedding.size()) {
        auto & embedding_out = lctx.embedding;

        embedding_out.resize(n_embd);
        memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
    }

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }

#if 0
    printf("\n%s: used_mem = %.3f MB, scratch -- %.3f MB %.3f MB\n", __func__,
            ggml_used_mem(ctx0)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0);
#endif

    ggml_free(ctx0);

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
            size_t char_len = Min(text.size() - offs, utf8_len(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(std::move(sym));
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

            //printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

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
                    llama_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
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

    if (text.size() == 0) {
        return output;
    }

    if (bos) {
        output.push_back(1);
    }

    tokenizer.tokenize(text, output);
    return output;
}

//
// sampling
//

static void sample_top_k(std::vector<std::pair<float, llama_vocab::id>> & logits_id, int top_k) {
    // find the top k tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<float, llama_vocab::id> & a, const std::pair<float, llama_vocab::id> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);
}

static llama_vocab::id llama_sample_top_p_top_k(
        llama_context & lctx,
        const std::vector<llama_vocab::id> & last_n_tokens,
        int top_k,
        float top_p,
        float temp,
        float repeat_penalty) {
    auto & rng = lctx.rng;

    const int n_logits = lctx.model.hparams.n_vocab;

    const auto & logits = lctx.logits;
    const auto * plogits = logits.data() + logits.size() - n_logits;

    if (temp <= 0) {
        // select the token with the highest logit directly
        float max_logit = plogits[0];
        llama_vocab::id max_id = 0;

        for (int i = 1; i < n_logits; ++i) {
            if (plogits[i] > max_logit) {
                max_logit = plogits[i];
                max_id = i;
            }
        }
        return max_id;
    }

    std::vector<std::pair<float, llama_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const float scale = 1.0f/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (plogits[i] < 0.0f) {
                    logits_id.push_back(std::make_pair(plogits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(plogits[i]*scale/repeat_penalty, i));
                }
            } else {
                logits_id.push_back(std::make_pair(plogits[i]*scale, i));
            }
        }
    }

    sample_top_k(logits_id, top_k);

    float maxl = -std::numeric_limits<float>::infinity();
    for (const auto & kv : logits_id) {
        maxl = Max(maxl, kv.first);
    }

    // compute probs for the top k tokens
    std::vector<float> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        const float p = expf(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0) {
        double cumsum = 0.0;
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) 10; i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //printf("\n\n");
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}

//
// quantization
//

// TODO: reuse code from the llama_model_load() somehow
static bool llama_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, int itype) {
    ggml_type type = GGML_TYPE_Q4_1;

    switch (itype) {
        case 2: type = GGML_TYPE_Q4_0; break;
        case 3: type = GGML_TYPE_Q4_1; break;
        default: fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype); return 1;
    };

    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1) {
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, type);
        return false;
    }

    llama_vocab vocab;

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic == LLAMA_FILE_MAGIC_UNVERSIONED) {
            fprintf(stderr, "%s: invalid model file '%s' (too old, regenerate your model files!)\n",
                    __func__, fname_inp.c_str());
            return false;
        }
        if (magic != LLAMA_FILE_MAGIC) {
            return report_bad_magic(fname_inp.c_str(), magic, LLAMA_FILE_MAGIC);
        }

        fout.write((char *) &magic, sizeof(magic));

        uint32_t format_version;
        finp.read((char *) &format_version, sizeof(format_version));

        if (format_version != LLAMA_FILE_VERSION) {
            fprintf(stderr, "%s: invalid model file '%s' (unsupported format version %" PRIu32 ", expected %d)\n",
                    __func__, fname_inp.c_str(), format_version, LLAMA_FILE_VERSION);
            return false;
        }

        fout.write((char *) &format_version, sizeof(format_version));
    }

    llama_hparams hparams;

    // load hparams
    {
        finp.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //finp.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        finp.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        finp.read((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        finp.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        finp.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        finp.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        finp.read((char *) &hparams.f16,     sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_mult  = %d\n", __func__, hparams.n_mult);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: f16     = %d\n", __func__, hparams.f16);

        fout.write((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //fout.write((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fout.write((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fout.write((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        fout.write((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fout.write((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fout.write((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fout.write((char *) &itype,           sizeof(hparams.f16));
    }

    // load vocab
    {
        const int32_t n_vocab = hparams.n_vocab;

        if (n_vocab != hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname_inp.c_str(), n_vocab, hparams.n_vocab);
            return false;
        }

        std::vector<char> word(32);
        vocab.id_to_token.resize(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read ((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            word.resize(len);
            finp.read ((char *) &word[0], len);
            fout.write((char *) &word[0], len);

            float score;
            finp.read ((char *) &score, sizeof(score));
            fout.write((char *) &score, sizeof(score));

            vocab.token_to_id[word.data()] = i;

            auto &tok_score = vocab.id_to_token[i];
            tok_score.tok = word.data();
            tok_score.score = score;
        }
    }

    // load weights
    {
        size_t total_size_org = 0;
        size_t total_size_new = 0;

        std::vector<float> work;

        std::vector<uint8_t>     data_u8;
        std::vector<ggml_fp16_t> data_f16;
        std::vector<float>       data_f32;

        std::vector<int64_t> hist_all(1 << 4, 0);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            finp.read(reinterpret_cast<char *>(&length), sizeof(length));
            finp.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (finp.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            finp.read (&name[0], length);

            {
                // ensure tensor data is aligned
                uint64_t offset = finp.tellg();
                offset = (offset + 31) & -32;
                finp.seekg(offset);
            }

            {
                static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                printf("%48s - [%5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ftype_str[ftype]);
            }

            // regexes of tensor names to be quantized
            const std::vector<std::string> k_names = {
                ".*weight",
            };

            bool quantize = false;
            for (const auto & s : k_names) {
                if (std::regex_match(name, std::regex(s))) {
                    quantize = true;
                    break;
                }
            }

            // quantize only 2D tensors
            quantize &= (n_dims == 2);

            if (quantize) {
                if (ftype != 0 && ftype != 1) {
                    fprintf(stderr, "%s: unsupported ftype %d for integer quantization\n", __func__, ftype);
                    return false;
                }

                if (ftype == 1) {
                    data_f16.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                    data_f32.resize(nelements);
                    for (int i = 0; i < nelements; ++i) {
                        data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                    }
                } else {
                    data_f32.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
                }

                ftype = itype;
            } else {
                const int bpe = (ftype == 0) ? sizeof(float) : sizeof(uint16_t);

                data_u8.resize(nelements*bpe);
                finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
            }

            fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fout.write(reinterpret_cast<char *>(&length), sizeof(length));
            fout.write(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
            for (int i = 0; i < n_dims; ++i) {
                fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            }
            fout.write(&name[0], length);

            {
                // ensure tensor data is aligned
                uint64_t offset = fout.tellp();
                offset = (offset + 31) & -32;
                fout.seekp(offset);
            }

            if (quantize) {
                printf("quantizing .. ");
                work.resize(nelements); // for quantization

                size_t cur_size = 0;
                std::vector<int64_t> hist_cur(1 << 4, 0);

                switch (type) {
                    case GGML_TYPE_Q4_0:
                        {
                            cur_size = ggml_quantize_q4_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                        } break;
                    case GGML_TYPE_Q4_1:
                        {
                            cur_size = ggml_quantize_q4_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                        } break;
                    default:
                        {
                            fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, type);
                            return false;
                        }
                }

                fout.write(reinterpret_cast<char *>(work.data()), cur_size);
                total_size_new += cur_size;

                printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
                for (int i = 0; i < (int) hist_cur.size(); ++i) {
                    hist_all[i] += hist_cur[i];
                }

                for (int i = 0; i < (int) hist_cur.size(); ++i) {
                    printf("%5.3f ", hist_cur[i] / float(nelements));
                }
                printf("\n");
            } else {
                printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
                fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
                total_size_new += data_u8.size();
            }

            total_size_org += nelements * sizeof(float);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
        printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

        {
            int64_t sum_all = 0;
            for (int i = 0; i < (int) hist_all.size(); ++i) {
                sum_all += hist_all[i];
            }

            printf("%s: hist: ", __func__);
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

//
// interface implementation
//

struct llama_context * llama_init_from_file(
                             const char * path_model,
            struct llama_context_params   params) {
    ggml_time_init();

    llama_context * ctx = new llama_context;

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    ctx->rng = std::mt19937(params.seed);
    ctx->logits_all = params.logits_all;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!llama_model_load(path_model, *ctx, params.n_ctx, params.n_parts, memory_type,
                          params.vocab_only, params.progress_callback,
                          params.progress_callback_user_data)) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        llama_free(ctx);
        return nullptr;
    }

    if (params.use_mlock) {
        char *err;
        if (!ggml_mlock(ctx->model.ctx,
                        ctx->model.mm_addr,
                        ctx->model.mm_length,
                        &err)) {
            fprintf(stderr, "%s\n", err);
            free(err);
            llama_free(ctx);
            return nullptr;
        }
    }

    // reserve memory for context buffers
    if (!params.vocab_only) {
        if (!kv_cache_init(ctx->model.hparams, ctx->model.kv_self, memory_type, ctx->model.hparams.n_ctx)) {
            fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_nbytes(ctx->model.kv_self.k) + ggml_nbytes(ctx->model.kv_self.v);
            fprintf(stderr, "%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
        }

        const auto & hparams = ctx->model.hparams;

        // resized during inference
        if (params.logits_all) {
            ctx->logits.reserve(hparams.n_ctx*hparams.n_vocab);
        } else {
            ctx->logits.reserve(hparams.n_ctx);
        }

        if (params.embedding){
            ctx->embedding.resize(hparams.n_embd);
        }

        ctx->buf_compute.resize(MEM_REQ_EVAL.at(ctx->model.type));

        ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0.at(ctx->model.type));
        ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1.at(ctx->model.type));
    }

    return ctx;
}

void llama_free(struct llama_context * ctx) {
    kv_cache_free(ctx->model.kv_self);

    if (ctx->model.ctx) {
        ggml_free(ctx->model.ctx);
    }

    if (ctx->model.mm_addr) {
        munmap_file(ctx->model.mm_addr, ctx->model.mm_length);
    }

    delete ctx;
}

int llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
               int   itype) {
    if (!llama_model_quantize_internal(fname_inp, fname_out, itype)) {
        fprintf(stderr, "%s: failed to quantize\n", __func__);
        return 1;
    }

    return 0;
}

// Returns the KV cache that will contain the context for the
// ongoing prediction with the model.
const uint8_t * llama_get_kv_cache(struct llama_context * ctx) {
    return ctx->model.kv_self.buf.data();
}

// Returns the size of the KV cache
size_t llama_get_kv_cache_size(struct llama_context * ctx) {
    return ctx->model.kv_self.buf.size();
}

int llama_get_kv_cache_token_count(struct llama_context * ctx) {
    return ctx->model.kv_self.n;
}

// Sets the KV cache containing the current context for the model
void llama_set_kv_cache(
        struct llama_context * ctx,
               const uint8_t * kv_cache,
                      size_t   n_size,
                         int   n_token_count) {
    // Make sure we have the same kv cache setup
    LLAMA_ASSERT(ctx->model.kv_self.buf.size() == n_size);
    memcpy(ctx->model.kv_self.buf.data(), kv_cache, n_size);
    ctx->model.kv_self.n = n_token_count;
}

int llama_eval(
        struct llama_context * ctx,
           const llama_token * tokens,
                         int   n_tokens,
                         int   n_past,
                         int   n_threads) {
    if (!llama_eval_internal(*ctx, tokens, n_tokens, n_past, n_threads)) {
        fprintf(stderr, "%s: failed to eval\n", __func__);
        return 1;
    }
    // get a more accurate load time, upon first eval
    if (!ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }
    return 0;
}

int llama_tokenize(
        struct llama_context * ctx,
                  const char * text,
                 llama_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto res = llama_tokenize(ctx->vocab, text, add_bos);

    if (n_max_tokens < (int) res.size()) {
        fprintf(stderr, "%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int llama_n_vocab(struct llama_context * ctx) {
    return ctx->vocab.id_to_token.size();
}

int llama_n_ctx(struct llama_context * ctx) {
    return ctx->model.hparams.n_ctx;
}

int llama_n_embd(struct llama_context * ctx) {
    return ctx->model.hparams.n_embd;
}

float * llama_get_logits(struct llama_context * ctx) {
    return ctx->logits.data();
}

float * llama_get_embeddings(struct llama_context * ctx) {
    return ctx->embedding.data();
}

const char * llama_token_to_str(struct llama_context * ctx, llama_token token) {
    if (token >= llama_n_vocab(ctx)) {
        return nullptr;
    }

    return ctx->vocab.id_to_token[token].tok.c_str();
}

llama_token llama_token_bos() {
    return 1;
}

llama_token llama_token_eos() {
    return 2;
}

llama_token llama_sample_top_p_top_k(
          llama_context * ctx,
      const llama_token * last_n_tokens_data,
                    int   last_n_tokens_size,
                    int   top_k,
                  float   top_p,
                  float   temp,
                  float   repeat_penalty) {
    const int64_t t_start_sample_us = ggml_time_us();

    llama_token result = 0;

    // TODO: avoid this ...
    const auto last_n_tokens = std::vector<llama_token>(last_n_tokens_data, last_n_tokens_data + last_n_tokens_size);

    result = llama_sample_top_p_top_k(
            *ctx,
            last_n_tokens,
            top_k,
            top_p,
            temp,
            repeat_penalty);

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;

    return result;
}


void llama_print_timings(struct llama_context * ctx) {
    const int64_t t_end_us = ggml_time_us();

    const int32_t n_sample = Max(1, ctx->n_sample);
    const int32_t n_eval   = Max(1, ctx->n_eval);
    const int32_t n_p_eval = Max(1, ctx->n_p_eval);

    fprintf(stderr, "\n");
    fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0);
    fprintf(stderr, "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * ctx->t_sample_us, n_sample, 1e-3 * ctx->t_sample_us / n_sample);
    fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__, 1e-3 * ctx->t_p_eval_us, n_p_eval, 1e-3 * ctx->t_p_eval_us / n_p_eval);
    fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * ctx->t_eval_us,   n_eval,   1e-3 * ctx->t_eval_us   / n_eval);
    fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us)/1000.0);
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
