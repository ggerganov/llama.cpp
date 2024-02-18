#include "ggml.h"
#include "llama.h"
#include "common.h"

#include <unordered_map>
#include <vector>
#include <cassert>
#include <climits>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <random>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <string>

// GGUF keys & tensor names.

#define KV_GENERAL_ARCHITECTURE          "general.architecture"
#define KV_GENERAL_NAME                  "general.name"

#define KV_TOKENIZER_MODEL               "tokenizer.ggml.model"
#define KV_TOKENIZER_LIST                "tokenizer.ggml.tokens"
#define KV_TOKENIZER_TOKEN_TYPE          "tokenizer.ggml.token_type"
#define KV_TOKENIZER_SCORES              "tokenizer.ggml.scores"
#define KV_TOKENIZER_BOS_ID              "tokenizer.ggml.bos_token_id"
#define KV_TOKENIZER_EOS_ID              "tokenizer.ggml.eos_token_id"
#define KV_TOKENIZER_UNK_ID              "tokenizer.ggml.unknown_token_id"
#define KV_TOKENIZER_SEP_ID              "tokenizer.ggml.seperator_token_id"
#define KV_TOKENIZER_PAD_ID              "tokenizer.ggml.padding_token_id"
#define KV_TOKENIZER_HF_JSON             "tokenizer.huggingface.json"

#define KV_CONTEXT_LENGTH                "llama.context_length"
#define KV_EMBEDDING_LENGTH              "llama.embedding_length"
#define KV_BLOCK_COUNT                   "llama.block_count"
#define KV_FEED_FORWARD_LENGTH           "llama.feed_forward_length"
#define KV_ATTENTION_HEAD_COUNT          "llama.attention.head_count"
#define KV_ATTENTION_HEAD_COUNT_KV       "llama.attention.head_count_kv"
#define KV_ATTENTION_LAYERNORM_RMS_EPS   "llama.attention.layer_norm_rms_epsilon"
#define KV_ROPE_DIMENSION_COUNT          "llama.rope.dimension_count"

#define TN_TOKEN_EMBD  "token_embd.weight"
#define TN_OUTPUT_NORM "output_norm.weight"
#define TN_OUTPUT      "output.weight"
#define TN_ATTN_NORM   "blk.%d.attn_norm.weight"
#define TN_ATTN_Q      "blk.%d.attn_q.weight"
#define TN_ATTN_K      "blk.%d.attn_k.weight"
#define TN_ATTN_V      "blk.%d.attn_v.weight"
#define TN_ATTN_OUTPUT "blk.%d.attn_output.weight"
#define TN_FFN_NORM    "blk.%d.ffn_norm.weight"
#define TN_FFN_GATE    "blk.%d.ffn_gate.weight"
#define TN_FFN_DOWN    "blk.%d.ffn_down.weight"
#define TN_FFN_UP      "blk.%d.ffn_up.weight"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define LLAMA_FILE_MAGIC_GGJT        0x67676a74u // 'ggjt'
#define LLAMA_FILE_VERSION_GGJT_V3   3

#define TOKENIZER_NAME "llama"
#define UNKNOWN_TOKEN_ID 0
#define BOS_TOKEN_ID 1
#define EOS_TOKEN_ID 2

//////////////////////////////////////// llama2.c model structs and functions to load models, alloc memory etc.
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

struct TransformerWeights {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float* wq; // (layer, dim, dim)
    float* wk; // (layer, dim, dim)
    float* wv; // (layer, dim, dim)
    float* wo; // (layer, dim, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    // float* freq_cis_real; // (seq_len, dim/2)
    // float* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;

    ~TransformerWeights() {
        delete[] token_embedding_table;
        delete[] rms_att_weight;
        delete[] rms_ffn_weight;
        delete[] wq;
        delete[] wk;
        delete[] wv;
        delete[] wo;
        delete[] w1;
        delete[] w2;
        delete[] w3;
        delete[] rms_final_weight;
        delete[] wcls;
    }
};

static void malloc_weights(TransformerWeights* w, Config* p, bool shared_weights) {
    // we calloc instead of malloc to keep valgrind happy
    w->token_embedding_table = new float[p->vocab_size * p->dim]();
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->token_embedding_table\n",__func__,p->vocab_size , p->dim, p->vocab_size * p->dim);

    w->rms_att_weight = new float[p->n_layers * p->dim]();
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->rms_att_weight\n",__func__,p->n_layers, p->dim, p->n_layers * p->dim);

    w->rms_ffn_weight = new float[p->n_layers * p->dim]();
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->rms_ffn_weight\n",__func__,p->n_layers , p->dim, p->n_layers * p->dim);

    w->wq = new float[p->n_layers * p->dim * p->dim]();
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wq\n",__func__,p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->wk = new float[p->n_layers * p->dim * p->dim]();
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wk\n",__func__,p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->wv = new float[p->n_layers * p->dim * p->dim]();
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wv\n",__func__, p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->wo = new float[p->n_layers * p->dim * p->dim]();
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wo\n",__func__,p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->w1 = new float[p->n_layers * p->hidden_dim * p->dim]();
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->w1\n",__func__,p->n_layers, p->hidden_dim, p->dim, p->n_layers * p->hidden_dim * p->dim);

    w->w2 = new float[p->n_layers * p->hidden_dim * p->dim]();
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->w2\n",__func__,p->n_layers, p->dim, p->hidden_dim, p->n_layers * p->hidden_dim * p->dim);

    w->w3 = new float[p->n_layers * p->hidden_dim * p->dim]();
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->w3\n",__func__,p->n_layers, p->hidden_dim, p->dim, p->n_layers * p->hidden_dim * p->dim);

    w->rms_final_weight = new float[p->dim]();
    printf("[%s:AK] Allocating [%d] float space for w->rms_final_weight\n",__func__,p->dim);

    if (shared_weights) {
        w->wcls = NULL;
    } else {
        w->wcls = new float[p->vocab_size * p->dim]();
        printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->wcls\n",__func__,p->vocab_size , p->dim, p->vocab_size * p->dim);
    }
}

static int checkpoint_init_weights(TransformerWeights *w, Config* p, FILE* f, bool shared_weights) {
    if (fread(w->token_embedding_table, sizeof(float), p->vocab_size * p->dim, f) != static_cast<size_t>(p->vocab_size * p->dim)) return 1;
    if (fread(w->rms_att_weight, sizeof(float), p->n_layers * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim)) return 1;
    if (fread(w->wq, sizeof(float), p->n_layers * p->dim * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->dim)) return 1;
    if (fread(w->wk, sizeof(float), p->n_layers * p->dim * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->dim)) return 1;
    if (fread(w->wv, sizeof(float), p->n_layers * p->dim * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->dim)) return 1;
    if (fread(w->wo, sizeof(float), p->n_layers * p->dim * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->dim)) return 1;
    if (fread(w->rms_ffn_weight, sizeof(float), p->n_layers * p->dim, f) != static_cast<size_t>(p->n_layers * p->dim)) return 1;
    if (fread(w->w1, sizeof(float), p->n_layers * p->dim * p->hidden_dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->hidden_dim)) return 1;
    if (fread(w->w2, sizeof(float), p->n_layers * p->hidden_dim * p->dim, f) != static_cast<size_t>(p->n_layers * p->hidden_dim * p->dim)) return 1;
    if (fread(w->w3, sizeof(float), p->n_layers * p->dim * p->hidden_dim, f) != static_cast<size_t>(p->n_layers * p->dim * p->hidden_dim)) return 1;
    if (fread(w->rms_final_weight, sizeof(float), p->dim, f) != static_cast<size_t>(p->dim)) return 1;

    // Skip freq_cis_real & freq_cis_imag
    int head_size = p->dim / p->n_heads;
    fseek(f, p->seq_len * head_size * sizeof(float), SEEK_CUR);

    if (!shared_weights && fread(w->wcls, sizeof(float), p->vocab_size * p->dim, f) != static_cast<size_t>(p->vocab_size * p->dim)) return 1;

    // Check we didn't forget to read anything
    auto curr = ftell(f);
    fseek(f, 0, SEEK_END);
    auto end = ftell(f);
    if (curr != end) {
        printf("Error: failed to read the checkpoint file to the end (curr = %ld, end =  %ld)\n", curr, end);
        return 1;
    }

    return 0;
}

static void print_sample_weights(TransformerWeights *w){
    printf("----- Quick print of first of the weight vales of all the variables\n");
    printf("%f\n", w->token_embedding_table[0]);
    printf("%f\n", w->rms_att_weight[0]);
    printf("%f\n", w->rms_ffn_weight[0]);

    printf("%f\n", w->wq[0]);
    printf("%f\n", w->wk[0]);
    printf("%f\n", w->wv[0]);
    printf("%f\n", w->wo[0]);
    printf("%f\n", w->w1[0]);
    printf("%f\n", w->w2[0]);
    printf("%f\n", w->w3[0]);
    printf("%f\n", w->rms_att_weight[0]);
    if (w->wcls) printf("%f\n", w->wcls[0]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////// ggml structs and functions required to load models, configs and save the model.

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;

    struct token_data {
        token text;
        float score;
        ttype type;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data> id_to_token;
};

struct my_llama_hparams {
    uint32_t n_vocab = 32000;
    uint32_t n_ctx   = 512;   // this is provided as user input?
    uint32_t n_embd  = 4096;
    uint32_t n_ff    = 11008;
    uint32_t n_mult  = 4;
    uint32_t n_head  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;
    bool operator!=(const my_llama_hparams& other) const {
        return memcmp(this, &other, sizeof(my_llama_hparams));
    }
};

struct my_llama_layer {
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

struct my_llama_model {
    struct ggml_context * ctx = NULL;

    std::string name;

    my_llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<my_llama_layer> layers;

    uint32_t train_its = 0;
    uint32_t train_samples = 0;
    uint32_t train_tokens = 0;
};

struct train_params {
    const char * fn_vocab_model;
    const char * fn_llama2c_model;
    const char * fn_llama2c_output_model;
    const char * fn_train_data;
    const char * fn_checkpoint_in;
    const char * fn_checkpoint_out;
    const char * fn_model_out;

    uint32_t seed;

    int n_ctx;
    int n_embd;
    int n_mult;
    int n_head;
    int n_layer;
    int n_rotmax;

    int n_threads;
    int n_batch;
    int n_examples;
    int n_predict;

    int print_info_interval;
    int print_details_interval;

    bool samples_start_after_nl;
    bool use_adam;
    bool use_flash;
    bool use_scratch;

    // only adam
    int   warmup;
    int   cos_decay_steps;
    float cos_decay_restart;
    float cos_decay_alpha;

    int   lbfgs_n_iter;
    int   adam_n_iter;
    float adam_alpha;
    float adam_decay;

    int mem_model_gb;
    int mem_compute_gb;
    int mem_compute0_gb;
    int mem_compute1_gb;
};

static void print_params(struct my_llama_hparams * params) {
    printf("%s: n_vocab: %u\n", __func__, params->n_vocab);
    printf("%s: n_ctx:   %u\n", __func__, params->n_ctx);
    printf("%s: n_embd:  %u\n", __func__, params->n_embd);
    printf("%s: n_mult:  %u\n", __func__, params->n_mult);
    printf("%s: n_head:  %u\n", __func__, params->n_head);
    printf("%s: n_ff:    %u\n", __func__, params->n_ff);
    printf("%s: n_layer: %u\n", __func__, params->n_layer);
    printf("%s: n_rot:   %u\n", __func__, params->n_rot);
}

static void init_model(struct my_llama_model * model) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;

    const uint32_t n_ff = hparams.n_ff;
    struct ggml_context * ctx = model->ctx;

    model->train_its = 0;
    model->train_samples = 0;
    model->train_tokens = 0;

    model->tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    printf("[%s:GG] Allocating [%u] x [%u] = [%u] float space for model->tok_embeddings\n",__func__,n_embd , n_vocab, n_embd * n_vocab);

    model->norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    printf("[%s:GG] Allocating [%u] float space for model->norm\n",__func__,n_embd);

    model->output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    printf("[%s:GG] Allocating [%u] x[%u] = [%u] float space for model->output\n",__func__,n_embd, n_vocab, n_embd * n_vocab);

    // printing the per-layer allocations here so we dont print in the for loop.
    printf("[%s:GG] Allocating [%u] x[%u] = [%u] float space for layer.wq for [%u] layers\n",__func__, n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%u] x[%u] = [%u] float space for layer.wk for [%u] layers\n",__func__, n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%u] x[%u] = [%u] float space for layer.wv for [%u] layers\n",__func__, n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%u] x[%u] = [%u] float space for layer.wo for [%u] layers\n",__func__, n_embd, n_embd, n_embd * n_embd, n_layer);

    printf("[%s:GG] Allocating [%u] float space for layer.ffn_norm for [%u] layers\n",__func__,n_embd, n_layer);

    printf("[%s:GG] Allocating [%u] x[%u] = [%u] float space for layer.w1 for [%u] layers\n",__func__, n_ff, n_embd, n_embd * n_ff, n_layer);
    printf("[%s:GG] Allocating [%u] x[%u] = [%u] float space for layer.w2 for [%u] layers\n",__func__, n_embd, n_ff, n_ff * n_embd, n_layer);
    printf("[%s:GG] Allocating [%u] x[%u] = [%u] float space for layer.w3 for [%u] layers\n",__func__, n_ff, n_embd, n_embd * n_ff, n_layer);

    ggml_set_name(model->tok_embeddings, "tok_embeddings.weight");
    ggml_set_name(model->norm,           "norm.weight");
    ggml_set_name(model->output,         "output.weight");

    model->layers.resize(n_layer);
    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];

        std::string layers_i = "layers." + std::to_string(i);

        layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        layer.wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
        layer.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);

        layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        layer.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
        layer.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_embd);
        layer.w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);

        ggml_set_name(layer.attention_norm, (layers_i + ".attention_norm.weight").c_str());

        ggml_set_name(layer.wq, (layers_i + ".attention.wq.weight").c_str());
        ggml_set_name(layer.wk, (layers_i + ".attention.wk.weight").c_str());
        ggml_set_name(layer.wv, (layers_i + ".attention.wv.weight").c_str());
        ggml_set_name(layer.wo, (layers_i + ".attention.wo.weight").c_str());

        ggml_set_name(layer.ffn_norm, (layers_i + ".ffn_norm.weight").c_str());

        ggml_format_name(layer.w1, "%s.feed_forward.w1.weight", layers_i.c_str());
        ggml_format_name(layer.w2, "%s.feed_forward.w2.weight", layers_i.c_str());
        ggml_format_name(layer.w3, "%s.feed_forward.w3.weight", layers_i.c_str());
    }
}

static float get_f32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1) {
    float * ptr = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    return *ptr;
}

static int32_t get_i32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1) {
    int32_t * ptr = (int32_t *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    return *ptr;
}

static void print_row(struct ggml_tensor * probs, int i) {
    for (int k = 0; k < probs->ne[0]; ++k) {
        float p = get_f32_2d(probs, k, i);
        printf(" %f", p);
    }
    printf("\n");
}

static void print_matrix(struct ggml_tensor * probs) {
    assert(ggml_is_matrix(probs));
    for (int i = 0; i < probs->ne[1]; ++i) {
        for (int k = 0; k < probs->ne[0]; ++k) {
            float p = get_f32_2d(probs, k, i);
            printf(" %.2f", p);
        }
        printf("\n");
    }
}

#ifdef __GNUC__
#ifdef __MINGW32__
__attribute__((format(gnu_printf, 1, 2)))
#else
__attribute__((format(printf, 1, 2)))
#endif
#endif
static std::string format(const char * fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            size = 0;
        } else {
            seek(0, SEEK_END);
            size = tell();
            seek(0, SEEK_SET);
        }
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    void read_raw(void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, size, 1, fp);
        if (ferror(fp)) {
            die_fmt("fread failed: %s", strerror(errno));
        }
        if (ret != 1) {
            die("unexpectedly reached end of file");
        }
    }

    std::uint32_t read_u32() {
        std::uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }
    std::float_t read_f32() {
        std::float_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    std::string read_string(std::uint32_t len) {
        std::vector<char> chars(len);
        read_raw(chars.data(), len);
        return std::string(chars.data(), len);
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

static bool is_ggml_file(const char * filename) {
    llama_file file(filename, "rb");
    if (file.size < 4) {
        return false;
    }
    std::string magic = file.read_string(4);
    return magic == GGUF_MAGIC;
}

static std::string llama_escape_whitespaces(const std::string & text) {
    std::ostringstream out;
    for (char c : text) {
        if (c == ' ') out << "\xe2\x96\x81";
        else out << c;
    }
    return out.str();
}

static void load_vocab(const char *filename, Config *config, struct llama_vocab *vocab) {
    if (is_ggml_file(filename)) {
        struct ggml_context * ctx_data = NULL;

        struct gguf_init_params params = {
            /*.no_alloc = */ false,
            /*.ctx      = */ &ctx_data,
        };

        struct gguf_context * ctx = gguf_init_from_file(filename, params);
        GGML_ASSERT(ctx != NULL);

        const int model_idx = gguf_find_key(ctx, KV_TOKENIZER_MODEL);
        GGML_ASSERT(model_idx >= 0);
        std::string tokenizer_name = gguf_get_val_str(ctx, model_idx);
        GGML_ASSERT(tokenizer_name == TOKENIZER_NAME);

        const int token_idx = gguf_find_key(ctx, KV_TOKENIZER_LIST);
        GGML_ASSERT(token_idx >= 0);

        const int score_idx = gguf_find_key(ctx, KV_TOKENIZER_SCORES);
        GGML_ASSERT(score_idx >= 0);
        const float * scores = (const float * ) gguf_get_arr_data(ctx, score_idx);

        const int toktype_idx = gguf_find_key(ctx, KV_TOKENIZER_TOKEN_TYPE);
        GGML_ASSERT(toktype_idx >= 0);
        const int * toktypes = (const int * ) gguf_get_arr_data(ctx, toktype_idx);

        const uint32_t n_vocab = gguf_get_arr_n(ctx, token_idx);

        vocab->id_to_token.resize(n_vocab);

        for (uint32_t i = 0; i < n_vocab; i++) {
            std::string word = gguf_get_arr_str(ctx, token_idx, i);

            vocab->token_to_id[word] = i;

            auto & token_data = vocab->id_to_token[i];
            token_data.text  = std::move(word);
            token_data.score = scores[i];
            token_data.type  = (llama_token_type) toktypes[i];
        }
        ggml_free(ctx_data);
        gguf_free(ctx);
    } else {
        // assume llama2.c vocabulary
        printf("Assuming llama2.c vocabulary since %s is not a gguf file\n", filename);
        llama_file file(filename, "rb");
        if (!file.fp) {
            die_fmt("%s: %s", strerror(errno), filename);
        }
        const int  n_vocab = config->vocab_size;
        /* uint32_t max_token_length =  */ file.read_u32(); // unused
        vocab->id_to_token.resize(n_vocab);
        for (llama_vocab::id id=0; id<n_vocab; ++id) {
            float_t score = file.read_f32();
            uint32_t len = file.read_u32();
            std::string text = file.read_string(len);

            unsigned char byte_val;
            llama_vocab::ttype type = LLAMA_TOKEN_TYPE_NORMAL;
            if (id == UNKNOWN_TOKEN_ID) {
                text = "<unk>";
                type = LLAMA_TOKEN_TYPE_UNKNOWN;
            } else if (id == BOS_TOKEN_ID) {
                text = "<s>";
                type = LLAMA_TOKEN_TYPE_CONTROL;
            } else if (id == EOS_TOKEN_ID) {
                text = "</s>";
                type = LLAMA_TOKEN_TYPE_CONTROL;
            } else if (text.empty()) {
                type = LLAMA_TOKEN_TYPE_CONTROL;
            } else if (sscanf(text.c_str(), "<0x%02hhX>", &byte_val) == 1) {
                // Text of byte tokens is already in the expected format.
                type = LLAMA_TOKEN_TYPE_BYTE;
            } else {
                type = LLAMA_TOKEN_TYPE_NORMAL;
            }
            text = llama_escape_whitespaces(text);

            vocab->id_to_token[id].text = text;
            vocab->id_to_token[id].score = score;
            vocab->id_to_token[id].type = type;
            vocab->token_to_id.emplace(text, id);
        }
    }
}

static void convert_weights_ak_to_gg(struct ggml_tensor * gg_weights, const float * karpathy_weights) {
    int ct;
    switch (ggml_n_dims(gg_weights)) {
        case 1:
            ct = 0;
            for (int i0 = 0; i0 < gg_weights->ne[0]; i0++){
                float * ptr = (float *) ((char *) gg_weights->data + i0*gg_weights->nb[0]);
                *ptr = karpathy_weights[ct];
                ct++;
            }
            break;
        case 2:
            ct = 0;
            for (int i1 = 0; i1 < gg_weights->ne[1]; i1++) {
                for (int i0 = 0; i0 < gg_weights->ne[0]; i0++) {
                    float * ptr = (float *) ((char *) gg_weights->data + i0*gg_weights->nb[0] + i1*gg_weights->nb[1]);
                    *ptr = karpathy_weights[ct];
                    ct++;
                }
            }
            break;
        case 3:
            ct = 0;
            for (int i2 = 0; i2 < gg_weights->ne[2]; i2++) {
                for (int i1 = 0; i1 < gg_weights->ne[1]; i1++) {
                    for (int i0 = 0; i0 < gg_weights->ne[0]; i0++) {
                        float * ptr = (float *) ((char *) gg_weights->data + i0*gg_weights->nb[0] + i1*gg_weights->nb[1] + i2*gg_weights->nb[2]);
                        *ptr = karpathy_weights[ct];
                        ct++;
                    }
                }
            }
            break;
    }
}

static void save_as_llama_model(
    struct llama_vocab * vocab, struct my_llama_model * model, TransformerWeights* w, const char * filename
) {
    // convert AK weights into GG weights one by one.
    // w->token_embedding_table -> model->tok_embeddings
    // float*                   -> struct ggml_tensor
    convert_weights_ak_to_gg(model->tok_embeddings, w->token_embedding_table);
    convert_weights_ak_to_gg(model->output, w->wcls ? w->wcls : w->token_embedding_table);

    convert_weights_ak_to_gg(model->norm, w->rms_final_weight);
    //print_row(model->norm, 0);

    // for rms-att-weight
    int row_length = model->hparams.n_embd;
    int n_ff = model->hparams.n_ff;

    for (uint32_t i = 0; i < model->hparams.n_layer; ++i){
        auto & layer = model->layers[i];
        // 1d
        convert_weights_ak_to_gg(layer.attention_norm, &w->rms_att_weight[i*row_length]);
        convert_weights_ak_to_gg(layer.ffn_norm      , &w->rms_ffn_weight[i*row_length]);

        // from 3d matrix layer x dim x dim to 2d matrix dim x dim
        convert_weights_ak_to_gg(layer.wq            , &w->wq[i*row_length*row_length]);
        convert_weights_ak_to_gg(layer.wk            , &w->wk[i*row_length*row_length]);
        convert_weights_ak_to_gg(layer.wv            , &w->wv[i*row_length*row_length]);
        convert_weights_ak_to_gg(layer.wo            , &w->wo[i*row_length*row_length]);

        convert_weights_ak_to_gg(layer.w1            , &w->w1[i*row_length*n_ff]);
        convert_weights_ak_to_gg(layer.w2            , &w->w2[i*n_ff*row_length]);
        convert_weights_ak_to_gg(layer.w3            , &w->w3[i*row_length*n_ff]);
    }

    struct gguf_context * ctx = gguf_init_empty();

    std::vector<const char*> tokens;
    std::vector<float> scores;
    std::vector<llama_token_type> token_types;
    for (const llama_vocab::token_data & token_data : vocab->id_to_token) {
        tokens.push_back(token_data.text.c_str());
        scores.push_back(token_data.score);
        token_types.push_back(token_data.type);
    }
    gguf_set_arr_str(ctx, KV_TOKENIZER_LIST, tokens.data(), tokens.size());
    gguf_set_arr_data(ctx, KV_TOKENIZER_SCORES, GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(ctx, KV_TOKENIZER_TOKEN_TYPE, GGUF_TYPE_INT32, token_types.data(), token_types.size());

    gguf_set_val_str(ctx, KV_TOKENIZER_MODEL, TOKENIZER_NAME);

    gguf_set_val_str(ctx, KV_GENERAL_ARCHITECTURE, "llama");
    gguf_set_val_str(ctx, KV_GENERAL_NAME, "llama");

    // special tokens
    gguf_set_val_u32(ctx, KV_TOKENIZER_UNK_ID, UNKNOWN_TOKEN_ID);
    gguf_set_val_u32(ctx, KV_TOKENIZER_BOS_ID, BOS_TOKEN_ID);
    gguf_set_val_u32(ctx, KV_TOKENIZER_EOS_ID, EOS_TOKEN_ID);
    gguf_set_val_u32(ctx, KV_TOKENIZER_SEP_ID, -1);
    gguf_set_val_u32(ctx, KV_TOKENIZER_PAD_ID, -1);

    gguf_set_val_u32(ctx, KV_CONTEXT_LENGTH, model->hparams.n_ctx);
    gguf_set_val_u32(ctx, KV_EMBEDDING_LENGTH, model->hparams.n_embd);
    gguf_set_val_u32(ctx, KV_FEED_FORWARD_LENGTH, model->hparams.n_ff);
    gguf_set_val_u32(ctx, KV_ATTENTION_HEAD_COUNT, model->hparams.n_head);
    // n_head_kv is optional, default to n_head
    // gguf_set_val_u32(ctx, KV_ATTENTION_HEAD_COUNT_KV, ...);
    gguf_set_val_u32(ctx, KV_BLOCK_COUNT, model->hparams.n_layer);
    gguf_set_val_u32(ctx, KV_ROPE_DIMENSION_COUNT, model->hparams.n_rot);
    gguf_set_val_f32(ctx, KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);

    // write tensors
    ggml_set_name(model->tok_embeddings, TN_TOKEN_EMBD);
    gguf_add_tensor(ctx, model->tok_embeddings);

    ggml_set_name(model->norm, TN_OUTPUT_NORM);
    gguf_add_tensor(ctx, model->norm);

    ggml_set_name(model->output, TN_OUTPUT);
    gguf_add_tensor(ctx, model->output);

    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
        auto & layer = model->layers[i];

        ggml_format_name(layer.wq, TN_ATTN_Q, i);
        gguf_add_tensor(ctx, layer.wq);

        ggml_format_name(layer.wk, TN_ATTN_K, i);
        gguf_add_tensor(ctx, layer.wk);

        ggml_format_name(layer.wv, TN_ATTN_V, i);
        gguf_add_tensor(ctx, layer.wv);

        ggml_format_name(layer.wo, TN_ATTN_OUTPUT, i);
        gguf_add_tensor(ctx, layer.wo);

        ggml_format_name(layer.attention_norm, TN_ATTN_NORM, i);
        gguf_add_tensor(ctx, layer.attention_norm);

        ggml_format_name(layer.w1, TN_FFN_GATE, i);
        gguf_add_tensor(ctx, layer.w1);

        ggml_format_name(layer.w2, TN_FFN_DOWN, i);
        gguf_add_tensor(ctx, layer.w2);

        ggml_format_name(layer.w3, TN_FFN_UP, i);
        gguf_add_tensor(ctx, layer.w3);

        ggml_format_name(layer.ffn_norm, TN_FFN_NORM, i);
        gguf_add_tensor(ctx, layer.ffn_norm);
    }

    gguf_write_to_file(ctx, filename, false);
    gguf_free(ctx);
}

static struct train_params get_default_train_params() {
    struct train_params params;
    params.fn_vocab_model    = "models/7B/ggml-model-f16.gguf";
    params.fn_llama2c_output_model = "ak_llama_model.bin";
    params.fn_train_data     = "shakespeare.txt";
    params.fn_checkpoint_in  = "checkpoint.bin";
    params.fn_checkpoint_out = "checkpoint.bin";
    params.fn_model_out      = "ggml-checkpoint-f32.bin";

    params.seed       =   -1;

    params.n_ctx      =  128;
    params.n_embd     =  256;
    params.n_mult     =  256;
    params.n_head     =    8;
    params.n_layer    =   16;
    params.n_rotmax   =   64;

    params.n_threads  =    6;
    params.n_batch    =    8;
    params.n_examples =    8;
    params.n_predict  = 1024;

    params.print_info_interval    = 1;
    params.print_details_interval = 2;

    params.samples_start_after_nl = false;
    params.use_adam               = true;
    params.use_flash              = true;
    params.use_scratch            = true;

    // only adam
    params.warmup            =  100;
    params.cos_decay_steps   = 1000;
    params.cos_decay_restart = 1.1f;
    params.cos_decay_alpha   = 0.0f;

    params.lbfgs_n_iter      = 16;
    params.adam_n_iter       = 16;
    params.adam_alpha        = 1e-3f;
    params.adam_decay        = 1e-3f;

    params.mem_model_gb   = 2;
    params.mem_compute_gb = 24;
    params.mem_compute0_gb = 8;
    params.mem_compute1_gb = 2;

    return params;
}

static void print_usage(int /*argc*/, char ** argv, const struct train_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                       show this help message and exit\n");
    fprintf(stderr, "  --copy-vocab-from-model FNAME    path of gguf llama model or llama2.c vocabulary from which to copy vocab (default '%s')\n", params->fn_vocab_model);
    fprintf(stderr, "  --llama2c-model FNAME            [REQUIRED] model path from which to load Karpathy's llama2.c model\n");
    fprintf(stderr, "  --llama2c-output-model FNAME     model path to save the converted llama2.c model (default %s')\n", params->fn_llama2c_output_model);
    fprintf(stderr, "\n");
}

static bool params_parse(int argc, char ** argv, struct train_params * params) {
    bool invalid_param = false;
    bool reqd_param_found = false;
    std::string arg;
    struct train_params default_params = get_default_train_params();
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "--copy-vocab-from-model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_vocab_model = argv[i];
        } else if (arg == "--llama2c-model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            reqd_param_found = true;
            params->fn_llama2c_model = argv[i];
        } else if (arg == "--llama2c-output-model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_llama2c_output_model = argv[i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, &default_params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, &default_params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv, &default_params);
        exit(1);
    }
    if (!reqd_param_found){
        fprintf(stderr, "error: please specify a llama2.c .bin file to be converted with argument --llama2c-model\n");
        print_usage(argc, argv, &default_params);
        exit(1);
    }

    return true;
}

static std::string basename(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

int main(int argc, char ** argv) {
    struct train_params params = get_default_train_params();
    if (!params_parse(argc, argv, &params)) {
        return 1;
    }
    Config config;
    TransformerWeights weights = {};
    {
        FILE *file = fopen(params.fn_llama2c_model, "rb");
        if (!file) { printf("Unable to open the checkpoint file %s!\n", params.fn_llama2c_model); return 1; }
        // read in the config header
        if(fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
        auto shared_weights = config.vocab_size > 0;
        config.vocab_size = abs(config.vocab_size);

        // read in the Transformer weights
        malloc_weights(&weights, &config, shared_weights);
        if(checkpoint_init_weights(&weights, &config, file, shared_weights)) { return 1; }
        fclose(file);
    }

    struct llama_vocab vocab;
    load_vocab(params.fn_vocab_model, &config, &vocab);

    struct my_llama_model model;
    model.hparams.n_vocab = config.vocab_size; //llama_n_vocab(lctx);
    model.hparams.n_ctx   = params.n_ctx;
    model.hparams.n_embd  = config.dim; //params.n_embd;
    model.hparams.n_ff    = config.hidden_dim;
    model.hparams.n_mult  = 32;//params.n_mult;
    model.hparams.n_head  = config.n_heads; //params.n_head;
    model.hparams.n_layer = config.n_layers; //params.n_layer;
    model.hparams.n_rot   = std::min((uint32_t)params.n_rotmax, model.hparams.n_embd / model.hparams.n_head);
    print_params(&model.hparams);
    struct ggml_init_params lcparams;
    lcparams.mem_size   = 1024ll*1024ll*1024ll*((size_t) params.mem_model_gb);
    lcparams.mem_buffer = NULL;
    lcparams.no_alloc   = false;

    model.ctx = ggml_init(lcparams);

    init_model(&model);
    model.name = basename(params.fn_llama2c_model);
    save_as_llama_model(&vocab, &model, &weights, params.fn_llama2c_output_model);

    printf("Saving llama.c model file %s in ggml format at %s\n", params.fn_llama2c_model, params.fn_llama2c_output_model);

    ggml_free(model.ctx);
    return 0;
}
