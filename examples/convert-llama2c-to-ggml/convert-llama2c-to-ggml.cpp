#include "ggml.h"
#include "llama.h"

#include <unordered_map>
#include <vector>
#include <cassert>
#include <climits>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

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

typedef struct {
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
    //float* wcls;
} TransformerWeights;

void malloc_weights(TransformerWeights* w, Config* p) {
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
}

int checkpoint_init_weights(TransformerWeights *w, Config* p, FILE* f) {
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
    return 0;
}

void free_weights(TransformerWeights* w) {
    delete w->token_embedding_table;
    delete w->rms_att_weight;
    delete w->rms_ffn_weight;
    delete w->wq;
    delete w->wk;
    delete w->wv;
    delete w->wo;
    delete w->w1;
    delete w->w2;
    delete w->w3;
    delete w->rms_final_weight;
}

void print_sample_weights(TransformerWeights *w){
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

uint32_t get_n_ff(const struct my_llama_hparams* hparams) {
    const uint32_t n_ff = ((2*(4*hparams->n_embd)/3 + hparams->n_mult - 1)/hparams->n_mult)*hparams->n_mult;
    return n_ff;
}

void print_params(struct my_llama_hparams * params) {
    printf("%s: n_vocab: %d\n", __func__, params->n_vocab);
    printf("%s: n_ctx:   %d\n", __func__, params->n_ctx);
    printf("%s: n_embd:  %d\n", __func__, params->n_embd);
    printf("%s: n_mult:  %d\n", __func__, params->n_mult);
    printf("%s: n_head:  %d\n", __func__, params->n_head);
    printf("%s: n_ff:    %d\n", __func__, get_n_ff(params));
    printf("%s: n_layer: %d\n", __func__, params->n_layer);
    printf("%s: n_rot:   %d\n", __func__, params->n_rot);
}

void init_model(struct my_llama_model * model) {
    const auto & hparams = model->hparams;

    const uint32_t n_embd  = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;

    const uint32_t n_ff = get_n_ff(&hparams);
    struct ggml_context * ctx = model->ctx;

    model->train_its = 0;
    model->train_samples = 0;
    model->train_tokens = 0;

    model->tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    printf("[%s:GG] Allocating [%d] x [%d] = [%d] float space for model->tok_embeddings\n",__func__,n_embd , n_vocab, n_embd * n_vocab);

    model->norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    printf("[%s:GG] Allocating [%d] float space for model->norm\n",__func__,n_embd);

    model->output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for model->output\n",__func__,n_embd, n_vocab, n_embd * n_vocab);

    // printing the per-layer allocations here so we dont print in the for loop.
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wq for [%d] layers\n",__func__, n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wk for [%d] layers\n",__func__, n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wv for [%d] layers\n",__func__, n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wo for [%d] layers\n",__func__, n_embd, n_embd, n_embd * n_embd, n_layer);

    printf("[%s:GG] Allocating [%d] float space for layer.ffn_norm for [%d] layers\n",__func__,n_embd, n_layer);

    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.w1 for [%d] layers\n",__func__, n_ff, n_embd, n_embd * n_ff, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.w2 for [%d] layers\n",__func__, n_embd, n_ff, n_ff * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.w3 for [%d] layers\n",__func__, n_ff, n_embd, n_embd * n_ff, n_layer);

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

float get_f32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1) {
    float * ptr = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    return *ptr;
}

int32_t get_i32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1) {
    int32_t * ptr = (int32_t *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    return *ptr;
}

void print_row(struct ggml_tensor * probs, int i) {
    for (int k = 0; k < probs->ne[0]; ++k) {
        float p = get_f32_2d(probs, k, i);
        printf(" %f", p);
    }
    printf("\n");
}

void print_matrix(struct ggml_tensor * probs) {
    assert(probs->n_dims == 2);
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
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error(std::string("unexpectedly reached end of file"));
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

    void write_raw(const void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, size, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) {
        write_raw(&val, sizeof(val));
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

void write_tensor(struct llama_file * file, struct ggml_tensor * tensor) {
    if (tensor == NULL) {
        file->write_u32(0);
        file->write_u32(0);
        file->write_u32(GGML_TYPE_F32);
        file->seek((0-file->tell()) & 31, SEEK_CUR);
        return;
    }
    const char * name = ggml_get_name(tensor);
    uint32_t name_len = strlen(name);
    uint32_t nd = tensor->n_dims;
    uint32_t ne[4] = { (uint32_t)tensor->ne[0],
                       (uint32_t)tensor->ne[1],
                       (uint32_t)tensor->ne[2],
                       (uint32_t)tensor->ne[3] };
    file->write_u32(nd);
    file->write_u32(name_len);
    file->write_u32(tensor->type);
    file->write_raw(ne, sizeof(ne[0]) * nd);
    file->write_raw(name, name_len);
    file->seek((0-file->tell()) & 31, SEEK_CUR);
    file->write_raw(tensor->data, ggml_nbytes(tensor));
}

bool is_ggml_file(const char *filename) {
    llama_file file(filename, "rb");
    if (file.size < 4) {
        return false;
    }
    uint32_t magic = file.read_u32();
    return magic == GGUF_MAGIC;
}

void load_vocab(const char *filename, Config *config, struct llama_vocab *vocab) {
    // heuristic to infer whether vocab is from ggml or from llama2.c vocabulary
    if (is_ggml_file(filename)) {

        struct llama_context_params llama_params = llama_context_default_params();
        llama_params.vocab_only = true;

        struct llama_model * lmodel = llama_load_model_from_file(filename, llama_params);
        struct llama_context * lctx = llama_new_context_with_model(lmodel, llama_params);

        const int n_vocab = llama_n_vocab(lctx);
        vocab->id_to_token.resize(n_vocab);
        for (int i=0; i<n_vocab; ++i) {
            vocab->id_to_token[i].text  = llama_token_get_text(lctx, i);
            vocab->id_to_token[i].score = llama_token_get_score(lctx, i);
            vocab->id_to_token[i].type  = llama_token_get_type(lctx, i);
            vocab->token_to_id.emplace(vocab->id_to_token[i].text, i);
        }
        llama_free(lctx);
        llama_free_model(lmodel);
    } else { // assume llama2.c vocabulary
        printf("Assuming llama2.c vocabulary since %s is not a ggml file\n", filename);
        llama_file file(filename, "rb");
        const int  n_vocab = config->vocab_size;
        /* uint32_t max_token_length =  */ file.read_u32(); // unused
        vocab->id_to_token.resize(n_vocab);
        for (int i=0; i<n_vocab; ++i) {
            float_t score = file.read_f32();
            uint32_t len = file.read_u32();
            std::string text = file.read_string(len);
            vocab->id_to_token[i].text = text;
            vocab->id_to_token[i].score = score;
            vocab->id_to_token[i].type = LLAMA_TOKEN_TYPE_UNDEFINED;
            vocab->token_to_id.emplace(text, i);
        }
    }
}

void stuff_karpathy_weights_into_gg(struct ggml_tensor * gg_weights, float * karpathy_weights){
    int ct;
    switch (gg_weights->n_dims){
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

void save_as_llama_model(struct llama_vocab * vocab, struct my_llama_model * model, TransformerWeights* w, const char * filename) {
    struct llama_file file(filename, "wb");
    if (file.fp == NULL) {
        return;
    }

#pragma message("TODO: implement file saving using gguf")
    (void) vocab;
    (void) model;
    (void) w;
//    // write_magic
//    file.write_u32(LLAMA_FILE_MAGIC);   // magic
//    file.write_u32(LLAMA_FILE_VERSION); // version
//    // write_hparams
//    file.write_u32(model->hparams.n_vocab);
//    file.write_u32(model->hparams.n_embd);
//    file.write_u32(model->hparams.n_mult);
//    file.write_u32(model->hparams.n_head);
//    file.write_u32(model->hparams.n_layer);
//    file.write_u32(model->hparams.n_rot);
//    file.write_u32(LLAMA_FTYPE_ALL_F32);
//
//    // write_vocab - for now we are just writing the existing BPE voc. assuming karpathy's vocabulary is the same. idk.
//    uint32_t n_vocab = model->hparams.n_vocab;
//    for (uint32_t i = 0; i < n_vocab; i++) {
//        const auto & token_data = vocab->id_to_token.at(i);
//        file.write_u32((uint32_t) token_data.tok.size());
//        file.write_raw(token_data.tok.data(), token_data.tok.size());
//        file.write_raw(&token_data.score, sizeof(token_data.score));
//    }
//
//    // stuff AK weights into GG weights one by one.
//    // w->token_embedding_table -> model->tok_embeddings
//    // float*                   -> struct ggml_tensor
//    stuff_karpathy_weights_into_gg(model->tok_embeddings, w->token_embedding_table);
//    stuff_karpathy_weights_into_gg(model->output, w->token_embedding_table);
//
//    stuff_karpathy_weights_into_gg(model->norm, w->rms_final_weight);
//    //print_row(model->norm, 0);
//
//    // for rms-att-weight
//    int row_length = model->hparams.n_embd;
//    const auto & hparams = model->hparams;
//    //int n_ff = model->hparams.n_embd;
//    int n_ff = get_n_ff(&hparams);
//
//    for (uint32_t i = 0; i < model->hparams.n_layer; ++i){
//        auto & layer = model->layers[i];
//        // 1d
//        stuff_karpathy_weights_into_gg(layer.attention_norm, &w->rms_att_weight[i*row_length]);
//        stuff_karpathy_weights_into_gg(layer.ffn_norm      , &w->rms_ffn_weight[i*row_length]);
//
//        // from 3d matrix layer x dim x dim to 2d matrix dim x dim
//        stuff_karpathy_weights_into_gg(layer.wq            , &w->wq[i*row_length*row_length]);
//        stuff_karpathy_weights_into_gg(layer.wk            , &w->wk[i*row_length*row_length]);
//        stuff_karpathy_weights_into_gg(layer.wv            , &w->wv[i*row_length*row_length]);
//        stuff_karpathy_weights_into_gg(layer.wo            , &w->wo[i*row_length*row_length]);
//
//        stuff_karpathy_weights_into_gg(layer.w1            , &w->w1[i*row_length*n_ff]);
//        stuff_karpathy_weights_into_gg(layer.w2            , &w->w2[i*n_ff*row_length]);
//        stuff_karpathy_weights_into_gg(layer.w3            , &w->w3[i*row_length*n_ff]);
//    }
//    // write tensors
//    write_tensor(&file, model->tok_embeddings);
//    write_tensor(&file, model->norm);
//    write_tensor(&file, model->output); // ?
//    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
//        auto & layer = model->layers[i];
//
//        write_tensor(&file, layer.attention_norm);
//        write_tensor(&file, layer.wq);
//        write_tensor(&file, layer.wk);
//        write_tensor(&file, layer.wv);
//        write_tensor(&file, layer.wo);
//        write_tensor(&file, layer.ffn_norm);
//        write_tensor(&file, layer.w1);
//        write_tensor(&file, layer.w2);
//        write_tensor(&file, layer.w3);
//    }
}

struct train_params get_default_train_params() {
    struct train_params params;
    params.fn_vocab_model    = "models/ggml-vocab.bin";
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

void print_usage(int /*argc*/, char ** argv, const struct train_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                       show this help message and exit\n");
    fprintf(stderr, "  --copy-vocab-from-model FNAME    llama2.c vocabulary or ggml model path from which to copy vocab (default '%s')\n", params->fn_vocab_model);
    fprintf(stderr, "  --llama2c-model FNAME            [REQUIRED] model path from which to load Karpathy's llama2.c model\n");
    fprintf(stderr, "  --llama2c-output-model FNAME     model path to save the converted llama2.c model (default %s')\n", params->fn_llama2c_output_model);
    fprintf(stderr, "\n");
}

bool params_parse(int argc, char ** argv, struct train_params * params) {
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

int main(int argc, char ** argv) {
    struct train_params params = get_default_train_params();
    if (!params_parse(argc, argv, &params)) {
        return 1;
    }
    Config config;
    TransformerWeights weights;
    {
        FILE *file = fopen(params.fn_llama2c_model, "rb");
        if (!file) { printf("Unable to open the checkpoint file %s!\n", params.fn_llama2c_model); return 1; }
        // read in the config header
        if(fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
        // read in the Transformer weights
        malloc_weights(&weights, &config);
        if(checkpoint_init_weights(&weights, &config, file)) { return 1; }
        fclose(file);
    }

    struct llama_vocab vocab;
    load_vocab(params.fn_vocab_model, &config, &vocab);

    struct my_llama_model model;
    model.hparams.n_vocab = config.vocab_size; //llama_n_vocab(lctx);
    model.hparams.n_ctx   = params.n_ctx;
    model.hparams.n_embd  = config.dim; //params.n_embd;
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
    save_as_llama_model(&vocab, &model, &weights, params.fn_llama2c_output_model);

    printf("Saving llama.c model file %s in ggml format at %s\n", params.fn_llama2c_model, params.fn_llama2c_output_model);

    ggml_free(model.ctx);
    free_weights(&weights);
    return 0;
}
