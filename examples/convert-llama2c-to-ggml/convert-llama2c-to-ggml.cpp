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

//////////////////////////////////////// llama.c model structs and functions to load models, alloc memory etc.
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
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

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
    int head_size = p->dim / p->n_heads;
    if (fread(w->freq_cis_real, sizeof(float), p->seq_len * head_size / 2, f) != static_cast<size_t>(p->seq_len * head_size / 2)) return 1;
    if (fread(w->freq_cis_imag, sizeof(float), p->seq_len * head_size / 2, f) != static_cast<size_t>(p->seq_len * head_size / 2)) return 1;
    return 0;
}

void malloc_weights(TransformerWeights* w, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    w->token_embedding_table = new float[p->vocab_size * p->dim]();//calloc(p->vocab_size * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->token_embedding_table\n",__func__,p->vocab_size , p->dim, p->vocab_size * p->dim);
    
    w->rms_att_weight = new float[p->n_layers * p->dim](); //calloc(p->n_layers * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->rms_att_weight\n",__func__,p->n_layers, p->dim, p->n_layers * p->dim);

    w->rms_ffn_weight = new float[p->n_layers * p->dim](); //calloc(p->n_layers * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->rms_ffn_weight\n",__func__,p->n_layers , p->dim, p->n_layers * p->dim);

    w->wq = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wq\n",__func__,p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->wk = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wk\n",__func__,p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->wv = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wv\n",__func__, p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->wo = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->wo\n",__func__,p->n_layers, p->dim, p->dim, p->n_layers * p->dim * p->dim);

    w->w1 = new float[p->n_layers * p->hidden_dim * p->dim](); //calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->w1\n",__func__,p->n_layers, p->hidden_dim, p->dim, p->n_layers * p->hidden_dim * p->dim);

    w->w2 = new float[p->n_layers * p->hidden_dim * p->dim](); //calloc(p->n_layers * p->dim * p->hidden_dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->w2\n",__func__,p->n_layers, p->hidden_dim, p->dim, p->n_layers * p->hidden_dim * p->dim);

    w->w3 = new float[p->n_layers * p->hidden_dim * p->dim](); //calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] x [%d] = [%d] float space for w->w3\n",__func__,p->n_layers, p->hidden_dim, p->dim, p->n_layers * p->hidden_dim * p->dim);

    w->rms_final_weight = new float[p->dim](); //calloc(p->dim, sizeof(float));
    printf("[%s:AK] Allocating [%d] float space for w->rms_final_weight\n",__func__,p->dim);

    w->freq_cis_real = new float[p->seq_len * p->dim / 2](); //calloc(p->seq_len * p->dim / 2, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->freq_cis_real\n",__func__,p->seq_len, p->dim / 2, p->seq_len * p->dim / 2);

    w->freq_cis_imag = new float[p->seq_len * p->dim / 2](); //calloc(p->seq_len * p->dim / 2, sizeof(float));
    printf("[%s:AK] Allocating [%d] x [%d] = [%d] float space for w->freq_cis_imag\n\n",__func__,p->seq_len, p->dim / 2, p->seq_len * p->dim / 2);

    // ensure all mallocs went fine
    // if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight 
    //  || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 || 
    //     !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag) {
    //     printf("malloc failed!\n");
    //     exit(1);
    // }
}

void free_weights(TransformerWeights* w) {
    free(w->token_embedding_table);
    free(w->rms_att_weight);
    free(w->rms_ffn_weight);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->w1);
    free(w->w2);
    free(w->w3);
    free(w->rms_final_weight);
    free(w->freq_cis_real);
    free(w->freq_cis_imag);
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
    printf("%f\n", w->freq_cis_real[0]);
    printf("%f\n", w->freq_cis_imag[0]);
    printf("------------------------------------------------------------------\n");

    
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////// ggml structs and functions required to load models, configs and save the model.

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

    struct ggml_tensor * freq_cis_real;
    struct ggml_tensor * freq_cis_imag;

    std::vector<my_llama_layer> layers;

    uint32_t train_its = 0;
    uint32_t train_samples = 0;
    uint32_t train_tokens = 0;
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

    model->freq_cis_real         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd/2);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for model->freq_cis_real\n",__func__,n_embd, n_embd / 2, n_embd * n_embd / 2);
    
    model->freq_cis_imag         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd/2);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for model->freq_cis_imag\n",__func__,n_embd, n_embd / 2, n_embd * n_embd / 2);

    // printing the per-layer allocations here so we dont print in the for loop.
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wq for [%d] layers\n",__func__,n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wk for [%d] layers\n",__func__,n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wv for [%d] layers\n",__func__,n_embd, n_embd, n_embd * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.wo for [%d] layers\n",__func__,n_embd, n_embd, n_embd * n_embd, n_layer);

    printf("[%s:GG] Allocating [%d] float space for layer.ffn_norm for [%d] layers\n",__func__,n_embd, n_layer);

    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.w1 for [%d] layers\n",__func__,n_embd, n_ff, n_embd * n_ff, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.w2 for [%d] layers\n",__func__,n_ff, n_embd, n_ff * n_embd, n_layer);
    printf("[%s:GG] Allocating [%d] x[%d] = [%d] float space for layer.w3 for [%d] layers\n",__func__,n_embd, n_ff, n_embd * n_ff, n_layer);
    

    ggml_set_name(model->tok_embeddings, "tok_embeddings.weight");
    ggml_set_name(model->norm,           "norm.weight");
    ggml_set_name(model->output,         "output.weight");
    ggml_set_name(model->freq_cis_real,         "output.freq_cis_real");
    ggml_set_name(model->freq_cis_imag,         "output.freq_cis_imag");

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

        layer.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);
        layer.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff, n_embd);
        layer.w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);

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

void set_f32_3d(struct ggml_tensor * tensor, int64_t i0, int64_t i1, int64_t i2, float value) {
    float * ptr = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
    *ptr = value;
}

void set_f32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1, float value) {
    float * ptr = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    *ptr = value;
}

void set_i32_2d(struct ggml_tensor * tensor, int64_t i0, int64_t i1, int32_t value) {
    int32_t * ptr = (int32_t *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
    *ptr = value;
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
        printf(" %.2f", p);
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

void print_token(struct llama_context * ctx, llama_token token) {
    printf("%s", llama_token_to_str(ctx, token));
}

void print_tokens(struct llama_context* ctx, struct ggml_tensor * tokens) {
    for (int i=0; i<tokens->ne[0]; ++i) {
        int token = ggml_get_i32_1d(tokens, i);
        print_token(ctx, token);
    }
}

void print_tokens_batch(struct llama_context* ctx, struct ggml_tensor * tokens) {
    for (int i1=0; i1<tokens->ne[1]; ++i1) {
        //int num_newline = 0;
        for (int i0=0; i0<tokens->ne[0]; ++i0) {
            int token = get_i32_2d(tokens, i0, i1);
            print_token(ctx, token);
            // bool isnl = (token == llama_token_nl());
            // if (isnl) {
            //     ++num_newline;
            // }
            // if (isnl) {
            //     if (num_newline < 2) {
            //         print_token(ctx, token);
            //     } else {
            //         printf("\\n");
            //     }
            // } else {
            //     print_token(ctx, token);
            // }
        }
        printf("\n--\n");
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

int tokenize_file(struct llama_context * lctx, const char * filename, std::vector<llama_token>& out) {
    struct llama_file f(filename, "rb");

    std::vector<char> buf;
    buf.resize(f.size+1);

    f.read_raw(buf.data(), f.size);
    buf[f.size] = '\0';

    out.resize(buf.size());

    int n_tokens = llama_tokenize(lctx, buf.data(), out.data(), buf.size(), false);
    if (n_tokens >= 0) {
        out.resize(n_tokens);
    }

    bool verify = false;
    if (verify) {
        const char * in  = buf.data();
        const char * end = buf.data() + buf.size();
        for (int i = 0; i < (int) out.size(); ++i) {
            const char * s = llama_token_to_str(lctx, out[i]);
            int len = strlen(s);
            if (in >= end) {
                printf("%s: unexpected end of original text.\n", __func__);
                break;
            }
            const bool matches = (strncmp(in, s, len) == 0);
            if (matches) {
                in += len;
            } else {
                printf("%s: mismatch: expected '%s', but got '%s'\n", __func__, std::string(in, len).c_str(), s);
            }
        }
    }

    return n_tokens;
}

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

void read_tensor(struct llama_file * file, struct ggml_tensor * tensor) {
    int32_t nd = file->read_u32();
    GGML_ASSERT(nd == tensor->n_dims);

    uint32_t name_len       = file->read_u32();
    enum     ggml_type type = (enum ggml_type) file->read_u32();
    GGML_ASSERT(type == tensor->type);

    uint32_t ne[4];
    file->read_raw(ne, sizeof(ne[0]) * nd);
    for (int i=0; i<nd; ++i) {
        GGML_ASSERT(ne[i] == tensor->ne[i]);
    }

    std::string name = file->read_string(name_len);
    GGML_ASSERT(strncmp(ggml_get_name(tensor), name.c_str(), sizeof(tensor->name)-1) == 0);

    file->seek((0-file->tell()) & 31, SEEK_CUR);
    file->read_raw(tensor->data, ggml_nbytes(tensor));
}

void stuff_karpathy_weights_into_gg(struct ggml_tensor * gg_weights, float * karpathy_weights){
    
    int ct;
    switch (gg_weights->n_dims){
        case 1:
            ct = 0;
            for (int i0 = 0; i0 < gg_weights->ne[0]; i0++){
                float * ptr = (float *) ((char *) gg_weights->data + i0*gg_weights->nb[0]);
                *ptr = karpathy_weights[ct];
            }
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
    // write_magic
    file.write_u32(LLAMA_FILE_MAGIC);   // magic
    file.write_u32(LLAMA_FILE_VERSION); // version
    // write_hparams
    file.write_u32(model->hparams.n_vocab);
    file.write_u32(model->hparams.n_embd);
    file.write_u32(model->hparams.n_mult);
    file.write_u32(model->hparams.n_head);
    file.write_u32(model->hparams.n_layer);
    file.write_u32(model->hparams.n_rot);
    file.write_u32(LLAMA_FTYPE_ALL_F32);

    // write_vocab - for now we are just writing the existing BPE voc. assuming karpathy's vocabulary is the same. idk.
    uint32_t n_vocab = model->hparams.n_vocab;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const auto & token_score = vocab->id_to_token.at(i);
        file.write_u32((uint32_t) token_score.tok.size());
        file.write_raw(token_score.tok.data(), token_score.tok.size());
        file.write_raw(&token_score.score, sizeof(token_score.score));
    }

    // stuff AK weights into GG weights one by one.
    // w->token_embedding_table -> model->tok_embeddings
    // float*                   -> struct ggml_tensor
    stuff_karpathy_weights_into_gg(model->tok_embeddings, w->token_embedding_table);
    // print_row(model->tok_embeddings, 0);

    stuff_karpathy_weights_into_gg(model->norm, w->rms_final_weight);         
    stuff_karpathy_weights_into_gg(model->freq_cis_real, w->freq_cis_real);
    stuff_karpathy_weights_into_gg(model->freq_cis_imag, w->freq_cis_imag);

    // for rms-att-weight 
    int row_length = model->hparams.n_embd;
    for (uint32_t i = 0; i < model->hparams.n_layer; ++i){
        auto & layer = model->layers[i];
        // 2d        
        stuff_karpathy_weights_into_gg(layer.attention_norm, &w->rms_att_weight[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.ffn_norm      , &w->rms_ffn_weight[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.wq            , &w->wq[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.wk            , &w->wk[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.wv            , &w->wv[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.wo            , &w->wo[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.w1            , &w->w1[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.w2            , &w->w2[i*row_length]);
        stuff_karpathy_weights_into_gg(layer.w3            , &w->w3[i*row_length]);
    }
    
    // write tensors
    write_tensor(&file, model->tok_embeddings);
    write_tensor(&file, model->norm);
    write_tensor(&file, model->output); // ?
    write_tensor(&file, model->freq_cis_real);
    write_tensor(&file, model->freq_cis_imag);
    for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {        
        auto & layer = model->layers[i];

        write_tensor(&file, layer.attention_norm);
        write_tensor(&file, layer.wq);
        write_tensor(&file, layer.wk);
        write_tensor(&file, layer.wv);
        write_tensor(&file, layer.wo);
        write_tensor(&file, layer.ffn_norm);
        write_tensor(&file, layer.w1);
        write_tensor(&file, layer.w2);
        write_tensor(&file, layer.w3);
    }
}

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

struct train_params get_default_train_params() {
    struct train_params params;
    params.fn_vocab_model    = "ggml-vic7b-uncensored-q4_0.bin";
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

void train_print_usage(int /*argc*/, char ** argv, const struct train_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                       show this help message and exit\n");
    fprintf(stderr, "  --vocab-model FNAME              model path from which to load vocab (default '%s')\n", params->fn_vocab_model);
    fprintf(stderr, "  --llama2c-model FNAME            model path from which to load Karpathy's llama2.c model\n");   
    fprintf(stderr, "  --llama2c-output-model FNAME     model path to save the converted llama2.c model (default %s')\n", params->fn_llama2c_output_model);   
    fprintf(stderr, "\n");
}

bool train_params_parse(int argc, char ** argv, struct train_params * params) {
    bool invalid_param = false;
    std::string arg;
    struct train_params default_params = get_default_train_params();
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "--vocab-model") {
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
            params->fn_llama2c_model = argv[i]; 
        } else if (arg == "--llama2c-output-model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_llama2c_output_model = argv[i]; 
        } else if (arg == "-h" || arg == "--help") {
            train_print_usage(argc, argv, &default_params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            train_print_usage(argc, argv, &default_params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        train_print_usage(argc, argv, &default_params);
        exit(1);
    }

    return true;
}

int main(int argc, char ** argv) {
    struct train_params params = get_default_train_params();
    if (!train_params_parse(argc, argv, &params)) {
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

    struct llama_context_params llama_params = llama_context_default_params();
    llama_params.vocab_only = true;

    struct llama_model * lmodel = llama_load_model_from_file(params.fn_vocab_model, llama_params);
    struct llama_context * lctx = llama_new_context_with_model(lmodel, llama_params);

    struct llama_vocab vocab;
    {
        std::vector<const char *> strings;
        std::vector<float> scores;
        int n_vocab = llama_n_vocab(lctx);        
        strings.resize(n_vocab, NULL);
        scores.resize(n_vocab, 0);
        n_vocab = llama_get_vocab(lctx, strings.data(), scores.data(), n_vocab);
        GGML_ASSERT(n_vocab == llama_n_vocab(lctx));
        vocab.id_to_token.resize(n_vocab);
        for (int i=0; i<n_vocab; ++i) {
            std::string tok   = std::string(strings[i]);
            float       score = scores[i];            
            vocab.id_to_token[i].tok   = tok;
            vocab.id_to_token[i].score = score;
            vocab.token_to_id.emplace(tok, i);
        }
    }
    struct my_llama_model model;
    model.hparams.n_vocab = config.vocab_size; //llama_n_vocab(lctx);
    model.hparams.n_ctx   = params.n_ctx;
    model.hparams.n_embd  = config.dim; //params.n_embd;
    model.hparams.n_mult  = params.n_mult; 
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

    llama_free(lctx);
    llama_free_model(lmodel);
    ggml_free(model.ctx);
    free_weights(&weights);
    return 0;
}