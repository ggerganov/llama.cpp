#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <cstring>
#include <string>

#include "llama.h"
#include "ggml.h"

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
} TransformerWeights;

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

struct train_params {
    const char * fn_vocab_model;
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

void save_as_llama_model(struct llama_vocab * vocab, struct my_llama_model * model, const char * filename) {
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
    // write_vocab
    uint32_t n_vocab = model->hparams.n_vocab;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const auto & token_score = vocab->id_to_token.at(i);
        file.write_u32((uint32_t) token_score.tok.size());
        file.write_raw(token_score.tok.data(), token_score.tok.size());
        file.write_raw(&token_score.score, sizeof(token_score.score));
    }
    // write tensors
    write_tensor(&file, model->tok_embeddings);
    write_tensor(&file, model->norm);
    write_tensor(&file, model->output);
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

void print_config(Config* p){
    printf("----- Configs extracted from the header -------\n");
    printf("config.dim %d\n", p->dim);
    printf("config.hidden_dim %d\n", p->hidden_dim);
    printf("config.n_layers %d\n", p->n_layers);
    printf("config.n_heads %d\n", p->n_heads );
    printf("config.n_kv_heads %d\n", p->n_kv_heads);
    printf("config.vocab_size %d\n", p->vocab_size);
    printf("config.seq_len %d\n", p->seq_len);
    printf("----------------------------------------------\n");
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
void malloc_weights(TransformerWeights* w, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    w->token_embedding_table = new float[p->vocab_size * p->dim]();//calloc(p->vocab_size * p->dim, sizeof(float));
    w->rms_att_weight = new float[p->n_layers * p->dim](); //calloc(p->n_layers * p->dim, sizeof(float));
    w->rms_ffn_weight = new float[p->n_layers * p->dim](); //calloc(p->n_layers * p->dim, sizeof(float));
    w->wq = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->wk = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->wv = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->wo = new float[p->n_layers * p->dim * p->dim](); //calloc(p->n_layers * p->dim * p->dim, sizeof(float));
    w->w1 = new float[p->n_layers * p->hidden_dim * p->dim](); //calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));
    w->w2 = new float[p->n_layers * p->hidden_dim * p->dim](); //calloc(p->n_layers * p->dim * p->hidden_dim, sizeof(float));
    w->w3 = new float[p->n_layers * p->hidden_dim * p->dim](); //calloc(p->n_layers * p->hidden_dim * p->dim, sizeof(float));
    w->rms_final_weight = new float[p->dim](); //calloc(p->dim, sizeof(float));
    w->freq_cis_real = new float[p->seq_len * p->dim / 2](); //calloc(p->seq_len * p->dim / 2, sizeof(float));
    w->freq_cis_imag = new float[p->seq_len * p->dim / 2](); //calloc(p->seq_len * p->dim / 2, sizeof(float));
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

int main(int argc, char *argv[]) {

    // poor man's C argparse
    char *checkpoint = NULL;
    char *tokenizer = NULL;
    // float temperature = 0.9f;
    // 'checkpoint' is necessary arg
    if (argc < 3) {
        printf("Usage: %s <checkpoint_file> <tokenizer_file>\n", argv[0]);
        return 1;
    }
    checkpoint = argv[1];
    tokenizer = argv[2];
    // if (argc < 3) {
    //     printf("Usage: %s <checkpoint_file>\n", argv[0]);
    //     return 1;
    // }
    // temperature is optional
    // if (argc >= 3) {
    //     temperature = atof(argv[2]);
    // }
    // seed is optional
    // if (argc >= 4) {
    //     unsigned int seed = atoi(argv[3]);
    //     srand(seed);
    // } else {
    //     time_t current_time; 
    //     time(&current_time);
    //     srand((unsigned int)current_time);
    // }

    // read in the Karpathy model.bin file
    Config config; // Configs are stashed in the bin file as header
    TransformerWeights weights;

    {
        FILE *file = fopen(checkpoint, "rb");
        if (!file) {
            printf("Unable to open the checkpoint file %s!\n", checkpoint);
            return 1;
        }
        else{
            printf("model file opened for reading...\n");
        }
        // read in the config header
        if(fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
        printf("config file read..\n");
        print_config(&config);
        // read in the Transformer weights
        malloc_weights(&weights, &config);
        printf("reading the opened model file...\n");
        if(checkpoint_init_weights(&weights, &config, file)) { return 1; }
        print_sample_weights(&weights);
        printf("Closing model file..bye...\n");
        fclose(file);
    }

    // read in the tokenizer.bin file
    char** vocab_ak = (char**)malloc(config.vocab_size * sizeof(char*));
    {
        FILE *file = fopen(tokenizer, "rb");
        if (!file) {
            printf("Unable to open the tokenizer file tokenizer.bin! Run "
            "python tokenizer.py to convert tokenizer.model -> tokenizer.bin\n");
            return 1;
        }
        int len;
        printf("karpathy vocab size = %d\n", config.vocab_size);

        for (int i = 0; i < config.vocab_size; i++) {
            if(fread(&len, sizeof(int), 1, file) != 1) { return 1; }
            vocab_ak[i] = (char *)malloc(len + 1);
            if(fread(vocab_ak[i], len, 1, file) != 1) { return 1; }
            vocab_ak[i][len] = '\0'; // add the string terminating token
            printf("len = %d, %s\n", len, vocab_ak[i]);

        }
        fclose(file);
    }

    //TODO:-------------------------------------------------------------------------------
    struct my_llama_model model;
    struct train_params params = get_default_train_params();
    struct llama_context_params llama_params = llama_context_default_params();
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

    save_as_llama_model(&vocab, &model, params.fn_model_out);

    printf("\n");
    free_weights(&weights);
    free(vocab_ak);
    return 0;

}