#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>

struct callback_data {
    std::vector<uint8_t> data;
    int n_tokens = 0;
    int n_embd = 0;
    bool is_eval_pos = true;
    // each element of the vector correspond to one layer
    std::vector<float *> v_pos;  // vector of matrices of size [n_embd, n_tokens]
    std::vector<float *> v_neg;  // vector of matrices of size [n_embd, n_tokens]
    std::vector<float *> v_diff; // vector of matrices of size [n_embd, n_tokens]
    std::vector<float *> v_final; // vector of finished vectors of size [n_embd]
};

struct ctrl_params {
    std::string outfile = "control_vector.gguf";
    std::string positive = "happy"; // TODO support multiple positive prompts
    std::string negative = "sad"; // TODO support multiple negative prompts
};

static void print_usage(const char * executable) {
    printf("\n");
    printf("usage: %s [options] -m <model> [gpt-opts]", executable);
    printf("\n");
    printf("Creates a GGUF control vector for a given model.");
    printf("\n");
    printf("options:\n");
    printf("  -h, --help          show this help message and exit\n");
    printf("  --outfile           output file (default: 'control_vector.gguf')\n");
    printf("  --positive          positive prompt (default: 'happy')\n");
    printf("  --negative          negative prompt (default: 'sad')\n");
    printf("\n");
    printf("gpt-opts: other options from main\n");
    printf("\n");
}

static int ctrlvec_params_parse_ex(int argc, char ** argv, ctrl_params & params) {
    std::string arg;
    const std::string arg_prefix = "--";
    int skipme = 0;

    int arg_idx = 1;
    for(; arg_idx < argc && strncmp(argv[arg_idx], arg_prefix.c_str(), 2) == 0; ++arg_idx) {
        arg = argv[arg_idx];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
        if (arg == "--version") {
            fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
            fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
            exit(0);
        }
        if (arg == "--outfile") {
            if (++arg_idx < argc && strncmp(argv[arg_idx], arg_prefix.c_str(), 2) != 0) {
                params.outfile = argv[arg_idx];
                // FIXME hack to skip these args in gpt_parse_params
                skipme += 2;
            }
            else {
                throw std::invalid_argument("error: missing argument for " + arg);
            }
        }
        if (arg == "--positive") {
            if (++arg_idx < argc && strncmp(argv[arg_idx], arg_prefix.c_str(), 2) != 0) {
                params.positive = argv[arg_idx];
                // FIXME hack to skip these args in gpt_parse_params
                skipme += 2;
            }
            else {
                throw std::invalid_argument("error: missing argument for " + arg);
            }
        }
        if (arg == "--negative") {
            if (++arg_idx < argc && strncmp(argv[arg_idx], arg_prefix.c_str(), 2) != 0) {
                params.negative = argv[arg_idx];
                // FIXME hack to skip these args in gpt_parse_params
                skipme += 2;
            }
            else {
                throw std::invalid_argument("error: missing argument for " + arg);
            }
        }

        // we do not handle any other unknown arguments here because they will be handled by gpt_parse_params
    }
    return skipme;
}

static int ctrlvec_params_parse(int argc, char ** argv, ctrl_params & params) {
    int skipme = 0;
    try {
        skipme = ctrlvec_params_parse_ex(argc, argv, params);
    }
    catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        print_usage(argv[0]);
        exit(EXIT_FAILURE);    
    }
    return skipme;
}

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    static const char * l_out_name = "l_out";
    const bool is_l_out = strncmp(t->name, l_out_name, strlen(l_out_name)) == 0;
    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return is_l_out;
    }

    if (!is_l_out || t->ne[1] != cb_data->n_tokens) {
        return true;
    }

    char src1_str[128] = {0};
    if (src1) {
        sprintf(src1_str, "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    printf("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
           t->name, ggml_type_name(t->type), ggml_op_desc(t),
           src0->name, ggml_ne_string(src0).c_str(),
           src1 ? src1_str : "",
           ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (t->type == GGML_TYPE_F32) {
        float * data = (float *) (is_host ? t->data : cb_data->data.data());
        float * dest = (float *) malloc(ggml_nbytes(t));
        memcpy(dest, data, ggml_nbytes(t));
        if (cb_data->is_eval_pos) {
            cb_data->v_pos.push_back(dest);
        } else {
            cb_data->v_neg.push_back(dest);
        }
    }

    return true;
}

static bool get_hidden_layers(llama_context * ctx, std::vector<llama_token> & tokens) {
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0))) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return false;
    }
    return true;
}

static void padding_seq(llama_context * ctx, std::vector<llama_token> & tokens, size_t len) {
    // TODO: customize padding token
    std::vector<llama_token> pad_tokens = ::llama_tokenize(ctx, " ", false);
    llama_token pad_tok = pad_tokens.back();
    while (tokens.size() < len) {
        tokens.push_back(pad_tok);
    }
}

static void calc_diff(callback_data & cb_data) {
    // TODO: assert cb_data.v_pos.size() == cb_data.v_neg.size()
    const size_t n_elems = cb_data.n_embd * cb_data.n_tokens;
    for (size_t il = 0; il < cb_data.v_pos.size(); il++) {
        auto & inp_pos = cb_data.v_pos[il];
        auto & inp_neg = cb_data.v_neg[il];
        float * dest = (float *) malloc(n_elems * sizeof(float *));
        for (size_t i = 0; i < n_elems; i++) {
            dest[i] = inp_pos[i] - inp_neg[i];
        }
        cb_data.v_diff.push_back(dest);
    }
}

// BEGIN NON-GGML IMPLEMENTATION

// TODO translate to ggml
// this probably doesn't want to be here - put it into the compute graph as a step in processing each layer
static float* square_diff(callback_data & cb_data, size_t idx) {
    float* result = new float[cb_data.n_embd * cb_data.n_embd];
    std::memset(result, 0, cb_data.n_embd * cb_data.n_embd * sizeof(float));
    for (size_t i = 0; i < cb_data.n_embd; i++) {
        for (size_t j = 0; j < cb_data.n_embd; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < cb_data.n_tokens; k++) {
                sum += cb_data.v_diff[idx][i * cb_data.n_tokens + k] * cb_data.v_diff[idx][j * cb_data.n_tokens + k];
            }
            result[i * cb_data.n_embd + j] = sum;
        }
    }
    return result;
}

// TODO translate to ggml
static void normalize_inplace(std::vector<float> & vec) {
    // inefficient(?) norm computation
    float norm = 0.0f;
    for (const float& val : vec) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    for (float& val : vec) {
        val /= norm;
    }
}

// TODO translate to ggml
static std::vector<float> mul_mat(const float * mat, const std::vector<float> & vec, size_t dim) {
    std::vector<float> result(dim, 0.0f);
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            result[i] += mat[i * dim + j] * vec[j];
        }
    }
    return result;
}

// TODO translate to ggml
static std::vector<float> power_iteration(callback_data & cb_data, const float * matrix, int maxIterations = 1000, float tolerance = 1e-8) {
    std::vector<float> b_tensor = std::vector<float>();
    
    // random vector gen/norm
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < cb_data.n_embd; ++i) {
        b_tensor.push_back(distribution(generator));
    }
    normalize_inplace(b_tensor);

    for (int iter = 0; iter < maxIterations; ++iter) {

        // store the previous one so we can check for convergence
        std::vector<float> b_prev_tensor = b_tensor;

        // matrix multiplication and renormalize
        b_tensor = mul_mat(matrix, b_tensor, cb_data.n_embd);
        normalize_inplace(b_tensor);

        // convergence check
        float diff = 0.0;
        for (int i = 0; i < cb_data.n_embd; ++i) {
            diff += std::pow(b_tensor[i] - b_prev_tensor[i], 2);
        }
        if (std::sqrt(diff) < tolerance) {
            break;
        }
    }

    return b_tensor;
}

// TODO translate to ggml
static void pca(callback_data & cb_data) {
    for (int i = 0; i < cb_data.v_diff.size(); i++) {
        float* matrix = square_diff(cb_data, i);
        std::vector<float> eigenvector = power_iteration(cb_data, matrix);
        cb_data.v_final.push_back(&eigenvector[0]);
        delete[] matrix;
        printf("Done with layer %d\n", i);
    }
    printf("Done with PCA.");
}

template <typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

static void export_gguf(callback_data & cb_data, const std::string fname, const std::string model_hint) {
    struct gguf_context * ctx = gguf_init_empty();

    const std::string arch = "controlvector";
    gguf_set_val_str(ctx, "general.architecture", arch.c_str());
    gguf_set_val_str(ctx, (arch + ".model_hint").c_str(), model_hint.c_str());
    gguf_set_val_i32(ctx, (arch + ".layer_count").c_str(), cb_data.v_final.size());

    //size_t buf_size = 3u*cb_data.n_embd*sizeof(float); // TODO how much size do i need?
    size_t buf_size = 128u*1024u*4096u; // FIXME placehokder

    // TODO customize mem size - I have no idea what this is supposed to be
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx_data = ggml_init(params);

    for (int i = 0; i < cb_data.v_final.size(); ++i) {
        // TODO this number is probably not right - figure out which layer is which
        // the python implementation uses a dict to handle this, we don't know if it's 1, 2, 3, 4... or other
        const std::string name = "direction." + to_string(i+1);

        struct ggml_tensor * cur = ggml_new_tensor_1d(ctx_data, GGML_TYPE_F32, cb_data.n_embd);

        ggml_set_name(cur, name.c_str());

        // TODO figure out how to set data - it's whining about buf != NULL when using the below commented line
        //ggml_backend_tensor_set(cur, cb_data.v_final[i], 0, cb_data.n_embd * sizeof(float));
        {
            float * data = (float *) cur->data;
            for(int j = 0; j < ggml_nelements(cur); j++) {
                data[j] = cb_data.v_final[i][j];
            }
        }

        gguf_add_tensor(ctx, cur);
        printf("Added tensor %d\n", i);
    }

    printf("Writing file...\n"); 

    gguf_write_to_file(ctx, fname.c_str(), false);

    printf("%s: wrote file '%s'\n", __func__, fname.c_str());

    ggml_free(ctx_data);
    gguf_free(ctx);
}

// END NON-GGML IMPLEMENTATION

int main(int argc, char ** argv) {
    ctrl_params cparams;
    int skipme = ctrlvec_params_parse(argc, argv, cparams);

    // FIXME hack to skip the ctrlvec args in parsing gpt params
    argc -= skipme;
    argv += skipme;

    callback_data cb_data;
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    print_build_info();
    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    llama_model * model;
    llama_context * ctx;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", gpt_params_get_system_info(params).c_str());
    }

    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));

    /* TODO this just tokenizes the exact pos/neg strings, correct?
     * instead we want to create a bunch of starter prompts for it to work off
     * we need to run get_hidden_layers many many times and then figure out how to combine the resulting vectors
     * see the blogpost + python implementation for reference
     * 
     * https://vgel.me/posts/representation-engineering/
     * https://github.com/vgel/repeng/blob/main/repeng/extract.py
     */
    std::vector<llama_token> tokens_pos = ::llama_tokenize(ctx, cparams.positive, add_bos);
    std::vector<llama_token> tokens_neg = ::llama_tokenize(ctx, cparams.negative, add_bos);
    size_t max_seq_len = std::max(tokens_pos.size(), tokens_neg.size());
    padding_seq(ctx, tokens_pos, max_seq_len);
    padding_seq(ctx, tokens_neg, max_seq_len);
    cb_data.n_tokens = max_seq_len;
    cb_data.n_embd = llama_n_embd(model);

    cb_data.is_eval_pos = true;
    get_hidden_layers(ctx, tokens_pos);
    cb_data.is_eval_pos = false;
    get_hidden_layers(ctx, tokens_neg);

    printf("%f %f \n", cb_data.v_pos[0][4096], cb_data.v_pos[0][4096]);
    printf("%f %f \n", cb_data.v_neg[0][4096], cb_data.v_neg[0][4096]);

    calc_diff(cb_data);
    printf("%f %f \n", cb_data.v_diff[0][4096], cb_data.v_diff[0][4096]);

    pca(cb_data);

    // TODO figure out how to extract this from model - there's no API exposed to get model arch string
    // we need get_arch_name() from llama.cpp
    // TODO also has support been implemeneted for arches other than llama yet? see #5970
    std::string model_hint = "llama";
    export_gguf(cb_data, cparams.outfile, model_hint);

    //llama_print_timings(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
