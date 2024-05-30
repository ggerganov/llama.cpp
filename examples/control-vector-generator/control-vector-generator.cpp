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
    for (size_t i = 0; i < cb_data.v_diff.size(); i++) {
        float* matrix = square_diff(cb_data, i);
        std::vector<float> eigenvector = power_iteration(cb_data, matrix);
        cb_data.v_final.push_back(&eigenvector[0]);
        delete[] matrix;
        // TODO make your print outputs nicer
        std::cout << "Done with layer " << i << "\n";
    }
}

template <typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

static void export_gguf(callback_data & cb_data, const std::string fname) {
    struct gguf_context * ctx = gguf_init_empty();

    gguf_set_val_str(ctx, "general.architecture", "controlvector");
    gguf_set_val_str(ctx, "controlvector.model_hint", "mistral"); // TODO steal this from the model somehow (arch)
    gguf_set_val_i32(ctx, "controlvector.layer_count", cb_data.v_final.size());

    //size_t buf_size = 3u*cb_data.n_embd*sizeof(float); // TODO how much size do i need???
    size_t buf_size = 128u*1024u*4096u;
    std::vector<uint8_t> buf(buf_size); 

    // TODO customize mem size - I have no idea
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx_data = ggml_init(params);

    // TODO direction tensor invalid??? probably because you start at 0. see below
    for (int i = 0; i < cb_data.v_final.size(); i++) {
        const std::string name = "direction." + to_string(i+1); // TODO figure out how to get the number for direction - dl repeng locally and debug
        // clone the repo and use importlib
        // git clone https://github.com/vgel/repeng.git

        struct ggml_tensor * cur = ggml_new_tensor_1d(ctx_data, GGML_TYPE_F32, cb_data.n_embd);

        std::cout << "Made it past tensor creation";

        ggml_set_name(cur, name.c_str());
        std::cout << "Made it past tensor name set";

        // whining about buf != NULL
        // TODO figure out how to set data
        //ggml_backend_tensor_set(cur, cb_data.v_final[i], 0, cb_data.n_embd * sizeof(float)); // if this doesn't work refer to gguf.cpp example
        {
            float * data = (float *) cur->data;
            for(int j = 0; j < ggml_nelements(cur); j++) {
                data[j] = cb_data.v_final[i][j];
            }
        }
        std::cout << "Made it past tensor backend set";

        gguf_add_tensor(ctx, cur);
        std::cout << "Added tensor " << i << "\n";
    }

    std::cout << "Writing file\n";

    gguf_write_to_file(ctx, fname.c_str(), false);

    printf("%s: wrote file '%s;\n", __func__, fname.c_str());

    ggml_free(ctx_data);
    gguf_free(ctx);
}

// END NON-GGML IMPLEMENTATION

int main(int argc, char ** argv) {
    callback_data cb_data;
    std::string prompt_pos = "happy";
    std::string prompt_neg = "sad";

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
    std::vector<llama_token> tokens_pos = ::llama_tokenize(ctx, prompt_pos, add_bos);
    std::vector<llama_token> tokens_neg = ::llama_tokenize(ctx, prompt_neg, add_bos);
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
    // TODO --outfile
    std::cout << "Done with PCA" << "\n";
    export_gguf(cb_data, "controlvector.gguf");

    //llama_print_timings(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
