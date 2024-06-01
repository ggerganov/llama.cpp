#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

// TODO read everything over and make sure it makes sense because you're dropping logic errors left and right

struct diff_wrapper {
    float * diff;   // matrix of size [n_rows, cb_data.n_embd] with zero rows stripped
    size_t n_rows;  // number of rows in the matrix for size calculation
};

/* TODO part of multithreading
struct tokens_pair {
    size_t max_seq_len;
    std::string positive;
    std::string negative;
    std::vector<llama_token> tokens_pos;
    std::vector<llama_token> tokens_neg;
}; */

struct callback_data {
    std::vector<uint8_t> data;

    int n_tokens = 0;
    int n_embd = 0;
    bool is_eval_pos = true;

    // each element of the vector correspond to one layer
    std::vector<float *> v_pos;       // vector of matrices of size [n_embd, n_tokens]
    std::vector<float *> v_neg;       // vector of matrices of size [n_embd, n_tokens]
    std::vector<float *> v_final;     // vector of finished vectors of size [n_embd]
    std::vector<diff_wrapper> v_diff; // vector of matrices of size [n_embd, m] where m ~ n_tokens * n_completions

    // each element of the outer vector correspond to one layer, each element of the inner vector correspond to one prompt pass
    std::vector<std::vector<diff_wrapper>> v_diffs_wrapped; // vector of compiled diff matrices to be concatenated

    ~callback_data() {
        for (auto ptr : v_pos) free(ptr);
        for (auto ptr : v_neg) free(ptr);
        for (auto ptr : v_diff) free(ptr.diff);
        for (auto ptr : v_final) free(ptr);
        for (auto & vec : v_diffs_wrapped) for (auto ptr : vec) free(ptr.diff);
    }
};

struct ctrl_params {
    /* default meta parameters */
    bool always_reload = false;
    // TODO part of multithreading
    // bool max_batch = false;
    int n_completions = 64;
    int n_threads = 8;

    /* default filepaths */
    std::string outfile = "control_vector.gguf";
    std::string completions_file = "examples/control-vector-generator/completions.txt";
    std::string positive_prompts_file = "examples/control-vector-generator/positive.txt";
    std::string negative_prompts_file = "examples/control-vector-generator/negative.txt";

    /* pair of prompts to be used for generating the vectors */
    std::vector<std::string> positive_prompts;
    std::vector<std::string> negative_prompts;

    /* pair of prompts to be used for testing */
    std::vector<std::string> positive_entries;
    std::vector<std::string> negative_entries;
};

static void print_usage(const char * executable) {
    printf("\n");
    printf("usage: %s [options] -m <model> [gpt-opts]", executable);
    printf("\n");
    printf("Creates a GGUF control vector for a given model.");
    printf("\n");
    printf("options:\n");
    printf("  -h,  --help               show this help message and exit\n");
    printf("  -o,  --outfile            output file\n");
    printf("                              default: 'control_vector.gguf'\n");
    printf("  -pf, --positive-file      positive prompts file, one prompt per line\n");
    printf("                              default: 'examples/control-vector-generator/positive.txt'\n");
    printf("  -nf, --negative-file      negative prompts file, one prompt per line\n");
    printf("                              default: 'examples/control-vector-generator/negative.txt'\n");
    printf("  -cf, --completions-file   completions file\n");
    printf("                              default: 'examples/control-vector-generator/completions.txt'\n");
    printf("  -nc, --num-completions N  number of lines of completions file to use\n");
    printf("                              default: 64\n");
    printf("  -t,  --num-threads N      number of threads to use (do not confuse with gpt-opts -t)\n");
    printf("                              default: 8\n");
    printf("       --always-reload      reload the model for every new template to parse\n");
    // TODO part of multithreading
    //printf("       --max-batch          maximize batch sizes, rather than optimizing for multithreading\n");
    printf("\n");
    printf("gpt-opts:\n");
    printf("  other options from main\n");
    printf("\n");
}

static int ctrlvec_params_parse_ex(int argc, char ** argv, ctrl_params & params) {
    std::string arg;
    const std::string arg_prefix = "-";
    // hack to skip ctrlvec args in gpt_parse_params but we'll leave it as is
    int skipme = 0;

    for(int arg_idx = 1; arg_idx < argc; ++arg_idx) {
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
        if (arg == "--outfile" || arg == "-o") {
            if (++arg_idx < argc && strncmp(argv[arg_idx], arg_prefix.c_str(), 2) != 0) {
                params.outfile = argv[arg_idx];
                skipme += 2;
            } else {
                throw std::invalid_argument("error: missing argument for " + arg);
            }
        }
        if (arg == "--completions-file" || arg == "-cf") {
            if (++arg_idx < argc && strncmp(argv[arg_idx], arg_prefix.c_str(), 2) != 0) {
                params.completions_file = argv[arg_idx];
                skipme += 2;
            } else {
                throw std::invalid_argument("error: missing argument for " + arg);
            }
        }
        if (arg == "--positive-file" || arg == "-pf") {
            if (++arg_idx < argc && strncmp(argv[arg_idx], arg_prefix.c_str(), 2) != 0) {
                params.positive_prompts_file = argv[arg_idx];
                skipme += 2;
            } else {
                throw std::invalid_argument("error: missing argument for " + arg);
            }
        }
        if (arg == "--negative-file" || arg == "-nf") {
            if (++arg_idx < argc && strncmp(argv[arg_idx], arg_prefix.c_str(), 2) != 0) {
                params.negative_prompts_file = argv[arg_idx];
                skipme += 2;
            } else {
                throw std::invalid_argument("error: missing argument for " + arg);
            }
        }
        if (arg == "--num-completions" || arg == "-nc") {
            if (++arg_idx < argc && strncmp(argv[arg_idx], arg_prefix.c_str(), 2) != 0) {
                try {
                    params.n_completions = std::stoi(argv[arg_idx]);
                }
                catch (const std::invalid_argument & ex) {
                    throw std::invalid_argument("error: invalid argument for " + arg);
                }
                skipme += 2;
            } else {
                throw std::invalid_argument("error: missing argument for " + arg);
            }
        }
        if (arg == "--num-threads" || arg == "-t") {
            if (++arg_idx < argc && strncmp(argv[arg_idx], arg_prefix.c_str(), 2) != 0) {
                try {
                    params.n_threads = std::stoi(argv[arg_idx]);
                }
                catch (const std::invalid_argument & ex) {
                    throw std::invalid_argument("error: invalid argument for " + arg);
                }
                skipme += 2;
            } else {
                throw std::invalid_argument("error: missing argument for " + arg);
            }
        }
        if (arg == "--always-reload") {
            params.always_reload = true;
            skipme += 1;
        }
        /* TODO part of multithreading
        if (arg == "--max-batch") {
            params.max_batch = true;
            skipme += 1;
        } */
        // TODO it might be nice QoL to have single positive/negative args
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

static std::vector<std::string> ctrlvec_load_prompt_file(std::string path) {
    std::vector<std::string> output;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + path);
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) { // skip empty lines
            output.push_back(line);
        }
    }
    file.close();
    return output;
}

static std::string format_template(std::string persona, std::string suffix) {
    //const std::string user_tag = "[INST]";
    //const std::string asst_tag = "[/INST]";
    //return user_tag + " Act as if you're extremely " + persona + ". " + asst_tag + " " + suffix;
    // TODO make this dynamic - allow the user to change it somehow - and adapt based on model
    return persona + " " + suffix; // entry in positive/negative.txt must already be formatted i.e. "[INST] Act as if you're extremely happy. [/INST]"
}

static void populate_entries(ctrl_params & cparams, std::string positive, std::string negative) {
    std::string line;
    std::ifstream completions_file(cparams.completions_file);
    int i = 0;
    if (completions_file.is_open()) {
        while (std::getline(completions_file, line) && i < cparams.n_completions) {
            // TODO replicate the truncations done by the python implementation
            cparams.positive_entries.push_back(format_template(positive, line));
            cparams.negative_entries.push_back(format_template(negative, line));
            i++;
        }
        completions_file.close();
    } else {
        throw std::invalid_argument("error: invalid completions file or file could not be opened");
    }
}

/* TODO part of multithreading
static size_t tokenize_pair(tokens_pair & tp, llama_context * ctx, const std::string & pos, const std::string & neg, const bool add_bos) {
    tp.positive = pos;
    tp.negative = neg;
    tp.tokens_pos = ::llama_tokenize(ctx, pos, add_bos);
    tp.tokens_neg = ::llama_tokenize(ctx, neg, add_bos);
    tp.max_seq_len = std::max(tp.tokens_pos.size(), tp.tokens_neg.size());
    padding_seq(ctx, tp.tokens_pos, tp.max_seq_len);
    padding_seq(ctx, tp.tokens_neg, tp.max_seq_len);
    return 2 * max_seq_len;
}

// current batching strategy works as follows:
// each batch runs on one model load, since we reload the model after every batch to clear context
// therefore each batch must be small enough to fit in the context size
// we try to make the batches multiples of thread count so threads are used most efficiently
static std::vector<std::vector<tokens_pair>> batch_prompts(llama_context * ctx, ctrl_params & cparams, int n_ctx, const bool add_bos) {
    std::vector<std::vector<tokens_pair>> batched_prompts;
    std::vector<tokens_pair> thread_batch;
    std::vector<tokens_pair> batch;
    size_t n_batch_tokens = 0;

    for (size_t i = 0; i < cparams.positive_entries.size(); ++i) {
        tokens_pair tp;
        size_t n_tokens = tokenize_pair(tp, ctx, cparams.positive_entries[i], cparams.negative_entries[i], add_bos);
        n_batch_tokens += n_tokens;

        if (n_batch_tokens > n_ctx) {
            if (cparams.max_batch) {
                batch.insert(batch.end(), thread_batch.begin(), thread_batch.end());
                thread_batch.clear();
            }
            batched_prompts.push_back(batch);
            batch.clear();
            n_batch_tokens = n_tokens;
        }

        thread_batch.push_back(tp);
        
        if (thread_batch.size() >= cparams.n_threads) {
            batch.insert(batch.end(), thread_batch.begin(), thread_batch.end());
            thread_batch.clear();;
        }
    }

    if (!thread_batch.empty()) {
        batch.insert(batch.end(), thread_batch.begin(), thread_batch.end());
    }
    if (!batch.empty()) {
        batched_prompts.push_back(batch);
    }

    return batched_prompts;
} */

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

static bool is_row_all_zeros(float * diff, size_t row, size_t cols, float eps = 1e-6) {
    for (size_t i = 0; i < cols; ++i) if (diff[row * cols + i] > eps) return false;
    return true;
}

static void calc_diff(callback_data & cb_data) {
    // TODO: assert cb_data.v_pos.size() == cb_data.v_neg.size()
    const size_t n_elems = cb_data.n_embd * cb_data.n_tokens;
    cb_data.v_diffs_wrapped.resize(cb_data.v_pos.size());
    for (size_t il = 0; il < cb_data.v_pos.size(); il++) {
        auto & inp_pos = cb_data.v_pos[il];
        auto & inp_neg = cb_data.v_neg[il];
        float * dest = (float *) malloc(n_elems * sizeof(float));
        for (size_t i = 0; i < n_elems; i++) {
            dest[i] = inp_pos[i] - inp_neg[i];
        }

        // TODO can we make this faster? like check during the above operation rather than on a second pass?

        // strip zero rows
        std::vector<size_t> nonzero_rows;
        for (int i = 0; i < cb_data.n_tokens; ++i) {
            if (!is_row_all_zeros(dest, i, cb_data.n_embd)) {
                nonzero_rows.push_back(i);
            }
        }

        /* debug
        if(cb_data.n_tokens != nonzero_rows.size()) {
            std::cout << "original n_tokens: " << cb_data.n_tokens << std::endl;
            std::cout << "zero rows in layer " << il << ": " << cb_data.n_tokens - nonzero_rows.size() << std::endl;
        } */

        struct diff_wrapper dw;
        dw.n_rows = nonzero_rows.size();
        dw.diff = (float *) malloc(dw.n_rows * cb_data.n_embd * sizeof(float));

        size_t offset = 0;
        for (size_t i = 0; i < dw.n_rows; ++i) {
            float * origin = dest + nonzero_rows[i] * cb_data.n_embd;
            memcpy(dw.diff + offset, origin, cb_data.n_embd * sizeof(float));
            offset += cb_data.n_embd;
        }

        cb_data.v_diffs_wrapped[il].push_back(dw);
        free(dest);
    }
}

// TODO do we want to multithread this? it takes very little time as it is
static void concatenate_diffs(callback_data & cb_data) {
    for (size_t i = 0; i < cb_data.v_diffs_wrapped.size(); ++i) {
        std::vector<diff_wrapper> & vec = cb_data.v_diffs_wrapped[i];
        size_t n_rows_total = 0;
        for (size_t j = 0; j < vec.size(); ++j) {
            n_rows_total += vec[j].n_rows;
        }
        // std::cout << "n_rows_total: " << n_rows_total << std::endl;
        float * diff = (float *) malloc(n_rows_total * cb_data.n_embd * sizeof(float));
        size_t offset = 0;
        for (size_t j = 0; j < vec.size(); ++j) {
            float * origin = vec[j].diff;
            memcpy(diff + offset, origin, vec[j].n_rows * cb_data.n_embd * sizeof(float));
            offset += vec[j].n_rows * cb_data.n_embd;
        }
        struct diff_wrapper dw;
        dw.n_rows = n_rows_total;
        dw.diff = diff;
        cb_data.v_diff.push_back(dw);
    }
}

// BEGIN NON-GGML IMPLEMENTATION

// TODO translate to ggml
// this probably doesn't want to be a separate function - put it into the compute graph as a step in processing each layer
static float* square_diff(callback_data & cb_data, size_t idx) {
    float* result = new float[cb_data.n_embd * cb_data.n_embd];
    std::memset(result, 0, cb_data.n_embd * cb_data.n_embd * sizeof(float));
    for (size_t i = 0; i < (size_t) cb_data.n_embd; i++) {
        for (size_t j = 0; j < (size_t) cb_data.n_embd; j++) {
            float sum = 0.0f;
            // watch out for indexing - can't just use cb_data.n_tokens
            for (size_t k = 0; k < cb_data.v_diff[idx].n_rows; k++) {
                sum += cb_data.v_diff[idx].diff[i + cb_data.n_embd * k] * cb_data.v_diff[idx].diff[j + cb_data.n_embd * k];
            }
            result[i * cb_data.n_embd + j] = sum;
        }
    }
    return result;
}

// TODO translate to ggml (this is a built-in function in ggml)
static void normalize_inplace(std::vector<float> & vec) {
    // inefficient(?) norm computation
    float norm = 0.0f;
    for (const float& val : vec) {
        norm += val * val;
    }
    if(norm == 0) throw std::runtime_error("norm is zero"); 
    norm = std::sqrt(norm);
    for (float& val : vec) {
        val /= norm;
    }
}

// TODO translate to ggml (this is a built-in function in ggml)
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
static void pca(callback_data & cb_data, int n_threads) {
    int n_layers = cb_data.v_diff.size();
    std::vector<std::thread> threads;
    cb_data.v_final.reserve(n_layers);
    auto worker_function = [&](int worker_id) {
        for (int il = worker_id; il < n_layers; il += n_threads) {
            float * matrix = square_diff(cb_data, il);
            std::vector<float> eigenvector = power_iteration(cb_data, matrix);
            cb_data.v_final[il] = (float *) malloc(eigenvector.size() * sizeof(float));
            memcpy(cb_data.v_final[il], eigenvector.data(), eigenvector.size() * sizeof(float));
            printf("Done with layer %d\n", il);
            printf("il = %d | %f %f \n", il, cb_data.v_final[il][0], cb_data.v_final[il][1]);
        }
    };
    printf("Running PCA...\n");
    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back(worker_function, i);
    }
    for (auto & th : threads) th.join();
    printf("Done with PCA.\n");
}

template <typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

static void export_gguf(callback_data & cb_data, int n_layers, const std::string fname, const std::string model_hint) {
    struct gguf_context * ctx = gguf_init_empty();

    size_t v_final_size_eff = n_layers - 1;
    
    const std::string arch = "controlvector";
    gguf_set_val_str(ctx, "general.architecture", arch.c_str());
    gguf_set_val_str(ctx, (arch + ".model_hint").c_str(), model_hint.c_str());
    gguf_set_val_i32(ctx, (arch + ".layer_count").c_str(), v_final_size_eff);

    struct ggml_init_params params = {
        /*.mem_size   =*/ (ggml_tensor_overhead() * v_final_size_eff)
                            + (cb_data.n_embd * v_final_size_eff * sizeof(float)),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx_data = ggml_init(params);

    for (size_t i = 0; i < v_final_size_eff; ++i) {
        // TODO this number is probably not right - figure out which layer is which
        // the python implementation uses a dict to handle this, we don't know if it's 1, 2, 3, 4... or other
        const std::string name = "direction." + to_string(i+1);

        struct ggml_tensor * cur = ggml_new_tensor_1d(ctx_data, GGML_TYPE_F32, cb_data.n_embd);

        ggml_set_name(cur, name.c_str());

        float * data = (float *) cur->data;
        for(int j = 0; j < cb_data.n_embd; j++) {
            data[j] = cb_data.v_final[i][j];
        }

        gguf_add_tensor(ctx, cur);
        printf("Added tensor %zu\n", i);
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
    argc -= skipme;
    argv += skipme;

    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    // load prompts
    cparams.positive_prompts = ctrlvec_load_prompt_file(cparams.positive_prompts_file);
    cparams.negative_prompts = ctrlvec_load_prompt_file(cparams.negative_prompts_file);
    if (cparams.positive_prompts.size() != cparams.negative_prompts.size()) {
        fprintf(stderr, "number of positive and negative prompts must be equal");
        return 1;
    }
    if (cparams.positive_prompts.empty()) {
        fprintf(stderr, "must provide at least one prompt pair");
        return 1;
    }

    callback_data cb_data;

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    print_build_info();
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model to get hparams
    llama_model * model;
    llama_context * ctx;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    int n_ctx = llama_n_ctx(ctx);
    int n_layers = llama_n_layer(model);
    int n_embd = llama_n_embd(model);
    cb_data.n_embd = n_embd;
    int n_prompts = cparams.positive_prompts.size();
    
    // vector of finished vectors of size [n_embd], we have (n_layers - 1) vectors in total
    std::vector<float *> v_final(n_layers - 1, NULL);
    for (size_t i = 0; i < v_final.size(); ++i) {
        v_final[i] = (float *) calloc(n_embd, sizeof(float));
    }

    // create templated prompts
    for (int i = 0; i < n_prompts; ++i) {
        populate_entries(cparams, cparams.positive_prompts[i], cparams.negative_prompts[i]);
    }

    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));

    /* TODO part of multithreading
    std::vector<std::vector<tokens_pair>> & batched_prompts = batch_prompts(ctx, cparams, n_ctx, add_bos);
    std::vector<std::thread> threads;
    auto worker_function = [&](tokens_pair & tp) {
        printf("Evaluating prompt: \"%s\" - \"%s\" (%ld tokens)\n", tp.positive.c_str(), tp.negative.c_str(), tp.max_seq_len);
        // TODO so how do we deal with this?
        // TODO we only have one cb_data object that everything gets passed to. so we need to be able to write to a different object per thread
        // TODO but there's only one cb_eval function used as callback by the model... help wanted
    };
    printf("Batching prompts...\n");
    for (int i = 0; i < batched_prompts.size(); ++i) {
        for (int j = 0; j < batched_prompts[i].size(); ++j) {
            threads.emplace_back(worker_function, batched_prompts[i][j]);
        }
        for (auto & th : threads) th.join();
        
        // reload model for next batch
        llama_free(ctx);
        llama_free_model(model);
        std::tie(model, ctx) = llama_init_from_gpt_params(params);
    }
    printf("Done with batching prompts.\n");
    */

    int token_ct = 0;

    for(size_t i = 0; i < cparams.positive_entries.size(); ++i) {
        std::string positive_prompt = cparams.positive_entries[i];
        std::string negative_prompt = cparams.negative_entries[i];
        std::vector<llama_token> tokens_pos = ::llama_tokenize(ctx, positive_prompt, add_bos);
        std::vector<llama_token> tokens_neg = ::llama_tokenize(ctx, negative_prompt, add_bos);
        size_t max_seq_len = std::max(tokens_pos.size(), tokens_neg.size());
        padding_seq(ctx, tokens_pos, max_seq_len);
        padding_seq(ctx, tokens_neg, max_seq_len);
        cb_data.n_tokens = max_seq_len;

        // need to reload the model so it doesn't run out of context
        // this should scale with -c option passed by main
        token_ct += 2 * max_seq_len;
        if (token_ct > n_ctx || cparams.always_reload) {
            //break;
            llama_free(ctx);
            llama_free_model(model);
            std::tie(model, ctx) = llama_init_from_gpt_params(params);
            token_ct = 2 * max_seq_len;
        }
        if (token_ct > n_ctx) {
            fprintf(stderr, "context size exceeded on iteration %zu\n", i);
            break;
        }

        printf("Evaluating prompt: \"%s\" - \"%s\" (%ld tokens)\n", positive_prompt.c_str(), negative_prompt.c_str(), max_seq_len);

        cb_data.is_eval_pos = true;
        get_hidden_layers(ctx, tokens_pos);
        cb_data.is_eval_pos = false;
        get_hidden_layers(ctx, tokens_neg);

        // TODO check whether the same tokens correspond to zero rows because we don't seem to be getting many zero rows anymore
        // we get a lot of zero rows for the first few prompts and then they drop off
        // likewise most of the zero rows are in the first few layers for each prompt

        calc_diff(cb_data);

        // reset for next iteration
        cb_data.v_pos.clear();
        cb_data.v_neg.clear();
    }

    concatenate_diffs(cb_data);
    pca(cb_data, cparams.n_threads);
    printf("v_final %f %f \n", cb_data.v_final[0][0], cb_data.v_final[0][1]);

    llama_free(ctx);
    llama_free_model(model);

    // TODO figure out how to extract this from model - there's no API exposed to get model arch string
    // we need get_arch_name() from llama.cpp
    // TODO also has support been implemeneted for arches other than llama yet? see #5970
    std::string model_hint = "llama";
    export_gguf(cb_data, n_layers, cparams.outfile, model_hint);

    llama_backend_free();

    return 0;
}
