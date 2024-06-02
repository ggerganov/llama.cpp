#include "common.h"
#include "llama.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

// TODO read everything over and make sure it makes sense because I'm dropping logic errors left and right - Christian

struct callback_data {
    std::vector<uint8_t> data;

    int n_tokens = 0;
    int n_embd = 0;
    bool is_eval_pos = true;

    // each element of the vector correspond to one layer
    std::vector<struct ggml_tensor *> v_pos;       // vector of matrices of size [n_embd, n_tokens]
    std::vector<struct ggml_tensor *> v_neg;       // vector of matrices of size [n_embd, n_tokens]
    std::vector<struct ggml_tensor *> v_final;     // vector of finished vectors of size [n_embd]
    std::vector<struct ggml_tensor *> v_diff; // vector of matrices of size [n_embd, m] where m ~ n_tokens * n_completions

    // each element of the outer vector correspond to one layer, each element of the inner vector correspond to one prompt pass
    std::vector<std::vector<struct ggml_tensor *>> v_diffs_wrapped; // vector of compiled diff matrices to be concatenated

    // TODO ggml destructor?
    ~callback_data() {
        for (auto ptr : v_pos) free(ptr);
        for (auto ptr : v_neg) free(ptr);
        for (auto ptr : v_diff) free(ptr);
        for (auto ptr : v_final) free(ptr);
        for (auto & vec : v_diffs_wrapped) for (auto ptr : vec) free(ptr);
    }
};

struct ctrl_params {
    /* default meta parameters */
    bool always_reload = false;
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

template <typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

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
    printf("       --always-reload      reload the model for every new template to parse (not recommended)\n");
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

    // TODO does this look right?
    struct ggml_tensor * t_host;
    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        struct ggml_init_params params = {
            /*.mem_size   =*/ n_bytes,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context * ctx_data = ggml_init(params);
        t_host = ggml_new_tensor_2d(ctx_data, t->type, t->ne[0], t->ne[1]);
        ggml_backend_tensor_get(t, t_host->data, 0, n_bytes);
    }
    else t_host = t;

    if (t_host->type == GGML_TYPE_F32) {
        if (cb_data->is_eval_pos) {
            cb_data->v_pos.push_back(t_host);
        } else {
            cb_data->v_neg.push_back(t_host);
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

static bool is_row_all_zeros(struct ggml_tensor * diff, size_t row, size_t cols, float eps = 1e-6) {
    for (size_t i = 0; i < cols; ++i) if (ggml_get_f32_nd(diff, row, i, 0, 0) > eps) return false;
    return true;
}

static void calc_diff(callback_data & cb_data) {
    // TODO: assert cb_data.v_pos.size() == cb_data.v_neg.size()
    cb_data.v_diffs_wrapped.resize(cb_data.v_pos.size());
    for (size_t il = 0; il < cb_data.v_pos.size(); il++) {
        auto & inp_pos = cb_data.v_pos[il];
        auto & inp_neg = cb_data.v_neg[il];
        auto n_bytes = ggml_nbytes(inp_pos);

        struct ggml_init_params params = {
            /*.mem_size   =*/ n_bytes,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context * ctx_data = ggml_init(params);

        // TODO is this the best way to get dimension? i don't know which way n_embd/n_tokens go
        // for that matter can we get rid of n_embd/n_tokens fields in favor of ne[0]/ne[1]?
        struct ggml_tensor * dest = ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, inp_pos->ne[0], inp_pos->ne[1]);
        for (size_t i = 0; i < cb_data.n_embd; i++) {
            for (size_t j = 0; j < cb_data.n_tokens; j++) {
                ggml_set_f32_nd(dest, i, j, 0, 0, ggml_get_f32_nd(inp_pos, i, j, 0, 0) - ggml_get_f32_nd(inp_neg, i, j, 0, 0));
            }
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

        // TODO I don't know if this is the right dimension but presumably it is
        struct ggml_tensor * diff = ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, nonzero_rows.size(), inp_pos->ne[1]);

        size_t offset = 0;
        for (size_t i = 0; i < nonzero_rows.size(); ++i) {
            float * origin = (float *)(dest->data) + nonzero_rows[i] * cb_data.n_embd;
            memcpy((float *)(diff->data) + offset, origin, cb_data.n_embd * sizeof(float));
            offset += cb_data.n_embd;
        }

        cb_data.v_diffs_wrapped[il].push_back(diff);
        free(dest);
    }
}

static void concatenate_diffs(callback_data & cb_data) {
    for (size_t i = 0; i < cb_data.v_diffs_wrapped.size(); ++i) {
        std::vector<struct ggml_tensor *> & vec = cb_data.v_diffs_wrapped[i];
        size_t n_rows_total = 0;
        for (size_t j = 0; j < vec.size(); ++j) {
            // TODO likewise no clue if this is right
            n_rows_total += vec[j]->ne[0];
        }

        struct ggml_init_params params = {
            /*.mem_size   =*/ cb_data.n_embd * n_rows_total * sizeof(float),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context * ctx_data = ggml_init(params);

        // std::cout << "n_rows_total: " << n_rows_total << std::endl;
        struct ggml_tensor * diff = ggml_new_tensor_2d(ctx_data, GGML_TYPE_F32, cb_data.n_embd, n_rows_total);
        size_t offset = 0;
        for (size_t j = 0; j < vec.size(); ++j) {
            float * origin = (float *)(vec[j]->data);
            // TODO again not sure about dimension
            memcpy((float *)(diff->data) + offset, origin, vec[j]->ne[0] * cb_data.n_embd * sizeof(float));
            offset += vec[j]->ne[0] * cb_data.n_embd;
        }
        cb_data.v_diff.push_back(diff);
    }
}

struct pca_model {
    struct ggml_tensor * v_diff_original;
    struct ggml_tensor * square;
    struct ggml_tensor * eigenvector;

    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_pca_model(pca_model & model, struct ggml_tensor * v_diff_original, const int n_embd) {
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0); // init device 0
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
    model.backend = ggml_backend_metal_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }

    const int num_tensors = 3;

    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    model.ctx = ggml_init(params);

    model.v_diff_original = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, v_diff_original->ne[0], v_diff_original->ne[1]);
    model.square = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_embd, n_embd);
    model.eigenvector = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, n_embd);

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    ggml_backend_tensor_set(model.v_diff_original, v_diff_original->data, 0, ggml_nbytes(v_diff_original));
    // no need to load anything into square yet

    // initialize model.eigenvector to random vector
    std::vector<float> random_vec = std::vector<float>();
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < n_embd; ++i) {
        random_vec.push_back(distribution(generator));
    }

    // we don't normalize it at first but that shouldn't be a problem
    ggml_backend_tensor_set(model.eigenvector, random_vec.data(), 0, ggml_nbytes(model.eigenvector));
}

struct ggml_cgraph * square_diff_graph(const pca_model & model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };
    struct ggml_context * ctx0 = ggml_init(params0);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * square = ggml_mul_mat(ctx0, model.v_diff_original, model.v_diff_original);

    ggml_build_forward_expand(gf, square);

    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor * compute_square(const pca_model & model, ggml_gallocr_t allocr, int n_threads) {
    struct ggml_cgraph * gf = square_diff_graph(model);

    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

    ggml_backend_graph_compute(model.backend, gf);

    return gf->nodes[gf->n_nodes - 1];
}

struct ggml_cgraph * power_iteration_graph(const pca_model & model, float tolerance) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };
    struct ggml_context * ctx0 = ggml_init(params0);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * b_tensor = ggml_mul_mat(ctx0, model.square, model.eigenvector);
    // TODO difference between ggml_norm and ggml_norm_inplace?
    // also is this the right way to do multi-step graphs?
    b_tensor = ggml_norm_inplace(ctx0, b_tensor, tolerance);

    ggml_build_forward_expand(gf, b_tensor);

    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor * compute_piter(const pca_model & model, ggml_gallocr_t allocr, int n_threads, float tolerance) {
    struct ggml_cgraph * gf = power_iteration_graph(model, tolerance);

    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

    ggml_backend_graph_compute(model.backend, gf);

    return gf->nodes[gf->n_nodes - 1];
}

static void power_iteration(callback_data & cb_data, int idx, int n_threads, int maxIterations = 1000, float tolerance = 1e-8) {
    pca_model model;
    load_pca_model(model, cb_data.v_diff[idx], cb_data.n_embd);

    ggml_gallocr_t allocr = NULL;

    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = square_diff_graph(model);
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

        fprintf(stderr, "%s: square diff, compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);
    }

    struct ggml_tensor * square = compute_square(model, allocr, n_threads);
    ggml_backend_tensor_get(square, model.square->data, 0, ggml_nbytes(square));

    // yes?
    ggml_gallocr_free(allocr);

    for (int iter = 0; iter < maxIterations; ++iter) {

        // TODO do I need to reset it like this every time?
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        struct ggml_tensor * host_new_eigenvector = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, cb_data.n_embd);
        struct ggml_tensor * b_tensor = compute_piter(model, allocr, n_threads, tolerance);
        ggml_backend_tensor_get(b_tensor, host_new_eigenvector->data, 0, ggml_nbytes(b_tensor));

        // convergence check
        float diff = 0.0;
        for (int i = 0; i < cb_data.n_embd; ++i) {
            diff += std::pow(((float *)(host_new_eigenvector->data))[i] - ((float *)(model.eigenvector->data))[i], 2);
        }

        // update eigenvector
        ggml_backend_tensor_set(model.eigenvector, host_new_eigenvector->data, 0, ggml_nbytes(model.eigenvector));

        try {
            if (std::sqrt(diff) < tolerance) {
                break;
            }
        }
        catch (std::exception & e) {
            // catch division by zero I guess
            break;
        }
    }

    // push back v_final with eigenvector
    ggml_backend_tensor_get(model.eigenvector, cb_data.v_final[idx]->data, 0, ggml_nbytes(model.eigenvector));
}

static void pca(callback_data & cb_data, int n_threads) {
    printf("Running PCA...\n");
    for (int il = 0; il < cb_data.v_diff.size(); ++il) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ cb_data.n_embd * sizeof(float),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context * ctx_data = ggml_init(params);
        cb_data.v_final.push_back(ggml_new_tensor_1d(ctx_data, GGML_TYPE_F32, cb_data.n_embd));
        power_iteration(cb_data, il, n_threads);
        printf("Done with layer %d\n", il);
        printf("il = %d | %f %f \n", il, ggml_get_f32_1d(cb_data.v_final[il], 0), ggml_get_f32_1d(cb_data.v_final[il], 1));
    }
    printf("Done with PCA.\n");
    printf("Done with PCA.\n");
}

static void export_gguf(callback_data & cb_data, int n_layers, const std::string fname, const std::string model_hint) {
    struct gguf_context * ctx = gguf_init_empty();

    size_t v_final_size_eff = n_layers - 1;
    
    const std::string arch = "controlvector";
    gguf_set_val_str(ctx, "general.architecture", arch.c_str());
    gguf_set_val_str(ctx, (arch + ".model_hint").c_str(), model_hint.c_str());
    gguf_set_val_i32(ctx, (arch + ".layer_count").c_str(), v_final_size_eff);

    for (size_t i = 0; i < v_final_size_eff; ++i) {
        // TODO this number is probably not right - figure out which layer is which
        // i'm pretty sure it's right now
        const std::string name = "direction." + to_string(i+1);

        ggml_set_name(cb_data.v_final[i], name.c_str());

        gguf_add_tensor(ctx, cb_data.v_final[i]);
        printf("Added tensor %zu\n", i);
    }

    printf("Writing file...\n"); 

    gguf_write_to_file(ctx, fname.c_str(), false);

    printf("%s: wrote file '%s'\n", __func__, fname.c_str());

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

    // create templated prompts
    for (int i = 0; i < n_prompts; ++i) {
        populate_entries(cparams, cparams.positive_prompts[i], cparams.negative_prompts[i]);
    }

    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));

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
        for (auto ptr : cb_data.v_pos) free(ptr);
        for (auto ptr : cb_data.v_neg) free(ptr);
        cb_data.v_pos.clear();
        cb_data.v_neg.clear();
    }

    concatenate_diffs(cb_data);
    pca(cb_data, cparams.n_threads);
    //printf("v_final %f %f \n", cb_data.v_final[0][0], cb_data.v_final[0][1]);

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
