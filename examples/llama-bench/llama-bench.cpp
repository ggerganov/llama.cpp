#include <algorithm>
#include <cassert>
#include <chrono>
#include <array>
#include <cinttypes>
#include <regex>
#include <stdio.h>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <numeric>
#include <map>
#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "build-info.h"

// utils
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

template<class T>
static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

template<class T>
static std::vector<T> split(const std::string & str, char delim) {
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        token_stream >> value;
        values.push_back(value);
    }
    return values;
}

template<typename T>
T avg(const std::vector<T> & v) {
    if (v.empty()) {
        return 0;
    }
    T sum = std::accumulate(v.begin(), v.end(), T(0));
    return sum / (T)v.size();
}

template<typename T>
T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) {
        return 0;
    }
    T mean = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    T stdev = std::sqrt(sq_sum / (T)(v.size() - 1) - mean * mean * (T)v.size() / (T)(v.size() - 1));
    return stdev;
}

// command line params
enum output_formats {CSV, JSON, MARKDOWN};

struct cmd_params {
    std::vector<std::string> model;
    std::vector<int> n_prompt;
    std::vector<int> n_gen;
    std::vector<int> n_batch;
    std::vector<bool> f32_kv;
    std::vector<int> n_threads;
    std::vector<int> n_gpu_layers;
    std::vector<int> main_gpu;
    std::vector<bool> mul_mat_q;
    std::vector<bool> low_vram;
    std::vector<std::array<float, LLAMA_MAX_DEVICES>> tensor_split;
    int reps;
    bool verbose;
    output_formats output_format;
};

static const cmd_params cmd_params_defaults = {
    /* model         */ {"models/7B/ggml-model-q4_0.bin"},
    /* n_prompt      */ {512},
    /* n_gen         */ {128},
    /* n_batch       */ {512},
    /* f32_kv        */ {false},
    /* n_threads     */ {get_num_physical_cores()},
    /* n_gpu_layers  */ {99},
    /* main_gpu      */ {0},
    /* mul_mat_q     */ {true},
    /* low_vram      */ {false},
    /* tensor_split  */ {{}},
    /* reps          */ 5,
    /* verbose       */ false,
    /* output_format */ MARKDOWN
};

static void print_usage(int /* argc */, char ** argv) {
    fprintf(stdout, "usage: %s [options]\n", argv[0]);
    fprintf(stdout, "\n");
    fprintf(stdout, "options:\n");
    fprintf(stdout, "  -h, --help\n");
    fprintf(stdout, "  -m, --model <filename>            (default: %s)\n", join(cmd_params_defaults.model, ",").c_str());
    fprintf(stdout, "  -p, --n-prompt <n>                (default: %s)\n", join(cmd_params_defaults.n_prompt, ",").c_str());
    fprintf(stdout, "  -n, --n-gen <n>                   (default: %s)\n", join(cmd_params_defaults.n_gen, ",").c_str());
    fprintf(stdout, "  -b, --batch-size <n>              (default: %s)\n", join(cmd_params_defaults.n_batch, ",").c_str());
    fprintf(stdout, "  --memory-f32 <0|1>                (default: %s)\n", join(cmd_params_defaults.f32_kv, ",").c_str());
    fprintf(stdout, "  -t, --threads <n>                 (default: %s)\n", join(cmd_params_defaults.n_threads, ",").c_str());
    fprintf(stdout, "  -ngl N, --n-gpu-layers <n>        (default: %s)\n", join(cmd_params_defaults.n_gpu_layers, ",").c_str());
    fprintf(stdout, "  -mg i, --main-gpu <n>             (default: %s)\n", join(cmd_params_defaults.main_gpu, ",").c_str());
    fprintf(stdout, "  -lv, --low-vram <0|1>             (default: %s)\n", join(cmd_params_defaults.low_vram, ",").c_str());
    fprintf(stdout, "  -mmq, --mul-mat-q <0|1>           (default: %s)\n", join(cmd_params_defaults.mul_mat_q, ",").c_str());
    fprintf(stdout, "  -ts, --tensor_split <ts>                       \n");
    fprintf(stdout, "  -r, --repetitions <n>             (default: %d)\n", cmd_params_defaults.reps);
    fprintf(stdout, "  -o, --output <csv|json|md>        (default: %s)\n", cmd_params_defaults.output_format == CSV ? "csv" : cmd_params_defaults.output_format == JSON ? "json" : "md");
    fprintf(stdout, "  -v, --verbose                     (default: %s)\n", cmd_params_defaults.verbose ? "1" : "0");
    fprintf(stdout, "\n");
    fprintf(stdout, "Multiple values can be given for each parameter by separating them with ',' or by repeating the parameter.\n");

}

static cmd_params parse_cmd_params(int argc, char ** argv) {
    cmd_params params;
    std::string arg;
    bool invalid_param = false;
    const std::string arg_prefix = "--";
    const char split_delim = ',';

    params.verbose = cmd_params_defaults.verbose;
    params.output_format = cmd_params_defaults.output_format;
    params.reps = cmd_params_defaults.reps;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<std::string>(argv[i], split_delim);
            params.model.insert(params.model.end(), p.begin(), p.end());
        } else if (arg == "-p" || arg == "--n-prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.n_prompt.insert(params.n_prompt.end(), p.begin(), p.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.n_gen.insert(params.n_gen.end(), p.begin(), p.end());
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.n_batch.insert(params.n_batch.end(), p.begin(), p.end());
        } else if (arg == "--memory-f32") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.f32_kv.insert(params.f32_kv.end(), p.begin(), p.end());
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.n_threads.insert(params.n_threads.end(), p.begin(), p.end());
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<int>(argv[i], split_delim);
            params.n_gpu_layers.insert(params.n_gpu_layers.end(), p.begin(), p.end());
        } else if (arg == "-mg" || arg == "--main-gpu") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.main_gpu = split<int>(argv[i], split_delim);
        } else if (arg == "-lv" || arg == "--low-vram") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<bool>(argv[i], split_delim);
            params.low_vram.insert(params.low_vram.end(), p.begin(), p.end());
        } else if (arg == "-mmq" || arg == "--mul-mat-q") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = split<bool>(argv[i], split_delim);
            params.mul_mat_q.insert(params.mul_mat_q.end(), p.begin(), p.end());
        } else if (arg == "-ts" || arg == "--tensor-split") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            for (auto ts : split<std::string>(argv[i], split_delim)) {
                // split string by ; and /
                const std::regex regex{R"([;/]+)"};
                std::sregex_token_iterator it{ts.begin(), ts.end(), regex, -1};
                std::vector<std::string> split_arg{it, {}};
                GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

                std::array<float, LLAMA_MAX_DEVICES> tensor_split;
                for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
                    if (i < split_arg.size()) {
                        tensor_split[i] = std::stof(split_arg[i]);
                    } else {
                        tensor_split[i] = 0.0f;
                    }
                }
                params.tensor_split.push_back(tensor_split);
            }
        } else if (arg == "-r" || arg == "--repetitions") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.reps = std::stoi(argv[i]);
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            if (argv[i] == std::string("csv")) {
                params.output_format = CSV;
            } else if (argv[i] == std::string("json")) {
                params.output_format = JSON;
            } else if (argv[i] == std::string("md")) {
                params.output_format = MARKDOWN;
            } else {
                invalid_param = true;
                break;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else {
            invalid_param = true;
            break;
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }

    // set defaults
    if (params.model.empty())        { params.model = cmd_params_defaults.model; }
    if (params.n_prompt.empty())     { params.n_prompt = cmd_params_defaults.n_prompt; }
    if (params.n_gen.empty())        { params.n_gen = cmd_params_defaults.n_gen; }
    if (params.n_batch.empty())      { params.n_batch = cmd_params_defaults.n_batch; }
    if (params.f32_kv.empty())       { params.f32_kv = cmd_params_defaults.f32_kv; }
    if (params.n_gpu_layers.empty()) { params.n_gpu_layers = cmd_params_defaults.n_gpu_layers; }
    if (params.main_gpu.empty())     { params.main_gpu = cmd_params_defaults.main_gpu; }
    if (params.mul_mat_q.empty())    { params.mul_mat_q = cmd_params_defaults.mul_mat_q; }
    if (params.low_vram.empty())     { params.low_vram = cmd_params_defaults.low_vram; }
    if (params.tensor_split.empty()) { params.tensor_split = cmd_params_defaults.tensor_split; }
    if (params.n_threads.empty())    { params.n_threads = cmd_params_defaults.n_threads; }

    return params;
}

struct cmd_params_instance {
    std::string model;
    int n_prompt;
    int n_gen;
    int n_batch;
    bool f32_kv;
    int n_gpu_layers;
    int main_gpu;
    bool mul_mat_q;
    bool low_vram;
    std::array<float, LLAMA_MAX_DEVICES> tensor_split;

    int n_threads;

    llama_context_params to_llama_params() const {
        llama_context_params lparams = llama_context_default_params();
        lparams.n_ctx = n_prompt + n_gen;
        lparams.n_batch = n_batch;
        lparams.f16_kv = !f32_kv;
        lparams.n_gpu_layers = n_gpu_layers;
        lparams.main_gpu = main_gpu;
        lparams.mul_mat_q = mul_mat_q;
        lparams.low_vram = low_vram;
        lparams.tensor_split = tensor_split.data();

        return lparams;
    }
};

static std::vector<cmd_params_instance> get_cmd_params_instances_int(const cmd_params & params, int n_gen, int n_prompt) {
    std::vector<cmd_params_instance> instances;

    for (const auto & m : params.model)
    for (const auto & nb : params.n_batch)
    for (const auto & fk : params.f32_kv)
    for (const auto & nl : params.n_gpu_layers)
    for (const auto & mg : params.main_gpu)
    for (const auto & mmq : params.mul_mat_q)
    for (const auto & lv : params.low_vram)
    for (const auto & ts : params.tensor_split)
    for (const auto & nt : params.n_threads) {
        cmd_params_instance instance;

        instance.model = m;
        instance.n_prompt = n_prompt;
        instance.n_gen = n_gen;
        instance.n_batch = nb;
        instance.f32_kv = fk;
        instance.n_gpu_layers = nl;
        instance.main_gpu = mg;
        instance.mul_mat_q = mmq;
        instance.low_vram = lv;
        std::copy(std::begin(ts), std::end(ts), std::begin(instance.tensor_split));
        instance.n_threads = nt;

        instances.push_back(instance);
    }
    return instances;
}

static std::vector<cmd_params_instance> get_cmd_params_instances(const cmd_params & params) {
    std::vector<cmd_params_instance> instances;

    for (const auto & n_prompt : params.n_prompt) {
        if (n_prompt == 0) {
            continue;
        }
        auto instances_prompt = get_cmd_params_instances_int(params, 0, n_prompt);
        instances.insert(instances.end(), instances_prompt.begin(), instances_prompt.end());
    }

    for (const auto & n_gen : params.n_gen) {
        if (n_gen == 0) {
            continue;
        }
        auto instances_gen = get_cmd_params_instances_int(params, n_gen, 0);
        instances.insert(instances.end(), instances_gen.begin(), instances_gen.end());
    }

    return instances;
}

// models params
struct model_params {
    std::string filename;
    std::string type;

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {"filename", "type"};
        return fields;
    }

    // TODO: use a map instead
    std::vector<std::string> get_values() const {
        return {filename, type};
    }
};

static bool ggml_cpu_has_metal() {
#if defined(GGML_USE_METAL)
    return true;
#else
    return false;
#endif
}

// backend params
struct backend_params {
    static const std::string build_commit;
    static const int build_number;
    static const bool cuda;
    static const bool opencl;
    static const bool metal;
    static const bool gpu_blas;
    static const bool blas;
    int n_batch;
    int n_threads;
    bool f32_kv;
    int n_gpu_layers;
    int main_gpu;
    bool mul_mat_q;
    bool low_vram;
    std::array<float, LLAMA_MAX_DEVICES> tensor_split;

    static std::string get_backend() {
        if (cuda) {
            return "CUDA";
        }
        if (opencl) {
            return "OpenCL";
        }
        if (metal) {
            return "Metal";
        }
        if (gpu_blas) {
            return "GPU BLAS";
        }
        if (blas) {
            return "BLAS";
        }
        return "CPU";
    }

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {
            "build_number", "build_commit",
            "cuda", "opencl", "metal", "gpu_blas", "blas",
            "n_batch", "n_threads", "f16_kv",
            "n_gpu_layers", "main_gpu", "mul_mat_q", "low_vram", "tensor_split"
        };
        return fields;
    }

    std::vector<std::string> get_values() const {
        std::string tensor_split_str;
        int max_nonzero = 0;
        for (int i = 0; i < LLAMA_MAX_DEVICES; i++) {
            if (tensor_split[i] > 0) {
                max_nonzero = i;
            }
        }
        for (int i = 0; i < max_nonzero; i++) {
            tensor_split_str += std::to_string(tensor_split[i]);
            if (i < max_nonzero - 1) {
                tensor_split_str += "/";
            }
        }
        std::vector<std::string> values = {
            std::to_string(build_number), build_commit,
            std::to_string(cuda), std::to_string(opencl), std::to_string(metal), std::to_string(gpu_blas), std::to_string(blas),
            std::to_string(n_batch), std::to_string(n_threads), std::to_string(!f32_kv),
            std::to_string(n_gpu_layers), std::to_string(main_gpu), std::to_string(mul_mat_q), std::to_string(low_vram), tensor_split_str
        };
        return values;
    }
};

const std::string backend_params::build_commit = BUILD_COMMIT;
const int backend_params::build_number         = BUILD_NUMBER;
const bool backend_params::cuda     = !!ggml_cpu_has_cublas();
const bool backend_params::opencl   = !!ggml_cpu_has_clblast();
const bool backend_params::metal    = !!ggml_cpu_has_metal();
const bool backend_params::gpu_blas = !!ggml_cpu_has_gpublas();
const bool backend_params::blas     = !!ggml_cpu_has_blas();

// benchmark params
struct bench_params {
    int n_prompt ;
    int n_gen;

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {"n_prompt", "n_gen"};
        return fields;
    }

    std::vector<std::string> get_values() const {
        return {std::to_string(n_prompt), std::to_string(n_gen)};
    }
};

// timing results
struct timing_samples {
    std::vector<uint64_t> t_ns;

    uint64_t avg() const {
        return ::avg(t_ns);
    }

    uint64_t stdev() const {
        return ::stdev(t_ns);
    }

    std::vector<double> get_ts(int n) const {
        std::vector<double> ts;
        std::transform(t_ns.begin(), t_ns.end(), std::back_inserter(ts), [n](uint64_t t) { return 1e9 * n / t; });
        return ts;
    }

    double avg_ts(uint64_t n) const {
        return ::avg(get_ts(n));
    }

    double stddev_ts(uint64_t n) const {
        return ::stdev(get_ts(n));
    }

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {"t_ns"};
        return fields;
    }
};

struct test {
    model_params mparams = {};
    bench_params bparams = {};
    backend_params bkparams = {};
    timing_samples tsamples = {};

    test(const cmd_params_instance & inst, const llama_model * lmodel, const llama_context * ctx) {
        mparams.filename = inst.model;
        char buf[128];
        llama_model_type(lmodel, buf, sizeof(buf));
        mparams.type = buf;

        bparams.n_prompt = inst.n_prompt;
        bparams.n_gen = inst.n_gen;

        bkparams.n_batch = inst.n_batch;
        bkparams.f32_kv = inst.f32_kv;
        bkparams.n_threads = inst.n_threads;
        bkparams.n_gpu_layers = inst.n_gpu_layers;
        bkparams.main_gpu = inst.main_gpu;
        bkparams.mul_mat_q = inst.mul_mat_q;
        bkparams.low_vram = inst.low_vram;
        bkparams.tensor_split = inst.tensor_split;

        (void) ctx;
    }
};

struct printer {
    FILE * fout;
    virtual void print_header(const cmd_params & params) { (void)params; };
    virtual void print_test(const test & t) = 0;
    virtual void print_footer() {};
};

struct csv_printer : public printer {
    virtual void print_header(const cmd_params & params) {
        std::vector<std::string> fields;
        fields.insert(fields.end(), model_params::get_fields().begin(), model_params::get_fields().end());
        fields.insert(fields.end(), bench_params::get_fields().begin(), bench_params::get_fields().end());
        fields.insert(fields.end(), backend_params::get_fields().begin(), backend_params::get_fields().end());
        fields.insert(fields.end(), timing_samples::get_fields().begin(), timing_samples::get_fields().end());
        fprintf(fout, "%s\n", join(fields, ",").c_str());
        (void) params;
    }

    void print_values(const std::vector<std::string> & values) {
        fprintf(fout, "%s", join(values, ",").c_str());
    }

    virtual void print_test(const test & t) {
        for (auto t_ns : t.tsamples.t_ns) {
            print_values(t.mparams.get_values());
            print_values(t.bparams.get_values());
            print_values(t.bkparams.get_values());
            print_values({std::to_string(t_ns)});
            fprintf(fout, "\n");
        }
    }
};

struct json_printer : public printer {
    void print_fields(const std::vector<std::string> & fields, const std::vector<std::string> & values) {
        assert(fields.size() == values.size());
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "    \"%s\": \"%s\",\n", fields.at(i).c_str(), values.at(i).c_str());
        }
    }

    virtual void print_test(const test & t) {
        fprintf(fout, "{\n");
        fprintf(fout, "  \"model\": {\n");
        print_fields(model_params::get_fields(), t.mparams.get_values());
        fprintf(fout, "  },\n");
        fprintf(fout, "  \"benchmark\": {\n");
        print_fields(bench_params::get_fields(), t.bparams.get_values());
        fprintf(fout, "  },\n");
        fprintf(fout, "  \"backend\": {\n");
        print_fields(backend_params::get_fields(), t.bkparams.get_values());
        fprintf(fout, "  },\n");
        fprintf(fout, "  \"samples\": {\n");
        fprintf(fout, "    \"ns\": [ %s ],\n", join(t.tsamples.t_ns, ", ").c_str());
        fprintf(fout, "    \"avg\": %" PRIu64 ",\n", t.tsamples.avg());
        fprintf(fout, "    \"stddev\": %" PRIu64 "\n", t.tsamples.stdev());
        fprintf(fout, "  }\n");
        fprintf(fout, "}\n");
    }
};

struct markdown_printer : public printer {
    std::vector<std::string> fields;

    static int get_field_width(const std::string & field) {
        if (field == "model") {
            return -30;
        }
        if (field == "t/s") {
            return 15;
        }
        int width = std::max((int)field.length(), 10);
        if (field == "backend") {
            return -width;
        }
        return width;
    }

    virtual void print_header(const cmd_params & params) {
        fields = { "model", "backend" };
        if (backend_params::get_backend() != "CPU") {
            fields.push_back("n_gpu_layers");
        }
        if (params.n_batch.size() > 1) {
            fields.push_back("n_batch");
        }
        if (params.n_threads.size() > 1 || backend_params::get_backend() == "CPU") {
            fields.push_back("n_threads");
        }
        if (params.f32_kv.size() > 1) {
            fields.push_back("f32_kv");
        }
        if (params.main_gpu.size() > 1) {
            fields.push_back("main_gpu");
        }
        if (params.mul_mat_q.size() > 1) {
            fields.push_back("mul_mat_q");
        }
        if (params.low_vram.size() > 1) {
            fields.push_back("low_vram");
        }
        fields.push_back("test");
        fields.push_back("t/s");

        fprintf(fout, "|");
        for (const auto & field : fields) {
            fprintf(fout, " %*s |", get_field_width(field), field.c_str());
        }
        fprintf(fout, "\n");
        fprintf(fout, "|");
        for (const auto & field: fields) {
            int width = get_field_width(field);
            fprintf(fout, " %s%s%s |", width < 0 ? ":" : "", std::string(std::abs(width) - 1, '-').c_str(), width > 0 ? ":" : "");
        }
        fprintf(fout, "\n");
        (void) params;
    }

    virtual void print_test(const test & t) {
        int n_tokens = t.bparams.n_prompt + t.bparams.n_gen;

        std::map<std::string, std::string> vmap;
        std::transform(model_params::get_fields().begin(), model_params::get_fields().end(), t.mparams.get_values().begin(),
            std::inserter(vmap, vmap.end()), std::make_pair<const std::string&, const std::string&>);
        std::transform(bench_params::get_fields().begin(), bench_params::get_fields().end(), t.bparams.get_values().begin(),
            std::inserter(vmap, vmap.end()), std::make_pair<const std::string&, const std::string&>);
        std::transform(backend_params::get_fields().begin(), backend_params::get_fields().end(), t.bkparams.get_values().begin(),
            std::inserter(vmap, vmap.end()), std::make_pair<const std::string&, const std::string&>);

        fprintf(fout, "|");
        for (const auto & field : fields) {
            std::string value;
            if (field == "model") {
                value = t.mparams.type;
            } else if (field == "backend") {
                value = backend_params::get_backend();
            } else if (field == "test") {
                char buf[128];
                if (t.bparams.n_prompt > 0 && t.bparams.n_gen == 0) {
                    snprintf(buf, sizeof(buf), "pp %d", t.bparams.n_prompt);
                } else if (t.bparams.n_gen > 0 && t.bparams.n_prompt == 0) {
                    snprintf(buf, sizeof(buf), "tg %d", t.bparams.n_gen);
                } else {
                    assert(false);
                    exit(1);
                }
                value = buf;
            } else if (field == "t/s") {
                char buf[128];
                snprintf(buf, sizeof(buf), "%.2f ± %.2f", t.tsamples.avg_ts(n_tokens), t.tsamples.stddev_ts(n_tokens));
                value = buf;
            } else if (vmap.find(field) != vmap.end()) {
                value = vmap.at(field);
            } else {
                assert(false);
                exit(1);
            }

            int width = get_field_width(field);
            if (field == "t/s") {
                // HACK: the utf-8 character is 2 bytes
                width += 1;
            }
            fprintf(fout, " %*s |", width, value.c_str());
        }
        fprintf(fout, "\n");
    }
};

void test_prompt(llama_context * ctx, int n_prompt, int n_past, int n_batch, int n_threads) {
    std::vector<llama_token> tokens(n_batch, llama_token_bos());
    int n_processed = 0;
    while (n_processed < n_prompt) {
        int n = std::min(n_prompt - n_processed, n_batch);
        llama_eval(ctx, tokens.data(), n, n_past + n_processed, n_threads);
        n_processed += n;
    }
}

void test_gen(llama_context * ctx, int n_gen, int n_past, int n_threads) {
    llama_token token = llama_token_bos();
    for (int i = 0; i < n_gen; i++) {
        llama_eval(ctx, &token, 1, n_past + i, n_threads);
    }
}

void llama_null_log_callback(enum llama_log_level level, const char * text, void * user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

int main(int argc, char ** argv) {
#if !defined(NDEBUG)
    fprintf(stderr, "warning: asserts enabled, performance may be affected\n");
#endif

#if (defined(_MSC_VER) && defined(_DEBUG)) || (!defined(_MSC_VER) && !defined(__OPTIMIZE__))
    fprintf(stderr, "warning: debug build, performance may be affected\n");
#endif

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    fprintf(stderr, "warning: sanitizer enabled, performance may be affected\n");
#endif

    cmd_params params = parse_cmd_params(argc, argv);

    // initialize llama.cpp
    if (!params.verbose) {
        llama_log_set(llama_null_log_callback, NULL);
    }
    bool numa = false;
    llama_backend_init(numa);

    // initialize printer
    std::unique_ptr<printer> p;
    switch (params.output_format) {
        case CSV:
            p.reset(new csv_printer());
            break;
        case JSON:
            p.reset(new json_printer());
            break;
        case MARKDOWN:
            p.reset(new markdown_printer());
            break;
    }
    p->fout = stdout;
    p->print_header(params);

    std::vector<cmd_params_instance> params_instances = get_cmd_params_instances(params);

    for (const auto & inst : params_instances) {
        // TODO: keep the model between tests when possible
        llama_context_params lparams = inst.to_llama_params();

        llama_model * lmodel  = llama_load_model_from_file(inst.model.c_str(), lparams);
        if (lmodel == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, inst.model.c_str());
            return 1;
        }

        llama_context * ctx = llama_new_context_with_model(lmodel, lparams);
        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, inst.model.c_str());
            llama_free_model(lmodel);
            return 1;
        }

        test t(inst, lmodel, ctx);

        // warmup run
        test_gen(ctx, 1, 0, t.bkparams.n_threads);

        for (int i = 0; i < params.reps; i++) {
            uint64_t t_start = get_time_ns();
            if (t.bparams.n_prompt > 0) {
                test_prompt(ctx, t.bparams.n_prompt, 0, t.bkparams.n_batch, t.bkparams.n_threads);
            }
            if (t.bparams.n_gen > 0) {
                test_gen(ctx, t.bparams.n_gen, t.bparams.n_prompt, t.bkparams.n_threads);
            }
            uint64_t t_ns = get_time_ns() - t_start;
            t.tsamples.t_ns.push_back(t_ns);
        }

        p->print_test(t);

        llama_print_timings(ctx);

        llama_free(ctx);
        llama_free_model(lmodel);
    }

    p->print_footer();

    llama_backend_free();

    return 0;
}
