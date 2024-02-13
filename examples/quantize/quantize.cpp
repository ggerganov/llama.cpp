#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <algorithm>

struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};

static const std::vector<struct quant_option> QUANT_OPTIONS = {
    { "Q4_0",   LLAMA_FTYPE_MOSTLY_Q4_0,   " 3.56G, +0.2166 ppl @ LLaMA-v1-7B", },
    { "Q4_1",   LLAMA_FTYPE_MOSTLY_Q4_1,   " 3.90G, +0.1585 ppl @ LLaMA-v1-7B", },
    { "Q5_0",   LLAMA_FTYPE_MOSTLY_Q5_0,   " 4.33G, +0.0683 ppl @ LLaMA-v1-7B", },
    { "Q5_1",   LLAMA_FTYPE_MOSTLY_Q5_1,   " 4.70G, +0.0349 ppl @ LLaMA-v1-7B", },
    { "IQ2_XXS",LLAMA_FTYPE_MOSTLY_IQ2_XXS," 2.06 bpw quantization",            },
    { "IQ2_XS", LLAMA_FTYPE_MOSTLY_IQ2_XS, " 2.31 bpw quantization",            },
    { "Q2_K",   LLAMA_FTYPE_MOSTLY_Q2_K,   " 2.63G, +0.6717 ppl @ LLaMA-v1-7B", },
    { "Q2_K_S", LLAMA_FTYPE_MOSTLY_Q2_K_S, " 2.16G, +9.0634 ppl @ LLaMA-v1-7B", },
    { "IQ3_XXS",LLAMA_FTYPE_MOSTLY_IQ3_XXS," 3.06 bpw quantization",            },
    { "Q3_K",   LLAMA_FTYPE_MOSTLY_Q3_K_M, "alias for Q3_K_M" },
    { "Q3_K_XS",LLAMA_FTYPE_MOSTLY_Q3_K_XS,"3-bit extra small quantization"   , },
    { "Q3_K_S", LLAMA_FTYPE_MOSTLY_Q3_K_S, " 2.75G, +0.5551 ppl @ LLaMA-v1-7B", },
    { "Q3_K_M", LLAMA_FTYPE_MOSTLY_Q3_K_M, " 3.07G, +0.2496 ppl @ LLaMA-v1-7B", },
    { "Q3_K_L", LLAMA_FTYPE_MOSTLY_Q3_K_L, " 3.35G, +0.1764 ppl @ LLaMA-v1-7B", },
    { "Q4_K",   LLAMA_FTYPE_MOSTLY_Q4_K_M, "alias for Q4_K_M", },
    { "Q4_K_S", LLAMA_FTYPE_MOSTLY_Q4_K_S, " 3.59G, +0.0992 ppl @ LLaMA-v1-7B", },
    { "Q4_K_M", LLAMA_FTYPE_MOSTLY_Q4_K_M, " 3.80G, +0.0532 ppl @ LLaMA-v1-7B", },
    { "Q5_K",   LLAMA_FTYPE_MOSTLY_Q5_K_M, "alias for Q5_K_M", },
    { "Q5_K_S", LLAMA_FTYPE_MOSTLY_Q5_K_S, " 4.33G, +0.0400 ppl @ LLaMA-v1-7B", },
    { "Q5_K_M", LLAMA_FTYPE_MOSTLY_Q5_K_M, " 4.45G, +0.0122 ppl @ LLaMA-v1-7B", },
    { "Q6_K",   LLAMA_FTYPE_MOSTLY_Q6_K,   " 5.15G, +0.0008 ppl @ LLaMA-v1-7B", },
    { "Q8_0",   LLAMA_FTYPE_MOSTLY_Q8_0,   " 6.70G, +0.0004 ppl @ LLaMA-v1-7B", },
    { "F16",    LLAMA_FTYPE_MOSTLY_F16,    "13.00G              @ 7B", },
    { "F32",    LLAMA_FTYPE_ALL_F32,       "26.00G              @ 7B", },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    { "COPY",   LLAMA_FTYPE_ALL_F32,       "only copy tensors, no quantizing", },
};


static bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out) {
    std::string ftype_str;

    for (auto ch : ftype_str_in) {
        ftype_str.push_back(std::toupper(ch));
    }
    for (auto & it : QUANT_OPTIONS) {
        if (it.name == ftype_str) {
            ftype = it.ftype;
            ftype_str_out = it.name;
            return true;
        }
    }
    try {
        int ftype_int = std::stoi(ftype_str);
        for (auto & it : QUANT_OPTIONS) {
            if (it.ftype == ftype_int) {
                ftype = it.ftype;
                ftype_str_out = it.name;
                return true;
            }
        }
    }
    catch (...) {
        // stoi failed
    }
    return false;
}

// usage:
//  ./quantize [--allow-requantize] [--leave-output-tensor] [--pure] models/llama/ggml-model.gguf [models/llama/ggml-model-quant.gguf] type [nthreads]
//
[[noreturn]]
static void usage(const char * executable) {
    printf("usage: %s [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights] [--exclude-weights] model-f32.gguf [model-quant.gguf] type [nthreads]\n\n", executable);
    printf("  --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit\n");
    printf("  --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing\n");
    printf("  --pure: Disable k-quant mixtures and quantize all tensors to the same type\n");
    printf("  --imatrix file_name: use data in file_name as importance matrix for quant optimizations\n");
    printf("  --include-weights tensor_name: use importance matrix for this/these tensor(s)\n");
    printf("  --exclude-weights tensor_name: use importance matrix for this/these tensor(s)\n");
    printf("Note: --include-weights and --exclude-weights cannot be used together\n");
    printf("\nAllowed quantization types:\n");
    for (auto & it : QUANT_OPTIONS) {
        if (it.name != "COPY") {
            printf("  %2d  or  ", it.ftype);
        } else {
            printf("          ");
        }
        printf("%-7s : %s\n", it.name.c_str(), it.desc.c_str());
    }
    exit(1);
}

static void load_imatrix(const std::string& imatrix_file, std::unordered_map<std::string, std::vector<float>>& imatrix_data) {
    std::ifstream in(imatrix_file.c_str(), std::ios::binary);
    if (!in) {
        printf("%s: failed to open %s\n",__func__,imatrix_file.c_str());
        return;
    }
    int n_entries;
    in.read((char*)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        printf("%s: no data in file %s\n", __func__, imatrix_file.c_str());
        return;
    }
    for (int i = 0; i < n_entries; ++i) {
        int len; in.read((char *)&len, sizeof(len));
        std::vector<char> name_as_vec(len+1);
        in.read((char *)name_as_vec.data(), len);
        if (in.fail()) {
            printf("%s: failed reading name for entry %d from %s\n",__func__,i+1,imatrix_file.c_str());
            return;
        }
        name_as_vec[len] = 0;
        std::string name{name_as_vec.data()};
        auto& e = imatrix_data[std::move(name)];
        int ncall;
        in.read((char*)&ncall, sizeof(ncall));
        int nval;
        in.read((char *)&nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            printf("%s: failed reading number of values for entry %d\n",__func__,i);
            imatrix_data = {};
            return;
        }
        e.resize(nval);
        in.read((char*)e.data(), nval*sizeof(float));
        if (in.fail()) {
            printf("%s: failed reading data for entry %d\n",__func__,i);
            imatrix_data = {};
            return;
        }
        if (ncall > 0) {
            for (auto& v : e) v /= ncall;
        }
    }
    printf("%s: loaded %d importance matrix entries from %s\n",__func__,int(imatrix_data.size()),imatrix_file.c_str());
}

static void prepare_imatrix(const std::string& imatrix_file,
        const std::vector<std::string>& included_weights,
        const std::vector<std::string>& excluded_weights,
        std::unordered_map<std::string, std::vector<float>>& imatrix_data) {
    if (!imatrix_file.empty()) {
        load_imatrix(imatrix_file, imatrix_data);
    }
    if (imatrix_data.empty()) {
        return;
    }
    if (!excluded_weights.empty()) {
        for (auto& name : excluded_weights) {
            for (auto it = imatrix_data.begin(); it != imatrix_data.end(); ) {
                auto pos = it->first.find(name);
                if (pos != std::string::npos) it = imatrix_data.erase(it);
                else ++it;
            }
        }
    }
    if (!included_weights.empty()) {
        std::unordered_map<std::string, std::vector<float>> tmp;
        for (auto& name : included_weights) {
            for (auto& e : imatrix_data) {
                auto pos = e.first.find(name);
                if (pos != std::string::npos) {
                    tmp.emplace(std::move(e));
                }
            }
        }
        imatrix_data = std::move(tmp);
    }
    if (!imatrix_data.empty()) {
        printf("%s: have %d importance matrix entries\n", __func__, int(imatrix_data.size()));
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
    }

    llama_model_quantize_params params = llama_model_quantize_default_params();

    int arg_idx = 1;
    std::string imatrix_file;
    std::vector<std::string> included_weights, excluded_weights;

    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        if (strcmp(argv[arg_idx], "--leave-output-tensor") == 0) {
            params.quantize_output_tensor = false;
        } else if (strcmp(argv[arg_idx], "--allow-requantize") == 0) {
            params.allow_requantize = true;
        } else if (strcmp(argv[arg_idx], "--pure") == 0) {
            params.pure = true;
        } else if (strcmp(argv[arg_idx], "--imatrix") == 0) {
            if (arg_idx < argc-1) {
                imatrix_file = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--include-weights") == 0) {
            if (arg_idx < argc-1) {
                included_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--exclude-weights") == 0) {
            if (arg_idx < argc-1) {
                excluded_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else {
            usage(argv[0]);
        }
    }

    if (argc - arg_idx < 2) {
        printf("%s: bad arguments\n", argv[0]);
        usage(argv[0]);
    }
    if (!included_weights.empty() && !excluded_weights.empty()) {
        usage(argv[0]);
    }

    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    prepare_imatrix(imatrix_file, included_weights, excluded_weights, imatrix_data);
    if (!imatrix_data.empty()) {
        params.imatrix = &imatrix_data;
    }

    llama_backend_init(false);

    // parse command line arguments
    const std::string fname_inp = argv[arg_idx];
    arg_idx++;
    std::string fname_out;

    std::string ftype_str;
    if (try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
        std::string fpath;
        const size_t pos = fname_inp.find_last_of("/\\");
        if (pos != std::string::npos) {
            fpath = fname_inp.substr(0, pos + 1);
        }
        // export as [inp path]/ggml-model-[ftype].gguf
        fname_out = fpath + "ggml-model-" + ftype_str + ".gguf";
        arg_idx++;
        if (ftype_str == "COPY") {
            params.only_copy = true;
        }
    }
    else {
        fname_out = argv[arg_idx];
        arg_idx++;

        if (argc <= arg_idx) {
            fprintf(stderr, "%s: missing ftype\n", __func__);
            return 1;
        }
        if (!try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
            fprintf(stderr, "%s: invalid ftype '%s'\n", __func__, argv[3]);
            return 1;
        }
        if (ftype_str == "COPY") {
           params.only_copy = true;
        }
        arg_idx++;
    }

    // parse nthreads
    if (argc > arg_idx) {
        try {
            params.nthread = std::stoi(argv[arg_idx]);
        }
        catch (const std::exception & e) {
            fprintf(stderr, "%s: invalid nthread '%s' (%s)\n", __func__, argv[arg_idx], e.what());
            return 1;
        }
    }

    if ((params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || params.ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) && imatrix_data.empty()) {
        fprintf(stderr, "\n===============================================================================================\n");
        fprintf(stderr, "Please do not use IQ2_XXS, IQ2_XS or Q2_K_S quantization without an importance matrix\n");
        fprintf(stderr, "===============================================================================================\n\n\n");
        return 1;
    }

    print_build_info();

    fprintf(stderr, "%s: quantizing '%s' to '%s' as %s", __func__, fname_inp.c_str(), fname_out.c_str(), ftype_str.c_str());
    if (params.nthread > 0) {
        fprintf(stderr, " using %d threads", params.nthread);
    }
    fprintf(stderr, "\n");

    const int64_t t_main_start_us = llama_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = llama_time_us();

        if (llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), &params)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = llama_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = llama_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    llama_backend_free();

    return 0;
}
