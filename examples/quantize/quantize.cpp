#include "build-info.h"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};

static const std::vector<struct quant_option> QUANT_OPTIONS = {
    {
        "q4_0",
        LLAMA_FTYPE_MOSTLY_Q4_0,
        "approx +0.2499 perplexity, 3.50G output @ 7B",
    },
    {
        "q4_1",
        LLAMA_FTYPE_MOSTLY_Q4_1,
        "approx +0.1846 perplexity, 3.90G output @ 7B",
    },
    {
        "q5_0",
        LLAMA_FTYPE_MOSTLY_Q5_0,
        "approx +0.0796 perplexity, 4.30G output @ 7B",
    },
    {
        "q5_1",
        LLAMA_FTYPE_MOSTLY_Q5_1,
        "approx +0.0415 perplexity, 4.70G output @ 7B",
    },
#ifdef GGML_USE_K_QUANTS
    {
        "q2_k",
        LLAMA_FTYPE_MOSTLY_Q2_K,
        "approx +0.8698 perplexity, 2.67G output @ 7B",
    },
    {
        "q3_k",
        LLAMA_FTYPE_MOSTLY_Q3_K_M,
        "alias for q3_k_m"
    },
    {
        "q3_k_s",
        LLAMA_FTYPE_MOSTLY_Q3_K_S,
        "approx +0.5505 perplexity, 2.75G output @ 7B",
    },
    {
        "q3_k_m",
        LLAMA_FTYPE_MOSTLY_Q3_K_M,
        "approx +0.2437 perplexity, 3.06G output @ 7B",
    },
    {
        "q3_k_l",
        LLAMA_FTYPE_MOSTLY_Q3_K_L,
        "approx +0.1803 perplexity, 3.35G output @ 7B",
    },
    {
        "q4_k",
        LLAMA_FTYPE_MOSTLY_Q4_K_M,
        "alias for q4_k_m",
    },
    {
        "q4_k_s",
        LLAMA_FTYPE_MOSTLY_Q4_K_S,
        "approx +0.1149 perplexity, 3.56G output @ 7B",
    },
    {
        "q4_k_m",
        LLAMA_FTYPE_MOSTLY_Q4_K_M,
        "approx +0.0535 perplexity, 3.80G output @ 7B",
    },
    {
        "q5_k",
        LLAMA_FTYPE_MOSTLY_Q5_K_M,
        "alias for q5_k_m",
    },
    {
        "q5_k_s",
        LLAMA_FTYPE_MOSTLY_Q5_K_S,
        "approx +0.0353 perplexity, 4.33G output @ 7B",
    },
    {
        "q5_k_m",
        LLAMA_FTYPE_MOSTLY_Q5_K_M,
        "approx +0.0142 perplexity, 4.45G output @ 7B",
    },
    {
        "q6_k",
        LLAMA_FTYPE_MOSTLY_Q6_K,
        "approx +0.0044 perplexity, 5.15G output @ 7B",
    },
#endif
    {
        "q8_0",
        LLAMA_FTYPE_MOSTLY_Q8_0,
        "approx +0.0004 perplexity, 6.70G output @ 7B",
    },
    {
        "f16",
        LLAMA_FTYPE_MOSTLY_F16,
        "no significant perplexity increase, 13.00G output @ 7B",
    },
    {
        "f32",
        LLAMA_FTYPE_ALL_F32,
        "full quality, 26.00G output @ 7B",
    },
};


bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out) {
    std::string ftype_str;

    for (auto ch : ftype_str_in) {
        ftype_str.push_back(std::tolower(ch));
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
//  ./quantize models/llama/ggml-model.bin [models/llama/ggml-model-quant.bin] type [nthreads]
//
void usage(const char * executable) {
    fprintf(stderr, "usage: %s [--help] [--allow-requantize] [--leave-output-tensor] model-f32.bin [model-quant.bin] type [nthreads]\n", executable);
    fprintf(stderr, "  --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit\n");
    fprintf(stderr, "  --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing\n");
    fprintf(stderr, "Allowed quantization types:\n");
    for (auto & it : QUANT_OPTIONS) {
        std::cout << "  " << std::setw(2) << it.ftype << "  or  " << std::setw(6) << it.name << "  :  " << it.desc << "\n";
    }
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
    }

    llama_model_quantize_params params = llama_model_quantize_default_params();

    int arg_idx = 1;

    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        if (strcmp(argv[arg_idx], "--leave-output-tensor") == 0) {
            params.quantize_output_tensor = false;
        } else if (strcmp(argv[arg_idx], "--allow-requantize") == 0) {
            params.allow_requantize = true;
        } else {
            usage(argv[0]);
        }
    }

    if (argc - arg_idx < 3) {
        usage(argv[0]);
    }

    llama_init_backend();

    // parse command line arguments
    const std::string fname_inp = argv[arg_idx];
    arg_idx++;
    std::string fname_out;

    std::string ftype_str;
    if (try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
        std::string fpath;
        const size_t pos = fname_inp.find_last_of('/');
        if (pos != std::string::npos) {
            fpath = fname_inp.substr(0, pos + 1);
        }
        // export as [inp path]/ggml-model-[ftype].bin
        fname_out = fpath + "ggml-model-" + ftype_str + ".bin";
        arg_idx++;
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

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

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

    return 0;
}
