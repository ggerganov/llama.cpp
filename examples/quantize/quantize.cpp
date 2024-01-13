#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

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
    { "Q2_K",   LLAMA_FTYPE_MOSTLY_Q2_K,   " 2.63G, +0.6717 ppl @ LLaMA-v1-7B", },
    { "Q2_K_S", LLAMA_FTYPE_MOSTLY_Q2_K_S, " 2.16G, +9.0634 ppl @ LLaMA-v1-7B", },
    { "Q3_K",   LLAMA_FTYPE_MOSTLY_Q3_K_M, "alias for Q3_K_M" },
    { "Q3_K_S", LLAMA_FTYPE_MOSTLY_Q3_K_S, " 2.75G, +0.5551 ppl @ LLaMA-v1-7B", },
    { "Q3_K_M", LLAMA_FTYPE_MOSTLY_Q3_K_M, " 3.07G, +0.2496 ppl @ LLaMA-v1-7B", },
    { "Q3_K_L", LLAMA_FTYPE_MOSTLY_Q3_K_L, " 3.35G, +0.1764 ppl @ LLaMA-v1-7B", },
    { "Q4_K",   LLAMA_FTYPE_MOSTLY_Q4_K_M, "alias for Q4_K_M", },
    { "Q4_K_S", LLAMA_FTYPE_MOSTLY_Q4_K_S, " 3.59G, +0.0992 ppl @ LLaMA-v1-7B", },
    { "Q4_K_M", LLAMA_FTYPE_MOSTLY_Q4_K_M, " 3.80G, +0.0532 ppl @ LLaMA-v1-7B", },
    { "Q5_K",   LLAMA_FTYPE_MOSTLY_Q5_K_M, "alias for Q5_K_M", },
    { "Q5_K_S", LLAMA_FTYPE_MOSTLY_Q5_K_S, " 4.33G, +0.0400 ppl @ LLaMA-v1-7B", },
    { "Q5_K_M", LLAMA_FTYPE_MOSTLY_Q5_K_M, " 4.45G, +0.0122 ppl @ LLaMA-v1-7B", },
    { "Q6_K",   LLAMA_FTYPE_MOSTLY_Q6_K,   " 5.15G, -0.0008 ppl @ LLaMA-v1-7B", },
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
    printf("usage: %s [--help] [--allow-requantize] [--leave-output-tensor] [--pure] model-f32.gguf [model-quant.gguf] type [nthreads]\n\n", executable);
    printf("  --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit\n");
    printf("  --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing\n");
    printf("  --pure: Disable k-quant mixtures and quantize all tensors to the same type\n");
    printf("\nAllowed quantization types:\n");
    for (auto & it : QUANT_OPTIONS) {
        if (it.name != "COPY") {
            printf("  %2d  or  ", it.ftype);
        } else {
            printf("          ");
        }
        printf("%-6s : %s\n", it.name.c_str(), it.desc.c_str());
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
        } else if (strcmp(argv[arg_idx], "--pure") == 0) {
            params.pure = true;
        } else {
            usage(argv[0]);
        }
    }

    if (argc - arg_idx < 2) {
        usage(argv[0]);
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
