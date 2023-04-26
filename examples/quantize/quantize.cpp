#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <map>
#include <string>

static const std::map<std::string, enum llama_ftype> LLAMA_FTYPE_MAP = {
  {"q4_0", LLAMA_FTYPE_MOSTLY_Q4_0},
  {"q4_1", LLAMA_FTYPE_MOSTLY_Q4_1},
  {"q4_2", LLAMA_FTYPE_MOSTLY_Q4_2},
  {"q4_3", LLAMA_FTYPE_MOSTLY_Q4_3},
  {"q5_0", LLAMA_FTYPE_MOSTLY_Q5_0},
  {"q5_1", LLAMA_FTYPE_MOSTLY_Q5_1},
  {"q8_0", LLAMA_FTYPE_MOSTLY_Q8_0},
};

// usage:
//  ./quantize models/llama/ggml-model.bin models/llama/ggml-model-quant.bin type
//
int main(int argc, char ** argv) {
    ggml_time_init();

    if (argc < 4) {
        fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type [nthread]\n", argv[0]);
        for (auto it = LLAMA_FTYPE_MAP.begin(); it != LLAMA_FTYPE_MAP.end(); it++) {
            fprintf(stderr, "  type = \"%s\" or %d\n", it->first.c_str(), it->second);
        }
        return 1;
    }

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    enum llama_ftype ftype;
    if (argv[3][0] == 'q') {
        auto it = LLAMA_FTYPE_MAP.find(argv[3]);
        if (it == LLAMA_FTYPE_MAP.end()) {
            fprintf(stderr, "%s: unknown ftype '%s'\n", __func__, argv[3]);
            return 1;
        }
        ftype = it->second;
    } else {
        ftype = (enum llama_ftype)atoi(argv[3]);
    }

    int nthread = argc > 4 ? atoi(argv[4]) : 0;

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), ftype, nthread)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    return 0;
}
