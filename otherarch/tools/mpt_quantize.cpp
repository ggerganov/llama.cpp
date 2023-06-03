#include "utils.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>

struct mpt_hparams {
    int32_t d_model      = 0;
    int32_t max_seq_len  = 0;
    int32_t n_heads      = 0;
    int32_t n_layers     = 0;
    int32_t n_vocab      = 0;
    float alibi_bias_max = 0;
    float clip_qkv       = 0;
    int32_t ftype        = 0;
};

// quantize a model
bool mpt_model_quantize(const std::string & fname_inp,
                        const std::string & fname_out, ggml_ftype ftype) {

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__,
                fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__,
                fname_out.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        finp.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n",
                    __func__, fname_inp.c_str());
            return false;
        }

        fout.write((char *)&magic, sizeof(magic));
    }

    mpt_hparams hparams;

    // load hparams
    {
        finp.read((char *) &hparams.d_model,        sizeof(hparams.d_model));
        finp.read((char *) &hparams.max_seq_len,    sizeof(hparams.max_seq_len));
        finp.read((char *) &hparams.n_heads,        sizeof(hparams.n_heads));
        finp.read((char *) &hparams.n_layers,       sizeof(hparams.n_layers));
        finp.read((char *) &hparams.n_vocab,        sizeof(hparams.n_vocab));
        finp.read((char *) &hparams.alibi_bias_max, sizeof(hparams.alibi_bias_max));
        finp.read((char *) &hparams.clip_qkv,       sizeof(hparams.clip_qkv));
        finp.read((char *) &hparams.ftype,          sizeof(hparams.ftype));

        const int32_t qntvr_src =    hparams.ftype / GGML_QNT_VERSION_FACTOR;
        const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;

        printf("%s: d_model        = %d\n", __func__, hparams.d_model);
        printf("%s: max_seq_len    = %d\n", __func__, hparams.max_seq_len);
        printf("%s: n_heads        = %d\n", __func__, hparams.n_heads);
        printf("%s: n_layers       = %d\n", __func__, hparams.n_layers);
        printf("%s: n_vocab        = %d\n", __func__, hparams.n_vocab);
        printf("%s: alibi_bias_max = %f\n", __func__, hparams.alibi_bias_max);
        printf("%s: clip_qkv       = %f\n", __func__, hparams.clip_qkv);
        printf("%s: ftype (src) = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr (src) = %d\n", __func__, qntvr_src);
        printf("%s: ftype (dst) = %d\n", __func__, ftype_dst);
        printf("%s: qntvr (dst) = %d\n", __func__, GGML_QNT_VERSION);

        fout.write((char *) &hparams.d_model,        sizeof(hparams.d_model));
        fout.write((char *) &hparams.max_seq_len,    sizeof(hparams.max_seq_len));
        fout.write((char *) &hparams.n_heads,        sizeof(hparams.n_heads));
        fout.write((char *) &hparams.n_layers,       sizeof(hparams.n_layers));
        fout.write((char *) &hparams.n_vocab,        sizeof(hparams.n_vocab));
        fout.write((char *) &hparams.alibi_bias_max, sizeof(hparams.alibi_bias_max));
        fout.write((char *) &hparams.clip_qkv,       sizeof(hparams.clip_qkv));
        fout.write((char *) &ftype_dst,              sizeof(ftype_dst));
    }

    // load vocab
    {
        const int32_t n_vocab = hparams.n_vocab;

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read((char *)&len, sizeof(len));
            fout.write((char *)&len, sizeof(len));

            word.resize(len);
            finp.read((char *)word.data(), len);
            fout.write((char *)word.data(), len);
        }
    }

    printf("%s: quantizing tensors\n", __func__);

    // regexes of tensor names to be quantized
    const std::vector<std::string> to_quant = {
        ".*weight",
    };

    if (!ggml_common_quantize_0(finp, fout, ftype, to_quant, {})) {
        fprintf(stderr, "%s: failed to quantize model '%s'\n", __func__,
                fname_inp.c_str());
        return false;
    }

    finp.close();
    fout.close();

    return true;
}

// usage:
//  ./mpt-quantize models/mpt/ggml-model.bin
//  models/mpt/ggml-model-quant.bin type
//
int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type\n",
                argv[0]);
        ggml_print_ftypes(stderr);
        return 1;
    }

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = {0, NULL, false};
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const ggml_ftype ftype = ggml_parse_ftype(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!mpt_model_quantize(fname_inp, fname_out, ggml_ftype(ftype))) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n",
                    __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__,
               t_quantize_us / 1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__,
               (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    return 0;
}