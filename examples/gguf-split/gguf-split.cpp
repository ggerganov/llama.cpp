#include "llama.h"
#include "ggml.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include <stdio.h>
#include <fcntl.h>
#include <string.h>

enum split_operation : uint8_t {
    SPLIT_OP_SPLIT,
    SPLIT_OP_MERGE,
};

static const char * const LLM_KV_GENERAL_SPLIT_I_SPLIT = "general.split";
static const char * const LLM_KV_GENERAL_SPLIT_N_SPLIT = "general.split_count";

static const int SPLIT_FILENAME_MAX = 256;

static const char * const SPLIT_FILENAME_FORMAT = "%s-%05d-of-%05d.gguf";

struct split_params {
    split_operation operation = SPLIT_OP_SPLIT;
    int n_split_tensors = 128;
    std::string input;
    std::string output;
};

static void split_print_usage(const char * executable) {
    const split_params default_params;
    printf("\n");
    printf("usage: %s [options] GGUF_IN GGUF_OUT\n", executable);
    printf("\n");
    printf("Apply a GGUF operation on IN to OUT.");
    printf("\n");
    printf("options:\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  --version             show version and build info\n");
    printf("  --split               split GGUF to multiple GGUF (default)\n");
    printf("  --split-max-tensors   max tensors in each split: default(%d)\n", default_params.n_split_tensors);
    printf("  --merge               merge multiple GGUF to a single GGUF\n");
    printf("\n");
}

static bool split_params_parse_ex(int argc, const char ** argv, split_params & params) {
    std::string arg;
    const std::string arg_prefix = "--";
    bool invalid_param = false;

    int arg_idx = 1;
    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        arg = argv[arg_idx];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        bool arg_found = false;
        if (arg == "-h" || arg == "--help") {
            split_print_usage(argv[0]);
            exit(0);
        }
        if (arg == "--version") {
            fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
            fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
            exit(0);
        }

        if (arg == "--merge") {
            arg_found = true;
            params.operation = SPLIT_OP_MERGE;
        }
        if (arg == "--split") {
            arg_found = true;
            params.operation = SPLIT_OP_SPLIT;
        }
        if (arg == "--split-max-tensors") {
            if (++arg_idx >= argc) {
                invalid_param = true;
                break;
            }
            arg_found = true;
            params.n_split_tensors = atoi(argv[arg_idx]);
        }

        if (!arg_found) {
            throw std::invalid_argument("error: unknown argument: " + arg);
        }
    }

    if (invalid_param) {
        throw std::invalid_argument("error: invalid parameter for argument: " + arg);
    }

    if (argc - arg_idx < 2) {
        printf("%s: bad arguments\n", argv[0]);
        split_print_usage(argv[0]);
        return false;
    }

    params.input = argv[arg_idx++];
    params.output = argv[arg_idx++];

    return true;
}

static bool split_params_parse(int argc, const char ** argv, split_params & params) {
    bool result = true;
    try {
        if (!split_params_parse_ex(argc, argv, params)) {
            split_print_usage(argv[0]);
            exit(1);
        }
    }
    catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        split_print_usage(argv[0]);
        exit(1);
    }
    return result;
}

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static std::string split_file_name(const std::string & path, int i_split, int n_split) {
    char f_split[SPLIT_FILENAME_MAX] = {0};
    snprintf(f_split, sizeof(f_split), SPLIT_FILENAME_FORMAT, path.c_str(), i_split + 1, n_split);
    return std::string(f_split);
}

struct split_strategy {
    const split_params params;
    std::ifstream & f_input;
    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_meta = NULL;
    const int n_tensors;

    const int n_split;
    int i_split = 0;

    int i_tensor = 0;

    std::vector<uint8_t> read_data;

    struct gguf_context * ctx_out;
    std::ofstream fout;

    split_strategy(const split_params & params,
            std::ifstream & f_input,
            struct gguf_context * ctx_gguf,
            struct ggml_context * ctx_meta) :
        params(params),
        f_input(f_input),
        ctx_gguf(ctx_gguf),
        ctx_meta(ctx_meta),
        n_tensors(gguf_get_n_tensors(ctx_gguf)),
        n_split(std::ceil(1. * n_tensors / params.n_split_tensors)) {
        }

    bool should_split() const {
        return i_tensor < n_tensors && i_tensor % params.n_split_tensors == 0;
    }

    void split_start() {
        ctx_out = gguf_init_empty();

        // Save all metadata in first split only
        if (i_split == 0) {
            gguf_set_kv(ctx_out, ctx_gguf);
        }
        gguf_set_val_u8(ctx_out, LLM_KV_GENERAL_SPLIT_I_SPLIT, i_split);
        gguf_set_val_u8(ctx_out, LLM_KV_GENERAL_SPLIT_N_SPLIT, n_split);

        // populate the original tensors, so we get an initial metadata
        for (int i = i_split * params.n_split_tensors; i < n_tensors && i < (i_split + 1) * params.n_split_tensors; ++i) {
            struct ggml_tensor * meta = ggml_get_tensor(ctx_meta, gguf_get_tensor_name(ctx_gguf, i));
            gguf_add_tensor(ctx_out, meta);
        }

        auto split_name = split_file_name(params.output, i_split, n_split);

        fprintf(stderr, "%s: %s ...", __func__, split_name.c_str());
        fout = std::ofstream(split_name, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors

        auto meta_size = gguf_get_meta_size(ctx_out);

        // placeholder for the meta data
        ::zeros(fout, meta_size);

        i_split++;
    }

    void next_tensor() {
        const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
        auto n_bytes = ggml_nbytes(t);

        if (read_data.size() < n_bytes) {
            read_data.resize(n_bytes);
        }

        auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor);
        f_input.seekg(offset);
        f_input.read((char *)read_data.data(), n_bytes);

        t->data = read_data.data();

        // write tensor data + padding
        fout.write((const char *)t->data, n_bytes);
        zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);

        i_tensor++;
    }

    void split_end() {
        // go back to beginning of file and write the updated metadata
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *)data.data(), data.size());

        fout.close();
        gguf_free(ctx_out);

        fprintf(stderr, "\033[3Ddone\n");
    }
};

static void gguf_split(const split_params & split_params) {
    struct ggml_context * ctx_meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };

    std::ifstream f_input(split_params.input.c_str(), std::ios::binary);
    if (!f_input.is_open()) {
        fprintf(stderr, "%s:  failed to open input GGUF from %s\n", __func__, split_params.input.c_str());
        exit(1);
    }

    auto * ctx_gguf = gguf_init_from_file(split_params.input.c_str(), params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s:  failed to load input GGUF from %s\n", __func__, split_params.input.c_str());
        exit(1);
    }

    split_strategy strategy(split_params, f_input, ctx_gguf, ctx_meta);
    fprintf(stderr, "%s: %s -> %s (%d tensors per file)\n",
            __func__, split_params.input.c_str(),
            split_file_name(split_params.output, strategy.i_split, strategy.n_split).c_str(),
            split_params.n_split_tensors);

    strategy.split_start();

    while (strategy.i_tensor < strategy.n_tensors) {
        strategy.next_tensor();
        if (strategy.should_split()) {
            strategy.split_end();
            strategy.split_start();
        }
    }
    strategy.split_end();

    gguf_free(ctx_gguf);
    f_input.close();

    fprintf(stderr, "%s: %d gguf split written with a total of %d tensors.\n",
            __func__, strategy.n_split, strategy.n_tensors);
}

static void gguf_merge(const split_params & split_params) {
    fprintf(stderr, "%s: %s -> %s\n",
            __func__, split_params.input.c_str(),
            split_params.output.c_str());
    int n_split = 1;
    int total_tensors = 0;

    auto * ctx_out = gguf_init_empty();
    std::ofstream fout(split_params.output.c_str(), std::ios::binary);
    fout.exceptions(std::ofstream::failbit); // fail fast on write errors

    std::vector<uint8_t> read_data;
    std::vector<ggml_context *> ctx_metas;
    std::vector<gguf_context *> ctx_ggufs;

    std::string split_prefix;

    // First pass to find KV and tensors metadata
    for (int i_split = 0; i_split < n_split; i_split++) {
        struct ggml_context * ctx_meta = NULL;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx_meta,
        };

        auto split_name = split_params.input;
        if (i_split > 0) {
            split_name = split_file_name(split_prefix, i_split, n_split);
        }
        fprintf(stderr, "%s: reading metadata %s ...", __func__, split_name.c_str());

        auto * ctx_gguf = gguf_init_from_file(split_name.c_str(), params);
        if (!ctx_gguf) {
            fprintf(stderr, "\n%s:  failed to load input GGUF from %s\n", __func__, split_params.input.c_str());
            exit(1);
        }
        ctx_ggufs.push_back(ctx_gguf);
        ctx_metas.push_back(ctx_meta);

        if (i_split == 0) {
            auto key_n_split = gguf_find_key(ctx_gguf, LLM_KV_GENERAL_SPLIT_N_SPLIT);
            if (key_n_split < 0) {
                fprintf(stderr,
                        "\n%s: input file does not contain %s metadata\n",
                        __func__,
                        LLM_KV_GENERAL_SPLIT_N_SPLIT);
                gguf_free(ctx_gguf);
                gguf_free(ctx_out);
                fout.close();
                exit(1);
            }

            n_split = gguf_get_val_u8(ctx_gguf, key_n_split);
            if (n_split < 1) {
                fprintf(stderr,
                        "\n%s: input file does not contain a valid split count %d\n",
                        __func__,
                        n_split);
                gguf_free(ctx_gguf);
                gguf_free(ctx_out);
                fout.close();
                exit(1);
            }

            // Do not trigger merge if we try to merge again the output
            gguf_set_val_u8(ctx_out, LLM_KV_GENERAL_SPLIT_N_SPLIT, 0);

            // Set metadata from the first split
            gguf_set_kv(ctx_out, ctx_gguf);
        }

        // Verify the file naming
        {
            int i_split_file = 0;
            int n_split_file = 0;
            const char * i_split_format = "-00000-of-00000.gguf";

            if (split_name.size() < strlen(i_split_format)) {
                fprintf(stderr, "\n%s: unexpected input file name: %s\n", __func__, split_params.input.c_str());
                for (auto * _ctx_gguf : ctx_ggufs) {
                    gguf_free(_ctx_gguf);
                }
                gguf_free(ctx_out);
                fout.close();
                exit(1);
            }

            split_prefix = split_name.substr(0, split_name.size() - strlen(i_split_format));

            const char * split_name_c_str = split_name.c_str();
            int n_part = sscanf(&split_name_c_str[0] + split_prefix.size(), "-%d-of-%d", &i_split_file, &n_split_file);

            if (n_part != 2 || i_split_file - 1 != i_split || n_split_file != n_split) {
                fprintf(stderr, "\n%s: unexpected input file name: %s"
                                " i_split=%d i_split_file=%d"
                                " n_split=%d n_split_file=%d\n", __func__,
                        split_params.input.c_str(),
                        i_split, i_split_file,
                        n_split, n_split_file);
                for (auto * _ctx_gguf : ctx_ggufs) {
                    gguf_free(_ctx_gguf);
                }
                gguf_free(ctx_out);
                fout.close();
                exit(1);
            }
        }

        auto n_tensors = gguf_get_n_tensors(ctx_gguf);
        for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
            const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
            gguf_add_tensor(ctx_out, t);
        }
        total_tensors += n_tensors;

        fprintf(stderr, "\033[3Ddone\n");
    }

    // placeholder for the meta data
    {
        auto meta_size = gguf_get_meta_size(ctx_out);
        ::zeros(fout, meta_size);
    }

    // Write tensors data
    for (int i_split = 0; i_split < n_split; i_split++) {
        auto split_name = split_file_name(split_prefix, i_split, n_split);
        std::ifstream f_input(split_name.c_str(), std::ios::binary);
        if (!f_input.is_open()) {
            fprintf(stderr, "%s:  failed to open input GGUF from %s\n", __func__, split_name.c_str());
            for (auto * _ctx_gguf : ctx_ggufs) {
                gguf_free(_ctx_gguf);
            }
            gguf_free(ctx_out);
            fout.close();
            exit(1);
        }
        fprintf(stderr, "%s: writing tensors %s ...", __func__, split_name.c_str());

        auto * ctx_gguf = ctx_ggufs[i_split];
        auto * ctx_meta = ctx_metas[i_split];

        auto n_tensors = gguf_get_n_tensors(ctx_gguf);
        for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
            const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);

            auto n_bytes = ggml_nbytes(t);

            if (read_data.size() < n_bytes) {
                read_data.resize(n_bytes);
            }

            auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor);
            f_input.seekg(offset);
            f_input.read((char *)read_data.data(), n_bytes);

            // write tensor data + padding
            fout.write((const char *)read_data.data(), n_bytes);
            zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
        }

        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
        f_input.close();
        fprintf(stderr, "\033[3Ddone\n");
    }

    {
        // go back to beginning of file and write the updated metadata
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *)data.data(), data.size());

        fout.close();
        gguf_free(ctx_out);
    }

    fprintf(stderr, "%s: %s merged from %d split with %d tensors.\n",
            __func__, split_params.output.c_str(), n_split, total_tensors);
}

int main(int argc, const char ** argv) {
    if (argc < 3) {
        split_print_usage(argv[0]);
    }

    split_params params;
    split_params_parse(argc, argv, params);

    switch (params.operation) {
        case SPLIT_OP_SPLIT: gguf_split(params);
            break;
        case SPLIT_OP_MERGE: gguf_merge(params);
            break;
        default:split_print_usage(argv[0]);
            exit(1);
    }

    return 0;
}
