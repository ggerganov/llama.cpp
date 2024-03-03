#include "common.h"
#include "llama.h"
#include "parser.hpp"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <algorithm>

static const size_t n_models = 2; // hard-limited to 2 input models for now

struct merge_params {
    std::string config_path = "config.txt";
    std::vector<std::string> model_paths;
    std::string output_path = "ggml-merged-f16.gguf";
    bool only_list_tensors_name = false;
    bool dry_run = false;
};

[[noreturn]]
static void usage(const char * executable, int exit_code) {
    struct merge_params defaults;
    printf("usage: %s -c CONFIG_FILE -o OUTPUT_FILE -m MODEL_PATH -m MODEL_PATH ...\n\n", executable);
    printf("\n");
    printf("Merging multiple models, inspired by mergekit.\n");
    printf("For more details, see \"config.example.txt\" file.\n");
    printf("\n");
    printf("NOTE:\n");
    printf("- Only support merging 2 models.\n");
    printf("- The embedding and output layers of the first model will be used.\n");
    printf("- Currently, we accept both quantized and non-quantized models as input. The output model will be re-quantized into the same format of the first model.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help                 Show this help message and exit\n");
    printf("  -c, --config CONFIG_FILE   Path to config file, in CSV format (default: %s)\n", defaults.config_path.c_str());
    printf("  -m, --model MODEL_PATH     Path to model. This option can be repeated multiple times and must be specified in the right order.\n");
    printf("  -o, --output OUTPUT_FILE   Path to the output model (default: %s)\n", defaults.output_path.c_str());
    printf("  --dry-run                  Only print out list of parsed and exit, useful for debugging\n");
    printf("  --print-list-tensor        Only print out list of tensors of the input model, useful for debugging (only one model is accepted)\n");
    printf("\n");
    printf("Example: ./merge -c config.txt -o output.gguf -m model_a.gguf -m model_b.gguf\n");
    exit(exit_code);
}

int main(int argc, char ** argv) {
    bool invalid_param = false;
    struct merge_params params;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0], 0);
        } else if (arg == "-c" || arg == "--config") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.config_path = argv[i];
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model_paths.push_back(argv[i]);
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.output_path = argv[i];
        } else if (arg == "--print-list-tensor") {
            params.only_list_tensors_name = true;
        } else if (arg == "--dry-run") {
            params.dry_run = true;
        }
    }

    try {
        if (invalid_param) {
            usage(argv[0], 1);
            throw std::invalid_argument("error: invalid parameter for argument: " + arg);
        } else if (!params.only_list_tensors_name && params.model_paths.size() < 2) {
            throw std::invalid_argument("error: require at least 2 models");
        }

        if (params.only_list_tensors_name) {
            if (params.model_paths.size() != 1) {
                throw std::invalid_argument("error: we can only list tensors of one single model");
            }
            print_model_tensors_name(params.model_paths[0]);
            return 0; // exit now
        }

        size_t n_layers = 0;
        auto instructions = parse_config(params.config_path, params.model_paths[0], n_layers);

        if (params.dry_run) {
            return 0;
        }

        std::vector<const char*> p_model_paths;
        for (auto & m : params.model_paths) {
            p_model_paths.push_back(m.data());
        }
        struct llama_merge_config config{
            {
                params.model_paths[0].c_str(),
                params.model_paths[1].c_str(),
            },
            instructions.data(),
            instructions.size(),
            n_layers,
            params.output_path.c_str(),
        };

        llama_merge_models(&config);
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << "\n\n";
    }

    return 0;
}
