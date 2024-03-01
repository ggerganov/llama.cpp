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

[[noreturn]]
static void usage(const char * executable, int exit_code) {
    printf("usage: %s -c CONFIG_FILE -o OUTPUT_FILE -m MODEL_PATH -m MODEL_PATH ...\n\n", executable);
    printf("\n");
    printf("Merging 2 models and change layers configuration.\n");
    printf("Merge config format is CSV, without header, one line represents one layer of the output model, columns in the order below:\n");
    printf("- Model A layer\n");
    printf("- Model A scale\n");
    printf("- Model B layer\n");
    printf("- Model B scale\n");
    printf("- ...\n");
    printf("\n");
    printf("For example:\n");
    printf("0,1.0,0,0.0    meaning: output layer 0 = A[0]*1.0 + B[0] * 0.0\n");
    printf("0,1.0,0,0.0    meaning: output layer 1 = A[0]*1.0 + B[0] * 0.0\n");
    printf("1,0.0,2,0.0    meaning: output layer 2 = A[1]*0.0 + B[2] * 0.0\n");
    printf("2,0.5,1,0.5    meaning: output layer 3 = A[2]*0.5 + B[1] * 0.5\n");
    printf("\n");
    printf("NOTE:\n");
    printf("- The embedding layer of the first model will be used\n");
    printf("- Currently, only F16 model type is supported\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help                 Show this help message and exit\n");
    printf("  -c, --config CONFIG_FILE   Path to config file (CSV format)\n");
    printf("  -m, --model MODEL_PATH     Path to model. This option can be repeated multiple times and must be specified in the right order.\n");
    printf("  -o, --output OUTPUT_FILE   Path to the output model\n");
    printf("\n");
    printf("Example: ./merge -c config.csv -o output.gguf -m model_a.gguf -m model_b.gguf\n");
    exit(exit_code);
}

inline std::vector<std::string> str_split(std::string str, const std::string & delimiter) {
    size_t pos = 0;
    std::string token;
    std::vector<std::string> output;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        output.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    output.push_back(str); // the rest
    return output;
}

static std::vector<struct llama_merge_layer> parse_config(std::string & config_path, size_t n_models, std::vector<int> & buf_srcs, std::vector<float> & buf_scales) {
    // read file
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file merge config file");
    }
    std::ostringstream content;
    content << file.rdbuf(); // Read the entire file into the stringstream
    file.close();

    // allocate memory
    auto lines = str_split(content.str(), "\n");
    buf_srcs.resize(lines.size()*n_models);
    buf_scales.resize(lines.size()*n_models);

    // process line by line, one line is one layer
    std::vector<struct llama_merge_layer> layers;
    for (size_t i_layer = 0; i_layer < lines.size(); i_layer++) {
        auto columns = str_split(lines[i_layer], ",");
        if (columns.size() != n_models*2) {
            std::stringstream ss;
            ss << "error: line " << i_layer+1 << " is malformed. Expect to have exactly " << n_models*2 << " columns, but got " << columns.size() << " columns";
            throw std::runtime_error(ss.str());
        }
        int * srcs     = buf_srcs.data()   + i_layer*n_models;
        float * scales = buf_scales.data() + i_layer*n_models;
        for (size_t i_model = 0; i_model < n_models; i_model++) {
            srcs[i_model]   = std::stoi(columns[i_model*2]);
            scales[i_model] = std::stof(columns[i_model*2 + 1]);
        }
        layers.push_back(llama_merge_layer{srcs, scales});
    }
    return layers;
}

int main(int argc, char ** argv) {
    bool invalid_param = false;
    std::string config_path;
    std::vector<std::string> model_paths;
    std::string output_path;

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
            config_path = argv[i];
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            model_paths.push_back(argv[i]);
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            output_path = argv[i];
        }
    }

    if (invalid_param) {
        throw std::invalid_argument("error: invalid parameter for argument: " + arg);
    } else if (config_path.empty()) {
        throw std::invalid_argument("error: missing config path");
    } else if (model_paths.size() < 2) {
        throw std::invalid_argument("error: require at least 2 models");
    } else if (output_path.empty()) {
        throw std::invalid_argument("error: missing output path");
    }

    // buffers to hold allocated data
    std::vector<int> buf_srcs;
    std::vector<float> buf_scales;

    auto layers = parse_config(config_path, model_paths.size(), buf_srcs, buf_scales);
    std::vector<const char*> p_model_paths;
    for (auto & m : model_paths) {
        p_model_paths.push_back(m.data());
    }
    const struct llama_merge_config config{
        p_model_paths.data(),
        p_model_paths.size(),
        layers.data(),
        layers.size(),
        output_path.data(),
    };

    llama_merge_models(&config);

    return 0;
}