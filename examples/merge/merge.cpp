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


// usage:
//  ./merge ./path/model_1 CONFIG1 ./path/model_2 CONFIG2
//
[[noreturn]]
static void usage(const char * executable) {
    printf("usage: %s ./path/model_1 CONFIG1 ./path/model_2 CONFIG2\n\n", executable);
    printf("  CONFIG must be in format: p0-p1,p2-p3,p4,... Example: 0-5,7,8-12\n");
    printf("  Optionally, you can specify the scaling for a range of layers, for example: 0-5*0.5,6-7*1. By default, scale will be 0.5. The number of layer start counting from 0.\n");
    printf("  The embedding layer of the first model will be used\n");
    printf("  NOTE: currently, only F16 model type is supported\n");
    exit(1);
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

static std::vector<struct llama_merge_config> parse_config(std::string & input) {
    std::vector<struct llama_merge_config> configs;
    auto intervals = str_split(input, ",");
    for (auto & interval : intervals) {
        auto components = str_split(interval, "*");
        if (components.empty()) {
            throw std::runtime_error("Config is incorrect");
        }
        float scale = components.size() == 2
            ? std::stof(components[1])
            : 0.5; // be default
        auto p0p1 = str_split(components[0], "-");
        if (p0p1.empty()) {
            throw std::runtime_error("Layer interval is invalid");
        }
        int p0 = std::stoi(p0p1[0]);
        int p1 = p0p1.size() == 2 ? std::stoi(p0p1[1]) : p0;
        if (p0 > p1) {
            throw std::runtime_error("Layer interval is invalid, the end layer number is bigger and start layer number (p0 > p1)");
        }
        for (int i = p0; i <= p1; i++) {
            struct llama_merge_config conf{i, scale, scale};
            configs.push_back(conf);
        }
        // TODO: maybe check for overlap intervals?
    }
    return configs;
}

int main(int argc, char ** argv) {
    llama_backend_init();

    if (argc < 6) {
        usage(argv[0]);
    }

    std::string fname_model1(argv[1]);
    std::string config_model1(argv[2]);
    std::string fname_model2(argv[3]);
    std::string config_model2(argv[4]);
    std::string fname_output(argv[5]);

    // TODO: add try catch
    auto configs1 = parse_config(config_model1);
    auto configs2 = parse_config(config_model2);
    std::vector<struct llama_merge_config> configs;

    if (configs1.size() != configs2.size()) {
        fprintf(stderr, "Number of layers between 2 configs does not match, config1 has %ld layers and config2 has %ld layers\n", configs1.size(), configs2.size());
    }

    // merge 2 configs
    printf("Merge configs:\n");
    for (auto c1 : configs1) {
        float scale2 = -1;
        for (auto c2 : configs2) {
            if (c2.i_layer == c1.i_layer) {
                scale2 = c2.scale2;
            }
        }
        if (scale2 < 0) {
            fprintf(stderr, "Cannot find config for layer %d in CONFIG2\n", c1.i_layer);
            exit(1);
        }
        struct llama_merge_config conf{c1.i_layer, c1.scale1, scale2};
        configs.push_back(conf);

        printf("  Layer %d: scale1 = %f, scale2 = %f\n", conf.i_layer, conf.scale1, conf.scale2);
    }

    llama_merge_models(
        fname_model1.c_str(),
        fname_model2.c_str(),
        configs.data(),
        configs.size(),
        fname_output.c_str()
    );
    llama_backend_free();
}