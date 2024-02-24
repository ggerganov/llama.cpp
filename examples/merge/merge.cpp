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
//  ./merge ./path/model_1 LAYERS_1 ./path/model_2 LAYERS_2
//
[[noreturn]]
static void usage(const char * executable) {
    printf("usage: %s ./path/model_1 LAYERS_1 ./path/model_2 LAYERS_2\n\n", executable);
    printf("  LAYERS must be in format: p0-p1,p2-p3,p4,... Example: 0-5,7,8-12\n");
    //printf("  Optionally, you can specify the scaling for a range of layers, for example: 0-5*0.5,6-7*1\n");
    printf("  The embedding layer of the first model will be used");
    exit(1);
}

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    std::vector<struct llama_merge_config> configs;
    for (int i = 0; i < 100; i++) {
        struct llama_merge_config conf{i, 0.0, 0.0};
        configs.push_back(conf);
    }
    llama_merge_models(
        "",
        "",
        configs.data(),
        100,
        "/tmp/dolphin-test-merge.gguf"
    );
    std::cout << "done\n";
    llama_backend_free();
}