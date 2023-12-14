#include "llama.h"

#include <cstdlib>
#include <tuple>

int main(void) {
    llama_backend_init(false);
    auto params = llama_model_params{};
    params.use_mmap = false;
    params.progress_callback = [](float progress, void * ctx){
        std::ignore = ctx;
        return progress > 0.50;
    };
    auto * model = llama_load_model_from_file("../models/7B/ggml-model-f16.gguf", params);
    llama_backend_free();
    return model == nullptr ? EXIT_SUCCESS : EXIT_FAILURE;
}
