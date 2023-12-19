#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(void) {
    const char * models_to_try[] = {
        // Same default as example/main for local use
        "./models/7B/ggml-model-f16.gguf",
        // Models for ./ci/run.sh
        "./models-mnt/open-llama/3B-v2/ggml-model-q2_k.gguf",
        "./models-mnt/open-llama/7B-v2/ggml-model-q2_k.gguf",
    };

    const char * chosen_model;
    for (size_t i = 0; i < sizeof(models_to_try) / sizeof(models_to_try[0]); i++) {
        const auto * model = models_to_try[i];

        auto * file = fopen(model, "r");
        if (file == nullptr) {
            continue;
        }

        chosen_model = model;
        fprintf(stderr, "using '%s'\n", model);
        fclose(file);
    }

    if (chosen_model == nullptr) {
        fprintf(stderr, "no model found\n");
        return EXIT_FAILURE;
    }

    llama_backend_init(false);
    auto params = llama_model_params{};
    params.use_mmap = false;
    params.progress_callback = [](float progress, void * ctx){
        (void) ctx;
        return progress > 0.05;
    };

    auto * model = llama_load_model_from_file(chosen_model, params);
    llama_backend_free();
    return model == nullptr ? EXIT_SUCCESS : EXIT_FAILURE;
}
