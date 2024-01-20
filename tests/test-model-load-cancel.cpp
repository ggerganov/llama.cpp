#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char *argv[] ) {
    char * model_path;
    if (argc > 1) {
        model_path = argv[1];
    } else {
        model_path = getenv("GG_RUN_CTEST_MODELFILE");
        if (!model_path || strlen(model_path) == 0) {
            fprintf(stderr, "error: no model file provided\n");
            exit(1);
        }
    }

    auto * file = fopen(model_path, "r");
    if (file == nullptr) {
        fprintf(stderr, "no model at '%s' found\n", model_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "using '%s'\n", model_path);
    fclose(file);

    llama_backend_init(false);
    auto params = llama_model_params{};
    params.use_mmap = false;
    params.progress_callback = [](float progress, void * ctx){
        (void) ctx;
        return progress > 0.50;
    };
    auto * model = llama_load_model_from_file(model_path, params);
    llama_backend_free();
    return model == nullptr ? EXIT_SUCCESS : EXIT_FAILURE;
}
