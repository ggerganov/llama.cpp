#include "jarvis.h"
#include "get-model.h"

#include <cstdlib>

int main(int argc, char *argv[] ) {
    auto * model_path = get_model_or_exit(argc, argv);
    auto * file = fopen(model_path, "r");
    if (file == nullptr) {
        fprintf(stderr, "no model at '%s' found\n", model_path);
        return EXIT_FAILURE;
    }

    fprintf(stderr, "using '%s'\n", model_path);
    fclose(file);

    jarvis_backend_init();
    auto params = jarvis_model_params{};
    params.use_mmap = false;
    params.progress_callback = [](float progress, void * ctx){
        (void) ctx;
        return progress > 0.50;
    };
    auto * model = jarvis_load_model_from_file(model_path, params);
    jarvis_backend_free();
    return model == nullptr ? EXIT_SUCCESS : EXIT_FAILURE;
}
