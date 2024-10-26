// ref: https://github.com/ggerganov/jarvis.cpp/issues/4952#issuecomment-1892864763

#include <cstdio>
#include <string>
#include <thread>

#include "jarvis.h"
#include "get-model.h"

// This creates a new context inside a pthread and then tries to exit cleanly.
int main(int argc, char ** argv) {
    auto * model_path = get_model_or_exit(argc, argv);

    std::thread([&model_path]() {
        jarvis_backend_init();
        auto * model = jarvis_load_model_from_file(model_path, jarvis_model_default_params());
        auto * ctx = jarvis_new_context_with_model(model, jarvis_context_default_params());
        jarvis_free(ctx);
        jarvis_free_model(model);
        jarvis_backend_free();
    }).join();

    return 0;
}
