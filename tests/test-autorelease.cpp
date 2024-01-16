// ref: https://github.com/ggerganov/llama.cpp/issues/4952#issuecomment-1892864763

#include <cstdio>
#include <string>
#include <pthread.h>

#include "llama.h"

static std::string g_fname;

static void * llamacpp_pthread(void * arg) {
    (void)arg;

    llama_backend_init(false);
    auto * model = llama_load_model_from_file(g_fname.c_str(), llama_model_default_params());
    auto * ctx = llama_new_context_with_model(model, llama_context_default_params());
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return NULL;
}

// This creates a new context inside a pthread and then tries to exit cleanly.
int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("Usage: %s model.gguf\n", argv[0]);
        return 0; // intentionally return success
    }

    g_fname = argv[1];

    pthread_t tid;
    pthread_create(&tid, NULL, llamacpp_pthread, NULL);
    pthread_join(tid, NULL);

    return 0;
}
