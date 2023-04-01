#include "ggml.h"
#include "rwkv.h"

#include <string>
#include <vector>
#include <thread>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

// --- Utilities ---

// Checks that x is not false. If x is false, prints fancy message to stderr and aborts the execution.
#define RWKV_ASSERT(x, ...) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "*** Assertion failed ***\n"); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

// Formats and prints a message to stderr. Trailing newline is added automatically.
#define RWKV_LOG(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while (0)

// --- Script ---

// Usage: main_rwkv.exe "C:\model.bin" <token index> "C:\state_in.bin" "C:\state_out.bin" "C:\logits_out.bin" [thread count]
// Token index is 0-based.
// Thread count is optional and defaults to std::thread::hardware_concurrency() / 2.
// To start from new state, pass empty string instead of input state file path.
int main(int argc, char ** argv) {
    ggml_run_test_suite();

    fprintf(stderr, "%s\n", rwkv_get_system_info_string());

    RWKV_ASSERT(argc - 1 == 5 || argc - 1 == 6, "Expected 5 or 6 arguments, got %d", argc - 1);
    char * model_path = argv[1];
    char * token_s = argv[2];
    char * state_in_path = argv[3];
    char * state_out_path = argv[4];
    char * logits_out_path = argv[5];

    int32_t token = strtol(token_s, (char **) NULL, 10);
    RWKV_LOG("Token index is %d", token);

    bool create_new_state = strcmp(state_in_path, "") == 0;

    int n_threads;

    if (argc - 1 == 6) {
        n_threads = strtol(argv[6], (char **) NULL, 10);
    } else {
        n_threads = 0;
    }

    if (n_threads == 0) {
        n_threads = std::max(1, (int32_t) std::thread::hardware_concurrency() / 2);
    } else {
        RWKV_ASSERT(n_threads > 0, "Thread couns %d is not positive", n_threads);
    }

    RWKV_LOG("Using %d threads", n_threads);

    struct rwkv_context * ctx = rwkv_init_from_file(model_path, n_threads);

    RWKV_ASSERT(ctx != NULL, "Failed to load the model");

    size_t state_buffer_size = rwkv_get_state_buffer_element_count(ctx) * sizeof(float);
    size_t logits_buffer_size = rwkv_get_logits_buffer_element_count(ctx) * sizeof(float);

    float * state_buffer = (float *) calloc(1, state_buffer_size);
    float * logits_buffer = (float *) calloc(1, logits_buffer_size);

    if (!create_new_state) {
        RWKV_LOG("Loading state from %s", state_in_path);

        FILE * state_in_file = fopen(state_in_path, "rb");
        RWKV_ASSERT(state_in_file != NULL, "Failed to open file %s", state_in_path);

        // TODO Saving/loading raw data makes state cache machine-dependent
        RWKV_ASSERT(fread(state_buffer, 1, state_buffer_size, state_in_file) == state_buffer_size, "Failed to read state from a file");

        fclose(state_in_file);
    }

    bool result = rwkv_eval(
        ctx,
        token,
        create_new_state ? NULL : state_buffer,
        state_buffer,
        logits_buffer
    );

    RWKV_ASSERT(result, "Failed to evaluate the model");

    {
        RWKV_LOG("Saving state to %s", state_out_path);

        FILE * state_out_file = fopen(state_out_path, "wb");
        RWKV_ASSERT(state_out_file != NULL, "Failed to open file %s", state_out_path);

        RWKV_ASSERT(fwrite(state_buffer, 1, state_buffer_size, state_out_file) == state_buffer_size, "Failed to write state to a file");

        fclose(state_out_file);
    }

    {
        RWKV_LOG("Saving logits to %s", logits_out_path);

        FILE * logits_out_file = fopen(logits_out_path, "wb");
        RWKV_ASSERT(logits_out_file != NULL, "Failed to open file %s", logits_out_path);

        RWKV_ASSERT(fwrite(logits_buffer, 1, logits_buffer_size, logits_out_file) == logits_buffer_size, "Failed to write logits to a file");

        fclose(logits_out_file);
    }

    rwkv_free(ctx);

    delete state_buffer;
    delete logits_buffer;

    RWKV_LOG("OK");

    return 0;
}
