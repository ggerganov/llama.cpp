#ifndef RWKV_H
#define RWKV_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef RWKV_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef RWKV_BUILD
#            define RWKV_API __declspec(dllexport)
#        else
#            define RWKV_API __declspec(dllimport)
#        endif
#    else
#        define RWKV_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define RWKV_API
#endif

// 'ggmf' in hex.
#define RWKV_FILE_MAGIC 0x67676d66

#define RWKV_FILE_VERSION_0 100
#define RWKV_FILE_VERSION_1 101
#define RWKV_FILE_VERSION_MIN RWKV_FILE_VERSION_0
#define RWKV_FILE_VERSION_MAX RWKV_FILE_VERSION_1
// Default file version is the latest version.
#define RWKV_FILE_VERSION RWKV_FILE_VERSION_MAX

#ifdef __cplusplus
extern "C" {
#endif

    // Represents an error encountered during a function call.
    // These are flags, so an actual value might contain multiple errors.
    enum rwkv_error_flags {
        RWKV_ERROR_NONE = 0,

        RWKV_ERROR_ARGS = 1 << 8,
        RWKV_ERROR_FILE = 2 << 8,
        RWKV_ERROR_MODEL = 3 << 8,
        RWKV_ERROR_MODEL_PARAMS = 4 << 8,
        RWKV_ERROR_GRAPH = 5 << 8,
        RWKV_ERROR_CTX = 6 << 8,

        RWKV_ERROR_ALLOC = 1,
        RWKV_ERROR_FILE_OPEN = 2,
        RWKV_ERROR_FILE_STAT = 3,
        RWKV_ERROR_FILE_READ = 4,
        RWKV_ERROR_FILE_WRITE = 5,
        RWKV_ERROR_FILE_MAGIC = 6,
        RWKV_ERROR_FILE_VERSION = 7,
        RWKV_ERROR_DATA_TYPE = 8,
        RWKV_ERROR_UNSUPPORTED = 9,
        RWKV_ERROR_SHAPE = 10,
        RWKV_ERROR_DIMENSION = 11,
        RWKV_ERROR_KEY = 12,
        RWKV_ERROR_DATA = 13,
        RWKV_ERROR_PARAM_MISSING = 14
    };

    struct rwkv_context;

    // Sets whether errors are automatically printed to stderr.
    // If this is set to false, you are responsible for calling rwkv_last_error manually if an operation fails.
    // - ctx: the context to suppress error messages for.
    //   If NULL, affects model load (rwkv_init_from_file) and quantization (rwkv_quantize_model_file) errors,
    //   as well as the default for new context.
    // - print_errors: whether error messages should be automatically printed.
    RWKV_API void rwkv_set_print_errors(struct rwkv_context * ctx, bool print_errors);

    // Gets whether errors are automatically printed to stderr.
    // - ctx: the context to retrieve the setting for, or NULL for the global setting.
    RWKV_API bool rwkv_get_print_errors(struct rwkv_context * ctx);

    // Retrieves and clears the error flags.
    // - ctx: the context the retrieve the error for, or NULL for the global error.
    RWKV_API enum rwkv_error_flags rwkv_get_last_error(struct rwkv_context * ctx);

    // Loads the model from a file and prepares it for inference.
    // Returns NULL on any error. Error messages would be printed to stderr.
    // - model_file_path: path to model file in ggml format.
    // - n_threads: count of threads to use, must be positive.
    RWKV_API struct rwkv_context * rwkv_init_from_file(const char * model_file_path, const uint32_t n_threads);

    // Evaluates the model for a single token.
    // Returns false on any error. Error messages would be printed to stderr.
    // - token: next token index, in range 0 <= token < n_vocab.
    // - state_in: FP32 buffer of size rwkv_get_state_buffer_element_count; or NULL, if this is a first pass.
    // - state_out: FP32 buffer of size rwkv_get_state_buffer_element_count. This buffer will be written to.
    // - logits_out: FP32 buffer of size rwkv_get_logits_buffer_element_count. This buffer will be written to.
    RWKV_API bool rwkv_eval(const struct rwkv_context * ctx, const uint32_t token, const float * state_in, float * state_out, float * logits_out);

    // Returns count of FP32 elements in state buffer.
    RWKV_API uint32_t rwkv_get_state_buffer_element_count(const struct rwkv_context * ctx);

    // Returns count of FP32 elements in logits buffer.
    RWKV_API uint32_t rwkv_get_logits_buffer_element_count(const struct rwkv_context * ctx);

    // Frees all allocated memory and the context.
    RWKV_API void rwkv_free(struct rwkv_context * ctx);

    // Quantizes FP32 or FP16 model to one of quantized formats.
    // Returns false on any error. Error messages would be printed to stderr.
    // - model_file_path_in: path to model file in ggml format, must be either FP32 or FP16.
    // - model_file_path_out: quantized model will be written here.
    // - format_name: must be one of available format names below.
    // Available format names:
    // - Q4_0
    // - Q4_1
    // - Q5_0
    // - Q5_1
    // - Q8_0
    RWKV_API bool rwkv_quantize_model_file(const char * model_file_path_in, const char * model_file_path_out, const char * format_name);

    // Returns system information string.
    RWKV_API const char * rwkv_get_system_info_string(void);

#ifdef __cplusplus
}
#endif

#endif