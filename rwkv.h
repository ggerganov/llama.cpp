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
#define RWKV_FILE_VERSION 100

#ifdef __cplusplus
extern "C" {
#endif

    struct rwkv_context;

    // Loads the model from a file and prepares it for inference.
    // Returns NULL on any error. Error messages would be printed to stderr.
    // - model_file_path: path to model file in ggml format.
    // - n_threads: count of threads to use, must be positive.
    RWKV_API struct rwkv_context * rwkv_init_from_file(const char * model_file_path, uint32_t n_threads);

    // Evaluates the model for a single token.
    // Returns false on any error. Error messages would be printed to stderr.
    // - token: next token index, in range 0 <= token < n_vocab.
    // - state_in: FP32 buffer of size rwkv_get_state_buffer_element_count; or NULL, if this is a first pass.
    // - state_out: FP32 buffer of size rwkv_get_state_buffer_element_count. This buffer will be written to.
    // - logits_out: FP32 buffer of size rwkv_get_logits_buffer_element_count. This buffer will be written to.
    RWKV_API bool rwkv_eval(struct rwkv_context * ctx, int32_t token, float * state_in, float * state_out, float * logits_out);

    // Returns count of FP32 elements in state buffer.
    RWKV_API uint32_t rwkv_get_state_buffer_element_count(struct rwkv_context * ctx);

    // Returns count of FP32 elements in logits buffer.
    RWKV_API uint32_t rwkv_get_logits_buffer_element_count(struct rwkv_context * ctx);

    // Frees all allocated memory and the context.
    RWKV_API void rwkv_free(struct rwkv_context * ctx);

    // Quantizes FP32 or FP16 model to one of INT4 formats.
    // Returns false on any error. Error messages would be printed to stderr.
    // - model_file_path_in: path to model file in ggml format, must be either FP32 or FP16.
    // - model_file_path_out: quantized model will be written here.
    // - q_type: set to 2 for GGML_TYPE_Q4_0, set to 3 for GGML_TYPE_Q4_1.
    RWKV_API bool rwkv_quantize_model_file(const char * model_file_path_in, const char * model_file_path_out, uint32_t q_type);

    // Returns system information string.
    RWKV_API const char * rwkv_get_system_info_string(void);

#ifdef __cplusplus
}
#endif

#endif
