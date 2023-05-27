#ifndef RWKV_H2
#define RWKV_H2

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef RWKV_SHARED2
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef RWKV_BUILD
#            define RWKV_V2_API __declspec(dllexport)
#        else
#            define RWKV_V2_API __declspec(dllimport)
#        endif
#    else
#        define RWKV_V2_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define RWKV_V2_API
#endif

// 'ggmf' in hex.
#define RWKV_V2_FILE_MAGIC 0x67676d66
#define RWKV_V2_FILE_VERSION 100

#ifdef __cplusplus
extern "C" {
#endif

    struct rwkv_v2_context;

    // Loads the model from a file and prepares it for inference.
    // Returns NULL on any error. Error messages would be printed to stderr.
    // - model_file_path: path to model file in ggml format.
    // - n_threads: count of threads to use, must be positive.
    RWKV_V2_API struct rwkv_v2_context * rwkv_v2_init_from_file(const char * model_file_path, uint32_t n_threads);

    // Evaluates the model for a single token.
    // Returns false on any error. Error messages would be printed to stderr.
    // - token: next token index, in range 0 <= token < n_vocab.
    // - state_in: FP32 buffer of size rwkv_v2_get_state_buffer_element_count; or NULL, if this is a first pass.
    // - state_out: FP32 buffer of size rwkv_v2_get_state_buffer_element_count. This buffer will be written to.
    // - logits_out: FP32 buffer of size rwkv_v2_get_logits_buffer_element_count. This buffer will be written to.
    RWKV_V2_API bool rwkv_v2_eval(struct rwkv_v2_context * ctx, int32_t token, float * state_in, float * state_out, float * logits_out);

    // Returns count of FP32 elements in state buffer.
    RWKV_V2_API uint32_t rwkv_v2_get_state_buffer_element_count(struct rwkv_v2_context * ctx);

    // Returns count of FP32 elements in logits buffer.
    RWKV_V2_API uint32_t rwkv_v2_get_logits_buffer_element_count(struct rwkv_v2_context * ctx);

    // Frees all allocated memory and the context.
    RWKV_V2_API void rwkv_v2_free(struct rwkv_v2_context * ctx);

    // Quantizes FP32 or FP16 model to one of quantized formats.
    // Returns false on any error. Error messages would be printed to stderr.
    // - model_file_path_in: path to model file in ggml format, must be either FP32 or FP16.
    // - model_file_path_out: quantized model will be written here.
    // - format_name: must be one of available format names below.
    // Available format names:
    // - Q4_0
    // - Q4_1
    // - Q4_2
    // - Q5_0
    // - Q5_1
    // - Q8_0
    RWKV_V2_API bool rwkv_v2_quantize_model_file(const char * model_file_path_in, const char * model_file_path_out, const char * format_name);

    // Returns system information string.
    RWKV_V2_API const char * rwkv_v2_get_system_info_string(void);

#ifdef __cplusplus
}
#endif

#endif