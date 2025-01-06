#ifndef OMNIVLMWRAPPER_H
#define OMNIVLMWRAPPER_H
#include <stdint.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define OMNIVLM_API __declspec(dllexport)
#        else
#            define OMNIVLM_API __declspec(dllimport)
#        endif
#    else
#        define OMNIVLM_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define OMNIVLM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct omni_streaming_sample;

OMNIVLM_API void omnivlm_init(const char* llm_model_path, const char* projector_model_path, const char* omni_vlm_version);

OMNIVLM_API const char* omnivlm_inference(const char* prompt, const char* imag_path);

OMNIVLM_API struct omni_streaming_sample* omnivlm_inference_streaming(const char* prompt, const char* imag_path);

OMNIVLM_API int32_t sample(struct omni_streaming_sample *);

OMNIVLM_API const char* get_str(struct omni_streaming_sample *);

OMNIVLM_API void omnivlm_free();

#ifdef __cplusplus
}
#endif

#endif