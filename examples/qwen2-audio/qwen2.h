#pragma once

#include "whisper.h"
#include "llama.h"
#include "grammar-parser.h"
#include "common.h"
#include "common-nexa.h"

#include <string>
#include <thread>

#include "audio-projector.h"

#ifdef OMNI_AUDIO_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef OMNI_AUDIO_BUILD
#            define OMNI_AUDIO_API __declspec(dllexport)
#        else
#            define OMNI_AUDIO_API __declspec(dllimport)
#        endif
#    else
#        define OMNI_AUDIO_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define OMNI_AUDIO_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct omni_context_params
{
    const char *model;
    const char *mmproj;
    const char *file;
    const char *prompt;
    int32_t n_gpu_layers;
};

OMNI_AUDIO_API bool omni_context_params_parse(int argc, char **argv, struct omni_context_params * params);

OMNI_AUDIO_API struct omni_context_params * omni_context_default_params();

OMNI_AUDIO_API struct omni_context * omni_init_context(struct omni_context_params * params);

OMNI_AUDIO_API void omni_free(struct omni_context * ctx_omni);

OMNI_AUDIO_API const char* omni_process_full(
    struct omni_context * ctx_omni,
    struct omni_context_params * params
);

#ifdef __cplusplus
}
#endif
