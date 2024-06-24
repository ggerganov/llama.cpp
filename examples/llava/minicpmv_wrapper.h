#ifndef MINICPMV_H
#define MINICPMV_H

#include "common.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MINICPMV_API __declspec(dllexport)
#        else
#            define MINICPMV_API __declspec(dllimport)
#        endif
#    else
#        define MINICPMV_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MINICPMV_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct minicpmv_context {
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

MINICPMV_API struct llama_model * llava_init(gpt_params * params);
MINICPMV_API struct minicpmv_context * llava_init_context(gpt_params * params, llama_model * model);
MINICPMV_API void llava_free(struct minicpmv_context * ctx_llava);

MINICPMV_API struct clip_ctx * clip_init_context(gpt_params * params);
MINICPMV_API struct uhd_image_embed * minicpmv_image_embed(gpt_params * params, const std::string & fname);

MINICPMV_API bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past);
MINICPMV_API bool eval_id(struct llama_context * ctx_llama, int id, int * n_past);
MINICPMV_API bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos);
MINICPMV_API void process_image(struct minicpmv_context * ctx_llava,  std::vector<std::vector<struct llava_image_embed *>> image_embed_slices, gpt_params * params, int &n_past);
MINICPMV_API const char * sample(struct llama_sampling_context * ctx_sampling, struct llama_context * ctx_llama, int * n_past);

#ifdef __cplusplus
}
#endif

#endif