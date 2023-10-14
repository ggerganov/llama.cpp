#ifndef LLAVA_H
#define LLAVA_H

#include "ggml.h"

struct clip_ctx;

#ifdef __cplusplus
extern "C" {
#endif

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;

//    int n_img_pos = 0;
//    float * image_embd = NULL;
};

struct llava_context * llava_init(gpt_params * params);
void llava_free(struct llava_context * ctx_llava);

//void llava_process_prompt(struct llava_context * ctx_llava, gpt_params * params, const char * prompt);


#ifdef __cplusplus
}
#endif

#endif
