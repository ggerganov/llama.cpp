#ifndef LLAVA_H
#define LLAVA_H

#include "ggml.h"
#include "common.h"

struct clip_ctx;

#ifdef __cplusplus
extern "C" {
#endif

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

struct llava_context * llava_init(gpt_params * params);
void llava_free(struct llava_context * ctx_llava);

/** build a llava image embedding from the passed-in clip image `img`. result is returned as image_embd_out, size n_image_pos_out */
bool llava_build_img_embed(const struct llama_context * ctx_llama, struct clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_image_pos_out);


#ifdef __cplusplus
}
#endif

#endif
