#ifndef LLAVA_H
#define LLAVA_H

#include "ggml.h"
#include "common.h"

struct clip_ctx;

#ifdef __cplusplus
extern "C" {
#endif

/** using ctx_clip, build a llava image embedding from the passed-in image `img` (see clip.h for methods to load img). 
 * result is returned as image_embd_out, size n_image_pos_out */
LLAMA_API bool llava_build_img_embed(const struct llama_context * ctx_llama, struct clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_image_pos_out);

/** write the image represented by image_embd (size n_image_pos) into the llama context with batch size n_batch, 
 * starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
LLAMA_API bool llava_eval_image_embd(struct llama_context * ctx_llama, float * image_embd, int n_image_pos, int n_batch, int * n_past);


#ifdef __cplusplus
}
#endif

#endif
