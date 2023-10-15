#ifndef LLAVA_H
#define LLAVA_H

#include "ggml.h"
#include "common.h"

struct clip_ctx;

#ifdef __cplusplus
extern "C" {
#endif

struct llava_image_embed {
    float * embed;
    int n_image_pos;
};

LLAMA_API bool llava_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip);

LLAMA_API struct llava_image_embed * llava_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length);
LLAMA_API struct llava_image_embed * llava_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path);
LLAMA_API void llava_image_embed_free(struct llava_image_embed * embed);

/** write the image represented by embed into the llama context with batch size n_batch, 
 * starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
LLAMA_API bool llava_eval_image_embed(struct llama_context * ctx_llama, const struct llava_image_embed * embed, int n_batch, int * n_past);


#ifdef __cplusplus
}
#endif

#endif
