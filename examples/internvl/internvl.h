#ifndef INTERNVL_H
#define INTERNVL_H

#include "ggml.h"

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define INTERNVL_API __declspec(dllexport)
#        else
#            define INTERNVL_API __declspec(dllimport)
#        endif
#    else
#        define INTERNVL_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define INTERNVL_API
#endif

struct clip_ctx;

#ifdef __cplusplus
extern "C" {
#endif

struct internvl_image_embed {
    float * embed;
    int n_image_pos;
};

/** sanity check for clip <-> internvl embed size match */
INTERNVL_API bool internvl_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip);

INTERNVL_API bool internvl_image_embed_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out);

/** build an image embed from image file bytes */
INTERNVL_API struct internvl_image_embed * internvl_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length);
/** build an image embed from a path to an image filename */
INTERNVL_API struct internvl_image_embed * internvl_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path);
INTERNVL_API void internvl_image_embed_free(struct internvl_image_embed * embed);
/** free an embedding made with internvl_image_embed_make_* */

/** write the image represented by embed into the llama context with batch size n_batch, starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
INTERNVL_API bool internvl_eval_image_embed(struct llama_context * ctx_llama, const struct internvl_image_embed * embed, int n_batch, int * n_past);


#ifdef __cplusplus
}
#endif

#endif
