#ifndef LLAVA_H
#define LLAVA_H

#include "ggml.h"

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

struct clip_ctx;
struct uhd_image_embed {
    std::vector<std::vector<struct llava_image_embed *>> image_embeds;
};

#ifdef __cplusplus
extern "C" {
#endif

struct llava_image_embed {
    float * embed;
    int n_image_pos;
};

/** sanity check for clip <-> llava embed size match */
MINICPMV_API bool llava_validate_embed_size(const struct llama_context * ctx_llama, const struct clip_ctx * ctx_clip);

MINICPMV_API bool llava_image_embed_make_with_clip_img(struct clip_ctx * ctx_clip, int n_threads, const struct clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out);

/** build an image embed from image file bytes */
MINICPMV_API struct uhd_image_embed * llava_image_embed_make_with_bytes_uhd(struct clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img);
/** build an image embed from a path to an image filename */
MINICPMV_API bool llava_image_embed_make_with_clip_img_ollama(struct clip_ctx * ctx_clip, int n_threads, const struct clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out);
MINICPMV_API struct uhd_image_embed *  llava_image_embed_make_with_filename_uhd(struct clip_ctx * ctx_clip, int n_threads, const char * image_path);
MINICPMV_API void llava_image_embed_free_uhd(struct uhd_image_embed *  embed);
/** free an embedding made with llava_image_embed_make_* */

/** write the image represented by embed into the llama context with batch size n_batch, starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
MINICPMV_API bool llava_eval_image_embed(struct llama_context * ctx_llama, const struct llava_image_embed * embed, int n_batch, int * n_past);

#ifdef __cplusplus
}
#endif

#endif
