#ifndef OMNIVLM_H
#define OMNIVLM_H

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

struct clip_ctx;
struct omni_image_embed {
    float * embed;
    int n_image_pos;
};

OMNIVLM_API bool omnivlm_validate_embed_size(const struct llama_context * ctx_llama, const struct clip_ctx * ctx_clip);

OMNIVLM_API bool omnivlm_image_embed_make_with_clip_img(struct clip_ctx * ctx_clip, int n_threads, const struct clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out);

/** build an image embed from image file bytes */
OMNIVLM_API struct omni_image_embed * omnivlm_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length);
/** build an image embed from a path to an image filename */
OMNIVLM_API struct omni_image_embed * omnivlm_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path);
/** free an embedding made with OMNIVLM_image_embed_make_* */
OMNIVLM_API void omnivlm_image_embed_free(struct omni_image_embed * embed);

/** write the image represented by embed into the llama context with batch size n_batch, starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
OMNIVLM_API bool omnivlm_eval_image_embed(struct llama_context * ctx_llama, const struct omni_image_embed * embed, int n_batch, int * n_past);

#ifdef __cplusplus
}
#endif

#endif
