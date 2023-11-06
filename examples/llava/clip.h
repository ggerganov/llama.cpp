#ifndef CLIP_H
#define CLIP_H

#include <stddef.h>
#include <stdint.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define CLIP_API __declspec(dllexport)
#        else
#            define CLIP_API __declspec(dllimport)
#        endif
#    else
#        define CLIP_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define CLIP_API
#endif

struct clip_ctx;

#ifdef __cplusplus
extern "C" {
#endif

struct clip_vision_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;
    float eps;
};

/** load mmproj model */
CLIP_API struct clip_ctx * clip_model_load(const char * fname, const int verbosity);
/** free mmproj model */
CLIP_API void clip_free(struct clip_ctx * ctx);

size_t clip_embd_nbytes(const struct clip_ctx * ctx);
int clip_n_patches(const struct clip_ctx * ctx);
int clip_n_mmproj_embd(const struct clip_ctx * ctx);

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;
    uint8_t * data = NULL;
    size_t size;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;
    float * data = NULL;
    size_t size;
};

struct clip_image_u8_batch {
    struct clip_image_u8 * data;
    size_t size;
};

struct clip_image_f32_batch {
    struct clip_image_f32 * data;
    size_t size;
};

struct clip_image_u8 * make_clip_image_u8();
struct clip_image_f32 * make_clip_image_f32();
CLIP_API void clip_image_u8_free(clip_image_u8 * img);
CLIP_API void clip_image_f32_free(clip_image_f32 * img);
CLIP_API bool clip_image_load_from_file(const char * fname, struct clip_image_u8 * img);
/** interpret bytes as an image file with length bytes_length, and use the result to populate img */
CLIP_API bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img);

bool clip_image_preprocess(const struct clip_ctx * ctx, const struct clip_image_u8 * img, struct clip_image_f32 * res, const bool pad2square);
bool clip_image_encode(const struct clip_ctx * ctx, const int n_threads, struct clip_image_f32 * img, float * vec);

bool clip_image_batch_encode(const struct clip_ctx * ctx, const int n_threads, const struct clip_image_f32_batch * imgs,
                             float * vec);

bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype);

#ifdef __cplusplus
}
#endif

#endif // CLIP_H
