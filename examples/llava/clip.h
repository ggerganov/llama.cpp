#ifndef CLIP_H
#define CLIP_H

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

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

    char mm_patch_merge_type[32]="flat"; // spatial_unpad or flat (default)
    int32_t image_grid_pinpoints[32];
    int32_t image_crop_resolution;

};

struct clip_ctx;
CLIP_API const struct clip_vision_hparams clip_get_vision_hparams(const struct clip_ctx * ctx);

CLIP_API struct clip_ctx * clip_model_load(const char * fname, int verbosity);
CLIP_API struct clip_ctx * clip_model_load_cpu(const char * fname, int verbosity);

CLIP_API void clip_free(struct clip_ctx * ctx);

CLIP_API size_t clip_embd_nbytes(const struct clip_ctx * ctx);

CLIP_API int clip_n_patches    (const struct clip_ctx * ctx);
CLIP_API int clip_n_mmproj_embd(const struct clip_ctx * ctx);

// RGB uint8 image
CLIP_API struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
 CLIP_API struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};


struct clip_image_u8_batch {
    struct clip_image_u8 * data;
    size_t size;
};

struct clip_image_f32_batch {
    struct clip_image_f32 * data;
    size_t size;
};
CLIP_API struct clip_image_grid_shape {
    int first;
    int second;
};

CLIP_API struct clip_image_u8  * clip_image_u8_init ();
CLIP_API struct clip_image_f32 * clip_image_f32_init();

CLIP_API void clip_image_u8_free (struct clip_image_u8 * img);
CLIP_API void clip_image_f32_free(struct clip_image_f32 * img);

CLIP_API bool clip_image_load_from_file(const char * fname, struct clip_image_u8 * img);
CLIP_API void clip_image_save_to_bmp(const clip_image_u8& img, const std::string& filename);
CLIP_API void clip_image_convert_f32_to_u8(const clip_image_f32& src, clip_image_u8& dst);

/** interpret bytes as an image file with length bytes_length, and use the result to populate img */
CLIP_API bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img);
/** preprocess img and store the result in res_tensor, pad2square may be overriden to false depending on model configuration */
CLIP_API bool clip_image_preprocess(struct clip_ctx * ctx, const clip_image_u8 * img, std::vector<clip_image_f32*>& res_tensor, bool pad2square);
CLIP_API struct clip_image_grid_shape get_anyres_image_grid_shape(const std::pair<int, int>& image_size, const std::vector<std::pair<int, int>>& grid_pinpoints, int image_patch_size);
CLIP_API struct ggml_tensor *clip_get_newline_tensor(const struct clip_ctx * ctx);

CLIP_API bool clip_image_encode      (struct clip_ctx * ctx, int n_threads, struct clip_image_f32 * img, float * vec);
CLIP_API bool clip_image_batch_encode(struct clip_ctx * ctx, int n_threads, const struct clip_image_f32_batch * imgs, float * vec);

CLIP_API bool clip_model_quantize(const char * fname_inp, const char * fname_out, int itype);

#ifdef __cplusplus
}
#endif

#endif // CLIP_H
