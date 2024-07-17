#include "clip.h"
#include "common.h"
#include "llama.h"
#include "llava.h"
#include "base64.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

struct clip_image_grid_shape {
    int first;
    int second;
};

struct uhd_image_embed {
    std::vector<std::vector<struct llava_image_embed *>> image_embeds;
};
/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static std::pair<int, int> select_best_resolution(const std::pair<int, int>& original_size, const std::vector<std::pair<int, int>>& possible_resolutions) {
    int original_width  = original_size.first;
    int original_height = original_size.second;

    std::pair<int, int> best_fit;
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width = resolution.first;
        int height = resolution.second;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width  = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // LOG_TEE("resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

/**
 * @brief Get the anyres image grid shape object
 *
 * @param image_size
 * @param grid_pinpoints
 * @param image_patch_size
 * @return <int, int>
 */
static struct clip_image_grid_shape get_anyres_image_grid_shape(const std::pair<int, int> & image_size, const std::vector<std::pair<int, int>> & grid_pinpoints, int image_patch_size) {
    /**
        Conversion from gguf flat array to vector:
        std::vector<std::pair<int, int>> possible_resolutions;
        for (int i = 0; i < 32 && params.image_grid_pinpoints[i] != 0; i+=2) {
            possible_resolutions.push_back({params.image_grid_pinpoints[i], params.image_grid_pinpoints[i+1]});
        }
     */
    auto best_resolution = select_best_resolution(image_size, grid_pinpoints);
    return {best_resolution.first / image_patch_size, best_resolution.second / image_patch_size};
}

// Take the image segments in a grid configuration and return the embeddings and the number of embeddings into preallocated memory (image_embd_out)
static bool clip_llava_handle_patches(clip_ctx * ctx_clip, std::vector<float *> & image_embd_v, struct clip_image_grid_shape grid_shape, float * image_embd_out, int * n_img_pos_out) {
    struct {
        struct ggml_context * ctx;
    } model;

    const int32_t image_size = clip_image_size(ctx_clip);
    const int32_t patch_size = clip_patch_size(ctx_clip);

    int32_t num_patches_per_side = image_size / patch_size; // 336 / 14 = 24 - used for embedding-patching boxes (24*24 = 576 patches)

    int num_patches_width  = grid_shape.first;  // grid 1-4
    int num_patches_height = grid_shape.second; // grid 1-4

    const size_t num_images = num_patches_width * num_patches_height + 1;

    // TODO: size calculation is not calculated - it's only tens of MB
    size_t ctx_size = 0;

    {
        ctx_size += clip_embd_nbytes(ctx_clip) * num_images * 8; // image_features
        ctx_size += 1024*1024 * ggml_type_size(GGML_TYPE_F32);
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };

    // Python reference code for full unpad:
    /*
        base_image_feature = image_feature[0]
        image_feature = image_feature[1:]
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_sizes[image_idx])
        image_feature = torch.cat((
            image_feature,
            self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1)
        ), dim=-1)
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    */
    // We now have two options: unpad or no unpad. Unpad removes tokens for faster llm eval.
    // In terms of result quality it appears to make no difference, so we'll start with the easier approach given 5D tensors are not supported in ggml yet.
    // Without unpad we have to split the sub-image embeddings into patches of 24 features each and permute them.
    // Once all images are processed to prepended the base_image_features without any changes.

    // Pytorch reference simplified, modified for ggml compatibility - confirmed identical output in python (for a 2x2 grid image (676x676 scaling))
    /*
        image_feature = image_feature.view(2, 2, 24, 24, 4096)
        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
        image_feature = image_feature.view(2, 24, 2, 24, 4096)
        image_feature = image_feature.flatten(0, 3)

        // Reshape to 4D tensor by merging the last two dimensions
        image_feature = image_feature.view(2, 2, 24, 24*4096)
        image_feature = image_feature.permute(0, 2, 1, 3).contiguous()
        image_feature = image_feature.view(-1, 4096)
    */

    model.ctx = ggml_init(params);

    struct ggml_tensor * image_features = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, clip_n_mmproj_embd(ctx_clip), clip_n_patches(ctx_clip), num_images - 1); // example: 4096 x 576 x 4
    // ggml_tensor_printf(image_features,"image_features",__LINE__,false,false);
    // fill it with the image embeddings, ignoring the base
    for (size_t i = 1; i < num_images; i++) {
        size_t offset = (i-1) * clip_embd_nbytes(ctx_clip);
        memcpy((uint8_t *)(image_features->data) + offset, image_embd_v[i], clip_embd_nbytes(ctx_clip));
    }

    struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);
    size_t size_ele = ggml_type_size(GGML_TYPE_F32);

    struct ggml_tensor *image_features_patchview = ggml_view_4d(model.ctx, image_features,
                                                                num_patches_per_side * clip_n_mmproj_embd(ctx_clip),
                                                                num_patches_per_side,
                                                                num_patches_width,
                                                                num_patches_height,
                                                                size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip),
                                                                size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip) * num_patches_per_side,
                                                                size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip) * num_patches_per_side * num_patches_width, 0);
    // ggml_tensor_printf(image_features_patchview,"image_features_patchview",__LINE__,false,false);
    struct ggml_tensor *permuted_cont = ggml_cont(model.ctx, ggml_permute(model.ctx, image_features_patchview, 0, 2, 1, 3));
    /**
     At the end of each row we have to add the row_end embeddings, which are the same as the newline embeddings
         image_feature = torch.cat((
        image_feature,
        self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
    ), dim=-1)
     *
     */

    // ggml_tensor_printf(permuted_cont,"permuted_cont",__LINE__,false,false);
    struct ggml_tensor *flatten = ggml_view_2d(model.ctx, permuted_cont, clip_n_mmproj_embd(ctx_clip), num_patches_height * num_patches_width * num_patches_per_side * num_patches_per_side,  size_ele * clip_n_mmproj_embd(ctx_clip), 0);
    // ggml_tensor_printf(flatten,"flatten",__LINE__,false,false);
    ggml_build_forward_expand(gf, flatten);
    ggml_graph_compute_with_ctx(model.ctx, gf, 1);
    struct ggml_tensor* result = gf->nodes[gf->n_nodes - 1];

    memcpy(image_embd_out, image_embd_v[0], clip_embd_nbytes(ctx_clip)); // main image as global context
    // append without newline tokens (default behavior in llava_arch when not using unpad ):
    memcpy(image_embd_out + clip_n_patches(ctx_clip) * clip_n_mmproj_embd(ctx_clip), (float*)result->data, clip_embd_nbytes(ctx_clip) * (num_images-1)); // grid patches
    *n_img_pos_out = static_cast<int>(result->ne[1]+clip_n_patches(ctx_clip));

    // Debug: Test single segments
    // Current findings: sending base image, sending a segment embedding all works similar to python
    // However, permuted embeddings do not work yet (stride issue?)
    // memcpy(image_embd_out, image_embd_v[0], clip_embd_nbytes(ctx_clip)); // main image as context
    // memcpy(image_embd_out, (float*)prepared_cont->data, clip_embd_nbytes(ctx_clip)); // main image as context
    // *n_img_pos_out=576;

    ggml_free(model.ctx);
    return true;
}


static bool encode_image_with_clip(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float * image_embd, int * n_img_pos) {
    // std::vector<clip_image_f32*> img_res_v; // format VectN x H x W x RGB (N x 336 x 336 x 3), so interleaved RGB - different to the python implementation which is N x 3 x 336 x 336
    clip_image_f32_batch img_res_v;
    img_res_v.size = 0;
    img_res_v.data = nullptr;
    if (!clip_image_preprocess(ctx_clip, img, &img_res_v)) {
        LOG_TEE("%s: unable to preprocess image\n", __func__);
        delete[] img_res_v.data;
        return false;
    }

    const int64_t t_img_enc_start_us = ggml_time_us();

    const char * mm_patch_merge_type = clip_patch_merge_type(ctx_clip);

    if (strcmp(mm_patch_merge_type, "spatial_unpad") != 0) {
        // flat / default llava-1.5 type embedding
        *n_img_pos = clip_n_patches(ctx_clip);
        bool encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[0], image_embd); // image_embd shape is 576 x 4096
        delete[] img_res_v.data;
        if (!encoded) {
            LOG_TEE("Unable to encode image\n");

            return false;
        }
    } else {
        // spatial_unpad llava-1.6 type embedding
        // TODO: CLIP needs batching support - in HF the llm projection is separate after encoding, which might be a solution to quickly get batching working
        std::vector<float *> image_embd_v;
        image_embd_v.resize(img_res_v.size);
        for (size_t i = 0; i < img_res_v.size; i++) {
            image_embd_v[i] = (float *)malloc(clip_embd_nbytes(ctx_clip)); // 576 patches * 4096 embeddings * 4 bytes = 9437184
            const bool encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[i], image_embd_v[i]); // image data is in 3x336x336 format and will be converted to 336x336x3 inside
            if (!encoded) {
                LOG_TEE("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int) i+1, (int) img_res_v.size);
                return false;
            }
        }
        const int64_t t_img_enc_batch_us = ggml_time_us();
        LOG_TEE("%s: %d segments encoded in %8.2f ms\n", __func__, (int)img_res_v.size, (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0);

        const int32_t * image_grid = clip_image_grid(ctx_clip);

        std::vector<std::pair<int, int>> grid_pinpoints;
        for (int i = 0; i < 32 && image_grid[i] != 0; i += 2) {
            grid_pinpoints.push_back({image_grid[i], image_grid[i+1]});
        }

        // free all img_res_v - not needed anymore
        delete[] img_res_v.data;
        img_res_v.size = 0;
        img_res_v.data = nullptr;

        const int32_t image_size = clip_image_size(ctx_clip);

        struct clip_image_grid_shape grid_shape = get_anyres_image_grid_shape({img->nx,img->ny}, grid_pinpoints, image_size);

        int n_img_pos_out;
        clip_llava_handle_patches(ctx_clip, image_embd_v, grid_shape, image_embd, &n_img_pos_out);
        *n_img_pos = n_img_pos_out;

        for (size_t i = 0; i < image_embd_v.size(); i++) {
            free(image_embd_v[i]);
        }
        image_embd_v.clear();

        // debug image/segment/normalization content:
        // clip_image_u8 * tmp = clip_image_u8_init();
        // clip_image_convert_f32_to_u8(*image_feature, *tmp);
        // clip_image_save_to_bmp(*tmp, "image_feature.bmp");
    }

    LOG_TEE("%s: image embedding created: %d tokens\n", __func__, *n_img_pos);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;

    LOG_TEE("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / *n_img_pos);

    return true;
}

bool llava_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip) {
        // make sure that the correct mmproj was used, i.e., compare apples to apples
    int n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));
    auto n_image_embd = clip_n_mmproj_embd(ctx_clip);
    if (n_image_embd != n_llama_embd) {
        LOG_TEE("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_image_embd, n_llama_embd);
        return false;
    }
    return true;
}

bool llava_image_embed_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out) {
    float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip)*6); // TODO: base on gridsize/llava model
    if (!image_embd) {
        LOG_TEE("Unable to allocate memory for image embeddings\n");
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_clip(ctx_clip, n_threads, img, image_embd, &n_img_pos)) {
        LOG_TEE("%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}

bool llava_eval_image_embed(llama_context * ctx_llama, const struct llava_image_embed * image_embed, int n_batch, int * n_past) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < image_embed->n_image_pos; i += n_batch) {
        int n_eval = image_embed->n_image_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {int32_t(n_eval), nullptr, (image_embed->embed+i*n_embd), nullptr, nullptr, nullptr, nullptr, *n_past, 1, 0, };
        if (llama_decode(ctx_llama, batch)) {
            LOG_TEE("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

struct llava_image_embed * llava_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length) {
    clip_image_u8 * img = clip_image_u8_init();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        clip_image_u8_free(img);
        LOG_TEE("%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }

    float* image_embed = NULL;
    int n_image_pos = 0;
    bool image_embed_result = llava_image_embed_make_with_clip_img(ctx_clip, n_threads, img, &image_embed, &n_image_pos);
    if (!image_embed_result) {
        clip_image_u8_free(img);
        LOG_TEE("%s: coulnd't embed the image\n", __func__);
        return NULL;
    }

    clip_image_u8_free(img);
    auto result = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

static bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long *sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        LOG_TEE("%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize); // Allocate memory to hold the file data
    if (buffer == NULL) {
        LOG_TEE("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file); // Read the file into the buffer
    if (ferror(file)) {
        die_fmt("read error: %s", strerror(errno));
    }
    if (ret != (size_t) fileSize) {
        die("unexpectedly reached end of file");
    }
    fclose(file); // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}

struct llava_image_embed * llava_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path) {
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path, &image_bytes, &image_bytes_length);
    if (!loaded) {
        LOG_TEE("%s: failed to load %s\n", __func__, image_path);
        return NULL;
    }

    llava_image_embed *embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, image_bytes, image_bytes_length);
    free(image_bytes);

    return embed;
}

void llava_image_embed_free(struct llava_image_embed * embed) {
    free(embed->embed);
    free(embed);
}

static bool encode_image_with_clip_uhd(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float * image_embd, int * n_img_pos, struct load_image_size * load_image_size) {
    // std::vector<clip_image_f32*> img_res_v; 
    // format VectN x H x W x RGB (N x 448 x 448 x 3)
    clip_image_f32 * img_res_v = clip_image_f32_init();
    uhd_normalize_image_u8_to_f32(ctx_clip, img, img_res_v);

    const int64_t t_img_enc_start_us = ggml_time_us();

    const char * mm_patch_merge_type = clip_patch_merge_type(ctx_clip);
    LOG_TEE("\n%s: mm_patch_merge_type is  %s.\n", __func__, mm_patch_merge_type);
    
    *n_img_pos = clip_n_patches(ctx_clip);
    bool encoded = clip_image_encode(ctx_clip, n_threads, img_res_v, image_embd, load_image_size); // image_embd shape is 96 x 4096
    if (!encoded) {
        LOG_TEE("Unable to encode image\n");
        return false;
    }
    LOG_TEE("%s: image embedding created: %d tokens\n", __func__, *n_img_pos);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;
    LOG_TEE("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / *n_img_pos);

    return true;
}

static int ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

static std::pair<int, int> uhd_find_best_resize(std::pair<int, int> original_size, int scale_resolution, int patch_size, bool allow_upscale = false) {
    int width = original_size.first;
    int height = original_size.second;
    if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
        float r = static_cast<float>(width) / height;
        height = static_cast<int>(scale_resolution / std::sqrt(r));
        width = static_cast<int>(height * r);
    }
    int best_width = ensure_divide(width, patch_size);
    int best_height = ensure_divide(height, patch_size);
    return std::make_pair(best_width, best_height);
}

static std::pair<int, int> uhd_get_refine_size(std::pair<int, int> original_size, std::pair<int, int> grid, int scale_resolution, int patch_size, bool allow_upscale = false) {
    int width, height;
    std::tie(width, height) = original_size;
    int grid_x, grid_y;
    std::tie(grid_x, grid_y) = grid;

    int refine_width = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);

    int grid_width = refine_width / grid_x;
    int grid_height = refine_height / grid_y;

   // auto best_grid_size = find_best_resize(std::make_tuple(grid_width, grid_height), scale_resolution, patch_size, allow_upscale); (old line)
    auto best_grid_size = uhd_find_best_resize(std::make_pair(grid_width, grid_height), scale_resolution, patch_size, allow_upscale); // (new line) => fixes conversion for make_tuple to make_pair
    int best_grid_width, best_grid_height;
    std::tie(best_grid_width, best_grid_height) = best_grid_size;

  //  std::pair<int, int> refine_size = std::make_tuple(best_grid_width * grid_x, best_grid_height * grid_y); (old line)
    std::pair<int, int> refine_size = std::make_pair(best_grid_width * grid_x, best_grid_height * grid_y); // (new line)
    return refine_size;
}

inline int clip(int x, int lower, int upper) {
    return std::max(lower, std::min(x, upper));
}

static bool bicubic_resize(const clip_image_u8 &img, clip_image_u8 &dst, int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)target_width;
    ty = (float)ny / (float)target_height;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (i = 0; i < target_height; i++) {
        for (j = 0; j < target_width; j++) {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            for (k = 0; k < 3; k++) {
                for (jj = 0; jj <= 3; jj++) {
                    d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                }
            }
        }
    }

    return true;
}

static clip_image_u8 * only_v2_5_reshape_by_patch(clip_image_u8 * image, int patch_size) {
    int width = image->nx;
    int height = image->ny;
    int num_patches = (height / patch_size) * (width / patch_size);
    clip_image_u8 * patch = clip_image_u8_init();
    patch->nx = patch_size * num_patches;
    patch->ny = patch_size;
    patch->buf.resize(3 * patch->nx * patch->ny);

    int patch_index = 0;

    for (int i = 0; i < height; i += patch_size) {
        for (int j = 0; j < width; j += patch_size) {
            for (int pi = 0; pi < patch_size; ++pi) {
                for (int pj = 0; pj < patch_size; ++pj) {
                    int input_index = ((i + pi) * width + (j + pj)) * 3;
                    int output_index = (pi * patch_size * num_patches + patch_index * patch_size + pj) * 3;
                    patch->buf[output_index] = image->buf[input_index];
                    patch->buf[output_index+1] = image->buf[input_index+1];
                    patch->buf[output_index+2] = image->buf[input_index+2];
                }
            }
            patch_index++;
        }
    }
    return patch;
}

// inspired from LLaVA-UHD:
//    -> https://arxiv.org/pdf/2403.11703
//    -> https://github.com/thunlp/LLaVA-UHD
//    -> https://github.com/thunlp/LLaVA-UHD/blob/302301bc2175f7e717fb8548516188e89f649753/llava_uhd/train/llava-uhd/slice_logic.py#L118
static std::vector<std::vector<clip_image_u8 *>> uhd_slice_image(const clip_image_u8 * img, const int max_slice_nums=9, const int scale_resolution=448, const int patch_size=14) {
    const std::pair<int, int> original_size={img->nx,img->ny};
    const int original_width = img->nx;
    const int original_height = img->ny;
    const float log_ratio = log(1.0*original_width/original_height); //
    const float ratio = 1.0 * original_width * original_height/ (scale_resolution * scale_resolution);
    const int multiple = fmin(ceil(ratio), max_slice_nums);

    std::vector<std::vector<clip_image_u8 *>> images;
    LOG_TEE("%s: multiple %d\n", __func__, multiple);
    images.push_back(std::vector<clip_image_u8 *>());

    if(multiple <= 1){
        auto best_size = uhd_find_best_resize(original_size, scale_resolution, patch_size, true);
        clip_image_u8 *source_image = clip_image_u8_init();
        bicubic_resize(*img, *source_image, best_size.first, best_size.second);
        // source_image = image.resize(best_size, Image.Resampling.BICUBIC)
        images[images.size()-1].push_back(source_image);
    }
    else if(multiple > 1){

        std::vector<int> candidate_split_grids_nums;
        for (int i : {multiple - 1, multiple, multiple + 1}) {
            if (i == 1 || i > max_slice_nums) {
                continue;
            }
            candidate_split_grids_nums.push_back(i);
        }

        auto best_size = uhd_find_best_resize(original_size, scale_resolution, patch_size);
        clip_image_u8 *source_image = clip_image_u8_init();
        bicubic_resize(*img, *source_image, best_size.first, best_size.second);
        // source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        LOG_TEE("%s: image_size: %d %d; source_image size: %d %d\n", __func__, img->nx, img->ny, best_size.first, best_size.second);
        images[images.size()-1].push_back(source_image);

        std::vector<std::pair<int, int>> candidate_grids;

        for (int split_grids_nums : candidate_split_grids_nums) {
            int m = 1;
            while (m <= split_grids_nums) {
                if (split_grids_nums % m == 0) {
                    candidate_grids.emplace_back(m, split_grids_nums / m);
                }
                ++m;
            }
        }

        std::pair<int, int> best_grid{1, 1};
        float min_error = std::numeric_limits<float>::infinity();

        for (const auto& grid : candidate_grids) {
            float error = std::abs(log_ratio - std::log(1.0 * grid.first / grid.second));
            if (error < min_error) {
                best_grid = grid;
                min_error = error;
            }
        }
        LOG_TEE("%s: image_size: %d %d; best_grid: %d %d\n", __func__, img->nx, img->ny, best_grid.first, best_grid.second);
        
        auto refine_size = uhd_get_refine_size(original_size, best_grid, scale_resolution, patch_size, true);
        clip_image_u8 *refine_image = clip_image_u8_init();
        bicubic_resize(*img, *refine_image, refine_size.first, refine_size.second);

        LOG_TEE("%s: refine_image_size: %d %d; refine_size: %d %d\n", __func__, refine_image->nx, refine_image->ny, refine_size.first, refine_size.second);

        // split_to_patches
        int width = refine_image->nx;
        int height = refine_image->ny;
        int grid_x = int(width / best_grid.first);
        int grid_y = int(height / best_grid.second);
        for (int patches_i = 0, ic = 0; patches_i < height && ic < best_grid.second; patches_i += grid_y, ic += 1){
            images.push_back(std::vector<clip_image_u8 *>());
            for(int patches_j = 0, jc = 0; patches_j < width && jc < best_grid.first; patches_j += grid_x, jc += 1){
                clip_image_u8 * patch = clip_image_u8_init();
                patch->nx = grid_x;
                patch->ny = grid_y;
                patch->buf.resize(3 * patch->nx * patch->ny);
                for (int y = patches_i; y < patches_i + grid_y; ++y) {
                    for (int x = patches_j; x < patches_j + grid_x; ++x) {
                        const int i = 3 * (y * refine_image->nx + x);
                        const int j = 3 * ((y-patches_i) * patch->nx + (x-patches_j));
                        patch->buf[j]   = refine_image->buf[i];
                        patch->buf[j+1] = refine_image->buf[i+1];
                        patch->buf[j+2] = refine_image->buf[i+2];
                    }
                }
                images[images.size()-1].push_back(patch);
            }
        }
    }
    return images;
}

struct uhd_image_embed * llava_image_embed_make_with_bytes_uhd(struct clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img) {
    std::vector<std::vector<clip_image_u8 *>> imgs = uhd_slice_image(img);
    for (size_t i = 0; i < imgs.size(); ++i){
        for (size_t j = 0; j < imgs[i].size(); ++j) {
            LOG_TEE("%s: %d %d\n", __func__,imgs[i][j]->nx,imgs[i][j]->ny);
        }
    }
    struct uhd_image_embed * results = new uhd_image_embed();

    for (size_t i = 0; i < imgs.size(); ++i){
        results->image_embeds.push_back(std::vector<llava_image_embed *>());
        for (size_t j = 0; j < imgs[i].size(); ++j) {
            float* image_embed = NULL;
            int n_image_pos = 0;
            int patch_size=14;
            struct load_image_size * load_image_size = load_image_size_init();
            load_image_size->image_size_width = imgs[i][j]->nx;
            load_image_size->image_size_height = imgs[i][j]->ny; 
            LOG_TEE("%s : %d %d\n", __func__, load_image_size->image_size_width, load_image_size->image_size_height);
            bool image_embed_result = llava_image_embed_make_with_clip_img_uhd(ctx_clip, n_threads, only_v2_5_reshape_by_patch(imgs[i][j], patch_size), &image_embed, &n_image_pos, load_image_size);
            if (!image_embed_result) {
                LOG_TEE("%s: coulnd't embed the image\n", __func__);
                return NULL;
            }

            auto result = (llava_image_embed*)malloc(sizeof(llava_image_embed));
            result->embed = image_embed;
            result->n_image_pos = n_image_pos;
            results->image_embeds[i].push_back(result);
        }
    }
    return results;
}

bool llava_image_embed_make_with_clip_img_uhd(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out, struct load_image_size * load_image_size) {
    float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip)*6); // TODO: base on gridsize/llava model
    if (!image_embd) {
        LOG_TEE("Unable to allocate memory for image embeddings\n");
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_clip_uhd(ctx_clip, n_threads, img, image_embd, &n_img_pos, load_image_size)) {
        LOG_TEE("%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}

bool llava_image_embed_make_with_clip_img_ollama(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out) {
    auto embeds = llava_image_embed_make_with_bytes_uhd(ctx_clip, n_threads, img);
    auto image_embed_slices = embeds->image_embeds;
    if (!image_embed_slices[0][0]){
        LOG_TEE("%s: failed to embeding image\n", __func__);
        return false;
    }
    std::string fname = "./examples/minicpm-v2.5/slice_token_for_ollama.raw";
    unsigned char* slice_token;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(fname.c_str(), &slice_token, &image_bytes_length);
    if (!loaded) {
        LOG_TEE("%s: failed to load %s\n", __func__, fname.c_str());
        return false;
    }

    float * all_image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip)*61);
    int all_n_img_pos=0;
    int token_len = clip_n_mmproj_embd(ctx_clip)*sizeof(float);

    std::memcpy(all_image_embd+token_len*all_n_img_pos++, slice_token, token_len);
    std::memcpy(all_image_embd+token_len*all_n_img_pos, image_embed_slices[0][0]->embed, 96*token_len);
    all_n_img_pos+=clip_n_patches(ctx_clip);
    std::memcpy(all_image_embd+token_len*all_n_img_pos++, slice_token+token_len, token_len);
    if (image_embed_slices.size() > 1) {
        std::memcpy(all_image_embd+token_len*all_n_img_pos++, slice_token+token_len*2, token_len);
        for (size_t i = 1; i < image_embed_slices.size(); ++i) {
            for (size_t j = 0; j < image_embed_slices[i].size(); ++j) {
                std::memcpy(all_image_embd+token_len*all_n_img_pos++, slice_token, token_len);
                std::memcpy(all_image_embd+token_len*all_n_img_pos, image_embed_slices[i][j]->embed, 96*token_len);
                all_n_img_pos+=clip_n_patches(ctx_clip);
                std::memcpy(all_image_embd+token_len*all_n_img_pos++, slice_token+token_len, token_len);
                if (j == image_embed_slices[i].size() - 1) {
                    std::memcpy(all_image_embd+token_len*all_n_img_pos++, slice_token+token_len*4, token_len);
                }
            }
        }
        std::memcpy(all_image_embd+token_len*all_n_img_pos++, slice_token+token_len*3, token_len);
    }
    *image_embd_out = all_image_embd;
    *n_img_pos_out = all_n_img_pos;
    return true;
}

struct uhd_image_embed * llava_image_embed_make_with_filename_uhd(struct clip_ctx * ctx_clip, int n_threads, const char * image_path) {
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path, &image_bytes, &image_bytes_length);
    if (!loaded) {
        LOG_TEE("%s: failed to load %s\n", __func__, image_path);
        return NULL;
    }
    clip_image_u8 * img = clip_image_u8_init();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        clip_image_u8_free(img);
        LOG_TEE("%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }

    struct uhd_image_embed * embeds = llava_image_embed_make_with_bytes_uhd(ctx_clip, n_threads, img);

    clip_image_u8_free(img);
    free(image_bytes);
    return embeds;
}

void llava_image_embed_free_uhd(struct uhd_image_embed * embed) {
    for (size_t i = 0; i < embed->image_embeds.size(); ++i){
        for (size_t j = 0; j < embed->image_embeds[i].size(); ++j){
            free(embed->image_embeds[i][j]->embed);
            free(embed->image_embeds[i][j]);
        }
        embed->image_embeds[i] = std::vector<struct llava_image_embed *>();
    }
    embed->image_embeds = std::vector<std::vector<struct llava_image_embed *>>();
}