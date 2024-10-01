#include "llama.h"
#include "llama-vision.h"
#include "llama-impl.h"

#include <string.h> // memcpy
#include <limits>
#include <cmath>

#ifndef NDEBUG
// for debugging
#include <fstream>
#include <cstdint>
#include <iostream>

// export clip_image_u8 to bmp file for debugging
// https://codereview.stackexchange.com/questions/195121/writing-a-bitmap-image-from-c
struct clip_image_size;
static int bmp_export(const struct clip_image_u8 &img, const std::string &location);
#endif

struct clip_image_size {
    int width;
    int height;
};

// RGB uint8 image
// Memory layout: RGBRGBRGB...
struct clip_image_u8 {
    int nx;
    int ny;
    std::vector<uint8_t> buf;
    clip_image_u8() {}
    clip_image_u8(const llama_img img) {
        nx = img.nx;
        ny = img.ny;
        buf.resize(nx*ny*3);
        memcpy(buf.data(), img.data, buf.size());
    }
};

struct clip_image_u8_batch {
    struct clip_image_u8 * data;
    size_t size;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;
    std::vector<float> buf;
};

using clip_image_f32_batch = std::vector<clip_image_f32>;
using clip_image_f8_batch  = std::vector<clip_image_u8>;

int clip_n_patches(const clip_context & ctx) {
    auto & hparams = ctx.model->hparams;
    int n_patches = (hparams.image_size / hparams.patch_size) * (hparams.image_size / hparams.patch_size);
    return n_patches;
}

int clip_n_mmproj_embd(const clip_context & ctx) {
    if (ctx.model->hparams.proj_type == CLIP_PROJECTOR_TYPE_MLP) {
        return ctx.model->mm_b_b->ne[0];
    } else {
        GGML_ASSERT(false && "invalid proj type");
    }
}

int clip_n_embd(const clip_context & ctx) {
    return clip_n_patches(ctx) * clip_n_mmproj_embd(ctx);
}

/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static clip_image_size select_best_resolution(const clip_image_size & original_size, const std::vector<clip_image_size>& possible_resolutions) {
    int original_width  = original_size.width;
    int original_height = original_size.height;

    clip_image_size best_fit;
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width   = resolution.width;
        int height  = resolution.height;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width  = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // LOG_DBG("resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

static bool bicubic_resize(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
    auto clip = [](int x, int lower, int upper) -> int {
        return std::max(lower, std::min(x, upper));
    };

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

static std::vector<clip_image_u8> divide_to_patches_u8(const clip_image_u8 & image, int patch_size) {
    std::vector<clip_image_u8> patches;
    int width = image.nx;
    int height = image.ny;
    for (int i = 0; i < height; i += patch_size) {
        for (int j = 0; j < width; j += patch_size) {
            clip_image_u8 patch;
            patch.nx = std::min(patch_size, width - j);
            patch.ny = std::min(patch_size, height - i);
            patch.buf.resize(3 * patch.nx * patch.ny);
            for (int y = 0; y < patch.ny; ++y) {
                for (int x = 0; x < patch.nx; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        patch.buf[3 * (y * patch.nx + x) + c] = image.buf[3 * ((i + y) * width + (j + x)) + c];
                    }
                }
            }
            patches.push_back(patch);
        }
    }
    return patches;
}

// llava-1.6 type of resize_and_pad (black)
static void resize_and_pad_image(const clip_image_u8 & image, clip_image_u8 & image_output, const clip_image_size & target_resolution) {
    int target_width  = target_resolution.width;
    int target_height = target_resolution.height;

    float scale_w = static_cast<float>(target_width) / image.nx;
    float scale_h = static_cast<float>(target_height) / image.ny;

    int new_width, new_height;

    if (scale_w < scale_h) {
        new_width = target_width;
        new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
    } else {
        new_height = target_height;
        new_width = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
    }

    clip_image_u8 resized_image;
    // bilinear_resize(image, resized_image, new_width, new_height);
    bicubic_resize(image, resized_image, new_width, new_height);

    clip_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.resize(3 * target_width * target_height, 0); // Initialize with black

    // Calculate padding offsets
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;

    // Copy the resized image into the center of the padded buffer
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] = resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    image_output = std::move(padded_image);
}

static void normalize_image_u8_to_f32(const clip_image_u8 src, clip_image_f32 dst, const std::array<float, 3> & mean, const std::array<float, 3> & std) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(src.buf.size());

    for (size_t i = 0; i < src.buf.size(); ++i) {
        int c = i % 3; // rgb
        dst.buf[i] = (static_cast<float>(src.buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

// returns the normalized float tensor for llava-1.5, for spatial_unpad with anyres processing for llava-1.6 it returns the normalized image patch tensors as a vector
// res_imgs memory is being allocated here, previous allocations will be freed if found
static bool clip_image_preprocess(const clip_context & ctx, const clip_image_u8 & img, clip_image_f32_batch & output_imgs) {
    bool pad_to_square = true;
    auto & params = ctx.model->hparams;
    // The model config actually contains all we need to decide on how to preprocess, here we automatically switch to the new llava-1.6 preprocessing
    if (params.mm_patch_merge_type == MM_PATCH_MERGE_SPATIAL_UNPAD) {
        pad_to_square = false;
    }

    // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
    // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

    clip_image_u8 temp;
    if (pad_to_square && img.nx != img.ny) {
        int longer_side = std::max(img.nx, img.ny);
        temp.nx = longer_side;
        temp.ny = longer_side;
        temp.buf.resize(3 * longer_side * longer_side);
        const uint8_t bc[3] = {122, 116, 104}; // background color in RGB from LLaVA (this is the mean rgb color * 255)

        // fill with background color
        for (size_t i = 0; i < temp.buf.size(); i++) {
            temp.buf[i] = bc[i % 3];
        }

        // copy from the input image
        for (int y = 0; y < img.ny; y++) {
            for (int x = 0; x < img.nx; x++) {
                const int i = 3 * (y * img.nx + x);
                const int j = 3 * (y * temp.nx + x);
                temp.buf[j]   = img.buf[i];
                temp.buf[j+1] = img.buf[i+1];
                temp.buf[j+2] = img.buf[i+2];
            }
        }
    } else {
        if (params.image_grid_pinpoints[0] != 0) {
            // "spatial_unpad" with "anyres" processing for llava-1.6
            std::vector<clip_image_size> possible_resolutions;
            for (int i = 0; i < 32 && params.image_grid_pinpoints[i] != 0; i += 2) {
                clip_image_size s;
                s.width  = params.image_grid_pinpoints[i];
                s.height = params.image_grid_pinpoints[i+1];
                possible_resolutions.push_back(s);
            }
            clip_image_size best_resolution = select_best_resolution({img.nx, img.ny}, possible_resolutions);
            // clip_image_save_to_bmp(*img, "input.bmp");
            resize_and_pad_image(img, temp, best_resolution);  // we do not pad with mean-bg color anymore in llava-1.6
            // clip_image_save_to_bmp(*temp, "resized.bmp");

            std::vector<clip_image_u8> patches = divide_to_patches_u8(temp, params.image_size); // prepare spatial sorted main patches of image_size each (336 in llava-1.6)

            clip_image_u8 image_original_resize;
            // bilinear_resize(*img, *image_original_resize, params.image_size, params.image_size); // in python this is "shortest_edge", but all CLIP are square
            bicubic_resize(img, image_original_resize, params.image_size, params.image_size); // in python this is "shortest_edge", but all CLIP are square
            patches.insert(patches.begin(), image_original_resize);
            // clip_image_f32_batch_init(patches.size());
            output_imgs.resize(patches.size());
            int num = 0;
            for (auto & patch : patches) {
                normalize_image_u8_to_f32(patch, output_imgs[num], params.image_mean, params.image_std);
                num++;
            }
            return true;
        } else {
            temp.nx = img.nx;
            temp.ny = img.ny;
            temp.buf.resize(img.buf.size());
            memcpy(temp.buf.data(), img.buf.data(), temp.buf.size());
        }
    }

    const int nx = temp.nx;
    const int ny = temp.ny;
    // bmp_export(temp, "resized_vanilla.bmp");

    const int nx2 = params.image_size;
    const int ny2 = params.image_size;
    clip_image_f32 res;
    res.nx = nx2;
    res.ny = ny2;
    res.buf.resize(3 * nx2 * ny2);

    const float scale = std::max(nx, ny) / (float)params.image_size;

    const int nx3 = int(nx / scale + 0.5f);
    const int ny3 = int(ny / scale + 0.5f);

    const auto & m3 = params.image_mean; // {0.48145466f, 0.4578275f, 0.40821073f};
    const auto & s3 = params.image_std;  // {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                // linear interpolation
                const float sx = (x + 0.5f) * scale - 0.5f;
                const float sy = (y + 0.5f) * scale - 0.5f;

                const int x0 = std::max(0, (int)std::floor(sx));
                const int y0 = std::max(0, (int)std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = temp.buf[j00];
                const float v01 = temp.buf[j01];
                const float v10 = temp.buf[j10];
                const float v11 = temp.buf[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res.buf[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }

    output_imgs.resize(1);
    output_imgs[0] = std::move(res);

    return true;
}

static ggml_cgraph * clip_image_build_graph(clip_context & ctx, int batch_size, clip_image_size & image_size) {
    auto & model = *ctx.model;
    auto & hparams = ctx.model->hparams;

    const int hidden_size   = hparams.hidden_size;
    const int n_head        = hparams.n_head;
    const int d_head        = hidden_size / n_head;
    const int patch_size    = hparams.patch_size;
    const float eps         = hparams.eps;
    const int num_patches   = ((image_size.width / patch_size) * (image_size.height / patch_size));
    const int num_positions = num_patches + (model.class_embedding ? 1 : 0);

    LLAMA_LOG_DEBUG("%s: num_patches = %d\n", __func__, num_patches);

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx.buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx.buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // input
    struct ggml_tensor * embeddings;
    {
        struct ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size.width, image_size.height, 3, batch_size);
        ggml_set_name(inp_raw, "inp_raw");
        ggml_set_input(inp_raw);

        struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

        inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

        if (model.patch_bias) {
            inp = ggml_add(ctx0, inp, model.patch_bias);
        }
        // auto * ne = inp->ne; printf("%d %d %d %d\n", ne[0], ne[1], ne[2], ne[3]);

        embeddings = inp;
        if (model.class_embedding) {
            embeddings = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, batch_size);
            ggml_set_name(embeddings, "embeddings");
            ggml_set_input(embeddings);
            embeddings = ggml_acc(ctx0, embeddings, model.class_embedding,
                    embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], 0);
            embeddings = ggml_acc(ctx0, embeddings, inp,
                    embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], model.class_embedding->nb[1]);
        }

        struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_positions);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        embeddings = ggml_add(ctx0,
            embeddings,
            ggml_get_rows(ctx0, model.position_embeddings, positions));
    }

    // pre-layernorm
    if (model.pre_norm_w) {
        embeddings = ggml_norm(ctx0, embeddings, eps);
        ggml_set_name(embeddings, "pre_ln");

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.pre_norm_w), model.pre_norm_b);
    }

    // loop over layers
    for (int il = 0; il < (int)hparams.n_layer - 2; il++) {
        struct ggml_tensor * cur = embeddings;

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_add(ctx0,
                ggml_mul(ctx0, cur, model.layers[il].norm_in_w),
                model.layers[il].norm_in_b);
        }

        // self-attention
        {

            struct ggml_tensor * Q = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.layers[il].q_w, cur),
                model.layers[il].q_b);

            Q = ggml_scale_inplace(ctx0, Q, 1.0f / sqrt((float)d_head));
            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_positions, batch_size);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * K = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.layers[il].k_w, cur),
                model.layers[il].k_b);

            K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * V = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.layers[il].v_w, cur),
                model.layers[il].v_b);

            V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, batch_size);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head * batch_size);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_soft_max_inplace(ctx0, KQ);
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_positions, n_head, batch_size);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cont_3d(ctx0, KQV, hidden_size, num_positions, batch_size);
        }

        // attention output
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].output_w, cur), model.layers[il].output_b);

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_add(ctx0,
                ggml_mul(ctx0, cur, model.layers[il].norm_out_w),
                model.layers[il].norm_out_b);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ffn_up_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ffn_up_b);

        if (hparams.use_gelu) {
            cur = ggml_gelu_inplace(ctx0, cur);
        } else {
            cur = ggml_gelu_quick_inplace(ctx0, cur);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ffn_down_b);

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);

        embeddings = cur;
    }

    // post-layernorm
    if (model.post_norm_w) {
        embeddings = ggml_norm(ctx0, embeddings, eps);
        ggml_set_name(embeddings, "post_ln");

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.post_norm_w), model.post_norm_b);
    }

    // llava projector
    {
        embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);

        struct ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_patches);
        ggml_set_name(patches, "patches");
        ggml_set_input(patches);

        // shape [1, 576, 1024]
        // ne is whcn, ne = [1024, 576, 1, 1]
        embeddings = ggml_get_rows(ctx0, embeddings, patches);

        if (hparams.proj_type == CLIP_PROJECTOR_TYPE_MLP) {
            embeddings = ggml_mul_mat(ctx0, model.mm_a_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_a_b);

            embeddings = ggml_gelu(ctx0, embeddings);
            embeddings = ggml_mul_mat(ctx0, model.mm_b_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_b_b);
        } else {
            GGML_ASSERT(false && "unsupported proj type");
        }
    }

    // build the graph
    ggml_build_forward_expand(gf, embeddings);
    ggml_free(ctx0);
    return gf;
}

static int32_t clip_image_batch_encode(clip_context & ctx, const clip_image_f32_batch & imgs, std::vector<float> & output) {
    int batch_size = imgs.size();
    auto & model = *ctx.model;
    auto & hparams = ctx.model->hparams;

    if (hparams.arch == VISION_ARCH_LLAVA) {
        GGML_ASSERT(batch_size == 1); // TODO: support multiple images
    }

    clip_image_size image_size{(int)hparams.image_size, (int)hparams.image_size};
    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size.width / patch_size) * (image_size.height / patch_size));
    const int num_positions = num_patches + (model.class_embedding ? 1 : 0);

    LLAMA_LOG_DEBUG("%s: image_size = %d\n", __func__, hparams.image_size);
    LLAMA_LOG_DEBUG("%s: num_positions = %d\n", __func__, num_positions);

    // build the inference graph
    ggml_cgraph * gf = clip_image_build_graph(ctx, batch_size, image_size);

    // alloc memory for graph
    bool ok = ggml_backend_sched_alloc_graph(ctx.sched, gf);
    if (!ok) {
        LLAMA_LOG_ERROR("failed to alloc memory for graph\n");
        return -1;
    }

    // set raw input
    {
        struct ggml_tensor * inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        float * data = (float *)malloc(ggml_nbytes(inp_raw));

        for (int i = 0; i < batch_size; i++) {
            const int nx = imgs[i].nx;
            const int ny = imgs[i].ny;
            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < 3; k++) {
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            data[(b * 3 * n) + k * n + y * nx + x] = imgs[b].buf[3 * (y * nx + x) + k];
                        }
                    }
                }
            }
        }
        ggml_backend_tensor_set(inp_raw, data, 0, ggml_nbytes(inp_raw));
        free(data);
    }

    if (model.class_embedding) {
        struct ggml_tensor * embeddings = ggml_graph_get_tensor(gf, "embeddings");

        void* zero_mem = malloc(ggml_nbytes(embeddings));
        memset(zero_mem, 0, ggml_nbytes(embeddings));
        ggml_backend_tensor_set(embeddings, zero_mem, 0, ggml_nbytes(embeddings));
        free(zero_mem);
    }

    {
        struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "positions");

        int* positions_data = (int*)malloc(ggml_nbytes(positions));
        for (int i = 0; i < num_positions; i++) {
            positions_data[i] = i;
        }
        ggml_backend_tensor_set(positions, positions_data, 0, ggml_nbytes(positions));
        free(positions_data);
    }

    {
        struct ggml_tensor * patches = ggml_graph_get_tensor(gf, "patches");
        int* patches_data = (int*)malloc(ggml_nbytes(patches));
        for (int i = 0; i < num_patches; i++) {
            patches_data[i] = i + 1;
        }
        ggml_backend_tensor_set(patches, patches_data, 0, ggml_nbytes(patches));
        free(patches_data);
    }

    // compute
    ggml_backend_sched_graph_compute_async(ctx.sched, gf);

    // the last node is the embedding tensor
    struct ggml_tensor * embeddings = ggml_graph_node(gf, -1);
    ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(ctx.sched, embeddings);

    // copy the embeddings to the location passed by the user
    output.resize(clip_n_embd(ctx));
    ggml_backend_tensor_get_async(backend_embd, embeddings, output.data(), 0, ggml_nbytes(embeddings));

    ggml_backend_sched_synchronize(ctx.sched);

    return 0;
}

static int32_t clip_image_encode(clip_context & ctx, const clip_image_f32 & img, std::vector<float> & output) {
    clip_image_f32_batch imgs{img};
    return clip_image_batch_encode(ctx, imgs, output);
}

static int32_t encode_image_with_clip(clip_context & ctx, const llama_img img, std::vector<float> & output_embd) {
    clip_image_u8 img_u8(img);
    clip_image_f32_batch img_res_v;
    auto & hparams = ctx.model->hparams;
    // bmp_export(img_u8, "test_inp.bmp");

    if (!clip_image_preprocess(ctx, img_u8, img_res_v)) {
        LLAMA_LOG_ERROR("%s: unable to preprocess image\n", __func__);
        return -2;
    }

    switch (hparams.mm_patch_merge_type) {
        case MM_PATCH_MERGE_FLAT:
            {
                // flat / default llava-1.5 type embedding
                // n_output = clip_n_patches(ctx);
                int32_t encoded = clip_image_encode(ctx, img_res_v[0], output_embd);
                if (encoded != 0) {
                    LLAMA_LOG_ERROR("Unable to encode image\n");
                    return encoded;
                }
            } break;
        case MM_PATCH_MERGE_SPATIAL_UNPAD:
            {
                // TODO: support llava-1.6
                (void)0;
            } break;
        default:
            GGML_ASSERT(false && "unsupported mm_patch_merge_type");
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////
// public API

int32_t llama_vision_encode_internal(clip_context & ctx, llama_img_batch * batch) {
    if (batch->n_imgs == 0) {
        return 0;
    }

    // TODO: batching is not working atm, should be fixed later
    const int n_embd = clip_n_embd(ctx);
    ctx.output.resize(n_embd * batch->n_imgs);
    ctx.n_output = batch->n_imgs;

    for (int i = 0; i < batch->n_imgs; i++) {
        std::vector<float> output_single;
        int32_t status = encode_image_with_clip(ctx, *batch->imgs[i], output_single);
        if (status != 0) {
            return status;
        }
        // copy output embeddings to result
        for (int k = 0; k < n_embd; k++) {
            ctx.output[n_embd*i + k] = output_single[k];
        }
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////
// for debugging
#ifndef NDEBUG

static int bmp_export(const struct clip_image_u8 &img, const std::string &location) {
    const uint32_t width = img.nx;
    const uint32_t height = img.ny;
    // swap red and blue channel
    std::vector<uint8_t> buffer(width*height*3);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            size_t base = x*3 + y*3*width;
            buffer[base+2] = img.buf[base];
            buffer[base+1] = img.buf[base+1];
            buffer[base]   = img.buf[base+2];
        }
    }
    const bool hasAlphaChannel = false;

    std::ofstream fout(location, std::ios::out | std::ios::binary);

    if (fout.fail()) {
        return 0;
    }

    //Padding
    const uint8_t padding = hasAlphaChannel ? 0 : (4 - (width * 3) % 4) % 4;

    //Bitmap file header.
    const char signature[2] = { 'B', 'M' };
    const uint32_t fileSize = buffer.size() * sizeof(uint8_t) + padding * (height - 1) + 14 + 124;
    const uint32_t offset = 14 + 124;

    //Bitmap information header file
    const uint32_t DIBSize = 124;
    const int32_t bitmapWidth = width;
    const int32_t bitmapHeight = height;
    const uint16_t numPlanes = 1;
    const uint16_t bitsPerPixel = (hasAlphaChannel) ? 32 : 24;
    const uint32_t compressionMethod = (hasAlphaChannel) ? 3 : 0; //BI_RGB = 0, BI_BITFIELDS = 3
    const uint32_t bitmapSize = buffer.size() * sizeof(uint8_t);
    const int32_t horizontalResolution = 2834;
    const int32_t verticalResolution = 2834;
    const uint32_t numColors = 0;
    const uint32_t impColorCount = 0;
    const uint32_t redBitmask = (hasAlphaChannel) ? 0x0000FF00 : 0; //ARGB32 pixel format
    const uint32_t greenBitmask = (hasAlphaChannel) ? 0x00FF0000 : 0;
    const uint32_t blueBitmask = (hasAlphaChannel) ? 0xFF000000 : 0;
    const uint32_t alphaBitmask = (hasAlphaChannel) ? 0x000000FF : 0;

    //Writing the file header and information header to the file
    std::vector<uint8_t> header(offset, 0);
    header[0] = signature[0];
    header[1] = signature[1];

#define BMP_HEADERS(i, variableName)    header[i] = variableName; header[i+1] = variableName >> 8; header[i+2] = variableName >> 16; header[i+3] = variableName >> 24;

    BMP_HEADERS(2, fileSize);
    BMP_HEADERS(6, 0);
    BMP_HEADERS(10, offset);
    BMP_HEADERS(14, DIBSize);
    BMP_HEADERS(18, bitmapWidth);
    BMP_HEADERS(22, bitmapHeight);

    header[26] = (uint8_t)numPlanes;
    header[27] = (uint8_t)(numPlanes >> 8);
    header[28] = (uint8_t)bitsPerPixel;
    header[29] = (uint8_t)(bitsPerPixel >> 8);

    BMP_HEADERS(30, compressionMethod);
    BMP_HEADERS(34, (unsigned char)bitmapSize);
    BMP_HEADERS(38, (unsigned char)horizontalResolution);
    BMP_HEADERS(42, (unsigned char)verticalResolution);
    BMP_HEADERS(46, (unsigned char)numColors);
    BMP_HEADERS(50, (unsigned char)impColorCount);
    BMP_HEADERS(54, (unsigned char)redBitmask);
    BMP_HEADERS(58, (unsigned char)greenBitmask);
    BMP_HEADERS(62, (unsigned char)blueBitmask);
    BMP_HEADERS(66, alphaBitmask);

#undef BMP_HEADERS

    fout.write((char *)header.data(), sizeof(uint8_t) * header.size());

    //Writing the pixel array
    const uint32_t bWidth = bitsPerPixel / 8 * width;

    for (int i = height - 1; i >= 0; i--) {
        std::vector<uint8_t> row(buffer.begin() + i * bWidth, buffer.begin() + i * bWidth + bWidth);
        fout.write((char *)row.data(), row.size() * sizeof(uint8_t));
        fout.seekp(padding * sizeof(uint8_t), std::ios::cur);
    }

    fout.close();
    return 1;
}

#endif

