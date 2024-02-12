#include "clip.h"
#include "common.h"
#include "llama.h"
#include "llava.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>

#include "base64.hpp"

// Take the image segments in a grid configuration and return the embeddings and the number of embeddings into preallocated memory (image_embd_out)
static bool handle_patches(clip_ctx * ctx_clip, std::vector<float *> & image_embd_v, struct clip_image_grid_shape grid_shape, float * image_embd_out, int * n_img_pos_out) {
    struct temp_model {
        struct ggml_tensor *newline;
        struct ggml_context * ctx;
    } model;

    auto & vparams = clip_get_vision_hparams(ctx_clip);
    auto num_patches_per_side = vparams.image_size / vparams.patch_size; // 336 / 14 = 24 - used for embedding-patching boxes (24*24 = 576 patches)
    int num_patches_width = grid_shape.first; // grid 1-4
    int num_patches_height = grid_shape.second; // grid 1-4

    // TODO: size calculation is not calculated - it's only tens of MB
    size_t ctx_size = 0;
    {
        ctx_size += clip_embd_nbytes(ctx_clip) * image_embd_v.size() * 8; // image_features
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
    ggml_context *ctx_noalloc = ggml_init({2048, NULL, true});

    ggml_tensor *newline_tmp = clip_get_newline_tensor(ctx_clip);
    model.newline = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, newline_tmp->ne[0]);
    if (newline_tmp->backend != GGML_BACKEND_CPU) {
        if (newline_tmp->buffer == NULL) {
            printf("newline_tmp tensor buffer is NULL\n");
        }
        ggml_backend_tensor_get(newline_tmp, model.newline->data, 0, ggml_nbytes(newline_tmp));
    } else
    {
        model.newline->data = newline_tmp->data;
        if (model.newline->data == NULL) {
            printf("newline_tmp tensor data is NULL\n");
        }
    }

    struct ggml_tensor * image_features = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, clip_n_mmproj_embd(ctx_clip), clip_n_patches(ctx_clip), image_embd_v.size() - 1); // example: 4096 x 576 x 4
    // ggml_tensor_printf(image_features,"image_features",__LINE__,false,false);
    // fill it with the image embeddings, ignoring the base
    for (int i = 1; i < image_embd_v.size(); i++)
    {
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
    // ggml_tensor_printf(permuted_cont,"permuted_cont",__LINE__,false,false);
    struct ggml_tensor *flatten = ggml_view_2d(model.ctx, permuted_cont, clip_n_mmproj_embd(ctx_clip), num_patches_height * num_patches_width * num_patches_per_side * num_patches_per_side,  size_ele * clip_n_mmproj_embd(ctx_clip), 0);
    // ggml_tensor_printf(flatten,"flatten",__LINE__,false,false);
    ggml_build_forward_expand(gf, flatten);
    ggml_graph_compute_with_ctx(model.ctx, gf, 1);
    struct ggml_tensor* result = gf->nodes[gf->n_nodes - 1];

    memcpy(image_embd_out, image_embd_v[0], clip_embd_nbytes(ctx_clip)); // main image as global context
    // append without newline tokens (default behavior in llava_arch when not using unpad ):
    memcpy(image_embd_out + clip_n_patches(ctx_clip) * clip_n_mmproj_embd(ctx_clip), (float*)result->data, clip_embd_nbytes(ctx_clip) * (image_embd_v.size()-1)); // grid patches
    *n_img_pos_out = result->ne[1]+clip_n_patches(ctx_clip);

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
    std::vector<clip_image_f32*> img_res_v; // format VectN x H x W x RGB (N x 336 x 336 x 3), so interleaved RGB - different to the python implementation which is N x 3 x 336 x 336
    if (!clip_image_preprocess(ctx_clip, img, img_res_v, /*pad2square =*/ true)) {
        fprintf(stderr, "%s: unable to preprocess image\n", __func__);
        for (auto img_res : img_res_v) {
            clip_image_f32_free(img_res);
        }
        return false;
    }

    const int64_t t_img_enc_start_us = ggml_time_us();
    auto & vparams = clip_get_vision_hparams(ctx_clip);
    // DEBUG print the "shape" and the first 10 rows and 10 cols of img_res_v in exp format
    // for (int i = 0; i < img_res_v.size(); i++)
    // {
    //     printf("img_res_v[%d] shape: %d x %d\n", i, img_res_v[i]->nx, img_res_v[i]->ny);
    //     for (int j = 0; j < 10; j++)
    //     {
    //         for (int k = 0; k < 10; k++)
    //         {
    //             printf("%e ", img_res_v[i]->buf[j*img_res_v[i]->ny + k]);
    //         }
    //         printf("\n");
    //     }
    // }

    if (strcmp(vparams.mm_patch_merge_type, "spatial_unpad") != 0)
    {
        // flat / default llava-1.5 type embedding
        *n_img_pos = clip_n_patches(ctx_clip);
        bool encoded = clip_image_encode(ctx_clip, n_threads, img_res_v[0], image_embd); // image_embd shape is 576 x 4096
        clip_image_f32_free(img_res_v[0]);
        if (!encoded) {
            fprintf(stderr, "Unable to encode image\n");

            return false;
        }
    } else
    {
        // spatial_unpad llava-1.6 type embedding
        // TODO: CLIP needs batching support - in HF the llm projection is separate after encoding, which might be a solution to quickly get batching working
        std::vector<float *> image_embd_v;
        image_embd_v.resize(img_res_v.size());
        for (int i = 0; i < img_res_v.size(); i++)
        {
            image_embd_v[i] = (float *)malloc(clip_embd_nbytes(ctx_clip)); // 576 patches * 4096 embeddings * 4 bytes = 9437184
            bool encoded = clip_image_encode(ctx_clip, n_threads, img_res_v[i], image_embd_v[i]); // image data is in 3x336x336 format and will be converted to 336x336x3 inside
            clip_image_f32_free(img_res_v[i]);
            if (!encoded) {
                fprintf(stderr, "Unable to encode image - spatial_unpad - subimage %d of %d\n", i+1, img_res_v.size());
                return false;
            }
        }
        const int64_t t_img_enc_batch_us = ggml_time_us();
        printf("%s: %d segments encoded in %8.2f ms\n", __func__, img_res_v.size(), (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0);


        std::vector<std::pair<int, int>> grid_pinpoints;
        for (int i = 0; i < 32 && vparams.image_grid_pinpoints[i] != 0; i+=2) {
            grid_pinpoints.push_back({vparams.image_grid_pinpoints[i], vparams.image_grid_pinpoints[i+1]});
        }
        img_res_v.clear();
        struct clip_image_grid_shape grid_shape = get_anyres_image_grid_shape({img->nx,img->ny}, grid_pinpoints, vparams.image_size);

        int n_img_pos_out;
        handle_patches(ctx_clip, image_embd_v, grid_shape, image_embd, &n_img_pos_out);
        *n_img_pos = n_img_pos_out;

        for (int i = 0; i < image_embd_v.size(); i++)
        {
            free(image_embd_v[i]);
        }
        image_embd_v.clear();

        // debug image/segment/normalization content:
        // clip_image_u8 * tmp = clip_image_u8_init();
        // clip_image_convert_f32_to_u8(*image_feature, *tmp);
        // clip_image_save_to_bmp(*tmp, "image_feature.bmp");

    }
    printf("%s: image embedding created: %d tokens\n", __func__, *n_img_pos);


    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;

    printf("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / *n_img_pos);

    return true;
}



bool llava_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip) {
        // make sure that the correct mmproj was used, i.e., compare apples to apples
    int n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));
    auto n_image_embd = clip_n_mmproj_embd(ctx_clip);
    if (n_image_embd != n_llama_embd) {
        printf("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_image_embd, n_llama_embd);
        return false;
    }
    return true;
}

static bool llava_image_embed_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out) {
    float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip)*6); // TODO: base on gridsize/llava model
    if (!image_embd) {
        fprintf(stderr, "Unable to allocate memory for image embeddings\n");
        free(image_embd);
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_clip(ctx_clip, n_threads, img, image_embd, &n_img_pos)) {
        fprintf(stderr, "%s: cannot encode image, aborting\n", __func__);
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
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

LLAVA_API struct llava_image_embed * llava_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length) {
    clip_image_u8 * img = clip_image_u8_init();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }

    float* image_embed = NULL;
    int n_image_pos = 0;
    bool image_embed_result = llava_image_embed_make_with_clip_img(ctx_clip, n_threads, img, &image_embed, &n_image_pos);
    if (!image_embed_result) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: coulnd't embed the image\n", __func__);
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
        fprintf(stderr, "%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize); // Allocate memory to hold the file data
    if (buffer == NULL) {
        fprintf(stderr, "%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
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

LLAVA_API struct llava_image_embed * llava_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path) {
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path, &image_bytes, &image_bytes_length);
    if (!loaded) {
        fprintf(stderr, "%s: failed to load %s\n", __func__, image_path);
        return NULL;
    }

    llava_image_embed *embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, image_bytes, image_bytes_length);
    free(image_bytes);

    return embed;
}

LLAVA_API void llava_image_embed_free(struct llava_image_embed * embed) {
    free(embed->embed);
    free(embed);
}
