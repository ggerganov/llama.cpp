#include "clip.h"
#include "llava-utils.h"
#include "common.h"
#include "llama.h"
#include "llava.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "base64.hpp"

static bool encode_image_with_clip(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float * image_embd, int * n_img_embd, int * n_img_pos) {
    clip_image_f32 img_res;
    if (!clip_image_preprocess(ctx_clip, img, &img_res, /*pad2square =*/ true)) {
        fprintf(stderr, "%s: unable to preprocess image\n", __func__);

        return false;
    }

    *n_img_pos = clip_n_patches(ctx_clip);
    *n_img_embd = clip_n_mmproj_embd(ctx_clip);

    const int64_t t_img_enc_start_us = ggml_time_us();
    if (!clip_image_encode(ctx_clip, n_threads, &img_res, image_embd)) {
        fprintf(stderr, "Unable to encode image\n");

        return false;
    }
    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;

    {
        printf("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / *n_img_pos);
    }

    return true;
}

bool llava_build_img_embed(const llama_context * ctx_llama, clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_image_pos_out) {

    float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip));
    if (!image_embd) {
        fprintf(stderr, "Unable to allocate memory for image embeddings\n");
        free(image_embd);
        return false;
    }

    int n_image_pos;
    int n_img_embd;
    if (!encode_image_with_clip(ctx_clip, n_threads, img, image_embd, &n_img_embd, &n_image_pos)) {
        fprintf(stderr, "%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    // make sure that the correct mmproj was used, i.e., compare apples to apples
    int n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));
    if (n_img_embd != n_llama_embd) {
        printf("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_img_embd, n_llama_embd);
        free(image_embd);
        return false;
    }

    *image_embd_out = image_embd;
    *n_image_pos_out = n_image_pos;

    return true;
}


struct llava_context * llava_init(gpt_params * params) {

    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }
    
    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    llama_backend_init(params->numa);

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return NULL;
    }

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings
    ctx_params.n_threads       = params->n_threads;
    ctx_params.n_threads_batch = params->n_threads_batch == -1 ? params->n_threads : params->n_threads_batch;

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;
    return ctx_llava;
}


void llava_free(struct llava_context * ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

