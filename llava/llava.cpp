#include "clip.h"
#include "llava-utils.h"
#include "common.h"
#include "llama.h"
#include "llava.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "base64.hpp"

static bool encode_image_with_clip(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float * image_embd, int * n_image_embd, int * n_img_pos) {
    clip_image_f32 img_res;
    if (!clip_image_preprocess(ctx_clip, img, &img_res, /*pad2square =*/ true)) {
        fprintf(stderr, "%s: unable to preprocess image\n", __func__);

        return false;
    }

    *n_img_pos = clip_n_patches(ctx_clip);
    *n_image_embd = clip_n_mmproj_embd(ctx_clip);

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

bool llava_build_img_embed(const llama_context * ctx_llama, clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out) {

    float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip));
    if (!image_embd) {
        fprintf(stderr, "Unable to allocate memory for image embeddings\n");
        free(image_embd);
        return false;
    }

    int n_img_pos;
    int n_image_embd;
    if (!encode_image_with_clip(ctx_clip, n_threads, img, image_embd, &n_image_embd, &n_img_pos)) {
        fprintf(stderr, "%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    // make sure that the correct mmproj was used, i.e., compare apples to apples
    int n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));
    if (n_image_embd != n_llama_embd) {
        printf("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_image_embd, n_llama_embd);
        free(image_embd);
        return false;
    }

    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}



bool llava_eval_image_embd(llama_context * ctx_llama, float * image_embd, int n_image_pos, int n_batch, int * n_past) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < n_image_pos; i += n_batch) {
        int n_eval = n_image_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {int32_t(n_eval), nullptr, (image_embd+i*n_embd), nullptr, nullptr, nullptr, *n_past, 1, 0, };
        if (llama_decode(ctx_llama, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}
