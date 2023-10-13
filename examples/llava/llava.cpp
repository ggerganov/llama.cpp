#include "clip.h"
#include "llava-utils.h"
#include "common.h"
#include "llama.h"
#include "llava.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "base64.hpp"

static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    printf("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static bool encode_image_with_clip(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float * image_embd, int * n_img_embd, int * n_img_pos, float * t_img_enc_ms) {
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
    *t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;
    return true;
}

static const char* IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char* IMG_BASE64_TAG_END = "\">";

static void find_image_tag_in_prompt(const std::string& prompt, size_t& begin_out, size_t& end_out) {
    begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
}

static bool prompt_contains_image(const std::string& prompt) {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    return (begin != std::string::npos);
}

// replaces the base64 image tag in the prompt with `replacement`
static bool get_image_from_prompt(const std::string& prompt, clip_image_u8 * img) {    
    size_t img_base64_str_start, img_base64_str_end;
    find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
    if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos) {
        fprintf(stderr, "%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
        return false;
    }

    auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
    auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
    auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count );
    printf("base64_str: '%s'\n", base64_str.c_str());

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    auto img_bytes_end = base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());
    auto img_bytes_len = img_bytes_end - img_bytes.begin();

    auto img_loaded_ok = clip_image_load_from_bytes(img_bytes.data(), img_bytes_len, img);
    if (!img_loaded_ok) {
        fprintf(stderr, "%s: could not load image from base64 string.\n", __func__);
        return false;
    }

    return true;
}

static std::string remove_image_from_prompt(const std::string& prompt, const char * replacement = "") {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    if (begin == std::string::npos || end == std::string::npos) {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(end+1);
    return pre + replacement + post;
}

struct llava_context * llava_init(gpt_params * params) {

    const char * clip_path = params->mmproj.c_str();
    const char * img_path = params->image.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }
    
    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    // load and preprocess the image
    clip_image_u8 img;

    if (prompt_contains_image(prompt)) {
        if (img_path) {
            printf("using base64 encoded image instead of command line image path\n");
        }
        if (!get_image_from_prompt(prompt, &img)) {
            fprintf(stderr, "%s: can't load image from prompt\n", __func__);
            clip_free(ctx_clip);
            return NULL;
        }
        prompt = remove_image_from_prompt(prompt);
    } else {
        if (!clip_image_load_from_file(img_path, &img)) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, img_path);
            clip_free(ctx_clip);
            return NULL;
        }
    }

    float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip));
    if (!image_embd) {
        fprintf(stderr, "Unable to allocate memory for image embeddings\n");
        return NULL;
    }

    int n_img_embd;
    int n_img_pos;
    float t_img_enc_ms;
    if (!encode_image_with_clip(ctx_clip, params->n_threads, &img, image_embd, &n_img_embd, &n_img_pos, &t_img_enc_ms)) {
        fprintf(stderr, "%s: cannot encode image, aborting\n", __func__);
        clip_free(ctx_clip);
        return NULL;
    }

    // we get the embeddings, free up the memory required for CLIP
    clip_free(ctx_clip);
    ctx_clip = NULL;

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

    // make sure that the correct mmproj was used, i.e., compare apples to apples
    int n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));
    if (n_img_embd != n_llama_embd) {
        printf("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_img_embd, n_llama_embd);

        llama_free(ctx_llama);
        llama_free_model(model);
        llama_backend_free();
        free(image_embd);

        return NULL;
    }

    {
        printf("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / n_img_pos);
    }


    auto ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;
    ctx_llava->image_embd = image_embd;
    ctx_llava->n_img_pos = n_img_pos;
    return ctx_llava;

}

void llava_free(struct llava_context * ctx_llava) {
    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
    free(ctx_llava->image_embd);
}

void llava_process_prompt(struct llava_context * ctx_llava, gpt_params * params, const char * prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    // GG: are we sure that the should be a trailing whitespace at the end of this string?
    eval_string(ctx_llava->ctx_llama, "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER: ", params->n_batch, &n_past);
    eval_image_embd(ctx_llava->ctx_llama, ctx_llava->image_embd, ctx_llava->n_img_pos, params->n_batch, &n_past);
    eval_string(ctx_llava->ctx_llama, prompt, params->n_batch, &n_past);
    eval_string(ctx_llava->ctx_llama, "\nASSISTANT:",        params->n_batch, &n_past);

    // generate the response

    printf("\n");

    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(ctx_llava->ctx_llama, *params, &n_past);
        if (strcmp(tmp, "</s>") == 0) break;

        printf("%s", tmp);
        fflush(stdout);
    }

    printf("\n");

}


int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        show_additional_info(argc, argv);
        return 1;
    }
    if (params.mmproj.empty() || (params.image.empty() && !prompt_contains_image(params.prompt))) {
        gpt_print_usage(argc, argv, params);
        show_additional_info(argc, argv);
        return 1;
    }

    auto ctx_llava = llava_init(&params);
    if (ctx_llava == NULL) {
        fprintf(stderr, "%s: error: failed to init llava\n", __func__);
        return 1;
    }

    // process the prompt
    // llava chat format is "<system_prompt>USER: <image_embeddings>\n<textual_prompt>\nASSISTANT:"
    llava_process_prompt(ctx_llava, &params, params.prompt.c_str());

    llama_print_timings(ctx_llava->ctx_llama);

    llava_free(ctx_llava);
    return 0;
}
