#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "clip.h"
#include "llava-utils.h"
#include "common.h"
#include "llama.h"


int main(int argc, char ** argv) {
    gpt_params params;

    if (argc < 4) {
        printf("usage: %s <path/to/llava-v1.5/ggml-model-q5_k.gguf> <path/to/llava-v1.5/mmproj-model-f16.gguf> <path/to/an/image.jpg> [a text prompt]\n", argv[0]);
        return 1;
    }

          params.model     = argv[1];
    const char * clip_path = argv[2];
    const char * img_path = argv[3];

    if (argc >= 5) {
        params.prompt = argv[4];
    }

    if (params.prompt.empty()) {
        params.prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    // load and preprocess the iamge
    clip_image_u8 img;
    clip_image_f32 img_res;
    clip_image_load_from_file(img_path, &img);
    clip_image_preprocess(ctx_clip, &img, &img_res);

    float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip));
    if (!image_embd) {
        fprintf(stderr, "Unable to allocate memory for CLIP embeddings\n");

        return 1;
    }

    if (!clip_image_encode(ctx_clip, params.n_threads, &img_res, image_embd)) {
        fprintf(stderr, "Unable to encode image\n");

        return 1;
    }

    // we get the embeddings, free up the memory required for CLIP
    clip_free(ctx_clip);

    llama_backend_init(params.numa);

    llama_model_params model_params = llama_model_default_params();
      // model_params.n_gpu_layers = 99; // offload all layers to the GPU
    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params                 = llama_context_default_params();
                         ctx_params.seed            = 1234;
                         ctx_params.n_ctx           = 2048;
                         ctx_params.n_threads       = params.n_threads;
                         ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;
    llama_context        * ctx_llama                = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // process the prompt
    // llava chat format is "user: <image embeddings>\n<textual prompt>\nassistant:"

    int n_past      = 0;
    int max_tgt_len = 256;
    eval_string(ctx_llama, "user: ", params.n_batch, &n_past);
    eval_image_embd(ctx_llama, image_embd, /*n_pos_image=*/ 576, params.n_batch, &n_past);
    eval_string(ctx_llama, params.prompt.c_str(), params.n_batch, &n_past);
eval_string(ctx_llama, "\nassistant:", params.n_batch, &n_past);

    // generate the response

    const char* tmp;
    for (int i=0; i<max_tgt_len; i++) {
        tmp = sample(ctx_llama, params, &n_past);
        if (strcmp(tmp, "</s>")==0) break;
        printf("%s", tmp);
        fflush(stdout);
    }
    printf("\n");

    llama_print_timings(ctx_llama);

    llama_free(ctx_llama);
    llama_free_model(model);
    llama_backend_free();
    free(image_embd);

    return 0;
}
