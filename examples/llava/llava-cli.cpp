#include <cstdio>
#include <cstdlib>

#include "ggml.h"
#include "common.h"
#include "clip.h"
#include "llava.h"
#include "llava-utils.h"

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};



static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    printf("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static struct llava_image_embed * load_image(llava_context * ctx_llava, gpt_params * params) {

    // load and preprocess the image
    llava_image_embed * embed = NULL;
    auto prompt = params->prompt;
    if (prompt_contains_image(prompt)) {
        if (!params->image.empty()) {
            printf("using base64 encoded image instead of command line image path\n");
        }
        embed = llava_image_embed_make_with_prompt_base64(ctx_llava->ctx_clip, params->n_threads, prompt);
        if (!embed) {
            fprintf(stderr, "%s: can't load image from prompt\n", __func__);
            return NULL;
        }
        params->prompt = remove_image_from_prompt(prompt);
    } else {
        embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->n_threads, params->image.c_str());
        if (!embed) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, params->image.c_str());
            return NULL;
        }
    }

    return embed;
}

static void process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, gpt_params * params, const char * prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    // llava chat format is "<system_prompt>USER: <image_embeddings>\n<textual_prompt>\nASSISTANT:"
    printf("evaluating system prompt\n");
    eval_string(ctx_llava->ctx_llama, "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:", params->n_batch, &n_past, true);
    printf("evaluating image embed\n");
    llava_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past);
    printf("evaluating prompt\n");
    eval_string(ctx_llava->ctx_llama, prompt, params->n_batch, &n_past, false);
    eval_string(ctx_llava->ctx_llama, "\nASSISTANT:",        params->n_batch, &n_past, false);
    printf("awaiting response\n");

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


static struct llava_context * llava_init(gpt_params * params) {
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


static void llava_free(struct llava_context * ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
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

    auto image_embed = load_image(ctx_llava, &params);

    // process the prompt
    process_prompt(ctx_llava, image_embed, &params, params.prompt.c_str());

    llama_print_timings(ctx_llava->ctx_llama);

    llava_image_embed_free(image_embed);
    llava_free(ctx_llava);
    return 0;
}
