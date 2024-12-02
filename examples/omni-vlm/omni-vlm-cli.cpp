#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "omni-vlm.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

using std::cout;
using std::endl;

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval))) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::common_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

static const char * sample(struct common_sampler * ctx_sampling,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = common_sampler_sample(ctx_sampling, ctx_llama, -1);
    common_sampler_accept(ctx_sampling, id, true);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(ctx_llama), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

static const std::string IMG_PAD = "<|image_pad|>";

static void find_image_tag_in_prompt(const std::string& prompt, size_t& idx) {
    // begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    // end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
    idx = prompt.find(IMG_PAD);
}

static bool prompt_contains_image(const std::string& prompt) {
    size_t begin;
    find_image_tag_in_prompt(prompt, begin);
    return (begin != std::string::npos);
}

// replaces the base64 image tag in the prompt with `replacement`
static omni_image_embed * omnivlm_image_embed_make_with_prompt(struct clip_ctx * ctx_clip, int n_threads, const std::string& prompt) {
    size_t idx;
    find_image_tag_in_prompt(prompt, idx);
    if (idx == std::string::npos) {
        LOG_ERR("%s: invalid base64 image tag. must be %s\n", __func__, IMG_PAD.c_str());
        return NULL;
    }

    auto base64_str = prompt.substr(idx, IMG_PAD.size());

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

    auto embed = omnivlm_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
    if (!embed) {
        LOG_ERR("%s: could not load image from base64 string.\n", __func__);
        return NULL;
    }

    return embed;
}

static std::string remove_image_from_prompt(const std::string& prompt, const char * replacement = "") {
    size_t begin;
    find_image_tag_in_prompt(prompt, begin);
    if (begin == std::string::npos) {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(begin + IMG_PAD.size());
    return pre + replacement + post;
}

struct omnivlm_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

static void print_usage(int argc, char ** argv) {
    LOG_ERR("\n example usage:\n");
    LOG_ERR("\n     %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG_ERR("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static struct omni_image_embed * load_image(omnivlm_context * ctx_omnivlm, common_params * params, const std::string & fname) {

    // load and preprocess the image
    omni_image_embed * embed = NULL;
    embed = omnivlm_image_embed_make_with_filename(ctx_omnivlm->ctx_clip, params->cpuparams.n_threads, fname.c_str());
    if (!embed) {
        fprintf(stderr, "%s: is %s really an image file?\n", __func__, fname.c_str());
        return NULL;
    }

    return embed;
}

static void process_prompt(struct omnivlm_context * ctx_omnivlm, struct omni_image_embed * image_embed, common_params * params, const std::string & prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    std::string system_prompt, user_prompt;
    size_t image_pos = prompt.find("<|image_pad|>");
    // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
    system_prompt = prompt.substr(0, image_pos);
    user_prompt = prompt.substr(image_pos + std::string("<|image_pad|>").length());
    if (params->verbose_prompt) {
        auto tmp = ::common_tokenize(ctx_omnivlm->ctx_llama, system_prompt, true, true);
        for (int i = 0; i < (int) tmp.size(); i++) {
            LOG_ERR("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_omnivlm->ctx_llama, tmp[i]).c_str());
        }
    }
    // LOG_ERR("user_prompt: %s\n", user_prompt.c_str());
    if (params->verbose_prompt) {
        auto tmp = ::common_tokenize(ctx_omnivlm->ctx_llama, user_prompt, true, true);
        for (int i = 0; i < (int) tmp.size(); i++) {
            LOG_ERR("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_omnivlm->ctx_llama, tmp[i]).c_str());
        }
    }

    eval_string(ctx_omnivlm->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, true);
    omnivlm_eval_image_embed(ctx_omnivlm->ctx_llama, image_embed, params->n_batch, &n_past);
    eval_string(ctx_omnivlm->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);

    // generate the response

    LOG("\n");

    params->sparams.temp = 0.0f;
    params->sparams.top_k = 1;
    params->sparams.top_p = 1.0f;
    struct common_sampler * ctx_sampling = common_sampler_init(ctx_omnivlm->model, params->sparams);
    if (!ctx_sampling) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(ctx_sampling, ctx_omnivlm->ctx_llama, &n_past);
        response += tmp;
        if (strcmp(tmp, "<|im_end|>") == 0) break;
        if (strcmp(tmp, "</s>") == 0) break;
        printf("%s", tmp);
        // LOG("%s", tmp);
        // if (strstr(response.c_str(), "<|im_end|>")) break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        // if (strstr(response.c_str(), "<|im_start|>")) break; // Yi-34B llava-1.6
        // if (strstr(response.c_str(), "USER:")) break; // mistral llava-1.6

        fflush(stdout);
    }

    common_sampler_free(ctx_sampling);
    printf("\n");
}

static struct llama_model * omnivlm_init(common_params * params) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = common_model_params_to_llama(*params);

    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

static struct omnivlm_context * omnivlm_init_context(common_params * params, llama_model * model) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, params->omni_vlm_version.c_str(), /*verbosity=*/ 0);
    // clip_set_omni_vlm_version(ctx_clip, params);

    llama_context_params ctx_params = common_context_params_to_llama(*params);
    ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        LOG_ERR("%s: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto * ctx_omnivlm = (struct omnivlm_context *)malloc(sizeof(omnivlm_context));

    ctx_omnivlm->ctx_llama = ctx_llama;
    ctx_omnivlm->ctx_clip = ctx_clip;
    ctx_omnivlm->model = model;
    return ctx_omnivlm;
}

static void omnivlm_free(struct omnivlm_context * ctx_omnivlm) {
    if (ctx_omnivlm->ctx_clip) {
        clip_free(ctx_omnivlm->ctx_clip);
        ctx_omnivlm->ctx_clip = NULL;
    }

    llama_free(ctx_omnivlm->ctx_llama);
    llama_free_model(ctx_omnivlm->model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, print_usage)) {
        return 1;
    }

    if (params.mmproj.empty() || (params.image.empty() && !prompt_contains_image(params.prompt))) {
        print_usage(argc, argv);
        return 1;
    }
    if (params.omni_vlm_version != "vlm-81-ocr" && params.prompt.empty()) {
        LOG_ERR("%s : prompt is empty.\n", __func__);
        print_usage(argc, argv);
        return 1;
    }

    if (params.omni_vlm_version == "vlm-81-ocr") {
        params.prompt = "<|im_start|>system\nYou are Nano-Omni-VLM, created by Nexa AI. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n <|vision_start|><|image_pad|><|vision_end|><|im_end|>";
    } else if (params.omni_vlm_version == "vlm-81-instruct" || params.omni_vlm_version == "nano-vlm-instruct") {
        params.prompt = "<|im_start|>system\nYou are Nano-Omni-VLM, created by Nexa AI. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n\n<|vision_start|><|image_pad|><|vision_end|>" + params.prompt + "<|im_end|>";
    } else {
        LOG_ERR("%s : error: you set wrong vlm version info:'%s'.\n", __func__, params.omni_vlm_version.c_str());
        print_usage(argc, argv);
        return 1;
    }

    auto * model = omnivlm_init(&params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init omnivlm model\n", __func__);
        return 1;
    }


    for (auto & image : params.image) {
        auto * ctx_omnivlm = omnivlm_init_context(&params, model);
        auto * image_embed = load_image(ctx_omnivlm, &params, image);
        if (!image_embed) {
            LOG_ERR("%s: failed to load image %s. Terminating\n\n", __func__, image.c_str());
            return 1;
        }
        // process the prompt
        process_prompt(ctx_omnivlm, image_embed, &params, params.prompt);

        llama_perf_context_print(ctx_omnivlm->ctx_llama);
        omnivlm_image_embed_free(image_embed);
        ctx_omnivlm->model = NULL;
        omnivlm_free(ctx_omnivlm);
    }

    llama_free_model(model);

    return 0;
}
