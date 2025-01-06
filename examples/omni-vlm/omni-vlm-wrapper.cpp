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
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>
#include <memory>

#include "omni-vlm-wrapper.h"

struct omnivlm_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

void* internal_chars = nullptr;

static struct common_params params;
static struct llama_model* model;
static struct omnivlm_context* ctx_omnivlm;
static std::unique_ptr<struct omni_streaming_sample> g_oss = nullptr;

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past);
static void omnivlm_free(struct omnivlm_context * ctx_omnivlm);

struct omni_streaming_sample {
    struct common_sampler * ctx_sampling_;
    std::string image_;
    std::string ret_str_;
    int32_t n_past_;
    int32_t dec_cnt_;

    omni_streaming_sample() = delete;
    omni_streaming_sample(const std::string& image)
            :image_(image) {
        n_past_ = 0;
        dec_cnt_ = 0;
        params.sparams.top_k = 1;
        params.sparams.top_p = 1.0f;
        ctx_sampling_ = common_sampler_init(model, params.sparams);
    }

    int32_t sample() {
        const llama_token id = common_sampler_sample(ctx_sampling_, ctx_omnivlm->ctx_llama, -1);
        common_sampler_accept(ctx_sampling_, id, true);
        if (llama_token_is_eog(llama_get_model(ctx_omnivlm->ctx_llama), id)) {
            ret_str_ = "</s>";
        } else {
            ret_str_ = common_token_to_piece(ctx_omnivlm->ctx_llama, id);
        }
        eval_id(ctx_omnivlm->ctx_llama, id, &n_past_);

        ++dec_cnt_;
        return id;
    }

    ~omni_streaming_sample() {
        common_sampler_free(ctx_sampling_);
        if(ctx_omnivlm != nullptr) {
            ctx_omnivlm->model = nullptr;
            omnivlm_free(ctx_omnivlm);
            free(ctx_omnivlm);
            ctx_omnivlm = nullptr;
        }
    }
};


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

    ctx_omnivlm = (struct omnivlm_context *)malloc(sizeof(omnivlm_context));

    ctx_omnivlm->ctx_llama = ctx_llama;
    ctx_omnivlm->ctx_clip = ctx_clip;
    ctx_omnivlm->model = model;
    return ctx_omnivlm;
}

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

static const char * sample(struct common_sampler * smpl,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = common_sampler_sample(smpl, ctx_llama, -1);
    common_sampler_accept(smpl, id, true);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(ctx_llama), id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

static const char* process_prompt(struct omnivlm_context * ctx_omnivlm, struct omni_image_embed * image_embed, common_params * params, const std::string & prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    // std::string full_prompt = "<|im_start|>system\nYou are Nano-Omni-VLM, created by Nexa AI. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" \
    //                             + prompt + "\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>";
    size_t image_pos = params->prompt.find("<|image_pad|>");
    std::string system_prompt, user_prompt;

    // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
    system_prompt = params->prompt.substr(0, image_pos);
    user_prompt = params->prompt.substr(image_pos + std::string("<|image_pad|>").length());
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

    params->sparams.top_k = 1;
    params->sparams.top_p = 1.0f;

    eval_string(ctx_omnivlm->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, true);
    omnivlm_eval_image_embed(ctx_omnivlm->ctx_llama, image_embed, params->n_batch, &n_past);
    eval_string(ctx_omnivlm->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);

    // generate the response

    LOG("\n");

    struct common_sampler * smpl = common_sampler_init(ctx_omnivlm->model, params->sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(smpl, ctx_omnivlm->ctx_llama, &n_past);
        if (strcmp(tmp, "<|im_end|>") == 0) break;
        if (strcmp(tmp, "</s>") == 0) break;
        // if (strstr(tmp, "###")) break; // Yi-VL behavior
        // printf("%s", tmp);
        response += tmp;
        // if (strstr(response.c_str(), "<|im_end|>")) break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        // if (strstr(response.c_str(), "<|im_start|>")) break; // Yi-34B llava-1.6
        // if (strstr(response.c_str(), "USER:")) break; // mistral llava-1.6

        fflush(stdout);
    }

    common_sampler_free(smpl);
    printf("\n");

    // const char* ret_char_ptr = (const char*)(malloc(sizeof(char)*response.size()));
    if(internal_chars != nullptr) { free(internal_chars); }
    internal_chars = malloc(sizeof(char)*(response.size()+1));
    strncpy((char*)(internal_chars), response.c_str(), response.size());
    ((char*)(internal_chars))[response.size()] = '\0';
    return (const char*)(internal_chars);
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

static void print_usage(int argc, char ** argv) {
    LOG_ERR("\n example usage:\n");
    LOG_ERR("\n     %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG_ERR("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

// inference interface definition
void omnivlm_init(const char* llm_model_path, const char* projector_model_path, const char* omni_vlm_version) {
    const char* argv = "omni-wrapper-py";
    char* nc_argv = const_cast<char*>(argv);
    if (!common_params_parse(1, &nc_argv, params, LLAMA_EXAMPLE_LLAVA, print_usage)) {
        print_usage(1, &nc_argv);
        throw std::runtime_error("init params error.");
    }
    params.model = llm_model_path;
    params.mmproj = projector_model_path;
    params.omni_vlm_version = omni_vlm_version;

    std::string omni_vlm_ver = params.omni_vlm_version;
    if(omni_vlm_ver != "vlm-81-ocr" && omni_vlm_ver != "vlm-81-instruct" && omni_vlm_ver != "nano-vlm-instruct") {
        fprintf(stderr, "%s: error: you set wrong omni_vlm_string: %s\n", __func__, omni_vlm_version);
        fprintf(stderr, "%s: Valid omni_vlm_version set is ('vlm-81-ocr', 'vlm-81-instruct', 'nano-vlm-instruct')\n", __func__);
        throw std::runtime_error("You set wrong vlm_version info strings.");
    }

    model = omnivlm_init(&params);
    if (model == nullptr) {
        fprintf(stderr, "%s: error: failed to init omnivlm model\n", __func__);
        throw std::runtime_error("Failed to init omnivlm model");
    }
}

const char* omnivlm_inference(const char *prompt, const char *imag_path) {
    ctx_omnivlm = omnivlm_init_context(&params, model);

    std::string image = imag_path;
    params.prompt = prompt;

    if (params.omni_vlm_version == "vlm-81-ocr") {
        params.prompt = "<|im_start|>system\nYou are Nano-Omni-VLM, created by Nexa AI. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n <|ocr_start|><|vision_start|><|image_pad|><|vision_end|><|ocr_end|><|im_end|>";
    } else if (params.omni_vlm_version == "vlm-81-instruct" || params.omni_vlm_version == "nano-vlm-instruct") {
        params.prompt = "<|im_start|>system\nYou are Nano-Omni-VLM, created by Nexa AI. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n\n<|vision_start|><|image_pad|><|vision_end|>" + params.prompt + "<|im_end|>";
    } else {
        LOG_ERR("%s : error: you set wrong vlm version info:'%s'.\n", __func__, params.omni_vlm_version.c_str());
        throw std::runtime_error("You set wrong vlm_version info strings.");
    }

    auto * image_embed = load_image(ctx_omnivlm, &params, image);
    if (!image_embed) {
        LOG_ERR("%s: failed to load image %s. Terminating\n\n", __func__, image.c_str());
        throw std::runtime_error("failed to load image " + image);
    }
    // process the prompt
    const char* ret_chars = process_prompt(ctx_omnivlm, image_embed, &params, params.prompt);

    // llama_perf_print(ctx_omnivlm->ctx_llama, LLAMA_PERF_TYPE_CONTEXT);
    omnivlm_image_embed_free(image_embed);
    ctx_omnivlm->model = nullptr;
    omnivlm_free(ctx_omnivlm);
    ctx_omnivlm = nullptr;

    return ret_chars;
}

void omnivlm_free() {
    if(internal_chars != nullptr) { free(internal_chars); }
    if(ctx_omnivlm != nullptr) {
        // this snipet should never be run!
        ctx_omnivlm->model = nullptr;
        omnivlm_free(ctx_omnivlm);
    }
    llama_free_model(model);
}


struct omni_streaming_sample* omnivlm_inference_streaming(const char *prompt, const char *imag_path) {
    if (g_oss) {
        g_oss.reset();
    }
    g_oss = std::make_unique<omni_streaming_sample>(std::string(imag_path));

    ctx_omnivlm = omnivlm_init_context(&params, model);

    params.prompt = prompt;

    if (params.omni_vlm_version == "vlm-81-ocr") {
        params.prompt = "<|im_start|>system\nYou are Nano-Omni-VLM, created by Nexa AI. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n <|ocr_start|><|vision_start|><|image_pad|><|vision_end|><|ocr_end|><|im_end|>";
    } else if (params.omni_vlm_version == "vlm-81-instruct" || params.omni_vlm_version == "nano-vlm-instruct") {
        params.prompt = "<|im_start|>system\nYou are Nano-Omni-VLM, created by Nexa AI. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n\n<|vision_start|><|image_pad|><|vision_end|>" + params.prompt + "<|im_end|>";
    } else {
        LOG_ERR("%s : error: you set wrong vlm version info:'%s'.\n", __func__, params.omni_vlm_version.c_str());
        throw std::runtime_error("You set wrong vlm_version info strings.");
    }

    return g_oss.get();
}

int32_t sample(omni_streaming_sample* oss) {
    const int max_tgt_len = params.n_predict < 0 ? 256 : params.n_predict;
    int32_t ret_id;
    if(oss->n_past_ == 0) {
        auto * image_embed = load_image(ctx_omnivlm, &params, oss->image_);
        if (!image_embed) {
            LOG_ERR("%s: failed to load image %s. Terminating\n\n", __func__, oss->image_.c_str());
            throw std::runtime_error("failed to load image " + oss->image_);
        }

        size_t image_pos = params.prompt.find("<|image_pad|>");
        std::string system_prompt, user_prompt;

        system_prompt = params.prompt.substr(0, image_pos);
        user_prompt = params.prompt.substr(image_pos + std::string("<|image_pad|>").length());
        if (params.verbose_prompt) {
            auto tmp = ::common_tokenize(ctx_omnivlm->ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_ERR("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_omnivlm->ctx_llama, tmp[i]).c_str());
            }
        }
        if (params.verbose_prompt) {
            auto tmp = ::common_tokenize(ctx_omnivlm->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_ERR("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_omnivlm->ctx_llama, tmp[i]).c_str());
            }
        }

        eval_string(ctx_omnivlm->ctx_llama, system_prompt.c_str(), params.n_batch, &(oss->n_past_), true);
        omnivlm_eval_image_embed(ctx_omnivlm->ctx_llama, image_embed, params.n_batch, &(oss->n_past_));
        eval_string(ctx_omnivlm->ctx_llama, user_prompt.c_str(), params.n_batch, &(oss->n_past_), false);

        omnivlm_image_embed_free(image_embed);

        ret_id = oss->sample();
        if (oss->ret_str_ == "<|im_end|>" || oss->ret_str_ == "</s>" ) {
            ret_id = -1;
        }
    } else {
        if(oss->dec_cnt_ == max_tgt_len) {
            ret_id = -2;
        } else {
            ret_id = oss->sample();
            if (oss->ret_str_ == "<|im_end|>" || oss->ret_str_ == "</s>" ) {
                ret_id = -1;
            }
        }
    }
    return ret_id;
}

const char* get_str(omni_streaming_sample* oss) {
    return oss->ret_str_.c_str();
}