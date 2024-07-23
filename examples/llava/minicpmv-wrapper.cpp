#include "ggml.h"
#include "common.h"
#include "clip.h"
#include "llava.h"
#include "minicpmv-wrapper.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

struct uhd_image_embed {
    std::vector<std::vector<struct llava_image_embed *>> image_embeds;
};

struct llama_model * llava_init(gpt_params * params) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

struct minicpmv_context * llava_init_context(gpt_params * params, llama_model * model) {
    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    if (params->n_ctx < 2048) {
        // warn user here, "Image processing requires at least 2048 context, setting context to 2048"
        LOG_TEE("%s: warn: Image processing requires at least 2048 context, setting context to 2048\n" , __func__);
        ctx_params.n_ctx = 2048;
    } else {
        ctx_params.n_ctx = params->n_ctx;
    }

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        LOG_TEE("%s: error: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto ctx_llava = (struct minicpmv_context *)malloc(sizeof(minicpmv_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->model = model;
    return ctx_llava;
}

void llava_free(struct minicpmv_context * ctx_llava) {
    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

struct clip_ctx * clip_init_context(gpt_params * params) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }
    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);
    return ctx_clip;
}

struct uhd_image_embed * minicpmv_image_embed(gpt_params * params, const std::string & fname){
    auto ctx_clip = clip_init_context(params);
    auto image_embed_and_slices = llava_image_embed_make_with_filename_uhd(ctx_clip, params->n_threads, fname.c_str());
    if (ctx_clip) {
        clip_free(ctx_clip);
        ctx_clip = NULL;
    }
    return image_embed_and_slices;
}


bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            LOG_TEE("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    return eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
}

void process_image(struct minicpmv_context * ctx_llava, struct uhd_image_embed * image_embed_slices, gpt_params * params, int &n_past) {
    std::string system_prompt;

    system_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n";
    LOG_TEE("%s: image token past: %d\n", __func__, n_past);
    eval_string(ctx_llava->ctx_llama, (system_prompt+"<image>").c_str(), params->n_batch, &n_past, false);
    llava_eval_image_embed(ctx_llava->ctx_llama, image_embed_slices->image_embeds[0][0], params->n_batch, &n_past);
    eval_string(ctx_llava->ctx_llama, std::string("</image>").c_str(), params->n_batch, &n_past, false);
    if (image_embed_slices->image_embeds.size() > 1) {
        eval_string(ctx_llava->ctx_llama, std::string("<slice>").c_str(), params->n_batch, &n_past, false);
        for (size_t i = 1; i < image_embed_slices->image_embeds.size(); ++i) {
            for (size_t j = 0; j < image_embed_slices->image_embeds[i].size(); ++j) {
                eval_string(ctx_llava->ctx_llama, std::string("<image>").c_str(), params->n_batch, &n_past, false);
                llava_eval_image_embed(ctx_llava->ctx_llama, image_embed_slices->image_embeds[i][j], params->n_batch, &n_past);
                eval_string(ctx_llava->ctx_llama, std::string("</image>").c_str(), params->n_batch, &n_past, false);
                if (j == image_embed_slices->image_embeds[i].size() - 1) {
                    eval_string(ctx_llava->ctx_llama, std::string("\n").c_str(), params->n_batch, &n_past, false);
                }
            }
        }
        eval_string(ctx_llava->ctx_llama, std::string("</slice>").c_str(), params->n_batch, &n_past, false);

    }
    LOG_TEE("%s: image token past: %d\n", __func__, n_past);
}

const char * sample(struct llama_sampling_context * ctx_sampling,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = llama_sampling_sample(ctx_sampling, ctx_llama, NULL);
    llama_sampling_accept(ctx_sampling, ctx_llama, id, true);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(ctx_llama), id)) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}