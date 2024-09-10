#include "ggml.h"
#include "common.h"
#include "clip.h"
#include "internvl.h"
#include "llama.h"

#include "base64.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            fprintf(stderr, "%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
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
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);

    // printf("prompt token ids: ");
    // for (int i = 0; i < (int) embd_inp.size(); i++) {
    //     printf("%d ", embd_inp[i]);
    // }
    // printf("\n");

    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

static const char * sample(struct llama_sampling_context * ctx_sampling,
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
static internvl_image_embed * internvl_image_embed_make_with_prompt_base64(struct clip_ctx * ctx_clip, int n_threads, const std::string& prompt) {
    size_t img_base64_str_start, img_base64_str_end;
    find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
    if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos) {
        fprintf(stderr, "%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
        return NULL;
    }

    auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
    auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
    auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count );

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

    auto embed = internvl_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
    if (!embed) {
        fprintf(stderr, "%s: could not load image from base64 string.\n", __func__);
        return NULL;
    }

    return embed;
}

static std::string remove_image_from_prompt(const std::string& prompt, const char * replacement = "") {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    if (begin == std::string::npos || end == std::string::npos) {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(end + strlen(IMG_BASE64_TAG_END));
    return pre + replacement + post;
}

struct internvl_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\n example usage:\n");
    LOG_TEE("\n     %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG_TEE("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static struct internvl_image_embed * load_image(internvl_context * ctx_internvl, gpt_params * params, const std::string & fname) {

    // load and preprocess the image
    internvl_image_embed * embed = NULL;
    auto prompt = params->prompt;
    if (prompt_contains_image(prompt)) {
        if (!params->image.empty()) {
            fprintf(stderr, "using base64 encoded image instead of command line image path\n");
        }
        embed = internvl_image_embed_make_with_prompt_base64(ctx_internvl->ctx_clip, params->n_threads, prompt);
        if (!embed) {
            fprintf(stderr, "%s: can't load image from prompt\n", __func__);
            return NULL;
        }
        params->prompt = remove_image_from_prompt(prompt);
    } else {
        embed = internvl_image_embed_make_with_filename(ctx_internvl->ctx_clip, params->n_threads, fname.c_str());
        if (!embed) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, fname.c_str());
            return NULL;
        }
    }

    return embed;
}

// prompt token ids = [user_id, tokens_id, assistant_id]
// total embedding = concat(img_embedding, tokens_id_embedding)
static void process_prompt(struct internvl_context * ctx_internvl, struct internvl_image_embed * image_embed, gpt_params * params, const std::string & prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx_internvl->ctx_llama));

    // llava chat format is "'<|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语).<|im_end|><|im_start|>user\n<image>\n请描述图片.<|im_end|><|im_start|>assistant\n'"
    std::size_t img_tok_pos = prompt.find("<image>");
    std::string prompt1;
    std::string prompt2;

    if (img_tok_pos != std::string::npos) {
        prompt1 = prompt.substr(0, img_tok_pos);
        prompt2 = prompt.substr(img_tok_pos + 7);
    }
    else {
        prompt1 = "";
        prompt2 = "\n" + prompt;
    }
    
    eval_string(ctx_internvl->ctx_llama, ("<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user\n" + prompt1 + "<img>").c_str(), params->n_batch, &n_past, true);
    // eval_string(ctx_internvl->ctx_llama, ("<|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语).<|im_end|><|im_start|>user\n" + prompt1 + "<img>").c_str(), params->n_batch, &n_past, true);
    internvl_eval_image_embed(ctx_internvl->ctx_llama, image_embed, params->n_batch, &n_past);
    eval_string(ctx_internvl->ctx_llama, ("</img>" + prompt2 + "<|im_end|><|im_start|>assistant\n").c_str(), params->n_batch, &n_past, false);
    // generate the response

    fprintf(stderr, "\n");

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params->sparams);

    if (params->n_predict == -1) {
        while (true) {
            const char *tmp = sample(ctx_sampling, ctx_internvl->ctx_llama, &n_past);
            if (strcmp(tmp, "</s>") == 0 || strcmp(tmp, "<|im_end|>") == 0)
                break;
            printf("%s", tmp);
            fflush(stdout);
        }
    } else {
        for (int i = 0; i < max_tgt_len; i++) {
            const char *tmp = sample(ctx_sampling, ctx_internvl->ctx_llama, &n_past);
            if (strcmp(tmp, "</s>") == 0 || strcmp(tmp, "<|im_end|>") == 0)
                break;
            printf("%s", tmp);
            fflush(stdout);
        }
    }

    llama_sampling_free(ctx_sampling);
    printf("\n");
    }

static struct llama_model * internvl_init(gpt_params * params) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return NULL;
    }
    
    return model;
}

static struct llama_context * llama_init_context(gpt_params * params, llama_model * model) {
    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    return ctx_llama;
}

static struct internvl_context * internvl_init_context(gpt_params * params, llama_model * model) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    // load visual model
    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    auto ctx_internvl = (struct internvl_context *)malloc(sizeof(internvl_context));

    ctx_internvl->ctx_llama = NULL;
    ctx_internvl->ctx_clip = ctx_clip;
    ctx_internvl->model = model;
    return ctx_internvl;
}

static void internvl_free(struct internvl_context * ctx_internvl) {
    if (ctx_internvl->ctx_clip) {
        clip_free(ctx_internvl->ctx_clip);
        ctx_internvl->ctx_clip = NULL;
    }

    llama_free(ctx_internvl->ctx_llama);
    llama_free_model(ctx_internvl->model);
    llama_backend_free();
}

static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv, params);
        return 1;
    }

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("llava", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
    llama_log_set(llama_log_callback_logTee, nullptr);
#endif // LOG_DISABLE_LOGS

    if (params.mmproj.empty() || (params.image.empty() && !prompt_contains_image(params.prompt))) {
        print_usage(argc, argv, params);
        return 1;
    }
    // printf("[debug by cxt] use prompt: %s\n", params.prompt.c_str());
    // printf("[debug by cxt] concat_image_text_embedding: %d\n", params.concat_image_text_embedding);
    // printf("[debug by cxt] bench_perf: %d\n", params.bench_perf);

    auto model = internvl_init(&params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init internvl\n", __func__);
        return 1;
    }

    // auto prompt_embed = load_adaptor(ctx_internvl, params.prompt.c_str(), false);
    // printf("%s: prompt context:%s, token size: %d\n", __func__, params.prompt.c_str(), prompt_embed->n_token);
    // for (int i=0; i<prompt_embed->n_token; i++) {
    //     int col_num=5;
    //     printf("[%d,:%d]: ", i, col_num);
    //     for (int j=0; j<col_num; j++) {
    //         printf("%f ", prompt_embed->embed[i*4096 + j]);
    //     }
    //     printf("  [%d,-%d:]: ", i, col_num);
    //     for (int j=0; j<col_num; j++) {
    //         printf("%f ", prompt_embed->embed[i*4096 + 4096 - col_num + j]);
    //     }
    //     printf("\n");
    // }
    // auto ctx_llama = llama_init_context(&params, model);

    auto ctx_internvl = internvl_init_context(&params, model);
    ctx_internvl->ctx_llama = llama_init_context(&params, model);
    for (auto & image : params.image) {
        for (int i=0; i<15; i++) {

        ctx_internvl->ctx_llama = llama_init_context(&params, model);
        // // clear kv cache
        // llama_kv_cache_clear(ctx_internvl->ctx_llama);

        const int64_t t_e2e_start_us = ggml_time_us();
        auto image_embed = load_image(ctx_internvl, &params, image);
        if (!image_embed) {
            std::cerr << "error: failed to load image " << image << ". Terminating\n\n";
            return 1;
        }

        // process the prompt
        process_prompt(ctx_internvl, image_embed, &params, params.prompt);

        const int64_t t_e2e_end_us = ggml_time_us();
        float t_e2e_cost_us = (t_e2e_end_us - t_e2e_start_us) / 1000.0;
        LOG_TEE("\n%s: %d e2e in %8.2f ms\n", __func__, i, t_e2e_cost_us);

        llama_print_timings(ctx_internvl->ctx_llama);

        // internvl_adaptor_embed_free(prompt_embed);

        internvl_image_embed_free(image_embed);
        // ctx_internvl->model = NULL;
        // internvl_free(ctx_internvl);

        }
    }

    llama_free_model(model);
    
    return 0;
}
