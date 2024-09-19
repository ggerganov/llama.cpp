#include "ggml.h"
#include "log.h"
#include "common.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"  // TODO: check if this head filde is necessary

#include <cstdio>
#include <cstdlib>
#include <vector>

static bool eval_tokens(struct llama_context *ctx_llama, std::vector<llama_token> tokens, int n_batch, int *n_past)
{
    int N = (int)tokens.size();
    // printf("token.size(): %d\n", N);
    // printf("n_batch: %d\n", n_batch);
    for (int i = 0; i < N; i += n_batch)
    {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch)
        {
            n_eval = n_batch;
        }
        // printf("n_eval: %d, n_past: %d\n", n_eval, *n_past);
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0)))
        {
            LOG_TEE("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context *ctx_llama, int id, int *n_past)
{
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context *ctx_llama, const char *str, int n_batch, int *n_past, bool add_bos)
{

    std::string              str2 = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    printf("!!prompt to eval!!: %s", str);
    printf("----------------------\n");
    // for (auto token : embd_inp){
    //     printf("%6d, ", token);
    // }
    printf("\n");
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

static const char *sample(struct llama_sampling_context *ctx_sampling, struct llama_context *ctx_llama, int *n_past)
{
    const llama_token id = llama_sampling_sample(ctx_sampling, ctx_llama, NULL);
    llama_sampling_accept(ctx_sampling, ctx_llama, id, true);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(ctx_llama), id))
    {
        ret = "<|end|>";
    }
    else
    {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

static const char *IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char *IMG_BASE64_TAG_END = "\">";

static void find_image_tag_in_prompt(const std::string &prompt, size_t &begin_out, size_t &end_out)
{
    begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
}

static bool prompt_contains_image(const std::string &prompt)
{
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    return (begin != std::string::npos);
}

// TODO: Implememt this function llava_image_embed_make_with_prompt_base64 for xgenmm
// static llava_image_embed *llava_image_embed_make_with_prompt_base64(struct clip_ctx *ctx_clip, int n_threads,
//                                                                     const std::string &prompt)
// {
//     size_t img_base64_str_start, img_base64_str_end;
//     find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
//     if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos)
//     {
//         LOG_TEE("%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN,
//                 IMG_BASE64_TAG_END);
//         return NULL;
//     }

//     auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
//     auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
//     auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count);

//     auto required_bytes = base64::required_encode_size(base64_str.size());
//     auto img_bytes = std::vector<unsigned char>(required_bytes);
//     base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

//     auto embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
//     if (!embed)
//     {
//         LOG_TEE("%s: could not load image from base64 string.\n", __func__);
//         return NULL;
//     }

//     return embed;
// }

static std::string remove_image_from_prompt(const std::string &prompt, const char *replacement = "")
{
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    if (begin == std::string::npos || end == std::string::npos)
    {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(end + strlen(IMG_BASE64_TAG_END));
    return pre + replacement + post;
}

struct llava_context
{
    struct clip_ctx      *ctx_clip = NULL;
    struct llama_context *ctx_llama = NULL;
    struct llama_model   *model = NULL;
};

// static void process_eval_image_embed(struct llava_context *ctx_llava, const struct llava_image_embed *embeds,
//                                      int n_batch, int *n_past, int idx)
// {
//     float *image_embed = (float *)malloc(clip_embd_nbytes(ctx_llava->ctx_clip));
//     std::memcpy(image_embed,
//                 embeds->embed + idx * clip_n_patches(ctx_llava->ctx_clip) * clip_n_mmproj_embd(ctx_llava->ctx_clip),
//                 clip_embd_nbytes(ctx_llava->ctx_clip));

//     auto slice_embed = (llava_image_embed *)malloc(sizeof(llava_image_embed));
//     slice_embed->embed = image_embed;
//     slice_embed->n_image_pos = clip_n_patches(ctx_llava->ctx_clip);
//     llava_eval_image_embed(ctx_llava->ctx_llama, slice_embed, n_batch, n_past);
//     llava_image_embed_free(slice_embed);
// }

static void print_usage(int argc, char **argv, const gpt_params &params)
{
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\n example usage:\n");
    LOG_TEE(
        "\n     %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image "
        "<path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in "
        "detail.\"]\n",
        argv[0]);
    LOG_TEE("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static struct llava_image_embed *load_image(llava_context *ctx_llava, gpt_params *params, const std::string &fname)
{
    // load and preprocess the image
    llava_image_embed *embed = NULL;
    auto               prompt = params->prompt;
    if (prompt_contains_image(prompt))
    {
        // if (!params->image.empty())
        // {
        //     LOG_TEE("using base64 encoded image instead of command line image path\n");
        // }
        // embed = llava_image_embed_make_with_prompt_base64(ctx_llava->ctx_clip, params->n_threads, prompt);
        // if (!embed)
        // {
        //     LOG_TEE("%s: can't load image from prompt\n", __func__);
        //     return NULL;
        // }
        // params->prompt = remove_image_from_prompt(prompt);
        printf("not implemented\n");
        exit(1);
    }
    else
    {
        embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->n_threads, fname.c_str());
        if (!embed)
        {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, fname.c_str());
            return NULL;
        }
    }

    return embed;
}

static void process_prompt(struct llava_context *ctx_llava, struct llava_image_embed *image_embed, gpt_params *params,
                           const std::string &prompt)
{
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    std::string system_prompt, user_prompt;
    size_t      image_pos = prompt.find("<image>");
    if (image_pos != std::string::npos)
    {
        // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for
        // the image
        system_prompt = prompt.substr(0, image_pos);
        user_prompt = prompt.substr(image_pos + std::string("<image>").length());
        LOG_TEE("system_prompt: %s\n", system_prompt.c_str());
        // phi3-tokenizer https://github.com/ggerganov/llama.cpp/issues/7938
        if (params->verbose_prompt)
        {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int)tmp.size(); i++)
            {
                LOG_TEE("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
        LOG_TEE("user_prompt: %s\n", user_prompt.c_str());
        if (params->verbose_prompt)
        {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int)tmp.size(); i++)
            {
                LOG_TEE("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    }
    else
    {
        // llava-1.5 native mode
        system_prompt =
            "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, "
            "detailed, and polite answers to the human's questions.\nUSER:";
        user_prompt = prompt + "\nASSISTANT:";
        if (params->verbose_prompt)
        {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int)tmp.size(); i++)
            {
                LOG_TEE("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    }
    eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, true);
    // image_embed
    // struct llava_image_embed
    // {
    //     float *embed;
    //     int    n_image_pos;
    // };
    llava_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past);
    eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);

    // generate the response

    LOG_TEE("\n");

    struct llama_sampling_context *ctx_sampling = llama_sampling_init(params->sparams);
    if (!ctx_sampling)
    {
        fprintf(stderr, "%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++)
    {
        // printf("i: %d\n", i);
        const char *tmp = sample(ctx_sampling, ctx_llava->ctx_llama, &n_past);
        response += tmp;
        // printf("%s", tmp);
        if (strcmp(tmp, "<|end|>") == 0){
            printf("\n STOP GENERATING because I saw <|end|>\n");
            break;
        }
        if (strcmp(tmp, "</s>") == 0) {
            printf("\n STOP GENERATING because I saw </s>\n");
            break;
        }
        if (strstr(tmp, "###")) break;  // Yi-VL behavior
        printf("%s", tmp);
        if (strstr(response.c_str(), "<|im_end|>"))
            break;  // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        if (strstr(response.c_str(), "<|im_start|>")) break;  // Yi-34B llava-1.6
        if (strstr(response.c_str(), "USER:")) break;         // mistral llava-1.6

        fflush(stdout);
    }

    llama_sampling_free(ctx_sampling);
    printf("\n");
}


static struct llama_model * llava_init(gpt_params * params) {
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

static struct llava_context *llava_init_context(gpt_params *params, llama_model *model)
{
    const char *clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty())
    {
        prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/1);

    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    ctx_params.n_ctx =
        params->n_ctx < 2048 ? 2048 : params->n_ctx;  // we need a longer context size to process image embeddings

    llama_context *ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL)
    {
        LOG_TEE("%s: error: failed to create the llama_context\n", __func__);
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
        printf(
            "YD:::Segmentation fault here; Because header.n_kv is empty\n clip_free->gguf_free(ctx->ctx_gguf)-> for "
            "(uint64_t i = 0; i < ctx->header.n_kv; ++i)\n");
        exit(1);
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }
    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

// static struct clip_ctx * clip_init_context(gpt_params * params) {
//     const char * clip_path = params->mmproj.c_str();

//     auto prompt = params->prompt;
//     if (prompt.empty()) {
//         prompt = "describe the image in detail.";
//     }
//     // std::cout << __LINE__ << std::endl;
//     auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);
//     return ctx_clip;
// }



// TODO: REMOVE THIS FUNCTION
// static void process_image(struct llava_context * ctx_llava, struct llava_image_embed * embeds, gpt_params * params, int &n_past) {
//     std::string system_prompt;
//     int idx = 0;
//     int num_image_embeds = embeds->n_image_pos / clip_n_patches(ctx_llava->ctx_clip);
//     int has_minicpmv_projector = clip_is_minicpmv(ctx_llava->ctx_clip);
//     if (has_minicpmv_projector == 2) {
//         system_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n";
//     }
//     else if (has_minicpmv_projector == 3) {
//         system_prompt = "<|im_start|>user\n";
//     }
//     LOG_TEE("%s: image token past: %d\n", __func__, n_past);
//     eval_string(ctx_llava->ctx_llama, (system_prompt+"<image>").c_str(), params->n_batch, &n_past, false);
//     process_eval_image_embed(ctx_llava, embeds, params->n_batch, &n_past, idx++);
//     eval_string(ctx_llava->ctx_llama, std::string("</image>").c_str(), params->n_batch, &n_past, false);
//     if (num_image_embeds > 1) {
//         size_t num_image_embeds_col = clip_uhd_num_image_embeds_col(ctx_llava->ctx_clip);
//         eval_string(ctx_llava->ctx_llama, std::string("<slice>").c_str(), params->n_batch, &n_past, false);
//         for (size_t i = 0; i < (num_image_embeds-1)/num_image_embeds_col; ++i) {
//             for (size_t j = 0; j < num_image_embeds_col; ++j) {
//                 eval_string(ctx_llava->ctx_llama, std::string("<image>").c_str(), params->n_batch, &n_past, false);
//                 process_eval_image_embed(ctx_llava, embeds, params->n_batch, &n_past, idx++);
//                 eval_string(ctx_llava->ctx_llama, std::string("</image>").c_str(), params->n_batch, &n_past, false);
//                 if (j == num_image_embeds_col - 1) {
//                     eval_string(ctx_llava->ctx_llama, std::string("\n").c_str(), params->n_batch, &n_past, false);
//                 }
//             }
//         }
//         eval_string(ctx_llava->ctx_llama, std::string("</slice>").c_str(), params->n_batch, &n_past, false);
//     }
//     LOG_TEE("%s: image token past: %d\n", __func__, n_past);
// }





// static struct llava_context * xgenmm_init(gpt_params * params, const std::string & fname, int &n_past){
//     auto ctx_clip = clip_init_context(params);
//     std::cout << "clip model has been loaded \n\n";

//     auto embeds = llava_image_embed_make_with_filename(ctx_clip, params->n_threads, fname.c_str());
//     if (!embeds) {
//         std::cerr << "error: failed to load image " << fname << ". Terminating\n\n";
//         return NULL;
//     }
//     std::cout<< "Start Processing Prompt: " << std::endl;
//     // TODO:
//     // process the prompt
//     if (params->prompt.empty() && params->interactive == false) {
//         LOG_TEE("prompt should be given or interactive mode should be on");
//         return NULL;
//     }

//     auto model = llava_init(params);
//     if (model == NULL) {
//         fprintf(stderr, "%s: error: failed to init minicpmv model\n", __func__);
//         return NULL;
//     }
//     const int64_t t_llava_init_start_us = ggml_time_us();
//     auto ctx_llava = llava_init_context(params, model);
//     ctx_llava->ctx_clip = ctx_clip;
//     const int64_t t_llava_init_end_us = ggml_time_us();
//     float t_llava_init_ms = (t_llava_init_end_us - t_llava_init_start_us) / 1000.0;
//     LOG_TEE("\n%s: llava init in %8.2f ms.\n", __func__, t_llava_init_ms);

//     const int64_t t_process_image_start_us = ggml_time_us();
//     process_prompt(ctx_llava, embeds, params, params->prompt);
//     // process_image(ctx_llava, embeds, params, n_past);
//     const int64_t t_process_image_end_us = ggml_time_us();
//     float t_process_image_ms = (t_process_image_end_us - t_process_image_start_us) / 1000.0;
//     LOG_TEE("\n%s: llama process image in %8.2f ms.\n", __func__, t_process_image_ms);

//     llava_image_embed_free(embeds);
//     return ctx_llava;
// }


// static struct llama_sampling_context * llama_init(struct llava_context * ctx_llava, gpt_params * params, std::string prompt, int &n_past, bool is_first = false){
//     std::string user_prompt = prompt;
//     int has_minicpmv_projector = clip_is_minicpmv(ctx_llava->ctx_clip);
//     if (!is_first) {
//         if (has_minicpmv_projector == 2) {
//             user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + prompt;
//         }
//         else if (has_minicpmv_projector == 3) {
//             user_prompt = "<|im_start|>user\n" + prompt;
//         }
//     }

//     eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);
//     if (has_minicpmv_projector == 2) {
//         eval_string(ctx_llava->ctx_llama, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", params->n_batch, &n_past, false);
//     }
//     else if (has_minicpmv_projector == 3) {
//         eval_string(ctx_llava->ctx_llama, "<|im_end|><|im_start|>assistant\n", params->n_batch, &n_past, false);
//     }

//     // generate the response

//     LOG_TEE("\n");

//     struct llama_sampling_context * ctx_sampling = llama_sampling_init(params->sparams);
//     return ctx_sampling;
// }

// static const char * llama_loop(struct llava_context * ctx_llava,struct llama_sampling_context * ctx_sampling, int &n_past){

//     const char * tmp = sample(ctx_sampling, ctx_llava->ctx_llama, &n_past);
//     return tmp;
// }

static void llama_log_callback_logTee(ggml_log_level level, const char *text, void *user_data)
{
    (void)level;
    (void)user_data;
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

    if (params.mmproj.empty() || (params.image.empty())) {
        gpt_params_print_usage(argc, argv, params);
        print_usage(argc, argv, params);
        return 1;
    }

    auto model = llava_init(&params);
    if (model == NULL)
    {
        fprintf(stderr, "%s: error: failed to init llava model\n", __func__);
        return 1;
    }

    if (prompt_contains_image(params.prompt))
    {
        auto ctx_llava = llava_init_context(&params, model);

        auto image_embed = load_image(ctx_llava, &params, "");

        // process the prompt
        process_prompt(ctx_llava, image_embed, &params, params.prompt);

        llama_print_timings(ctx_llava->ctx_llama);
        llava_image_embed_free(image_embed);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
    }
    else
    {
        for (auto &image : params.image)
        {
            printf("image: %s\n", image.c_str());
            auto ctx_llava = llava_init_context(&params, model);

            auto image_embed = load_image(ctx_llava, &params, image);
            printf("n_image_pos: %d\n", image_embed->n_image_pos);
            if (!image_embed)
            {
                std::cerr << "error: failed to load image " << image << ". Terminating\n\n";
                return 1;
            }

            // process the prompt
            process_prompt(ctx_llava, image_embed, &params, params.prompt);

            llama_print_timings(ctx_llava->ctx_llama);
            llava_image_embed_free(image_embed);
            ctx_llava->model = NULL;
            llava_free(ctx_llava);
        }
    }

    llama_free_model(model);

    // prompt_contains_image(params.prompt);
    // for (auto & image : params.image) {  // only single image for now
    //     int n_past = 0;
    //     auto ctx_llava = xgenmm_init(&params, image, n_past);  // generate vision tokens
    //     std::cout << "Start llava generation: " << std::endl;
    //     llama_print_timings(ctx_llava->ctx_llama);
    //     ctx_llava->model = NULL;
    //     llava_free(ctx_llava);
    // }
    printf("Remember to remove print_tensor function in xgenmm.cpp and clip.cpp\n");
    return 0;
}
