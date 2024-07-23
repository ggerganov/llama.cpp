#include "ggml.h"
#include "log.h"
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

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG_TEE("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG_TEE("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

static struct minicpmv_context * minicpmv_init(gpt_params * params, const std::string & fname, int &n_past){
    auto embeds = minicpmv_image_embed(params, fname);
    auto image_embed_slices = embeds->image_embeds;
    if (!image_embed_slices[0][0]) {
        std::cerr << "error: failed to load image " << fname << ". Terminating\n\n";
        return NULL;
    }

    // process the prompt
    if (params->prompt.empty() && params->interactive == false) {
        LOG_TEE("prompt should be given or interactive mode should be on");
        return NULL;
    }

    auto model = llava_init(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init minicpmv model\n", __func__);
        return NULL;
    }
    const int64_t t_llava_init_start_us = ggml_time_us();
    auto ctx_llava = llava_init_context(params, model);

    const int64_t t_llava_init_end_us = ggml_time_us();
    float t_llava_init_ms = (t_llava_init_end_us - t_llava_init_start_us) / 1000.0;
    LOG_TEE("\n%s: llava init in %8.2f ms.\n", __func__, t_llava_init_ms);

    const int64_t t_process_image_start_us = ggml_time_us();
    process_image(ctx_llava, embeds, params, n_past);
    const int64_t t_process_image_end_us = ggml_time_us();
    float t_process_image_ms = (t_process_image_end_us - t_process_image_start_us) / 1000.0;
    LOG_TEE("\n%s: llama process image in %8.2f ms.\n", __func__, t_process_image_ms);

    llava_image_embed_free_uhd(embeds);
    return ctx_llava;
}

static struct llama_sampling_context * llama_init(struct minicpmv_context * ctx_llava, gpt_params * params, std::string prompt, int &n_past, bool is_first = false){
    std::string user_prompt = prompt;
    if (!is_first) user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + prompt;

    eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);
    eval_string(ctx_llava->ctx_llama, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", params->n_batch, &n_past, false);
    // generate the response

    LOG_TEE("\n");

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params->sparams);
    return ctx_sampling;
}

static const char * llama_loop(struct minicpmv_context * ctx_llava,struct llama_sampling_context * ctx_sampling, int &n_past){
    
    const char * tmp = sample(ctx_sampling, ctx_llava->ctx_llama, &n_past);
    return tmp;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        show_additional_info(argc, argv);
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
        show_additional_info(argc, argv);
        return 1;
    }

    for (auto & image : params.image) {
        int n_past = 0;
        auto ctx_llava = minicpmv_init(&params, image, n_past);

        if (!params.prompt.empty()) {
            LOG_TEE("<user>%s\n", params.prompt.c_str());
            LOG_TEE("<assistant>");
            auto ctx_sampling = llama_init(ctx_llava, &params, params.prompt.c_str(), n_past, true);
            const int max_tgt_len = params.n_predict < 0 ? 256 : params.n_predict;
            std::string response = "";
            bool have_tmp = false;
            for (int i = 0; i < max_tgt_len; i++) {
                auto tmp = llama_loop(ctx_llava, ctx_sampling, n_past);
                response += tmp;
                if (strcmp(tmp, "</s>") == 0){
                    if(!have_tmp)continue;
                    else break;
                }
                if (strstr(tmp, "###")) break; // Yi-VL behavior
                have_tmp = true;
                printf("%s", tmp);
                if (strstr(response.c_str(), "<user>")) break; // minicpm-v 

                fflush(stdout);
            }
            llama_sampling_free(ctx_sampling);
        }else {
            while (true) {
                LOG_TEE("<user>");
                std::string prompt;
                std::getline(std::cin, prompt);
                LOG_TEE("<assistant>");
                auto ctx_sampling = llama_init(ctx_llava, &params, prompt, n_past, true);
                const int max_tgt_len = params.n_predict < 0 ? 256 : params.n_predict;
                std::string response = "";
                for (int i = 0; i < max_tgt_len; i++) {
                    auto tmp = llama_loop(ctx_llava, ctx_sampling, n_past);
                    response += tmp;
                    if (strcmp(tmp, "</s>") == 0) break;
                    if (strstr(tmp, "###")) break; // Yi-VL behavior
                    printf("%s", tmp);// mistral llava-1.6
                    if (strstr(response.c_str(), "<user>")) break; // minicpm-v 
                    fflush(stdout);
                }
                llama_sampling_free(ctx_sampling);
            }
        }
        printf("\n");
        llama_print_timings(ctx_llava->ctx_llama);        

        ctx_llava->model = NULL;
        llava_free(ctx_llava);
    }

    return 0;
}