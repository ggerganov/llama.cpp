#include "ggml.h"
#include "log.h"
#include "common.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
}

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

struct clip_image_u8 {
    int nx;
    int ny;
    std::vector<uint8_t> buf;
};

static std::vector<clip_image_u8 *> extract_frames(const std::string& video_path, const int frame_num) {
    AVFormatContext* format_ctx = nullptr;
    if (avformat_open_input(&format_ctx, video_path.c_str(), nullptr, nullptr) < 0) {
        LOG_TEE("Could not open video file.");
        return {};
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        LOG_TEE("Could not find stream information.");
        avformat_close_input(&format_ctx);
        return {};
    }

    const AVCodec* codec = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    int video_stream_index = -1;

    for (size_t i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            codec = avcodec_find_decoder(format_ctx->streams[i]->codecpar->codec_id);
            if (codec) {
                video_stream_index = i;
                break;
            }
        }
    }

    if (video_stream_index == -1) {
        LOG_TEE("Could not find video stream.");
        avformat_close_input(&format_ctx);
        return {};
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        LOG_TEE("Could not allocate video codec context.");
        avformat_close_input(&format_ctx);
        return {};
    }

    if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[video_stream_index]->codecpar) < 0) {
        LOG_TEE("Could not copy codec parameters to codec context.");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return {};
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        LOG_TEE("Could not open codec.");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return {};
    }

    AVFrame* frame = av_frame_alloc();
    AVFrame* frame_rgb = av_frame_alloc();
    if (!frame || !frame_rgb) {
        LOG_TEE("Could not allocate frames.");
        av_frame_free(&frame);
        av_frame_free(&frame_rgb);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return {};
    }

    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, codec_ctx->width, codec_ctx->height, 1);
    uint8_t* buffer = (uint8_t*)av_malloc(num_bytes * sizeof(uint8_t));
    av_image_fill_arrays(frame_rgb->data, frame_rgb->linesize, buffer, AV_PIX_FMT_RGB24, codec_ctx->width, codec_ctx->height, 1);

    struct SwsContext* sws_ctx = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
                                                codec_ctx->width, codec_ctx->height, AV_PIX_FMT_RGB24,
                                                SWS_BILINEAR, nullptr, nullptr, nullptr);

    std::vector<clip_image_u8 *> frames;

    AVPacket packet;
    int64_t last_pts = AV_NOPTS_VALUE;
    int64_t total_frames = format_ctx->streams[video_stream_index]->nb_frames;
    // LOG_TEE("total_frames: %lld\n", total_frames);

    int64_t frame_interval = (int64_t)codec_ctx->framerate.num / codec_ctx->framerate.den;
    // LOG_TEE("frame_interval: %lld\n", frame_interval);
    // LOG_TEE("codec_ctx->framerate.num: %lld\n", codec_ctx->framerate.num);
    // LOG_TEE("codec_ctx->framerate.den: %lld\n", codec_ctx->framerate.den);

    float frame_len = 1.0 * total_frames / frame_interval;
    LOG_TEE("frame_len: %f\n", frame_len);
    if (frame_len > frame_num - 1) {
        frame_len = fmax(0, frame_num - 1);
        frame_interval = (int64_t)(1.0 * total_frames / frame_len);
    }
    // LOG_TEE("frame_interval: %lld\n", frame_interval);
    int frame_idx = 0;
    while (av_read_frame(format_ctx, &packet) >= 0) {
        if (packet.stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, &packet) == 0) {
                for(;avcodec_receive_frame(codec_ctx, frame) == 0;frame_idx++) {
                    // int frame_idx = frame->pts/codec_ctx->framerate.den;
                    // LOG_TEE("frame_idx: %d %d\n", frame_idx, frame_idx % frame_interval);
                    if (frame->pts != last_pts && (frame_idx) % frame_interval == 0) {
                        sws_scale(sws_ctx, frame->data, frame->linesize, 0, codec_ctx->height,
                                  frame_rgb->data, frame_rgb->linesize);

                        clip_image_u8 * img = clip_image_u8_init();
                        img->nx = codec_ctx->width;
                        img->ny = codec_ctx->height;
                        img->buf.resize(num_bytes);
                        std::copy(buffer, buffer + num_bytes, img->buf.begin());

                        frames.push_back(img);
                        last_pts = frame->pts;
                    }
                }
            }
        }
        av_packet_unref(&packet);
    }

    av_free(buffer);
    av_frame_free(&frame_rgb);
    av_frame_free(&frame);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);
    sws_freeContext(sws_ctx);

    return frames;
}

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG_TEE("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> [--video <path/to/an/video.mp4>] [--image <path/to/an/image.jpg>] [--image <path/to/another/image.jpg>] [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG_TEE("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

static struct clip_ctx * clip_init_context(gpt_params * params) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }
    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);
    return ctx_clip;
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

static struct llava_context * llava_init_context(gpt_params * params) {
    auto model = llava_init(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init minicpmv model\n", __func__);
        return NULL;
    }

    const char * clip_path = params->mmproj.c_str();
    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }
    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

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

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
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

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    return eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
}

static void process_eval_image_embed(struct llava_context * ctx_llava, const struct llava_image_embed * embeds, int n_batch, int * n_past, int idx) {
    float * image_embed = (float *)malloc(clip_embd_nbytes(ctx_llava->ctx_clip));
    std::memcpy(image_embed, embeds->embed + idx * clip_n_patches(ctx_llava->ctx_clip) * clip_n_mmproj_embd(ctx_llava->ctx_clip), clip_embd_nbytes(ctx_llava->ctx_clip));
    
    auto slice_embed = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    slice_embed->embed = image_embed;
    slice_embed->n_image_pos = clip_n_patches(ctx_llava->ctx_clip);
    llava_eval_image_embed(ctx_llava->ctx_llama, slice_embed, n_batch, n_past);
    llava_image_embed_free(slice_embed);
}

static int process_image(struct llava_context * ctx_llava, struct llava_image_embed * embeds, gpt_params * params, int &n_past) {
    std::string system_prompt;
    bool res = false;
    int idx = 0;
    int num_image_embeds = embeds->n_image_pos / clip_n_patches(ctx_llava->ctx_clip);    
    LOG_TEE("%s: image token past: %d\n", __func__, n_past);
    eval_string(ctx_llava->ctx_llama, (system_prompt+"<image>").c_str(), params->n_batch, &n_past, false);
    process_eval_image_embed(ctx_llava, embeds, params->n_batch, &n_past, idx++);
    res = eval_string(ctx_llava->ctx_llama, std::string("</image>").c_str(), params->n_batch, &n_past, false);
    if (num_image_embeds > 1) {
        size_t num_image_embeds_col = clip_uhd_num_image_embeds_col(ctx_llava->ctx_clip);
        eval_string(ctx_llava->ctx_llama, std::string("<slice>").c_str(), params->n_batch, &n_past, false);
        for (size_t i = 0; i < (num_image_embeds-1)/num_image_embeds_col; ++i) {
            for (size_t j = 0; j < num_image_embeds_col; ++j) {
                eval_string(ctx_llava->ctx_llama, std::string("<image>").c_str(), params->n_batch, &n_past, false);
                process_eval_image_embed(ctx_llava, embeds, params->n_batch, &n_past, idx++);
                eval_string(ctx_llava->ctx_llama, std::string("</image>").c_str(), params->n_batch, &n_past, false);
                if (j == num_image_embeds_col - 1) {
                    eval_string(ctx_llava->ctx_llama, std::string("\n").c_str(), params->n_batch, &n_past, false);
                }
            }
        }
        res = eval_string(ctx_llava->ctx_llama, std::string("</slice>").c_str(), params->n_batch, &n_past, false);
    }
    LOG_TEE("%s: image token past: %d\n", __func__, n_past);
    if(!res) return 0;
    return n_past;
}

static bool process_prompt(int type, struct llava_context * ctx_llava, gpt_params * params, int &n_past, std::string prompt = ""){
    int has_minicpmv_projector = clip_is_minicpmv(ctx_llava->ctx_clip);
    if (type==0) {
        std::string system_prompt;
        if (has_minicpmv_projector == 1) {
            system_prompt = "<用户>";
        }
        else if (has_minicpmv_projector == 2) {
            system_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n";
        }
        else if (has_minicpmv_projector == 3) {
            system_prompt = "<|im_start|>user\n";
        }
        return eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, false);
    }
    else if (type==1) {
        std::string user_prompt = prompt;
        return eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);
    }
    else if (type==2) {
        if (has_minicpmv_projector == 1) {
            return eval_string(ctx_llava->ctx_llama, "<AI>\n", params->n_batch, &n_past, false);
        }
        else if (has_minicpmv_projector == 2) {
            return eval_string(ctx_llava->ctx_llama, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", params->n_batch, &n_past, false);
        }
        else if (has_minicpmv_projector == 3) {
            return eval_string(ctx_llava->ctx_llama, "<|im_end|><|im_start|>assistant\n", params->n_batch, &n_past, false);
        }
    }
    return 0;
}

static struct llava_image_embed * video_image_embed(struct clip_ctx * ctx_clip, gpt_params * params, const clip_image_u8 * img){
    float* image_embed = NULL;
    int n_image_pos = 0;
    clip_uhd_max_slice_nums(ctx_clip, 2);
    bool image_embed_result = llava_image_embed_make_with_clip_img(ctx_clip, params->n_threads, img, &image_embed, &n_image_pos);
    if (!image_embed_result) {
        LOG_TEE("%s: coulnd't embed the image\n", __func__);
        return NULL;
    }

    auto result = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

static struct llava_image_embed * interleaved_image_embed(struct clip_ctx * ctx_clip, gpt_params * params, const std::string & fname){
    clip_uhd_max_slice_nums(ctx_clip, 9);
    llava_image_embed * embed = llava_image_embed_make_with_filename(ctx_clip, params->n_threads, fname.c_str());
    if (!embed) {
        LOG_TEE("error: failed to embed image Terminating\n\n");
        return NULL;
    }
    return embed;
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

static struct llava_context * minicpmv_init(gpt_params * params, const std::string & fname, int &n_past){
    auto ctx_clip = clip_init_context(params);
    auto embeds = llava_image_embed_make_with_filename(ctx_clip, params->n_threads, fname.c_str());
    if (!embeds) {
        LOG_TEE("error: failed to load image %s. Terminating\n\n", fname.c_str());
        return NULL;
    }

    // process the prompt
    if (params->prompt.empty() && params->interactive == false) {
        LOG_TEE("prompt should be given or interactive mode should be on");
        return NULL;
    }

    const int64_t t_llava_init_start_us = ggml_time_us();
    auto ctx_llava = llava_init_context(params);
    ctx_llava->ctx_clip = ctx_clip;
    const int64_t t_llava_init_end_us = ggml_time_us();
    float t_llava_init_ms = (t_llava_init_end_us - t_llava_init_start_us) / 1000.0;
    LOG_TEE("\n%s: llava init in %8.2f ms.\n", __func__, t_llava_init_ms);

    const int64_t t_process_image_start_us = ggml_time_us();
    process_prompt(0, ctx_llava, params, n_past);
    process_image(ctx_llava, embeds, params, n_past);
    const int64_t t_process_image_end_us = ggml_time_us();
    float t_process_image_ms = (t_process_image_end_us - t_process_image_start_us) / 1000.0;
    LOG_TEE("\n%s: llama process image in %8.2f ms.\n", __func__, t_process_image_ms);

    llava_image_embed_free(embeds);
    return ctx_llava;
}

static int process_input(struct llava_context * ctx_llava, gpt_params * params, int type, std::string prompt, int &n_past, struct llava_image_embed * embeds = nullptr){
    if (type==0) {
        if (process_prompt(1, ctx_llava, params, n_past, prompt)) return 1;
    }
    else if (type == 1) {
        if(embeds != NULL){
            return (process_image(ctx_llava, embeds, params, n_past));
        }
    }
    return 0;
}

static struct llama_sampling_context * llama_init(struct llava_context * ctx_llava, gpt_params * params, std::string prompt, int &n_past, bool is_first = false){
    std::string user_prompt = prompt;
    if(is_first)process_prompt(0, ctx_llava, params, n_past);
    process_prompt(1, ctx_llava, params, n_past, prompt);
    process_prompt(2, ctx_llava, params, n_past);

    // generate the response
    LOG_TEE("\n");
    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params->sparams);
    return ctx_sampling;
}

static const char * llama_loop(struct llava_context * ctx_llava,struct llama_sampling_context * ctx_sampling, int &n_past){
    
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

    if (params.mmproj.empty()) {
        gpt_params_print_usage(argc, argv, params);
        show_additional_info(argc, argv);
        return 1;
    }
    
    int n_past = 0;
    struct llava_context * ctx_llava = nullptr;

    if (params.video.size() > 0){
        ctx_llava = llava_init_context(&params);
        auto video = params.video;        
        std::vector<clip_image_u8 *> frames = extract_frames(video.c_str(), params.frame_num);
        process_prompt(0, ctx_llava, &params, n_past);
        // LOG_TEE("frames.size: %zu\n", frames.size());
        for (size_t i = 0; i < frames.size(); ++i) {
            auto embeds = video_image_embed(ctx_llava->ctx_clip, &params, frames[i]);
            process_input(ctx_llava, &params, 1, "", n_past, embeds);
        }
        process_input(ctx_llava, &params, 0, params.prompt.c_str(), n_past);
        process_prompt(2, ctx_llava, &params, n_past);
        if (!params.prompt.empty()) {
            LOG_TEE("minicpmv_version: %d\n", clip_is_minicpmv(ctx_llava->ctx_clip));
            LOG_TEE("<user>%s\n", params.prompt.c_str());
            LOG_TEE("<assistant>");
        }
        struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);
        const int max_tgt_len = params.n_predict < 0 ? 8192 : params.n_predict;
        std::string response = "";
        bool have_tmp = false;
        for (int i = 0; i < max_tgt_len; i++) {
            auto tmp = llama_loop(ctx_llava, ctx_sampling, n_past);
            response += tmp;
            if (strcmp(tmp, "</s>") == 0){
                if(!have_tmp)continue;
                else break;
            }
            have_tmp = true;
            printf("%s", tmp);
            if (strstr(response.c_str(), "<user>")) break; // minicpm-v 

            fflush(stdout);
        }
        llama_sampling_free(ctx_sampling);
    }
    else {
        if (params.image.size() > 1) {
            ctx_llava = llava_init_context(&params);
            process_prompt(0, ctx_llava, &params, n_past);
            for (auto & image : params.image) {
                auto embeds = interleaved_image_embed(ctx_llava->ctx_clip, &params, image);
                process_input(ctx_llava, &params, 1, "", n_past, embeds);
            }
            process_prompt(2, ctx_llava, &params, n_past);
            if (!params.prompt.empty()) {
                LOG_TEE("minicpmv_version: %d\n", clip_is_minicpmv(ctx_llava->ctx_clip));
                LOG_TEE("<user>%s\n", params.prompt.c_str());
                LOG_TEE("<assistant>");
            }
            struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);
            const int max_tgt_len = params.n_predict < 0 ? 8192 : params.n_predict;
            std::string response = "";
            bool have_tmp = false;
            for (int i = 0; i < max_tgt_len; i++) {
                auto tmp = llama_loop(ctx_llava, ctx_sampling, n_past);
                response += tmp;
                if (strcmp(tmp, "</s>") == 0){
                    if(!have_tmp)continue;
                    else break;
                }
                have_tmp = true;
                printf("%s", tmp);
                if (strstr(response.c_str(), "<user>")) break; // minicpm-v 

                fflush(stdout);
            }
            llama_sampling_free(ctx_sampling);
        }
        else {
            auto image = params.image[0];
            ctx_llava = minicpmv_init(&params, image, n_past);
            
            if (!params.prompt.empty()) {
                LOG_TEE("minicpmv_version: %d\n", clip_is_minicpmv(ctx_llava->ctx_clip));
                LOG_TEE("<user>%s\n", params.prompt.c_str());
                LOG_TEE("<assistant>");
                auto ctx_sampling = llama_init(ctx_llava, &params, params.prompt.c_str(), n_past, false);
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
                    have_tmp = true;
                    printf("%s", tmp);
                    if (strstr(response.c_str(), "<user>")) break; // minicpm-v 

                    fflush(stdout);
                }
                llama_sampling_free(ctx_sampling);
            }
            else {
                while (true) {
                    LOG_TEE("minicpmv_version: %d\n", clip_is_minicpmv(ctx_llava->ctx_clip));
                    LOG_TEE("<user>");
                    std::string prompt;
                    std::getline(std::cin, prompt);
                    LOG_TEE("<assistant>");
                    auto ctx_sampling = llama_init(ctx_llava, &params, prompt, n_past, false);
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
        }
        printf("\n");
        llama_print_timings(ctx_llava->ctx_llama);        

        ctx_llava->model = NULL;
        llava_free(ctx_llava);
    }

    return 0;
}