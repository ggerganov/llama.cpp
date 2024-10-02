#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>


static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    std::vector<llama_pos> pos;
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        auto batch = llama_batch_get_one(&tokens[i], n_eval, *n_past, 0);
        // TODO: add mrope pos ids somewhere else
        pos.resize(batch.n_tokens * 3);
        for (int j = 0; j < batch.n_tokens * 3; j ++) {
            pos[j] = j % batch.n_tokens;
        }
        batch.pos = pos.data();
        
        if (llama_decode(ctx_llama, batch)) {
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
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

static const char * sample(struct gpt_sampler * smpl,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = gpt_sampler_sample(smpl, ctx_llama, -1);
    gpt_sampler_accept(smpl, id, true);
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
static llava_image_embed * llava_image_embed_make_with_prompt_base64(struct clip_ctx * ctx_clip, int n_threads, const std::string& prompt) {
    size_t img_base64_str_start, img_base64_str_end;
    find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
    if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos) {
        LOG_ERR("%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
        return NULL;
    }

    auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
    auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
    auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count );

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

    auto embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
    if (!embed) {
        LOG_ERR("%s: could not load image from base64 string.\n", __func__);
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

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

static void print_usage(int, char ** argv) {
    LOG("\n example usage:\n");
    LOG("\n     %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static struct llava_image_embed * load_image(llava_context * ctx_llava, gpt_params * params, const std::string & fname) {

    // load and preprocess the image
    llava_image_embed * embed = NULL;
    auto prompt = params->prompt;
    if (prompt_contains_image(prompt)) {
        if (!params->image.empty()) {
            LOG_INF("using base64 encoded image instead of command line image path\n");
        }
        embed = llava_image_embed_make_with_prompt_base64(ctx_llava->ctx_clip, params->cpuparams.n_threads, prompt);
        if (!embed) {
            LOG_ERR("%s: can't load image from prompt\n", __func__);
            return NULL;
        }
        params->prompt = remove_image_from_prompt(prompt);
    } else {
        embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->cpuparams.n_threads, fname.c_str());
        if (!embed) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, fname.c_str());
            return NULL;
        }
    }

    return embed;
}

static void process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, gpt_params * params, const std::string & prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    std::string system_prompt, user_prompt;
    size_t image_pos = prompt.find("<image>");
    if (image_pos != std::string::npos) {
        // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
        system_prompt = prompt.substr(0, image_pos);
        user_prompt = prompt.substr(image_pos + std::string("<image>").length());
        LOG_INF("system_prompt: %s\n", system_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
        LOG_INF("user_prompt: %s\n", user_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    } else {
        // llava-1.5 native mode
        system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
        user_prompt = prompt + "\nASSISTANT:";
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    }

    eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, true);
    if (image_embed != nullptr)
        llava_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past);
    eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);

    // generate the response

    LOG("\n");

    struct gpt_sampler * smpl = gpt_sampler_init(ctx_llava->model, params->sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(smpl, ctx_llava->ctx_llama, &n_past);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) break;
        if (strstr(tmp, "###")) break; // Yi-VL behavior
        LOG("%s", tmp);
        if (strstr(response.c_str(), "<|im_end|>")) break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        if (strstr(response.c_str(), "<|im_start|>")) break; // Yi-34B llava-1.6
        if (strstr(response.c_str(), "USER:")) break; // mistral llava-1.6

        fflush(stdout);
    }

    gpt_sampler_free(smpl);
    LOG("\n");
}

static struct llama_model * llava_init(gpt_params * params) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

static struct llava_context * llava_init_context(gpt_params * params, llama_model * model) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);


    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        LOG_ERR("%s: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto * ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

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

static void tmp_test_conv2d_reshape(struct llava_context * ctx_llava, gpt_params * params) {
    int image_size_width = 256;
    int image_size_height = 256;
    int batch_size = 1;

    static size_t buf_size = 512u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params init_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(init_params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size_width, image_size_height, 3, batch_size);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    auto image_pixels = batch_size * image_size_width * image_size_height * 3;
    auto one_ch = image_size_width * image_size_height;
    std::vector<float> dummy_img;
    dummy_img.resize(image_pixels);
    std::fill(dummy_img.begin(), dummy_img.begin() + one_ch, 0.1);
    std::fill(dummy_img.begin() + one_ch, dummy_img.begin() + one_ch * 2, 0.2);
    std::fill(dummy_img.begin() + one_ch * 2, dummy_img.end(), 0.3);
    memcpy(inp_raw->data, dummy_img.data(), image_pixels * ggml_element_size(inp_raw));

    int patch_size = 14;
    int hidden_size = 32;
    int patch_w = image_size_width / patch_size;
    int patch_h = image_size_height / patch_size;
    int num_patches = (image_size_width / patch_size) * (image_size_height / patch_size);
    struct ggml_tensor * kernel_0 = ggml_new_tensor_4d(
        ctx0, GGML_TYPE_F32, 
        patch_size, patch_size, 3, hidden_size);
    ggml_set_name(kernel_0, "conv2d_kernel_0");
    ggml_set_input(kernel_0);

    auto kernel_ne = patch_size * patch_size * 3 * hidden_size;
    std::vector<float> dummy_kernel;
    dummy_kernel.resize(kernel_ne);
    std::fill(dummy_kernel.begin(), dummy_kernel.end(), 0.0);
    memcpy(kernel_0->data, dummy_img.data(), kernel_ne * ggml_element_size(kernel_0));

    struct ggml_tensor * inp = ggml_conv_2d(ctx0, kernel_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    // inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
    // inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));  // swap axis 0 & 1, ignore axis 3 which is empty in this tensor
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));  // [w, h, c, b] -> [c, w, h, b]
    inp = ggml_reshape_4d(
        ctx0, inp, 
        hidden_size * 2, patch_w / 2, patch_h, batch_size);
    inp = ggml_reshape_4d(
        ctx0, inp, 
        hidden_size * 2, patch_w / 2, 2, batch_size * (patch_h / 2));
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 0, 2, 1, 3));
    inp = ggml_reshape_2d(
        ctx0, inp, 
        hidden_size * 4, (patch_w / 2) * batch_size * (patch_h / 2));
    
    ggml_build_forward_expand(gf, inp);
    ggml_graph_compute_with_ctx(ctx0, gf, 2);

    std::vector<float> embd;
    embd.resize(num_patches * hidden_size * batch_size);
    memcpy(
        embd.data(), 
        (float *) ggml_get_data(inp), 
        sizeof(float) * num_patches * hidden_size * batch_size);
    ggml_free(ctx0);

    std::ofstream outFile("conv2d.bin", std::ios::binary);
    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(embd.data()), embd.size() * sizeof(int));

        outFile.close();
        std::cout << "Data successfully written to conv2d.bin" << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}


static void tmp_test_4d_reshape(struct llava_context * ctx_llava, gpt_params * params) {
    int image_size_width = 32;
    int image_size_height = 32;
    int batch_size = 1;

    static size_t buf_size = 512u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params init_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(init_params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_raw = ggml_new_tensor_4d(
        ctx0, GGML_TYPE_F32, image_size_width, image_size_height, 8, batch_size);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    auto image_pixels = batch_size * image_size_width * image_size_height * 8;
    auto one_ch = image_size_width * image_size_height;
    std::vector<float> dummy_img;
    dummy_img.resize(image_pixels);
    for (int i = 0; i < 8; i++)
    {
        // std::fill(
        //     dummy_img.begin() + one_ch * i, 
        //     dummy_img.begin() + one_ch * (i + 1), 
        //     0.1 * i
        // );
        for (size_t y = 0; y < image_size_height; y++)
        {
            for (size_t x = 0; x < image_size_width; x++)
            {
                dummy_img[one_ch * i + image_size_width * y + x] = i * (image_size_width * y + x) / (float)(32 * 32);
            }
            
        }
        
    }    
    memcpy(inp_raw->data, dummy_img.data(), image_pixels * ggml_element_size(inp_raw));

    int patch_size = 1;
    int hidden_size = 8;
    int patch_w = image_size_width / patch_size;
    int patch_h = image_size_height / patch_size;
    int num_patches = (image_size_width / patch_size) * (image_size_height / patch_size);

    // inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
    // inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));  // swap axis 0 & 1, ignore axis 3 which is empty in this tensor
    // auto inp = ggml_cont(ctx0, ggml_permute(ctx0, inp_raw, 2, 0, 1, 3));  // [w, h, c, b] -> [c, w, h, b]
    auto inp = ggml_cont(ctx0, ggml_permute(ctx0, inp_raw, 1, 2, 0, 3));  // [w, h, c, b] -> [c, w, h, b] [(0-->1), (1-->2), (2-->0), (3-->3)]
    inp = ggml_reshape_4d(
        ctx0, inp, 
        hidden_size * 2, patch_w / 2, patch_h, batch_size);
    inp = ggml_reshape_4d(
        ctx0, inp, 
        hidden_size * 2, patch_w / 2, 2, batch_size * (patch_h / 2));
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 0, 2, 1, 3));
    inp = ggml_reshape_2d(
        ctx0, inp, 
        hidden_size * 4, (patch_w / 2) * batch_size * (patch_h / 2));
    
    ggml_build_forward_expand(gf, inp);
    ggml_graph_compute_with_ctx(ctx0, gf, 2);

    std::vector<float> embd;
    embd.resize(num_patches * hidden_size * batch_size);
    memcpy(
        embd.data(), 
        (float *) ggml_get_data(inp), 
        sizeof(float) * num_patches * hidden_size * batch_size);
    ggml_free(ctx0);

    std::ofstream outFile("reshape_4d.bin", std::ios::binary);
    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(embd.data()), embd.size() * sizeof(int));

        outFile.close();
        std::cout << "Data successfully written to reshape_4d.bin" << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}


static void tmp_test_rope(struct llava_context * ctx_llava, gpt_params * params) {
    
    int n_threads = 1;
    static size_t buf_size = 512u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params init_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(init_params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 128, 12, 30);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    std::vector<float> dummy_q;
    dummy_q.resize(128 * 12 * 30);
    std::fill(dummy_q.begin(), dummy_q.end(), 0.1);
    memcpy(inp_raw->data, dummy_q.data(), 128 * 12 * 30 * ggml_element_size(inp_raw));

    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 30);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    std::vector<int> pos_id;
    pos_id.resize(30);
    for (int i = 0; i < 30; i ++) pos_id[i] = i;
    memcpy(pos->data, pos_id.data(), (30) * ggml_element_size(pos));

    auto encode = ggml_rope_ext(
        ctx0, inp_raw, pos, nullptr,
        128, LLAMA_ROPE_TYPE_NEOX, 32768, 1000000, 1,
        0, 1, 32, 1);
    
    ggml_build_forward_expand(gf, encode);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    std::vector<float> embd;
    embd.resize(128 * 12 * 30);
    memcpy(
        embd.data(), 
        (float *) ggml_get_data(encode), 
        sizeof(float) * 128 * 12 * 30);
    ggml_free(ctx0);


    // Open a binary file for writing
    std::ofstream outFile("rope.bin", std::ios::binary);
    // Check if file is open
    if (outFile.is_open()) {
        // Write the vector to the file
        outFile.write(reinterpret_cast<const char*>(embd.data()), embd.size() * sizeof(int));

        // Close the file
        outFile.close();
        std::cout << "Data successfully written to output.bin" << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}


static void tmp_test_mrope(struct llava_context * ctx_llava, gpt_params * params) {
    
    int n_threads = 1;
    static size_t buf_size = 512u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params init_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(init_params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 128, 12, 30);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    std::vector<float> dummy_q;
    dummy_q.resize(128 * 12 * 30);
    std::fill(dummy_q.begin(), dummy_q.end(), 0.1);
    memcpy(inp_raw->data, dummy_q.data(), 128 * 12 * 30 * ggml_element_size(inp_raw));

    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 30 * 3);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    std::vector<int> pos_id;
    pos_id.resize(90);
    for (int i = 0; i < 30; i ++) pos_id[i] = i;
    for (int i = 30; i < 60; i ++) pos_id[i] = i - 0;
    for (int i = 60; i < 90; i ++) pos_id[i] = i - 0;
    memcpy(pos->data, pos_id.data(), 90 * ggml_element_size(pos));

    int sections[3] = {16, 24, 24};
    auto encode = ggml_mrope_ext(
        ctx0, inp_raw, pos, nullptr,
        128, sections, LLAMA_ROPE_TYPE_NEOX, 32768, 1000000, 1,
        0, 1, 32, 1);
    
    ggml_build_forward_expand(gf, encode);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    std::vector<float> embd;
    embd.resize(128 * 12 * 30);
    memcpy(
        embd.data(), 
        (float *) ggml_get_data(encode), 
        sizeof(float) * 128 * 12 * 30);
    ggml_free(ctx0);

    std::ofstream outFile("mrope.bin", std::ios::binary);
    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(embd.data()), embd.size() * sizeof(int));

        outFile.close();
        std::cout << "Data successfully written to mrope.bin" << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}


static void tmp_test_mrope_2d(struct llava_context * ctx_llava, gpt_params * params) {
    
    int n_threads = 1;
    static size_t buf_size = 512u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params init_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(init_params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 128, 12, 30);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    std::vector<float> dummy_q;
    dummy_q.resize(128 * 12 * 30);
    std::fill(dummy_q.begin(), dummy_q.end(), 0.1);
    memcpy(inp_raw->data, dummy_q.data(), 128 * 12 * 30 * ggml_element_size(inp_raw));

    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 30 * 3);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    std::vector<int> pos_id;
    pos_id.resize(90);
    for (int i = 0; i < 30; i ++) pos_id[i] = i;
    for (int i = 30; i < 60; i ++) pos_id[i] = i - 30;
    for (int i = 60; i < 90; i ++) pos_id[i] = i - 0;
    memcpy(pos->data, pos_id.data(), 90 * ggml_element_size(pos));

    int sections[3] = {32, 32, 0};
    auto encode = ggml_mrope_ext(
        ctx0, inp_raw, pos, nullptr,
        128/2, sections, LLAMA_ROPE_TYPE_NEOX, 32768, 1000000, 1,
        0, 1, 32, 1);
    
    ggml_build_forward_expand(gf, encode);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    std::vector<float> embd;
    embd.resize(128 * 12 * 30);
    memcpy(
        embd.data(), 
        (float *) ggml_get_data(encode), 
        sizeof(float) * 128 * 12 * 30);
    ggml_free(ctx0);

    std::ofstream outFile("mrope_2d.bin", std::ios::binary);
    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(embd.data()), embd.size() * sizeof(int));

        outFile.close();
        std::cout << "Data successfully written to mrope.bin" << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}

/*
    -----------------------------------------------------------------------------------------------------------------
*/

int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, print_usage)) {
        return 1;
    }

    gpt_init();

    if (params.mmproj.empty() || (params.image.empty() && !prompt_contains_image(params.prompt))) {
        print_usage(argc, argv);
        return 1;
    }

    auto * model = llava_init(&params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init llava model\n", __func__);
        return 1;
    }

    if (prompt_contains_image(params.prompt)) {
        auto * ctx_llava = llava_init_context(&params, model);

        auto * image_embed = load_image(ctx_llava, &params, "");

        // process the prompt
        process_prompt(ctx_llava, image_embed, &params, params.prompt);

        llama_perf_context_print(ctx_llava->ctx_llama);
        llava_image_embed_free(image_embed);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
    } else if (params.image.empty() | true) {
        // This section is for testing LLM parts of the model during development phase!
        auto ctx_llava = llava_init_context(&params, model);

        // process the prompt
        tmp_test_4d_reshape(ctx_llava, &params);
        // tmp_test_rope(ctx_llava, &params);
        // tmp_test_mrope(ctx_llava, &params);
        // tmp_test_mrope_2d(ctx_llava, &params);
        // process_prompt(ctx_llava, nullptr, &params, params.prompt);

        llama_perf_context_print(ctx_llava->ctx_llama);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
        
    } else {
        for (auto & image : params.image) {
            auto * ctx_llava = llava_init_context(&params, model);

            auto * image_embed = load_image(ctx_llava, &params, image);
            if (!image_embed) {
                LOG_ERR("%s: failed to load image %s. Terminating\n\n", __func__, image.c_str());
                return 1;
            }

            // process the prompt
            process_prompt(ctx_llava, image_embed, &params, params.prompt);

            llama_perf_context_print(ctx_llava->ctx_llama);
            llava_image_embed_free(image_embed);
            ctx_llava->model = NULL;
            llava_free(ctx_llava);
        }
    }

    llama_free_model(model);

    return 0;
}
