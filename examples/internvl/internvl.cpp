#include "clip.h"
#include "ggml.h"
#include "common.h"
#include "llama.h"
#include "internvl.h"
#include "base64.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>

#include <cstdarg>
#include <climits>


// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

struct clip_image_grid_shape {
    int first;
    int second;
};


static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const bool *)data)[i] ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}

static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    std::string result;
    for (size_t pos = 0; ; pos += search.length()) {
        auto new_pos = s.find(search, pos);
        if (new_pos == std::string::npos) {
            result += s.substr(pos, s.size() - pos);
            break;
        }
        result += s.substr(pos, new_pos - pos) + replace;
        pos = new_pos;
    }
    s = std::move(result);
}

static std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replace_all(val, "\\", "\\\\");
                        replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

static void print_tensor_info(const ggml_tensor * tensor, const char * prefix = "") {
    size_t tensor_size = ggml_nbytes(tensor);
    printf("%s: n_dims = %d, name = %s, tensor_size=%zu, shape:[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], type = %s\n",
            prefix, ggml_n_dims(tensor), tensor->name, tensor_size,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], ggml_type_name(tensor->type));
}

/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static std::pair<int, int> select_best_resolution(const std::pair<int, int>& original_size, const std::vector<std::pair<int, int>>& possible_resolutions) {
    int original_width  = original_size.first;
    int original_height = original_size.second;

    std::pair<int, int> best_fit;
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width = resolution.first;
        int height = resolution.second;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width  = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // fprintf(stderr, "resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

/**
 * @brief Get the anyres image grid shape object
 *
 * @param image_size
 * @param grid_pinpoints
 * @param image_patch_size
 * @return <int, int>
 */
static struct clip_image_grid_shape get_anyres_image_grid_shape(const std::pair<int, int> & image_size, const std::vector<std::pair<int, int>> & grid_pinpoints, int image_patch_size) {
    /**
        Conversion from gguf flat array to vector:
        std::vector<std::pair<int, int>> possible_resolutions;
        for (int i = 0; i < 32 && params.image_grid_pinpoints[i] != 0; i+=2) {
            possible_resolutions.push_back({params.image_grid_pinpoints[i], params.image_grid_pinpoints[i+1]});
        }
     */
    auto best_resolution = select_best_resolution(image_size, grid_pinpoints);
    return {best_resolution.first / image_patch_size, best_resolution.second / image_patch_size};
}

static bool encode_image_with_clip(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float * image_embd, int * n_img_pos) {
    // std::vector<clip_image_f32*> img_res_v; // format VectN x H x W x RGB (N x 336 x 336 x 3), so interleaved RGB - different to the python implementation which is N x 3 x 336 x 336
    const int64_t t_img_process_start_us = ggml_time_us();

    clip_image_f32_batch img_res_v;
    img_res_v.size = 0;
    img_res_v.data = nullptr;
    const int64_t t_img_preprocess_start_us = ggml_time_us();
    if (!clip_image_preprocess(ctx_clip, img, img_res_v)) {
        fprintf(stderr, "%s: unable to preprocess image\n", __func__);
        delete[] img_res_v.data;
        return false;
    }

    const int64_t t_img_process_end_us = ggml_time_us();
    float t_img_process_cost_us = (t_img_process_end_us - t_img_process_start_us) / 1000.0;
    LOG_TEE("\n%s: image process in %8.2f ms\n", __func__, t_img_process_cost_us);


    const int64_t t_img_enc_start_us = ggml_time_us();

    const char * mm_patch_merge_type = clip_patch_merge_type(ctx_clip);

    // flat / default internvl-1.5 type embedding
    *n_img_pos = clip_n_patches(ctx_clip);
    bool encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[0], image_embd); // image_embd shape is 576 x 4096
    delete[] img_res_v.data;
    if (!encoded) {
        fprintf(stderr, "Unable to encode image\n");

        return false;
    }

    LOG_TEE("%s: image embedding created: %d tokens\n", __func__, *n_img_pos);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_preprocess_ms = (t_img_enc_start_us - t_img_preprocess_start_us) / 1000.0;
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;
    LOG_TEE("\n%s: image preprocessed in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_preprocess_ms, t_img_preprocess_ms / *n_img_pos);
    LOG_TEE("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / *n_img_pos);

    return true;
}

bool internvl_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip) {
        // make sure that the correct mmproj was used, i.e., compare apples to apples
    int n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));
    auto n_image_embd = clip_n_mmproj_embd(ctx_clip);
    if (n_image_embd != n_llama_embd) {
        printf("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_image_embd, n_llama_embd);
        return false;
    }
    return true;
}

bool internvl_image_embed_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out) {
    // allocate memory for out embeddings, default batch_size is 6, buff_size = 268 * 4096 * sizeof(float) * 6
    float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip)*6); // TODO: base on gridsize/internvl model
    if (!image_embd) {
        fprintf(stderr, "Unable to allocate memory for image embeddings\n");
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_clip(ctx_clip, n_threads, img, image_embd, &n_img_pos)) {
        fprintf(stderr, "%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}

bool internvl_eval_image_embed(llama_context * ctx_llama, const struct internvl_image_embed * image_embed, int n_batch, int * n_past) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < image_embed->n_image_pos; i += n_batch) {
        int n_eval = image_embed->n_image_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {int32_t(n_eval), nullptr, (image_embed->embed+i*n_embd), nullptr, nullptr, nullptr, nullptr, *n_past, 1, 0, };
        if (llama_decode(ctx_llama, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

struct internvl_image_embed * internvl_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length) {
    
    const int64_t t_img_dec_start_us = ggml_time_us();

    clip_image_u8 * img = clip_image_u8_init();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }


    const int64_t t_img_dec_end_us = ggml_time_us();
    float t_img_dec_ms = (t_img_dec_end_us - t_img_dec_start_us) / 1000.0;
    LOG_TEE("\n%s: image encoded in %8.2f ms\n", __func__, t_img_dec_ms);

    float* image_embed = NULL;
    int n_image_pos = 0;
    bool image_embed_result = internvl_image_embed_make_with_clip_img(ctx_clip, n_threads, img, &image_embed, &n_image_pos);
    if (!image_embed_result) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: coulnd't embed the image\n", __func__);
        return NULL;
    }

    clip_image_u8_free(img);
    auto result = (internvl_image_embed*)malloc(sizeof(internvl_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

static bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long *sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize); // Allocate memory to hold the file data
    if (buffer == NULL) {
        fprintf(stderr, "%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file); // Read the file into the buffer
    if (ferror(file)) {
        die_fmt("read error: %s", strerror(errno));
    }
    if (ret != (size_t) fileSize) {
        die("unexpectedly reached end of file");
    }
    fclose(file); // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}

struct internvl_image_embed * internvl_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path) {
    unsigned char* image_bytes;
    long image_bytes_length;

    const int64_t t_img_load_start_us = ggml_time_us();

    auto loaded = load_file_to_bytes(image_path, &image_bytes, &image_bytes_length);
    if (!loaded) {
        fprintf(stderr, "%s: failed to load %s\n", __func__, image_path);
        return NULL;
    }

    const int64_t t_img_load_end_us = ggml_time_us();
    float t_img_load_ms = (t_img_load_end_us - t_img_load_start_us) / 1000.0;
    LOG_TEE("\n%s: image loaded in %8.2f ms\n", __func__, t_img_load_ms);

    const int64_t t_img_enc_start_us = ggml_time_us();

    internvl_image_embed *embed = internvl_image_embed_make_with_bytes(ctx_clip, n_threads, image_bytes, image_bytes_length);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;
    LOG_TEE("\n%s: image encoded in %8.2f ms\n", __func__, t_img_enc_ms);


    free(image_bytes);

    return embed;
}

void internvl_image_embed_free(struct internvl_image_embed * embed) {
    free(embed->embed);
    free(embed);
}
