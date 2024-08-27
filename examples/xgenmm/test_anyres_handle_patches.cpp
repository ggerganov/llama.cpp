#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "clip.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "xgenmm.h"
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <cstdlib>
#include <memory>
#include <string>

template <class T>
std::string type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void (*)(void*)>          own(
#ifndef _MSC_VER
        abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
#else
        nullptr,
#endif
        std::free);
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value) r += " const";
    if (std::is_volatile<TR>::value) r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

struct clip_image_u8
{
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

struct clip_image_f32
{
    int nx;
    int ny;

    std::vector<float> buf;
};

inline int  clip(int x, int lower, int upper) { return std::max(lower, std::min(x, upper)); }

static bool bicubic_resize(const clip_image_u8& img, clip_image_u8& dst, int target_width, int target_height)
{
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int   i, j, k, jj;
    int   x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)target_width;
    ty = (float)ny / (float)target_height;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (i = 0; i < target_height; i++)
    {
        for (j = 0; j < target_width; j++)
        {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            for (k = 0; k < 3; k++)
            {
                for (jj = 0; jj <= 3; jj++)
                {
                    d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] -
                         img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] -
                         img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] -
                         img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;

                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                }
            }
        }
    }

    return true;
}

enum projector_type
{
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_MLP_NORM,
    PROJECTOR_TYPE_LDP,
    PROJECTOR_TYPE_LDPV2,
    PROJECTOR_TYPE_RESAMPLER,
    PROJECTOR_TYPE_UNKNOWN,
};

static std::map<projector_type, std::string> PROJECTOR_TYPE_NAMES = {
    {PROJECTOR_TYPE_MLP, "mlp"},
    {PROJECTOR_TYPE_LDP, "ldp"},
    {PROJECTOR_TYPE_LDPV2, "ldpv2"},
    {PROJECTOR_TYPE_RESAMPLER, "resampler"},
};



struct clip_hparams
{
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;

    float eps;

    char mm_patch_merge_type[32] = "flat";  // spatial_unpad or flat (default)

    int32_t image_grid_pinpoints[32];
    int32_t image_crop_resolution;
};

struct clip_layer
{
    // attention
    struct ggml_tensor* k_w;
    struct ggml_tensor* k_b;
    struct ggml_tensor* q_w;
    struct ggml_tensor* q_b;
    struct ggml_tensor* v_w;
    struct ggml_tensor* v_b;

    struct ggml_tensor* o_w;
    struct ggml_tensor* o_b;

    // layernorm 1
    struct ggml_tensor* ln_1_w;
    struct ggml_tensor* ln_1_b;

    // ff
    struct ggml_tensor* ff_i_w;
    struct ggml_tensor* ff_i_b;

    struct ggml_tensor* ff_o_w;
    struct ggml_tensor* ff_o_b;

    // layernorm 2
    struct ggml_tensor* ln_2_w;
    struct ggml_tensor* ln_2_b;
};

struct clip_vision_model
{
    struct clip_hparams hparams;

    // embeddings
    struct ggml_tensor* class_embedding;
    struct ggml_tensor* patch_embeddings;
    struct ggml_tensor* patch_bias;
    struct ggml_tensor* position_embeddings;

    struct ggml_tensor* pre_ln_w;
    struct ggml_tensor* pre_ln_b;

    std::vector<clip_layer> layers;

    struct ggml_tensor* post_ln_w;
    struct ggml_tensor* post_ln_b;

    struct ggml_tensor* projection;

    // LLaVA projection
    struct ggml_tensor* mm_0_w = NULL;
    struct ggml_tensor* mm_0_b = NULL;
    struct ggml_tensor* mm_2_w = NULL;
    struct ggml_tensor* mm_2_b = NULL;

    struct ggml_tensor* image_newline = NULL;

    // Yi type models with mlp+normalization projection
    struct ggml_tensor* mm_1_w = NULL;  // Yi type models have 0, 1, 3, 4
    struct ggml_tensor* mm_1_b = NULL;
    struct ggml_tensor* mm_3_w = NULL;
    struct ggml_tensor* mm_3_b = NULL;
    struct ggml_tensor* mm_4_w = NULL;
    struct ggml_tensor* mm_4_b = NULL;

    // MobileVLM projection
    struct ggml_tensor* mm_model_mlp_1_w;
    struct ggml_tensor* mm_model_mlp_1_b;
    struct ggml_tensor* mm_model_mlp_3_w;
    struct ggml_tensor* mm_model_mlp_3_b;
    struct ggml_tensor* mm_model_block_1_block_0_0_w;
    struct ggml_tensor* mm_model_block_1_block_0_1_w;
    struct ggml_tensor* mm_model_block_1_block_0_1_b;
    struct ggml_tensor* mm_model_block_1_block_1_fc1_w;
    struct ggml_tensor* mm_model_block_1_block_1_fc1_b;
    struct ggml_tensor* mm_model_block_1_block_1_fc2_w;
    struct ggml_tensor* mm_model_block_1_block_1_fc2_b;
    struct ggml_tensor* mm_model_block_1_block_2_0_w;
    struct ggml_tensor* mm_model_block_1_block_2_1_w;
    struct ggml_tensor* mm_model_block_1_block_2_1_b;
    struct ggml_tensor* mm_model_block_2_block_0_0_w;
    struct ggml_tensor* mm_model_block_2_block_0_1_w;
    struct ggml_tensor* mm_model_block_2_block_0_1_b;
    struct ggml_tensor* mm_model_block_2_block_1_fc1_w;
    struct ggml_tensor* mm_model_block_2_block_1_fc1_b;
    struct ggml_tensor* mm_model_block_2_block_1_fc2_w;
    struct ggml_tensor* mm_model_block_2_block_1_fc2_b;
    struct ggml_tensor* mm_model_block_2_block_2_0_w;
    struct ggml_tensor* mm_model_block_2_block_2_1_w;
    struct ggml_tensor* mm_model_block_2_block_2_1_b;

    // MobileVLM_V2 projection
    struct ggml_tensor* mm_model_mlp_0_w;
    struct ggml_tensor* mm_model_mlp_0_b;
    struct ggml_tensor* mm_model_mlp_2_w;
    struct ggml_tensor* mm_model_mlp_2_b;
    struct ggml_tensor* mm_model_peg_0_w;
    struct ggml_tensor* mm_model_peg_0_b;

    // MINICPMV projection
    struct ggml_tensor* mm_model_pos_embed_k;
    struct ggml_tensor* mm_model_query;
    struct ggml_tensor* mm_model_proj;
    struct ggml_tensor* mm_model_kv_proj;
    struct ggml_tensor* mm_model_attn_q_w;
    struct ggml_tensor* mm_model_attn_q_b;
    struct ggml_tensor* mm_model_attn_k_w;
    struct ggml_tensor* mm_model_attn_k_b;
    struct ggml_tensor* mm_model_attn_v_w;
    struct ggml_tensor* mm_model_attn_v_b;
    struct ggml_tensor* mm_model_attn_o_w;
    struct ggml_tensor* mm_model_attn_o_b;
    struct ggml_tensor* mm_model_ln_q_w;
    struct ggml_tensor* mm_model_ln_q_b;
    struct ggml_tensor* mm_model_ln_kv_w;
    struct ggml_tensor* mm_model_ln_kv_b;
    struct ggml_tensor* mm_model_ln_post_w;
    struct ggml_tensor* mm_model_ln_post_b;
};

struct clip_ctx
{
    bool has_text_encoder = false;
    bool has_vision_encoder = false;
    bool has_llava_projector = false;
    bool has_minicpmv_projector = false;
    bool has_xgenmm_projector = false;
    int  minicpmv_version = 2;

    struct clip_vision_model vision_model;
    projector_type           proj_type = PROJECTOR_TYPE_MLP;

    float   image_mean[3];
    float   image_std[3];
    bool    use_gelu = false;
    int32_t ftype = 1;

    bool has_class_embedding = true;
    bool has_pre_norm = true;
    bool has_post_norm = false;
    bool has_patch_bias = false;

    struct gguf_context* ctx_gguf;
    struct ggml_context* ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    // memory buffers to evaluate the model
    ggml_backend_buffer_t params_buffer = NULL;

    ggml_backend_t backend = NULL;
    ggml_gallocr_t compute_alloc = NULL;

    struct clip_image_size* load_image_size;
};

static bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long* sizeOut)
{
    auto file = fopen(path, "rb");
    if (file == NULL)
    {
        LOG_TEE("%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char*)malloc(fileSize);  // Allocate memory to hold the file data
    if (buffer == NULL)
    {
        LOG_TEE("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file);  // Read the file into the buffer
    if (ferror(file))
    {
        die_fmt("read error: %s", strerror(errno));
    }
    if (ret != (size_t)fileSize)
    {
        die("unexpectedly reached end of file");
    }
    fclose(file);  // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}

void print_img(clip_image_u8* img)
{
    const int nx = img->nx;
    const int ny = img->ny;
    printf("num pixels: %d\n", img->buf.size());
    printf("raw img: nx:%d | ny:%d\n", nx, ny);

    const int n = nx * ny;
    for (int k = 0; k < 3; k++)
    {
        for (int y = 0; y < 5; y++)
        {
            for (int x = 0; x < 10; x++)
            {
                // data[(i * 3 * n) + k * n + y * nx + x] = imgs->data[i].buf[3 * (y * nx + x) + k];
                printf("%d ", img->buf[3 * (y * nx + x) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void img_to_csv(clip_image_u8* img, const char* filename)
{
        std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
    }
    const int nx = img->nx;
    const int ny = img->ny;

    for (int k = 0; k < 3; k++)
    {
        for (int y = 0; y < ny; y++)
        {
            for (int x = 0; x < nx; x++)
            {
                outFile << int(img->buf[3 * (y * nx + x) + k]);
                if (x < nx - 1)
                {
                    outFile << ",";
                }
            }
            outFile << std::endl;
        }
        outFile << std::endl;
    }

    outFile.close();
    printf("file saved to %s\n", filename);
}

void tensor_to_csv(clip_image_f32* img, const char* filename)
{
    
    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
    }
    const int nx = img->nx;
    const int ny = img->ny;

    for (int k = 0; k < 3; k++)
    {
        for (int y = 0; y < ny; y++)
        {
            for (int x = 0; x < nx; x++)
            {
                outFile << float(img->buf[3 * (y * nx + x) + k]);
                if (x < nx - 1)
                {
                    outFile << ",";
                }
            }
            outFile << std::endl;
        }
        outFile << std::endl;
    }

    outFile.close();
    printf("file saved to %s\n", filename);
}

struct clip_image_grid_shape
{
    int first;
    int second;
};

static std::pair<int, int> select_best_resolution(const std::pair<int, int>&              original_size,
                                                  const std::vector<std::pair<int, int>>& possible_resolutions)
{
    int original_width = original_size.first;
    int original_height = original_size.second;

    std::pair<int, int> best_fit;
    int                 max_effective_resolution = 0;
    int                 min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions)
    {
        int   width = resolution.first;
        int   height = resolution.second;
        float scale =
            std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // LOG_TEE("resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale,
        // downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution ||
            (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution))
        {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

static struct clip_image_grid_shape get_anyres_image_grid_shape(const std::pair<int, int>&              image_size,
                                                                const std::vector<std::pair<int, int>>& grid_pinpoints,
                                                                int image_patch_size)
{
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

int main(){


    const char*      clip_path = "/export/share/yutong/xgenmm/llamacpp_wd/llava-1.6/vit/mmproj-model-f16.gguf";
    struct clip_ctx * ctx = clip_model_load(clip_path, /*verbosity=*/2);
    printf("Model loaded\n");
    for (int i=0; i < 3; i++){
        ctx->image_mean[i] = 0.5;
        ctx->image_std[i] = 0.5;
    }
    LOG_TEE("v_image_mean       %f %f %f\n", ctx->image_mean[0], ctx->image_mean[1], ctx->image_mean[2]);
    LOG_TEE("v_image_std        %f %f %f\n", ctx->image_std[0], ctx->image_std[1], ctx->image_std[2]);
    // [[384, 768], [768, 384], [768, 768], [1152, 384], [384, 1152]]
    ctx->vision_model.hparams.image_grid_pinpoints[0] = 384;
    ctx->vision_model.hparams.image_grid_pinpoints[1] = 768;
    ctx->vision_model.hparams.image_grid_pinpoints[2] = 768;
    ctx->vision_model.hparams.image_grid_pinpoints[3] = 384;
    ctx->vision_model.hparams.image_grid_pinpoints[4] = 768;
    ctx->vision_model.hparams.image_grid_pinpoints[5] = 768;
    ctx->vision_model.hparams.image_grid_pinpoints[6] = 1152;
    ctx->vision_model.hparams.image_grid_pinpoints[7] = 384;
    ctx->vision_model.hparams.image_grid_pinpoints[8] = 384;
    ctx->vision_model.hparams.image_grid_pinpoints[9] = 1152;
    for (int i = 0; i < 10; i++)
    {
        printf("grid[%d]:%d ", i, ctx->vision_model.hparams.image_grid_pinpoints[i]);
    }
    printf("\n");
    ctx->vision_model.hparams.image_size = 384;
    printf("in test_anyres: params.image_size:%d\n", ctx->vision_model.hparams.image_size);
    /* 
        part of: 
            llava_image_embed_make_with_filename
    */
    const char*    image_path = "/export/home/llama.cpp/examples/xgenmm/imgs/image-1d100e9.jpg";  // Porcelain
    // const char*    image_path = "/export/home/llama.cpp/examples/xgenmm/imgs/image-1d100e9-1.jpg";
    unsigned char* image_bytes;
    long           image_bytes_length;
    auto           loaded = load_file_to_bytes(image_path, &image_bytes, &image_bytes_length);
    if (!loaded)
    {
        LOG_TEE("%s: failed to load %s\n", __func__, image_path);
        return NULL;
    }

    /*
    part of:
        llava_image_embed_make_with_bytes
    */
    clip_image_u8* img = clip_image_u8_init();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img))
    {
        clip_image_u8_free(img);
        LOG_TEE("%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }


    /*
        part of:
        encode_image_with_clip
    */
    clip_image_f32_batch img_res_v;
    img_res_v.size = 0;
    img_res_v.data = nullptr;
    if (!clip_image_preprocess(ctx, img, &img_res_v))
    {
        LOG_TEE("%s: unable to preprocess image\n", __func__);
        delete[] img_res_v.data;
        return false;
    }
    printf("img->nx:%ld | img->ny:%ld\n", img->nx, img->ny);
    printf("Bacth size: img_res_v.size:%ld\n", img_res_v.size);

    // std::cout << "decltype(img_res_v.data) is " << type_name<decltype(img_res_v.data)>() << '\n';

    // printf("Image Dimension in this batch: img_res_v.data->nx:%ld | img_res_v.data->nx:%ld\n", img_res_v.data->nx,
    //        img_res_v.data->ny);
    // printf("img_res_v.data->buf.size():%ld\n", img_res_v.data->buf.size());

    
    // std::cout << "decltype(img_res_v.data[0]) is " << type_name<decltype(img_res_v.data[0])>() << '\n';
    // std::cout << "decltype(img_res_v.data[0].buf[0]) is " << type_name<decltype(img_res_v.data[0].buf[0])>() << '\n';
    // for (size_t i = 0; i < img_res_v.size; i++) {
    //     const int nx = img_res_v.data[i].nx;
    //     const int ny = img_res_v.data[i].ny;
    //     const int vec_len = img_res_v.data[i].buf.size();
    //     printf("i:%d | nx:%d | ny:%d | vec len:%d\n", i, nx, ny, vec_len);
    // }

    const char* mm_patch_merge_type = clip_patch_merge_type(ctx);
    printf("mm_patch_merge_type:%s\n", mm_patch_merge_type);

    struct clip_ctx* ctx_clip = ctx;
    const int32_t* image_grid = clip_image_grid(ctx_clip);

    std::vector<std::pair<int, int>> grid_pinpoints;
    for (int i = 0; i < 32 && image_grid[i] != 0; i += 2)
    {
        grid_pinpoints.push_back({image_grid[i], image_grid[i + 1]});
    }
    for (const auto& point : grid_pinpoints)
    {
        std::cout << "(" << point.first << ", " << point.second << ")" << std::endl;
    }

    const int32_t image_size = clip_image_size(ctx_clip);
    printf("image_size:%d\n", image_size);

    struct clip_image_grid_shape grid_shape =
        get_anyres_image_grid_shape({img->nx, img->ny}, grid_pinpoints, image_size);

    printf("grid_shape.first:%d | grid_shape.second:%d\n", grid_shape.first, grid_shape.second);

    std::vector<float*> image_embd_v;
    image_embd_v.resize(img_res_v.size);
    printf("image_embd_v.size():%d\n", image_embd_v.size());
    for (size_t i = 0; i < img_res_v.size; i++)
    {
        image_embd_v[i] =
            (float*)malloc(clip_embd_nbytes(ctx_clip));  // 576 patches * 4096 embeddings * 4 bytes = 9437184
        const bool encoded = clip_image_encode(
            ctx_clip, 1, &img_res_v.data[i],
            image_embd_v[i]);  // image data is in 3x336x336 format and will be converted to 336x336x3 inside
        if (!encoded)
        {
            LOG_TEE("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int)i + 1, (int)img_res_v.size);
            return false;
        }
    }

    return 0;
}


// make test_anyres_handle_patches && ./bin/test_anyres_handle_patches