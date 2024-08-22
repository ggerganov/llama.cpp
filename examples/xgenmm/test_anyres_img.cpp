#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "clip.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "xgenmm.h"

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

struct clip_ctx {
    bool has_text_encoder    = false;
    bool has_vision_encoder  = false;
    bool has_llava_projector = false;
    bool has_minicpmv_projector = false;
    int minicpmv_version = 2;

    struct clip_vision_model vision_model;
    projector_type proj_type = PROJECTOR_TYPE_MLP;

    float image_mean[3];
    float image_std[3];
    bool use_gelu = false;
    int32_t ftype = 1;

    bool has_class_embedding = true;
    bool has_pre_norm = true;
    bool has_post_norm = false;
    bool has_patch_bias = false;

    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    // memory buffers to evaluate the model
    ggml_backend_buffer_t params_buffer  = NULL;

    ggml_backend_t backend       = NULL;
    ggml_gallocr_t compute_alloc = NULL;

    struct clip_image_size * load_image_size;
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

int main(){
    /*
    Pytorch Image Processing Pipeline
        n_px = hf_processor.image_processor.size['height']
        image_processor = Compose([
            Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC, antialias=True),
            Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        anyres_grids = [[384, 768], [768, 384], [768, 768], [1152, 384], [384, 1152]]
        grid_pinpoints = anyres_grids
        best_resolution = select_best_resolution(image.size, possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)
        processor_size = processor.transforms[0].size
        patches = divide_to_patches(image_padded, processor_size[0])
        image_original_resize = image.resize((processor_size[0], processor_size[0]))
        image_patches = [image_original_resize] + patches
        image_patches = [processor(image_patch) for image_patch in image_patches]
        return torch.stack(image_patches, dim=0)

        this part is already implemented in the clip_image_preprocess function in clip.cpp
    */

    const char*      clip_path = "/export/share/yutong/xgenmm/llamacpp_wd/llava-1.6/vit/mmproj-model-f16.gguf";
    // struct ggml_context* meta = NULL;

    // struct gguf_init_params params = {
    //     /*.no_alloc = */ true,
    //     /*.ctx      = */ &meta,
    // };

    // struct gguf_context* ctx = gguf_init_from_file(clip_path, params);
    // if (!ctx)
    // {
    //     throw std::runtime_error(
    //         format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, clip_path));
    // }
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
    printf("params.image_size:%d\n", ctx->vision_model.hparams.image_size);
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

    print_img(img);

    clip_image_u8* image_original_resize = clip_image_u8_init();
    bicubic_resize(*img, *image_original_resize, 384, 384);

    printf("**********************************\n");

    print_img(image_original_resize);
    img_to_csv(image_original_resize, "/export/home/llama.cpp/examples/xgenmm/imgs/image_original_resize.csv");
    printf("num pixels: %d\n", image_original_resize->buf.size());
    printf("raw img: nx:%d | ny:%d\n", image_original_resize->nx, image_original_resize->ny);

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
    // printf("img_res_v.size:%ld\n", img_res_v.size);
    printf("img_res_v->nx:%ld | img_res_v->ny:%ld\n", img_res_v.data->nx, img_res_v.data->ny);
    // std::cout << img_res_v.data->nx << " | " << img_res_v.data->ny << std::endl;
    // std::cout << img_res_v.data->buf.size() << std::endl;

    const char* mm_patch_merge_type = clip_patch_merge_type(ctx);
    printf("mm_patch_merge_type:%s\n", mm_patch_merge_type);

    std::string basename = "/export/home/llama.cpp/examples/xgenmm/imgs/image_res";
    for (size_t i = 0; i < img_res_v.size; i++) {
        const int nx = img_res_v.data[i].nx;
        const int ny = img_res_v.data[i].ny;
        printf("i:%d | nx:%d | ny:%d\n", i, nx, ny);

        const int n = nx * ny;

 
        for (int k = 0; k < 1; k++) {
            for (int y = 0; y < 5; y++) {
                for (int x = 0; x < 10; x++) {
                    // data[(i * 3 * n) + k * n + y * nx + x] = imgs->data[i].buf[3 * (y * nx + x) + k];
                    printf("%.4f ", img_res_v.data[i].buf[3 * (y * nx + x) + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
        std::string fileName = basename + "_" + std::to_string(i) + ".csv";
        tensor_to_csv(&img_res_v.data[i], fileName.c_str());
    }
    

    // /*
    // part of:
    // clip_image_encode
    // */
    // clip_image_f32_batch imgs{};
    // imgs.size = 1;
    // imgs.data = &img_res_v.data[0];


    // /*
    // part of:
    // clip_image_batch_encode
    // */
    // const clip_image_f32_batch * imgs_f32_const = &imgs;
    // int batch_size = imgs_f32_const->size;
    // if (ctx->has_llava_projector) {
    //     GGML_ASSERT(batch_size == 1); // TODO: support multiple images
    // }
    // if (ctx->has_minicpmv_projector) {
    //     GGML_ASSERT(batch_size == 1);
    // }


    

    return 0;
}


// make test_anyres_img && ./bin/test_anyres_img