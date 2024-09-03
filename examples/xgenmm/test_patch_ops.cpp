#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"



void print_tensor(ggml_tensor* tensor, const char* name = "", int verbosity = 0)
{
    if (tensor->ne[2] == 1)
    {
        printf("---> %s: (%ld, %ld)\n", name, tensor->ne[0], tensor->ne[1]);
    }
    else if (ggml_is_3d(tensor))
    {
        printf("---> %s: (%ld, %ld, %ld)\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2]);
    }
    else
    {
        printf("---> %s: (%ld, %ld, %ld, %ld)\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    }
    if (verbosity == 1)
    {
        printf("*********************************************************************\n");
        if (tensor->ne[2] == 1)
        {
            const float* mat = (float*)tensor->data;
            int          dim0 = tensor->ne[1];
            int          dim1 = tensor->ne[0];
            if (dim0 < 6 && dim1 < 6)
            {
                for (int i = 0; i < dim0; i++)
                {
                    for (int j = 0; j < dim1; j++)
                    {
                        printf("%+.4f ", mat[i * dim1 + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            else
            {
                for (int i = 0; i < std::min(dim0, 3); i++)
                {
                    for (int j = 0; j < std::min(dim1, 3); j++)
                    {
                        printf("%+.4f ", mat[i * dim1 + j]);
                    }
                    printf("... ");
                    for (int j = dim1 - 3; j < dim1; j++)
                    {
                        printf("%+.4f ", mat[i * dim1 + j]);
                    }
                    printf("\n");
                }
                if (dim0 > 3)
                {
                    printf("...................... omit ......................\n");
                    for (int i = dim0 - 3; i < dim0; i++)
                    {
                        for (int j = 0; j < std::min(dim1, 3); j++)
                        {
                            printf("%+.4f ", mat[i * dim1 + j]);
                        }
                        printf("... ");
                        for (int j = dim1 - 3; j < dim1; j++)
                        {
                            printf("%+.4f ", mat[i * dim1 + j]);
                        }
                        printf("\n");
                    }
                }
            }
        }
        else if (ggml_is_3d(tensor))
        {
            const float* data = (float*)tensor->data;
            int          dim0 = tensor->ne[2];
            int          dim1 = tensor->ne[1];
            int          dim2 = tensor->ne[0];
            if (dim0 < 6 && dim1 < 6 && dim2 < 6)
            {
                for (int i = 0; i < dim0; i++)
                {
                    printf("dim0 = %d\n", i);
                    for (int j = 0; j < dim1; j++)
                    {
                        for (int k = 0; k < dim2; k++)
                        {
                            printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                printf("\n");
            }
            else
            {
                for (int i = 0; i < std::min(dim0, 4); i++)
                {
                    printf("dim0 = %d\n", i);
                    for (int j = 0; j < std::min(dim1, 3); j++)
                    {
                        for (int k = 0; k < std::min(dim2, 3); k++)
                        {
                            printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("... ");
                        for (int k = dim2 - 3; k < dim2; k++)
                        {
                            printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("\n");
                    }
                    printf("........................ omit .....................\n");
                    for (int j = dim1 - 3; j < dim1; j++)
                    {
                        for (int k = 0; k < std::min(dim2, 3); k++)
                        {
                            printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("... ");
                        for (int k = dim2 - 3; k < dim2; k++)
                        {
                            printf("%+.4f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("\n");
                    }
                    printf("---------------------------------------------------\n");
                }
                printf("\n");
            }
        }
    }
    printf("*********************************************************************\n");
    printf("\n");
}

void tensor_to_csv(ggml_tensor* tensor, const char* filename)
{
    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
    }

    const float* mat = (float*)tensor->data;
    int          dim0 = tensor->ne[1];
    int          dim1 = tensor->ne[0];

    {
        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                outFile << float(mat[i * dim1 + j]);
                if (j < dim1 - 1)
                {
                    outFile << ",";
                }
            }
            outFile << std::endl;
        }
    }
    outFile.close();
    printf("file saved to %s\n", filename);
}

struct tensor_from_gguf
{
    struct ggml_tensor*  data;
    struct ggml_context* ctx;
};

bool load_tensor_from_file(const char* filename, tensor_from_gguf& tensor)
{
    struct gguf_init_params params = {
        /*.no_alloc   =*/false,
        /*.ctx        =*/&tensor.ctx,
    };
    gguf_context* ctx = gguf_init_from_file(filename, params);
    if (!ctx)
    {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    tensor.data = ggml_get_tensor(tensor.ctx, "data");

    return true;
}

int main(){
    tensor_from_gguf tensor;
    std::string      filename = "../examples/xgenmm/imgs/4patches_embeddings.gguf";
    bool is_successful = load_tensor_from_file(filename.c_str(), tensor);
    if (!is_successful)
    {
        fprintf(stderr, "%s: load_tensor_from_file() failed\n", __func__);
        return 1;
    }
    
    ggml_tensor* patch_embeds = tensor.data;
    // print_tensor(patch_embeds, "patch_embeds", 1);

    /*
        hardcoded values
    */
    int original_width = 955;
    int original_height = 289;
    int num_images = 4;  // 3 patches + 1 base
    int32_t num_patches_per_side = 384 / 14;
    int num_patches_width = 3; //grid_shape.first
    int num_patches_height = 1; // grid_shape.second



    size_t  size_ele = ggml_type_size(GGML_TYPE_F32);

    struct
    {
        struct ggml_context* ctx;
    } model;


    // TODO: size calculation is not calculated - it's only tens of MB
    size_t ctx_size = 0;

    {
        ctx_size +=
            num_patches_per_side * num_patches_per_side * 1152 * sizeof(float) * num_images * 8;  // image_features
        ctx_size += 1024 * 1024 * ggml_type_size(GGML_TYPE_F32);
    }

    struct ggml_init_params params
    {
        /*.mem_size   =*/ctx_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,  // NOTE: this should be false when using the legacy API
    };

    model.ctx = ggml_init(params);

    

    // FIXME: hardcoded for the patch size and vit embedding size
    struct ggml_tensor* image_features = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, 1152, 729, num_images - 1); 
    struct ggml_tensor* base_image_feature = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, 1152, 729, 1);
    // ggml_tensor_printf(image_features,"image_features",__LINE__,false,false);
    // fill it with the image embeddings, ignoring the base
    // for (size_t i = 1; i < num_images; i++)
    // {
    //     size_t offset = (i - 1) * 729 * 1152 * sizeof(float);
    //     // size_t offset = (i - 1) * clip_embd_nbytes(ctx_clip);
    //     // memcpy((uint8_t*)(image_features->data) + offset, image_embd_v[i], clip_embd_nbytes(ctx_clip));
    // }

    int dim0 = num_images - 1;
    int dim1 = num_patches_per_side * num_patches_per_side;
    int dim2 = 1152;
    float* patch_embeds_data = (float*)patch_embeds->data;
    float* image_features_data = (float*)image_features->data;
    float* base_image_feature_data = (float*)base_image_feature->data;
    for (int i=0; i < dim0; i++)
    {
        for (int j=0; j < dim1; j++)
        {
            for (int k=0; k < dim2; k++)
            {
                image_features_data[i * dim1 * dim2 + j * dim2 + k] =
                    patch_embeds_data[(i + 1) * dim1 * dim2 + j * dim2 + k];
                if (i == 0)
                {
                    base_image_feature_data[j * dim2 + k] = patch_embeds_data[j * dim2 + k];
                }
            }
        }
    }
    // print_tensor(image_features, "image_features", 1);
    

    struct ggml_tensor* image_features_patchview = ggml_view_4d(
        model.ctx, image_features, num_patches_per_side * 1152, num_patches_per_side,
        num_patches_width, num_patches_height, size_ele * num_patches_per_side * 1152,
        size_ele * num_patches_per_side * 1152 * num_patches_per_side,
        size_ele * num_patches_per_side * 1152 * num_patches_per_side * num_patches_width, 0);
    print_tensor(image_features_patchview, "image_features_patchview", 0); // (27 * 1152, 27, 3, 1)
    struct ggml_tensor* permuted_cont =
        ggml_cont(model.ctx, ggml_permute(model.ctx, image_features_patchview, 0, 2, 1, 3));

    print_tensor(permuted_cont, "permuted_cont", 0);  // (27 * 1152, 3, 27, 1)
    struct ggml_tensor* flatten =
        ggml_view_2d(model.ctx, permuted_cont, 1152,
                     num_patches_height * num_patches_width * num_patches_per_side * num_patches_per_side,
                     size_ele * 1152, 0);

    print_tensor(flatten, "flatten", 0);  //  (1152, 27 * 27 * 3)

    // struct ggml_tensor* tensor_3d =
    //     ggml_view_3d(model.ctx, flatten,
    //                  1152,                                         // ne0
    //                  num_patches_per_side * num_patches_per_side,  // ne1
    //                  num_patches_width * num_patches_height,       // ne2 = num_patches_width * num_patches_height,
    //                  size_ele * num_patches_width * num_patches_height,  // nb1 = sizeof(float) × ne2,
    //                  size_ele * num_patches_width * num_patches_height * num_patches_per_side *
    //                      num_patches_per_side,  // nb2 = sizeof(float)×ne1×ne2
    //                  0);
    struct ggml_tensor* tensor_3d =
        ggml_reshape_3d(model.ctx, flatten,
                        1152,                                        
                        num_patches_per_side * num_patches_per_side, 
                        num_patches_width * num_patches_height);
    tensor_3d = ggml_cont(model.ctx, tensor_3d);
    tensor_3d =  ggml_concat(model.ctx, base_image_feature, tensor_3d, 2);
    struct ggml_cgraph* gf = ggml_new_graph(model.ctx);
    ggml_build_forward_expand(gf, tensor_3d);
    ggml_graph_compute_with_ctx(model.ctx, gf, 1);
    struct ggml_tensor* result = gf->nodes[gf->n_nodes - 1];

    print_tensor(result, "result", 1);  // (1152, 27 * 27, 3)

    struct
    {
        struct ggml_context* ctx;
    } mask;

    // TODO: size calculation is not calculated - it's only tens of MB
    ctx_size = 0;

    {
        ctx_size +=
            num_patches_per_side * num_patches_width * num_patches_per_side * num_patches_height * sizeof(float) * 2;
    }

    params = 
    {
        /*.mem_size   =*/ctx_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,  // NOTE: this should be false when using the legacy API
    };
    mask.ctx = ggml_init(params);
    int current_height = num_patches_per_side * num_patches_height;
    int current_width = num_patches_per_side * num_patches_width;
    float original_aspect_ratio = (float)original_width / (float)original_height;
    float current_aspect_ratio = (float)current_width / (float)current_height;
    printf("original_height: %d, original_width: %d, original_aspect_ratio: %.2f\n", original_height, original_width,
           original_aspect_ratio);
    printf("current_height: %d, current_width: %d, current_aspect_ratio: %.2f\n", current_height, current_width,
           current_aspect_ratio);

    float scale_factor = 1.0;
    struct ggml_tensor* attention_mask = ggml_new_tensor_2d(mask.ctx, GGML_TYPE_F32, current_width, current_height);
    if (original_aspect_ratio > current_aspect_ratio){
        scale_factor = (float)current_width / (float)original_width;
        int new_height = int(original_height * scale_factor);
        int padding = (current_height - new_height) / 2;
        // printf("new_height: %d, padding: %d\n", new_height, padding);
        float* attention_mask_data = (float*)attention_mask->data;
        for (int i = 0; i < current_height; i++){
            for (int j = 0; j < current_width; j++){
                if (i < padding || i > padding + new_height){
                    attention_mask_data[i * current_width + j] = 0.0;
                } else {
                    attention_mask_data[i * current_width + j] = 1.0;
                }
            }
        }
    }else{
        scale_factor = current_height / original_height;
        int new_width = int(original_width * scale_factor);
        int padding = (current_width - new_width) / 2;
        float* attention_mask_data = (float*)attention_mask->data;
        for (int i = 0; i < current_height; i++){
            for (int j = 0; j < current_width; j++){
                if (j < padding || j > padding + new_width){
                    attention_mask_data[i * current_width + j] = 0.0;
                } else {
                    attention_mask_data[i * current_width + j] = 1.0;
                }
            }
        }
    }

    print_tensor(attention_mask, "attention_mask", 1);
    tensor_to_csv(attention_mask, "/export/home/llama.cpp/examples/xgenmm/imgs/attention_mask_4patchhes.csv");
    ggml_free(model.ctx);
    ggml_free(mask.ctx);
    ggml_free(tensor.ctx);
    return 0;
}


// make test_patch_ops && ./bin/test_patch_ops