#include "clip.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    const char * model_path = argv[1];
    const char * img_path = argv[2];
    const char * text = argv[3];

    auto ctx_clip = clip_model_load(model_path, 1);
    clip_image_u8 img;
    clip_image_f32 img_res;
    clip_image_load_from_file(img_path, &img);
    clip_image_preprocess(ctx_clip, &img, &img_res);
    float * vec = (float *)malloc(4096 * 257 * sizeof(float));
    clip_image_encode(ctx_clip, 4, &img_res, vec, false);
    
    /*
    float score;
    clip_compare_text_and_image(ctx_clip, 4, text, &img, &score);
    printf("score: %f\n", score);
    */

    clip_free(ctx_clip);
    free(vec);


    return 0;
}