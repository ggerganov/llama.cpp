#include "clip.h"
#include <stdio.h>

int main(int argc, char ** argv) {
    const char * model_path = argv[1];
    const char * img_path = argv[2];
    const char * text = argv[3];

    auto ctx_clip = clip_model_load(model_path, 1);
    clip_image_u8 img;
    //clip_tokens tokens;
    //clip_tokenize(ctx_clip, text, &tokens);
    //float vec[512];
    //clip_text_encode(ctx_clip, 4, &tokens, vec, false);
    clip_image_load_from_file(img_path, &img);
    float score;
    clip_compare_text_and_image(ctx_clip, 4, text, &img, &score);
    printf("score: %f\n", score);


    return 0;
}