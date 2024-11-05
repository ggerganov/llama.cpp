// WARNING: this .cpp file is only for debugging. do not user directly.
#include "omni-vlm-wrapper.h"

int main(int argc, char ** argv) {
    const char* llm_model = "<path to llm gguf.>";
    const char* mmproj_model = "<path to mm projector gguf>";
    const char* image_path = "<path where image is located.>";
    const char* prompt = "";

    omnivlm_init(llm_model, mmproj_model);
    omnivlm_inference(prompt, image_path);
    omnivlm_inference(prompt, image_path);
    omnivlm_free();

    return 0;
}
