// WARNING: this .cpp file is only for debugging. do not user directly.
#include "omni-vlm-wrapper.h"
#include <iostream>


using std::cout;
using std::endl;

int main(int argc, char ** argv) {
    const char* llm_model = "";
    const char* mmproj_model = "";
    const char* image_path = "";
    const char* prompt = "";

    omnivlm_init(llm_model, mmproj_model, "vlm-81-ocr");

    const char* res;
    res = omnivlm_inference(prompt, image_path);
    cout << "RES: " << res << endl;
    res = omnivlm_inference(prompt, image_path);
    cout << "RES: " << res << endl;
    omnivlm_free();

    return 0;
}
