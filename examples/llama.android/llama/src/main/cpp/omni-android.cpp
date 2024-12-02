#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <math.h>
#include <string>
#include <unistd.h>
#include "llama.h"
#include "omni-vlm-wrapper.cpp"
//#include "omni-vlm-cli.cpp"
#include <nlohmann/json.hpp>
#include <jni.h>
#include <string>
#include <iostream>
#include <thread>

#define TAG "llava-android.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern bool is_valid_utf8(const char* str);

extern std::string jstring2str(JNIEnv* env, jstring jstr);


// 用于捕获输出的函数
void redirect_output_to_logcat(const char* tag, int fd) {
    char buffer[1024];
    while (true) {
        ssize_t count = read(fd, buffer, sizeof(buffer) - 1);
        if (count <= 0) break;
        buffer[count] = '\0';
        __android_log_print(ANDROID_LOG_DEBUG, tag, "%s", buffer);
    }
}

// 初始化重定向
void setup_redirect_stdout_stderr() {
    int stdout_pipe[2];
    int stderr_pipe[2];

    pipe(stdout_pipe);
    pipe(stderr_pipe);

    // 重定向 stdout
    dup2(stdout_pipe[1], STDOUT_FILENO);
    close(stdout_pipe[1]);
    std::thread(redirect_output_to_logcat, "STDOUT", stdout_pipe[0]).detach();

    // 重定向 stderr
    dup2(stderr_pipe[1], STDERR_FILENO);
    close(stderr_pipe[1]);
    std::thread(redirect_output_to_logcat, "STDERR", stderr_pipe[0]).detach();
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    setup_redirect_stdout_stderr();
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaOmniVlmInference_init(JNIEnv *env, jobject /* this */, jstring jmodel, jstring jmmproj, jstring jtype) {
    const char* model_chars = env->GetStringUTFChars(jmodel, nullptr);
    const char* mmproj_chars = env->GetStringUTFChars(jmmproj, nullptr);
    const char* type = env->GetStringUTFChars(jtype, nullptr);

    omnivlm_init(model_chars, mmproj_chars, type);
}


extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaOmniVlmInference_image_1embed(JNIEnv *env, jobject /* this */, jstring jprompt, jstring jimage) {
    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);
    const char* imag_path = env->GetStringUTFChars(jimage, nullptr);

    ctx_omnivlm = omnivlm_init_context(&params, model);
    std::string image = imag_path;
    params.prompt = prompt;
    params.prompt = "<|im_start|>system\nYou are Nano-Omni-VLM, created by Nexa AI. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n\n<|vision_start|><|image_pad|><|vision_end|>" + params.prompt + "<|im_end|>";
    auto * image_embed = load_image(ctx_omnivlm, &params, image);

    return reinterpret_cast<jlong>(image_embed);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaOmniVlmInference_sampler_1init(JNIEnv *env, jobject /* this */, jstring jprompt, jstring jimage, jlong jnpast,jlong jimage_embed) {
    auto* n_past = reinterpret_cast<int*>(jnpast);

    auto* image_embed = reinterpret_cast<struct omni_image_embed *>(jimage_embed);

    if (image_embed == nullptr) {
        std::cout << "image_embed is null!" << std::endl;
    }

    size_t image_pos = params.prompt.find("<|image_pad|>");
    std::string system_prompt, user_prompt;

    system_prompt = params.prompt.substr(0, image_pos);
    user_prompt = params.prompt.substr(image_pos + std::string("<|image_pad|>").length());

    params.sparams.top_k = 1;
    params.sparams.top_p = 1.0f;
    eval_string(ctx_omnivlm->ctx_llama, system_prompt.c_str(), params.n_batch, n_past, true);
    omnivlm_eval_image_embed(ctx_omnivlm->ctx_llama, image_embed, params.n_batch, n_past);
    eval_string(ctx_omnivlm->ctx_llama, user_prompt.c_str(), params.n_batch, n_past, false);

    struct common_sampler * smpl = common_sampler_init(ctx_omnivlm->model, params.sparams);

    return reinterpret_cast<jlong>(smpl);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaOmniVlmInference_npast_1init(JNIEnv *env, jobject /* this */) {
    int* n_past = new int(0);
    return reinterpret_cast<jlong>(n_past);
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_nexa_NexaOmniVlmInference_inference(JNIEnv *env, jobject /* this */, jlong jnpast, jlong jsampler) {
    auto* n_past = reinterpret_cast<int*>(jnpast);
    auto * sampler =  reinterpret_cast<struct common_sampler *>(jsampler);
    const char * tmp = sample(sampler, ctx_omnivlm->ctx_llama, n_past);

    jstring new_token = nullptr;
    new_token = env->NewStringUTF(tmp);
    return new_token;
}


extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaOmniVlmInference_sampler_1free(JNIEnv *env, jobject /* this */, jlong jsampler) {
    struct common_sampler * sampler =  reinterpret_cast<struct common_sampler *>(jsampler);
    common_sampler_free(sampler);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaOmniVlmInference_free(JNIEnv *env, jobject /* this */) {
    omnivlm_free();
}
