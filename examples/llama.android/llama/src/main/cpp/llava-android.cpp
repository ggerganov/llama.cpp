#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <math.h>
#include <string>
#include <unistd.h>
#include "llama.h"
//#include "common.h"
#include "llava-cli.cpp"
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



// Helper function to throw a Java exception from JNI
void throwJavaException(JNIEnv* env, const char* className, const std::string& message) {
    // Find the exception class
    jclass exceptionClass = env->FindClass(className);
    if (exceptionClass != nullptr) {
        // Throw the exception with the given message
        env->ThrowNew(exceptionClass, message.c_str());
        env->DeleteLocalRef(exceptionClass); // Clean up the local reference
    } else {
        // If the specified exception class cannot be found, fall back to RuntimeException
        std::cerr << "Error: Cannot find exception class: " << className << std::endl;
        jclass runtimeExceptionClass = env->FindClass("java/lang/RuntimeException");
        if (runtimeExceptionClass != nullptr) {
            env->ThrowNew(runtimeExceptionClass, ("Fallback: " + message).c_str());
            env->DeleteLocalRef(runtimeExceptionClass); // Clean up
        } else {
            std::cerr << "Critical Error: Cannot find RuntimeException class" << std::endl;
        }
    }
}



extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaVlmInference_init_1params(JNIEnv *env, jobject /* this */, jstring jmodel, jstring jmmproj) {
    try {
        // Initialize timers and common components
        ggml_time_init();


        const char* model_chars = env->GetStringUTFChars(jmodel, nullptr);
        const char* mmproj_chars = env->GetStringUTFChars(jmmproj, nullptr);

        const char* argv = "-t 1";
        char* nc_argv = const_cast<char*>(argv);
        common_params* params = new common_params();
        common_params_parse(0, &nc_argv, *params, LLAMA_EXAMPLE_LLAVA, print_usage);

        params->model = std::string(model_chars);
        params->mmproj = std::string(mmproj_chars);

        env->ReleaseStringUTFChars(jmodel, model_chars);
        env->ReleaseStringUTFChars(jmmproj, mmproj_chars);

        return reinterpret_cast<jlong>(params);
    } catch (const nlohmann::json::exception& e) {
        throwJavaException(env, "java/lang/IllegalArgumentException",
            std::string("JSON parsing error: ") + e.what());
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException",
            std::string("Error loading model 1: ") + e.what());
    } catch (...) {
        throwJavaException(env, "java/lang/RuntimeException",
            "Unknown error occurred while loading model");
    }

    return 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaVlmInference_load_1model(JNIEnv *env, jobject /* this */, jlong jparams) {
    try {
        const auto params = reinterpret_cast<common_params*>(jparams);

        auto* model = llava_init(params);
        if (model == nullptr) {
            throwJavaException(env, "java/lang/RuntimeException", "Failed to initialize model");
            return 0;
        }

        return reinterpret_cast<jlong>(model);
    } catch (const nlohmann::json::exception& e) {
        throwJavaException(env, "java/lang/IllegalArgumentException",
            std::string("JSON parsing error: ") + e.what());
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException",
            std::string("Error loading model 1: ") + e.what());
    } catch (...) {
        throwJavaException(env, "java/lang/RuntimeException",
            "Unknown error occurred while loading model");
    }

    return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaVlmInference_update_1params(JNIEnv *env, jobject /* this */, jlong jparams, jfloat jtemp , jint jtopK, jfloat jtopP) {
    int32_t top_k = (int32_t) jtopK;
    float top_p = (float) jtopP;
    float temp = (float) jtemp;
    const auto params = reinterpret_cast<common_params*>(jparams);
    params->sparams.top_k = top_k;
    params->sparams.top_p = top_p;
    params->sparams.temp = temp;


}

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaVlmInference_free_1model(JNIEnv *env, jobject /* this */, jlong jmodel) {
    const auto llava_model = reinterpret_cast<llama_model *>(jmodel);

    llama_free_model(llava_model);
}


extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaVlmInference_llava_1init_1context(JNIEnv *env, jobject /* this */, jlong jparams, jlong jmodel) {
    try {
        const auto params = reinterpret_cast<common_params*>(jparams);

        const auto llava_model = reinterpret_cast<llama_model*>(jmodel);
        auto* ctx_llava = llava_init_context(params, llava_model);
        if (ctx_llava == nullptr) {
            throwJavaException(env, "java/lang/RuntimeException", "Failed to initialize llava ctx");
            return 0;
        }

        return reinterpret_cast<jlong>(ctx_llava);
    } catch (const nlohmann::json::exception& e) {
        throwJavaException(env, "java/lang/IllegalArgumentException",
            std::string("JSON parsing error: ") + e.what());
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException",
            std::string("Error loading model: ") + e.what());
    } catch (...) {
        throwJavaException(env, "java/lang/RuntimeException",
            "Unknown error occurred while loading model");
    }

    return 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaVlmInference_llava_1ctx_1free(JNIEnv *env, jobject /* this */, jlong llava_ctx_pointer) {
    try {
        auto* llava_ctx = reinterpret_cast<llava_context *>(llava_ctx_pointer);
        if (llava_ctx == nullptr) {
            throwJavaException(env, "java/lang/RuntimeException", "Null pointer");
            return 0;
        }

        llava_ctx->model = NULL;
        llava_free(llava_ctx);
    }  catch (...) {
        throwJavaException(env, "java/lang/RuntimeException",
            "Unknown error occurred while freeing ctx");
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nexa_NexaVlmInference_llava_1image_1embed_1free(JNIEnv *env, jobject /* this */, jlong llava_image_embed_pointer) {
    try {
        if (llava_image_embed_pointer == 0) {
            throwJavaException(env, "java/lang/RuntimeException", "Pointer is null.");
            return -1;
        }

        auto* llava_image_embed = reinterpret_cast<struct llava_image_embed *>(llava_image_embed_pointer);
        if (llava_image_embed == nullptr ) {
            throwJavaException(env, "java/lang/RuntimeException", "Pointer cast resulted in null.");
            return -1;
        }
        if(llava_image_embed->embed == nullptr ) {
            throwJavaException(env, "java/lang/RuntimeException", "Pointer cast resulted in null.");
            return -1;
        }

        llava_image_embed_free(llava_image_embed);
    } catch (const std::exception &e) {
        // 捕获标准异常
        throwJavaException(env, "java/lang/RuntimeException", e.what());
        return -1;
    } catch (...) {
        // 捕获未知异常
        throwJavaException(env, "java/lang/RuntimeException",
            "Unknown error occurred while freeing image");
        return -1;
    }

    return 0; // 成功
}


extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaVlmInference_load_1image(JNIEnv *env, jobject /* this */, jlong llava_ctx_pointer, jlong jparams, jstring imagePath) {
    try {
        auto* params = reinterpret_cast<common_params*>(jparams);
        auto* ctx_llava = reinterpret_cast<llava_context *>(llava_ctx_pointer);

        std::string image_str = jstring2str(env, imagePath);
        auto * image_embed = load_image(ctx_llava, params, image_str);
        if (image_embed == nullptr) {
            throwJavaException(env, "java/lang/RuntimeException", "Failed to initialize llava ctx");
            return 0;
        }

        return reinterpret_cast<jlong>(image_embed);
    } catch (const nlohmann::json::exception& e) {
        throwJavaException(env, "java/lang/IllegalArgumentException",
            std::string("JSON parsing error: ") + e.what());
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException",
            std::string("Error loading model: ") + e.what());
    } catch (...) {
        throwJavaException(env, "java/lang/RuntimeException",
            "Unknown error occurred while loading model");
    }
}


extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaVlmInference_llava_1eval(JNIEnv *env, jobject /* this */, jlong llava_ctx_pointer, jlong jparams,  jlong llava_image_embed_pointer, jstring jprompt) {

    try {
        auto* params = reinterpret_cast<common_params*>(jparams);
        auto* image_embed = reinterpret_cast<llava_image_embed *>(llava_image_embed_pointer);
        auto* ctx_llava = reinterpret_cast<llava_context *>(llava_ctx_pointer);

        int* n_past = new int(0);

        const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;
        std::string prompt = jstring2str(env, jprompt);

        std::string system_prompt, user_prompt;
        system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
        user_prompt = prompt + "\nASSISTANT:";

        eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, n_past, true);
        llava_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, n_past);
        eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, n_past, false);

        return reinterpret_cast<jlong>(n_past);
    } catch (const nlohmann::json::exception& e) {
        throwJavaException(env, "java/lang/IllegalArgumentException",
            std::string("JSON parsing error: ") + e.what());
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException",
            std::string("Error loading model: ") + e.what());
    } catch (...) {
        throwJavaException(env, "java/lang/RuntimeException",
            "Unknown error occurred while loading model");
    }

    return 0;
}


extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaVlmInference_llava_1sampler_1init(JNIEnv *env, jobject /* this */, jlong llava_ctx_pointer, jlong jparams) {

    try {
        auto* params = reinterpret_cast<common_params*>(jparams);
        auto* ctx_llava = reinterpret_cast<llava_context *>(llava_ctx_pointer);
        struct common_sampler * smpl = common_sampler_init(ctx_llava->model,params->sparams);

        if (smpl == nullptr) {
            throwJavaException(env, "java/lang/RuntimeException", "Failed to initialize llava ctx");
            return 0;
        }

        return reinterpret_cast<jlong>(smpl);
    } catch (const nlohmann::json::exception& e) {
        throwJavaException(env, "java/lang/IllegalArgumentException",
            std::string("JSON parsing error: ") + e.what());
    } catch (const std::exception& e) {
        throwJavaException(env, "java/lang/RuntimeException",
            std::string("Error loading model: ") + e.what());
    } catch (...) {
        throwJavaException(env, "java/lang/RuntimeException",
            "Unknown error occurred while loading model");
    }

    return 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nexa_NexaVlmInference_llava_1sample(JNIEnv *env, jobject /* this */, jlong llava_ctx_pointer, jlong sampler, jlong jnpast, jlong jcached_tokens) {
    auto* smpl = reinterpret_cast<common_sampler*>(sampler);
    auto* ctx_llava = reinterpret_cast<llava_context*>(llava_ctx_pointer);
    auto* cached_tokens = reinterpret_cast<std::string*>(jcached_tokens);
    auto* n_past = reinterpret_cast<int*>(jnpast);
    const char* tmp = sample(smpl, ctx_llava->ctx_llama, n_past);
    *cached_tokens += tmp;
    jstring new_token = nullptr;
    if (is_valid_utf8(cached_tokens->c_str())) {
        new_token = env->NewStringUTF(cached_tokens->c_str());
        cached_tokens->clear();
    } else {
        new_token = env->NewStringUTF("");
    }

    return new_token;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaVlmInference_llava_1sample_1free(JNIEnv *env, jobject /* this */, jlong sampler) {
    auto* smpl = reinterpret_cast<common_sampler*>(sampler);
    common_sampler_free(smpl);
}


extern "C" JNIEXPORT jlong JNICALL
Java_com_nexa_NexaVlmInference_cached_1token_1init(JNIEnv *env, jobject /* this */) {
    std::string* strPtr = new std::string("");
    return reinterpret_cast<jlong>(strPtr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_nexa_NexaVlmInference_cached_1token_1free(JNIEnv *env, jobject /* this */, jlong jcached_tokens) {
    std::string* str = reinterpret_cast<std::string*>(jcached_tokens);

    if (str) {
        delete str;
    }
}