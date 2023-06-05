#include <jni.h>
#include "llama.h"

//
// Created by gcpth on 05/06/2023.
//

extern "C"
JNIEXPORT void JNICALL
Java_com_layla_LlamaCpp_llama_1init_1backend(JNIEnv *env, jclass clazz) {
    llama_init_backend();
}
