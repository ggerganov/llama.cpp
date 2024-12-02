// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Includes the Jni utilities for Android to be able to create the
// relevant bindings for java, including JNIEXPORT, JNICALL , and
// other "j-variables".
#include <jni.h>

// The ML class exposing the Kompute ML workflow for training and
// prediction of inference data.
#include "KomputeModelML.hpp"

// Allows us to use the C++ sleep function to wait when loading the
// Vulkan library in android
#include <unistd.h>
#include <kompute/logger/Logger.hpp>

static std::vector<float>
jfloatArrayToVector(JNIEnv* env, const jfloatArray& fromArray)
{
    float* inCArray = env->GetFloatArrayElements(fromArray, NULL);
    if (NULL == inCArray)
        return std::vector<float>();
    int32_t length = env->GetArrayLength(fromArray);

    std::vector<float> outVector(inCArray, inCArray + length);
    return outVector;
}

static jfloatArray
vectorToJFloatArray(JNIEnv* env, const std::vector<float>& fromVector)
{
    jfloatArray ret = env->NewFloatArray(fromVector.size());
    if (NULL == ret)
        return NULL;
    env->SetFloatArrayRegion(ret, 0, fromVector.size(), fromVector.data());
    return ret;
}

extern "C"
{
    JNIEXPORT jfloatArray JNICALL
    Java_com_ethicalml_kompute_KomputeJni_kompute(JNIEnv* env,
                                                  jobject thiz,
                                                  jfloatArray xiJFloatArr,
                                                  jfloatArray xjJFloatArr,
                                                  jfloatArray yJFloatArr)
    {

        KP_LOG_INFO("Creating manager");

        std::vector<float> xiVector = jfloatArrayToVector(env, xiJFloatArr);
        std::vector<float> xjVector = jfloatArrayToVector(env, xjJFloatArr);
        std::vector<float> yVector = jfloatArrayToVector(env, yJFloatArr);

        KomputeModelML kml;
        kml.train(yVector, xiVector, xjVector);

        std::vector<float> pred = kml.predict(xiVector, xjVector);

        return vectorToJFloatArray(env, pred);
    }

    JNIEXPORT jfloatArray JNICALL
    Java_com_ethicalml_kompute_KomputeJni_komputeParams(JNIEnv* env,
                                                        jobject thiz,
                                                        jfloatArray xiJFloatArr,
                                                        jfloatArray xjJFloatArr,
                                                        jfloatArray yJFloatArr)
    {

        KP_LOG_INFO("Creating manager");

        std::vector<float> xiVector = jfloatArrayToVector(env, xiJFloatArr);
        std::vector<float> xjVector = jfloatArrayToVector(env, xjJFloatArr);
        std::vector<float> yVector = jfloatArrayToVector(env, yJFloatArr);

        KomputeModelML kml;
        kml.train(yVector, xiVector, xjVector);

        std::vector<float> params = kml.get_params();

        return vectorToJFloatArray(env, params);
    }
}
