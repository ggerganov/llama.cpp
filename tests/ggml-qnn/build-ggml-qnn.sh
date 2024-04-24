#!/bin/bash

set -e

#https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct
#https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
QNN_SDK_PATH=/opt/qcom/aistack/qnn/2.20.0.240223/

ANDROID_NDK=`pwd`/android-ndk-r26c
ANDROID_PLATFORM=android-34
TARGET=ggml-qnn-test


function dump_vars()
{
    echo -e "ANDROID_NDK:          ${ANDROID_NDK}"
    echo -e "QNN_SDK_PATH:         ${QNN_SDK_PATH}"
}


function show_pwd()
{
    echo -e "current working path:$(pwd)\n"
}


function check_qnn_sdk()
{
    if [ ! -d ${QNN_SDK_PATH} ]; then
        echo -e "QNN_SDK_PATH ${QNN_SDK_PATH} not exist, pls check...\n"
        exit 1
    fi
}


function check_and_download_ndk()
{
    is_android_ndk_exist=1

    if [ ! -d ${ANDROID_NDK} ]; then
        is_android_ndk_exist=0
    fi

    if [ ! -f ${ANDROID_NDK}/build/cmake/android.toolchain.cmake ]; then
        is_android_ndk_exist=0
    fi

    if [ ${is_android_ndk_exist} -eq 0 ]; then

        if [ ! -f android-ndk-r26c-linux.zip ]; then
            wget --no-config --quiet --show-progress -O android-ndk-r26c-linux.zip  https://dl.google.com/android/repository/android-ndk-r26c-linux.zip
        fi

        unzip android-ndk-r26c-linux.zip

        if [ $? -ne 0 ]; then
            printf "failed to download android ndk to %s \n" "${ANDROID_NDK}"
            exit 1
        fi

        printf "android ndk saved to ${ANDROID_NDK} \n\n"
    else
        printf "android ndk already exist:${ANDROID_NDK} \n\n"
    fi
}


function build_arm64
{
    cmake -H. -B./out/arm64-v8a -DTARGET_NAME=${TARGET} -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=${ANDROID_PLATFORM} -DANDROID_NDK=${ANDROID_NDK}  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DQNN_SDK_PATH=${QNN_SDK_PATH}

    cd ./out/arm64-v8a
    make

    ls -lah ${TARGET}
    /bin/cp ${TARGET} ../../
    cd -
}


function remove_temp_dir()
{
    if [ -d out ]; then
        echo "remove out directory in `pwd`"
        rm -rf out
    fi
}


show_pwd
check_and_download_ndk
check_qnn_sdk
dump_vars
remove_temp_dir
build_arm64
