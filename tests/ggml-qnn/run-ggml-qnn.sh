#!/bin/bash

#https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct
#https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
QNN_SDK_PATH=/opt/qcom/aistack/qnn/2.20.0.240223/
GGML_QNN_TEST=ggml-qnn-test
REMOTE_PATH=/data/local/tmp/


function check_qnn_sdk()
{
    if [ ! -d ${QNN_SDK_PATH} ]; then
        echo -e "QNN_SDK_PATH ${QNN_SDK_PATH} not exist, pls check or download it from https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct...\n"
        exit 1
    fi
}


function check_qnn_libs()
{
    #reuse the cached qnn libs in Android phone
    adb shell ls ${REMOTE_PATH}/libQnnCpu.so
    if [ $? -eq 0 ]; then
        printf "QNN libs already exist on Android phone\n"
    else
        adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnSystem.so              ${REMOTE_PATH}/
        adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnCpu.so                 ${REMOTE_PATH}/
        adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnGpu.so                 ${REMOTE_PATH}/

        #the QNN NPU(aka HTP/DSP) backend only verified on Xiaomi14(Qualcomm SM8650-AB Snapdragon 8 Gen 3) successfully
        adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp.so                 ${REMOTE_PATH}/
        adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpNetRunExtensions.so ${REMOTE_PATH}/
        adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpPrepare.so          ${REMOTE_PATH}/
        adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpV75Stub.so          ${REMOTE_PATH}/
        adb push ${QNN_SDK_PATH}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so     ${REMOTE_PATH}/
    fi
}


function show_usage()
{
    echo "Usage:"
    echo "  $0 GGML_OP_ADD      0/1/2"
    echo "  $0 GGML_OP_MUL      0/1/2"
    echo "  $0 GGML_OP_MUL_MAT  0/1/2"
    echo -e "\n\n\n"
}


function main()
{
    check_qnn_libs

    #upload the latest ggml_qnn_test
    adb push ${GGML_QNN_TEST} ${REMOTE_PATH}
    adb shell chmod +x ${REMOTE_PATH}/${GGML_QNN_TEST}

    case "$ggmlop" in
        GGML_OP_ADD)
            echo "adb shell ${REMOTE_PATH}/${GGML_QNN_TEST}  -t GGML_OP_ADD -b $qnnbackend"
            adb shell ${REMOTE_PATH}/${GGML_QNN_TEST}  -t GGML_OP_ADD -b $qnnbackend
        ;;

        GGML_OP_MUL)
            adb shell ${REMOTE_PATH}/${GGML_QNN_TEST}  -t GGML_OP_MUL -b $qnnbackend
        ;;

        GGML_OP_MUL_MAT)
            adb shell ${REMOTE_PATH}/${GGML_QNN_TEST}  -t GGML_OP_MUL_MAT -b $qnnbackend
        ;;

        *)
            printf " \n$arg not supported currently\n"
            show_usage
            exit 1
        ;;
    esac
}


check_qnn_sdk

unset ggmlop
unset qnnbackend
if [ $# == 0 ]; then
    show_usage
    exit 1
elif [ $# == 1 ]; then
    if [ "$1" == "-h" ]; then
        #avoid upload command line program to Android phone in this scenario
        show_usage
        exit 1
    elif [ "$1" == "help" ]; then
        #avoid upload command line program to Android phone in this scenario
        show_usage
        exit 1
    else
        ggmlop=$1
        qnnbackend=0
    fi
elif [ $# == 2 ]; then
    ggmlop=$1
    qnnbackend=$2
else
    show_usage
    exit 1
fi
main $arg
