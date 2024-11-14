#!/bin/bash

model_dir="/home/qianlangyu/model/InternVL-gguf"
projector_name="InternViT-300M-448px-f16.gguf"
# projector_name="InternViT-300M-448px-q4_k.gguf"
# llama_name="internlm2-1.8B-chat-F16.gguf"
llama_name="internlm2-1.8B-chat-q4_k.gguf"
img_dir="/home/qianlangyu/resource/imgs"
img_name="image1.jpg"
prompt="<image>\nPlease describe the image shortly."
# img_name="cat.jpeg"
# prompt="<image>\nWhat is in the image?"
# img_name="demo.jpg"
# prompt="<image>\nWho is the author of this book? \nAnswer the question using a single word or phrase."

program_dir="build/bin"
binName="llama-internvl-cli"
n_threads=4


deviceDir="/data/local/tmp"
saveDir="output"
if [ ! -d ${saveDir} ]; then
    mkdir ${saveDir}
fi


function android_run() {
    # # copy resource into device
    # adb push ${model_dir}/${projector_name} ${deviceDir}/${projector_name}
    # adb push ${model_dir}/${llama_name} ${deviceDir}/${llama_name}
    adb push ${img_dir}/${img_name} ${deviceDir}/${img_name}
    # copy program into device
    adb push ${program_dir}/${binName} ${deviceDir}/${binName}
    adb shell "chmod 0777 ${deviceDir}/${binName}"

    # run
    adb shell "echo cd ${deviceDir} LD_LIBRARY_PATH=/data/local/tmp ${deviceDir}/${binName} \
                                                 -m ${deviceDir}/${llama_name} \
                                                 --mmproj ${deviceDir}/${projector_name} \
                                                 -t ${n_threads} \
                                                 --image ${deviceDir}/${img_name} \
                                                 -p \"${prompt}\" \
                                                 > ${deviceDir}/${modelName}_${projector_name}_${llama_name}_${n_threads}_${img_name}_1.txt"
    adb shell "cd ${deviceDir}; pwd; LD_LIBRARY_PATH=/data/local/tmp ${deviceDir}/${binName} \
                                                 -m ${deviceDir}/${llama_name} \
                                                 --mmproj ${deviceDir}/${projector_name} \
                                                 -t ${n_threads} \
                                                 --image ${deviceDir}/${img_name} \
                                                 -p \"${prompt}\" \
                                                 >> ${deviceDir}/${modelName}_${projector_name}_${llama_name}_${n_threads}_${img_name}_1.txt 2>&1"
    adb pull ${deviceDir}/${modelName}_${projector_name}_${llama_name}_${n_threads}_${img_name}_1.txt ${saveDir}
}

android_run

echo "android_run is Done!"
