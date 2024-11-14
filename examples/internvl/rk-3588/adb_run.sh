#!/bin/bash

model_dir="/home/qianlangyu/model/dms-single-img"
projector_name="model.safetensors-q4_0.gguf"
# projector_name="InternViT-300M-448px-q4_k.gguf"
# llama_name="internlm2-1.8B-chat-F16.gguf"
llama_name="Dms-Single-Img-630M-q4_0.gguf"
img_dir="/home/qianlangyu/model/checkpoint-2000-merged/images/"
img_name="baixiancui_cloud_agent_1707907703002_37313.jpg"
prompt="<image>\n<dms>"
# img_name="cat.jpeg"
# prompt="<image>\nWhat is in the image?"
# img_name="demo.jpg"
# prompt="<image>\nWho is the author of this book? \nAnswer the question using a single word or phrase."

program_dir="build/bin"
binName="llama-internvl-cli"
n_threads=4


deviceDir="/data/qianlangyu/dms"
saveDir="output"
if [ ! -d ${saveDir} ]; then
    mkdir ${saveDir}
fi


function rk_run() {
    # # copy resource into device
    # adb push ${model_dir}/${projector_name} ${deviceDir}/${projector_name}
    # adb push ${model_dir}/${llama_name} ${deviceDir}/${llama_name}
    adb push ${img_dir}/${img_name} ${deviceDir}/${img_name}
    # copy program into device
    adb push ${program_dir}/${binName} ${deviceDir}/${binName}
    adb shell "chmod 0777 ${deviceDir}/${binName}"

    # run
    adb shell "echo cd ${deviceDir} LD_LIBRARY_PATH=/data/qianlangyu/dms/lib ${deviceDir}/${binName} \
                                                 -m ${deviceDir}/${llama_name} \
                                                 --mmproj ${deviceDir}/${projector_name} \
                                                 -t ${n_threads} \
                                                 --image ${deviceDir}/${img_name} \
                                                 -p \"${prompt}\" \
                                                 -b 512 -c 512 \
                                                 > ${deviceDir}/${modelName}_${projector_name}_${llama_name}_${n_threads}_${img_name}.txt"
    adb shell "cd ${deviceDir}; pwd; LD_LIBRARY_PATH=/data/qianlangyu/dms/lib ${deviceDir}/${binName} \
                                                 -m ${deviceDir}/${llama_name} \
                                                 --mmproj ${deviceDir}/${projector_name} \
                                                 -t ${n_threads} \
                                                 --image ${deviceDir}/${img_name} \
                                                 -p \"${prompt}\" \
                                                 -b 512 -c 512 \
                                                 >> ${deviceDir}/${modelName}_${projector_name}_${llama_name}_${n_threads}_${img_name}.txt 2>&1"
    adb pull ${deviceDir}/${modelName}_${projector_name}_${llama_name}_${n_threads}_${img_name}.txt ${saveDir}
}

rk_run

echo "rk_run is Done!"
