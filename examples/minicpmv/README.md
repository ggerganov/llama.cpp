# 所有命令在 llama.cpp 根目录执行，模型位于根目录上级目录处
# All command should be executed under the root path of llama.cpp repo. We assume the MiniCPM-V-2.5 model are put in its parent folder.

```bash
make
make minicpmv-cli

python ./examples/minicpmv/minicpmv-surgery.py -m ../MiniCPM-V-2_5
python ./examples/minicpmv/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-V-2_5 --minicpmv-projector ../MiniCPM-V-2_5/minicpmv.projector --output-dir ../MiniCPM-V-2_5/ --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5
python ./convert.py ../MiniCPM-V-2_5/model  --outtype f16 --vocab-type bpe
./minicpmv-cli -m ../MiniCPM-V-2_5/model/model-8B-F16.gguf --mmproj ../MiniCPM-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# or run quantize int4 version
./quantize ../MiniCPM-V-2_5/model/model-8B-F16.gguf ../MiniCPM-V-2_5/model/ggml-model-Q4_K_M.gguf Q4_K_M
./minicpmv-cli -m ../MiniCPM-V-2_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg  -p "What is in the image?"

# or run in interactive mode
./minicpmv-cli -m ../MiniCPM-V-2_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -i
```

# 在 Android 上运行
# Run on android

```bash
mkdir build-android
cd build-android
export NDK=/your_ndk_path
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS=-march=armv8.4a+dotprod ..
make
cd bin
# then llava-cli is the file you need to put on the phone.
```