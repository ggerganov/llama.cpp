### 所有命令在 llama.cpp 根目录执行，模型位于根目录上级目录处
### All command should be executed under the root path of llama.cpp repo. We assume the MiniCPM-V-2.5 model are put in its parent folder.

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

### Android

#### Build on Android using Termux
[Termux](https://github.com/termux/termux-app#installation) is a method to execute `llama.cpp` on an Android device (no root required).
```
apt update && apt upgrade -y
apt install git make cmake
```

It's recommended to move your model inside the `~/` directory for best performance:
```
cd storage/downloads
mv model.gguf ~/
```

#### Building the Project using Android NDK
Obtain the [Android NDK](https://developer.android.com/ndk) and then build with CMake.

Execute the following commands on your computer to avoid downloading the NDK to your mobile. Alternatively, you can also do this in Termux:

```bash
mkdir build-android
cd build-android
export NDK=/your_ndk_path
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_C_FLAGS=-march=armv8.4a+dotprod ..
make
```

Install [termux](https://github.com/termux/termux-app#installation) on your device and run `termux-setup-storage` to get access to your SD card (if Android 11+ then run the command twice).

Finally, copy these built `llama` binaries and the model file to your device storage. Because the file permissions in the Android sdcard cannot be changed, you can copy the executable files to the `/data/data/com.termux/files/home/bin` path, and then execute the following commands in Termux to add executable permission:

(Assumed that you have pushed the built executable files to the /sdcard/llama.cpp/bin path using `adb push`)
```
$cp -r /sdcard/llama.cpp/bin /data/data/com.termux/files/home/
$cd /data/data/com.termux/files/home/bin
$chmod +x ./*
```

Download models and push them to `/sdcard/llama.cpp/`, then move it to `/data/data/com.termux/files/home/model/`

```
$mv /sdcard/llama.cpp/ggml-model-Q4_K_M.gguf /data/data/com.termux/files/home/model/
$mv /sdcard/llama.cpp/mmproj-model-f16.gguf /data/data/com.termux/files/home/model/
```

Now, you can start chatting:
```
$cd /data/data/com.termux/files/home/bin
$./minicpmv-cli -m ../MiniCPM-V-2_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg  -p "What is in the image?"
```