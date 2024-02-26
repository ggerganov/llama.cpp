# llama.cpp for SYCL

- [Background](#background)
- [News](#news)
- [OS](#os)
- [Intel GPU](#intel-gpu)
- [Docker](#docker)
- [Linux](#linux)
- [Windows](#windows)
- [Environment Variable](#environment-variable)
- [Known Issue](#known-issue)
- [Q&A](#q&a)
- [Todo](#todo)

## Background

SYCL is a higher-level programming model to improve programming productivity on various hardware accelerators—such as CPUs, GPUs, and FPGAs. It is a single-source embedded domain-specific language based on pure C++17.

oneAPI is a specification that is open and standards-based, supporting multiple architecture types including but not limited to GPU, CPU, and FPGA. The spec has both direct programming and API-based programming paradigms.

Intel uses the SYCL as direct programming language to support CPU, GPUs and FPGAs.

To avoid to re-invent the wheel, this code refer other code paths in llama.cpp (like OpenBLAS, cuBLAS, CLBlast). We use a open-source tool [SYCLomatic](https://github.com/oneapi-src/SYCLomatic) (Commercial release [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html)) migrate to SYCL.

The llama.cpp for SYCL is used to support Intel GPUs.

For Intel CPU, recommend to use llama.cpp for X86 (Intel MKL building).

## News

- 2024.3
  - Support multiple cards: **--split-mode**: [none|layer]; not support [row], it's on developing.
  - Support to assign main GPU by **--main-gpu**, replace $GGML_SYCL_DEVICE.
  - Support detecting all GPUs with level-zero and same top **Max compute units**.
  - Support OPs
    - hardsigmoid
    - hardswish
    - pool2d

- 2024.1
  - Create SYCL backend for Intel GPU.
  - Support Windows build

## OS

|OS|Status|Verified|
|-|-|-|
|Linux|Support|Ubuntu 22.04, Fedora Silverblue 39|
|Windows|Support|Windows 11|


## Intel GPU

### Verified

|Intel GPU| Status | Verified Model|
|-|-|-|
|Intel Data Center Max Series| Support| Max 1550|
|Intel Data Center Flex Series| Support| Flex 170|
|Intel Arc Series| Support| Arc 770, 730M|
|Intel built-in Arc GPU| Support| built-in Arc GPU in Meteor Lake|
|Intel iGPU| Support| iGPU in i5-1250P, i7-1260P, i7-1165G7|

Note: If the EUs (Execution Unit) in iGPU is less than 80, the inference speed will be too slow to use.

### Memory

The memory is a limitation to run LLM on GPUs.

When run llama.cpp, there is print log to show the applied memory on GPU. You could know how much memory to be used in your case. Like `llm_load_tensors:            buffer size =  3577.56 MiB`.

For iGPU, please make sure the shared memory from host memory is enough. For llama-2-7b.Q4_0, recommend the host memory is 8GB+.

For dGPU, please make sure the device memory is enough. For llama-2-7b.Q4_0, recommend the device memory is 4GB+.

## Nvidia GPU

### Verified

|Intel GPU| Status | Verified Model|
|-|-|-|
|Ampere Series| Support| A100|

### oneMKL

The current oneMKL release does not contain the oneMKL cuBlas backend.
As a result for Nvidia GPU's oneMKL must be built from source.

```
git clone https://github.com/oneapi-src/oneMKL
cd oneMKL
mkdir build
cd build
cmake -G Ninja .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DENABLE_MKLGPU_BACKEND=OFF -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_CUBLAS_BACKEND=ON
ninja
// Add paths as necessary
```

## Docker

Note:
- Only docker on Linux is tested. Docker on WSL may not work.
- You may need to install Intel GPU driver on the host machine (See the [Linux](#linux) section to know how to do that)

### Build the image

You can choose between **F16** and **F32** build. F16 is faster for long-prompt inference.


```sh
# For F16:
#docker build -t llama-cpp-sycl --build-arg="LLAMA_SYCL_F16=ON" -f .devops/main-intel.Dockerfile .

# Or, for F32:
docker build -t llama-cpp-sycl -f .devops/main-intel.Dockerfile .

# Note: you can also use the ".devops/main-server.Dockerfile", which compiles the "server" example
```

### Run

```sh
# Firstly, find all the DRI cards:
ls -la /dev/dri
# Then, pick the card that you want to use.

# For example with "/dev/dri/card1"
docker run -it --rm -v "$(pwd):/app:Z" --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card1:/dev/dri/card1 llama-cpp-sycl -m "/app/models/YOUR_MODEL_FILE" -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33
```

## Linux

### Setup Environment

1. Install Intel GPU driver.

a. Please install Intel GPU driver by official guide: [Install GPU Drivers](https://dgpu-docs.intel.com/driver/installation.html).

Note: for iGPU, please install the client GPU driver.

b. Add user to group: video, render.

```sh
sudo usermod -aG render username
sudo usermod -aG video username
```

Note: re-login to enable it.

c. Check

```sh
sudo apt install clinfo
sudo clinfo -l
```

Output (example):

```
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Arc(TM) A770 Graphics


Platform #0: Intel(R) OpenCL HD Graphics
 `-- Device #0: Intel(R) Iris(R) Xe Graphics [0x9a49]
```

2. Install Intel® oneAPI Base toolkit.

a. Please follow the procedure in [Get the Intel® oneAPI Base Toolkit ](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html).

Recommend to install to default folder: **/opt/intel/oneapi**.

Following guide use the default folder as example. If you use other folder, please modify the following guide info with your folder.

b. Check

```sh
source /opt/intel/oneapi/setvars.sh

sycl-ls
```

There should be one or more level-zero devices. Please confirm that at least one GPU is present, like **[ext_oneapi_level_zero:gpu:0]**.

Output (example):
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.10.0.17_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i7-13700K OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [23.30.26918.50]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26918]

```

2. Build locally:

Note:
- You can choose between **F16** and **F32** build. F16 is faster for long-prompt inference.
- By default, it will build for all binary files. It will take more time. To reduce the time, we recommend to build for **example/main** only.

```sh
mkdir -p build
cd build
source /opt/intel/oneapi/setvars.sh

# For FP16:
#cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_SYCL_F16=ON

# Or, for FP32:
cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# For Nvidia GPUs
cmake .. -DLLAMA_SYCL=ON -DLLAMA_SYCL_TARGET=NVIDIA -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# Build example/main only
#cmake --build . --config Release --target main

# Or, build all binary
cmake --build . --config Release -v

cd ..
```

or

```sh
./examples/sycl/build.sh
```

### Run

1. Put model file to folder **models**

You could download [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf) as example.

2. Enable oneAPI running environment

```
source /opt/intel/oneapi/setvars.sh
```

3. List device ID

Run without parameter:

```sh
./build/bin/ls-sycl-device

# or running the "main" executable and look at the output log:

./build/bin/main
```

Check the ID in startup log, like:

```
found 4 SYCL devices:
  Device 0: Intel(R) Arc(TM) A770 Graphics,	compute capability 1.3,
    max compute_units 512,	max work group size 1024,	max sub group size 32,	global mem size 16225243136
  Device 1: Intel(R) FPGA Emulation Device,	compute capability 1.2,
    max compute_units 24,	max work group size 67108864,	max sub group size 64,	global mem size 67065057280
  Device 2: 13th Gen Intel(R) Core(TM) i7-13700K,	compute capability 3.0,
    max compute_units 24,	max work group size 8192,	max sub group size 64,	global mem size 67065057280
  Device 3: Intel(R) Arc(TM) A770 Graphics,	compute capability 3.0,
    max compute_units 512,	max work group size 1024,	max sub group size 32,	global mem size 16225243136

```

|Attribute|Note|
|-|-|
|compute capability 1.3|Level-zero running time, recommended |
|compute capability 3.0|OpenCL running time, slower than level-zero in most cases|

4. Set device ID and execute llama.cpp

Set device ID = 0 by **GGML_SYCL_DEVICE=0**

```sh
GGML_SYCL_DEVICE=0 ./build/bin/main -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33
```
or run by script:

```sh
./examples/sycl/run_llama2.sh
```

Note:

- By default, mmap is used to read model file. In some cases, it leads to the hang issue. Recommend to use parameter **--no-mmap** to disable mmap() to skip this issue.


5. Check the device ID in output

Like:
```
Using device **0** (Intel(R) Arc(TM) A770 Graphics) as main device
```

## Windows

### Setup Environment

1. Install Intel GPU driver.

Please install Intel GPU driver by official guide: [Install GPU Drivers](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html).

Note: **The driver is mandatory for compute function**.

2. Install Visual Studio.

Please install [Visual Studio](https://visualstudio.microsoft.com/) which impact oneAPI environment enabling in Windows.

3. Install Intel® oneAPI Base toolkit.

a. Please follow the procedure in [Get the Intel® oneAPI Base Toolkit ](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html).

Recommend to install to default folder: **C:\Program Files (x86)\Intel\oneAPI**.

Following guide uses the default folder as example. If you use other folder, please modify the following guide info with your folder.

b. Enable oneAPI running environment:

- In Search, input 'oneAPI'.

Search & open "Intel oneAPI command prompt for Intel 64 for Visual Studio 2022"

- In Run:

In CMD:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

c. Check GPU

In oneAPI command line:

```
sycl-ls
```

There should be one or more level-zero devices. Please confirm that at least one GPU is present, like **[ext_oneapi_level_zero:gpu:0]**.

Output (example):
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.10.0.17_160000]
[opencl:cpu:1] Intel(R) OpenCL, 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Iris(R) Xe Graphics OpenCL 3.0 NEO  [31.0.101.5186]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Iris(R) Xe Graphics 1.3 [1.3.28044]
```

4. Install cmake & make

a. Download & install cmake for Windows: https://cmake.org/download/

b. Download & install mingw-w64 make for Windows provided by w64devkit

- Download the latest fortran version of [w64devkit](https://github.com/skeeto/w64devkit/releases).

- Extract `w64devkit` on your pc.

- Add the **bin** folder path in the Windows system PATH environment, like `C:\xxx\w64devkit\bin\`.

### Build locally:

In oneAPI command line window:

```
mkdir -p build
cd build
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force

::  for FP16
::  faster for long-prompt inference
::  cmake -G "MinGW Makefiles" ..  -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx  -DCMAKE_BUILD_TYPE=Release -DLLAMA_SYCL_F16=ON

::  for FP32
cmake -G "MinGW Makefiles" ..  -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx  -DCMAKE_BUILD_TYPE=Release


::  build example/main only
::  make main

::  build all binary
make -j
cd ..
```

or

```
.\examples\sycl\win-build-sycl.bat
```

Note:

- By default, it will build for all binary files. It will take more time. To reduce the time, we recommend to build for **example/main** only.

### Run

1. Put model file to folder **models**

You could download [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf) as example.

2. Enable oneAPI running environment

- In Search, input 'oneAPI'.

Search & open "Intel oneAPI command prompt for Intel 64 for Visual Studio 2022"

- In Run:

In CMD:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

3. List device ID

Run without parameter:

```
build\bin\ls-sycl-device.exe

or

build\bin\main.exe
```

Check the ID in startup log, like:

```
found 4 SYCL devices:
  Device 0: Intel(R) Arc(TM) A770 Graphics,	compute capability 1.3,
    max compute_units 512,	max work group size 1024,	max sub group size 32,	global mem size 16225243136
  Device 1: Intel(R) FPGA Emulation Device,	compute capability 1.2,
    max compute_units 24,	max work group size 67108864,	max sub group size 64,	global mem size 67065057280
  Device 2: 13th Gen Intel(R) Core(TM) i7-13700K,	compute capability 3.0,
    max compute_units 24,	max work group size 8192,	max sub group size 64,	global mem size 67065057280
  Device 3: Intel(R) Arc(TM) A770 Graphics,	compute capability 3.0,
    max compute_units 512,	max work group size 1024,	max sub group size 32,	global mem size 16225243136

```

|Attribute|Note|
|-|-|
|compute capability 1.3|Level-zero running time, recommended |
|compute capability 3.0|OpenCL running time, slower than level-zero in most cases|

4. Set device ID and execute llama.cpp

Set device ID = 0 by **set GGML_SYCL_DEVICE=0**

```
set GGML_SYCL_DEVICE=0
build\bin\main.exe -m models\llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 33 -s 0
```
or run by script:

```
.\examples\sycl\win-run-llama2.bat
```

Note:

- By default, mmap is used to read model file. In some cases, it leads to the hang issue. Recommend to use parameter **--no-mmap** to disable mmap() to skip this issue.


5. Check the device ID in output

Like:
```
Using device **0** (Intel(R) Arc(TM) A770 Graphics) as main device
```

## Environment Variable

#### Build

|Name|Value|Function|
|-|-|-|
|LLAMA_SYCL|ON (mandatory)|Enable build with SYCL code path. <br>For FP32/FP16, LLAMA_SYCL=ON is mandatory.|
|LLAMA_SYCL_F16|ON (optional)|Enable FP16 build with SYCL code path. Faster for long-prompt inference. <br>For FP32, not set it.|
|CMAKE_C_COMPILER|icx|Use icx compiler for SYCL code path|
|CMAKE_CXX_COMPILER|icpx (Linux), icx (Windows)|use icpx/icx for SYCL code path|

#### Running


|Name|Value|Function|
|-|-|-|
|GGML_SYCL_DEVICE|0 (default) or 1|Set the device id used. Check the device ids by default running output|
|GGML_SYCL_DEBUG|0 (default) or 1|Enable log function by macro: GGML_SYCL_DEBUG|
|ZES_ENABLE_SYSMAN| 0 (default) or 1|Support to get free memory of GPU by sycl::aspect::ext_intel_free_memory.<br>Recommended to use when --split-mode = layer|

## Known Issue

- Hang during startup

  llama.cpp use mmap as default way to read model file and copy to GPU. In some system, memcpy will be abnormal and block.

  Solution: add **--no-mmap** or **--mmap 0**.

- Split-mode: [row] is not supported

  It's on developing.

## Q&A

- Error:  `error while loading shared libraries: libsycl.so.7: cannot open shared object file: No such file or directory`.

  Miss to enable oneAPI running environment.

  Install oneAPI base toolkit and enable it by: `source /opt/intel/oneapi/setvars.sh`.

- In Windows, no result, not error.

  Miss to enable oneAPI running environment.

- Meet compile error.

  Remove folder **build** and try again.

- I can **not** see **[ext_oneapi_level_zero:gpu:0]** afer install GPU driver in Linux.

  Please run **sudo sycl-ls**.

  If you see it in result, please add video/render group to your ID:

  ```
  sudo usermod -aG render username
  sudo usermod -aG video username
  ```

  Then **relogin**.

  If you do not see it, please check the installation GPU steps again.

## Todo

- Support multiple cards.
