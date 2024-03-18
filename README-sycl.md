# llama.cpp for SYCL

- [Background](#background)
- [News](#news)
- [OS](#os)
- [Supported Devices](#supported-devices)
- [Docker](#docker)
- [Linux](#linux)
- [Windows](#windows)
- [Environment Variable](#environment-variable)
- [Known Issue](#known-issue)
- [Q&A](#q&a)
- [Todo](#todo)

## Background

**SYCL** is a high-level parallel programming model designed to improve developers productivity writing code across various hardware accelerators such as CPUs, GPUs, and FPGAs. It is a single-source language designed for heterogeneous computing and based on standard C++17.

**oneAPI** is an open ecosystem and a standard-based specification, supporting multiple architectures including but not limited to intel CPUs, GPUs and FPGAs. The key components of the oneAPI ecosystem include : 

- **DPCPP** *(Data Parallel C++)* : The primary oneAPI SYCL implementation, which includes the icpx/icx Compilers.
- **oneAPI Libraries** : A set of highly optimized libraries targeting multiple domains *(e.g. oneMKL - Math Kernel Library)*.
- **oneAPI LevelZero** : A high performance low level interface for fine-grained control over intel iGPUs and dGPUs.
- **Nvidia & AMD Plugins** : These are plugins extending oneAPI's DPCPP support to SYCL on Nvidia and AMD GPU targets. 

### Llama.cpp + SYCL 
To avoid re-inventing the wheel, this SYCL "backend" follows the same design found in other llama.cpp BLAS-based paths such as * OpenBLAS, cuBLAS, CLBlast etc..*. The oneAPI's [SYCLomatic](https://github.com/oneapi-src/SYCLomatic) open-source migration tool (Commercial release [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html)) was used for this purpose.

The llama.cpp for SYCL is used to support: 
- Intel GPUs.
- Nvidia GPUs.

*Upcoming support : AMD GPUs*.

For **Intel CPUs**, it is recommend to use llama.cpp for [x86](README.md#intel-onemkl) approach.

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


## Supported devices

### intel GPUs

The BLAS acceleration oneAPI Math Kernel Library which comes with the oneAPI base-toolkit natively supports intel GPUs. In order to make it "visible" while building/running llama.cpp, simply run the following : 
```sh
source /opt/intel/oneapi/setvars.sh
```

- **Tested devices**

|Intel GPU| Status | Verified Model|
|-|-|-|
|Intel Data Center Max Series| Support| Max 1550|
|Intel Data Center Flex Series| Support| Flex 170|
|Intel Arc Series| Support| Arc 770, 730M|
|Intel built-in Arc GPU| Support| built-in Arc GPU in Meteor Lake|
|Intel iGPU| Support| iGPU in i5-1250P, i7-1260P, i7-1165G7|

*Notes :*

- Device memory can be a limitation when running a large model on an intel GPU. The loaded model size, *`llm_load_tensors : buffer_size`*, is displayed in the log when running `./bin/main`

- Please make sure the GPU shared memory from the host is large enough to account for the model's size. For e.g. the *llama-2-7b.Q4_0* requires at least 8.0GB for integrated GPUs and 4.0GB for discrete GPUs.

- If the iGPU has less than 80  EUs *(Execution Unit)*, the inference speed will likely be too slow for practical use.

### Nvidia GPUs
The BLAS acceleration on Nvidia GPUs through oneAPI can be obtained using the Nvidia plugins for oneAPI and the cuBLAS backend of the upstream oneMKL library. Details and instructions on how to setup the runtime and library can be found in [this section](#i-setup-environment)

Math Kernel Library which comes with the oneAPI base-toolkit natively supports intel GPUs. In order to make it "visible" while building/running llama.cpp, simply run the following :

- **Tested devices**

|Nvidia GPU| Status | Verified Model|
|-|-|-|
|Ampere Series| Support| A100, A4000|
|Ampere Series *(Mobile)*| Support| RTX 40 Series

*Notes :* 
  - Support for Nvidia targets through oneAPI is currently limited to Linux platforms.

  - Please make sure the native oneAPI MKL *(dedicated to intel CPUs and GPUs)* is not "visible" at this stage to properly setup and use the built-from-source oneMKL with cuBLAS backend in llama.cpp for Nvidia GPUs. 


## Docker
The docker build option is currently limited to *intel GPU* targets.
### Build image 
```sh
docker build -t llama-cpp-sycl --build-arg="LLAMA_SYCL_F16=[OFF|ON]" -f .devops/main-intel.Dockerfile .
```

*Note* : you can also use the `.devops/server-intel.Dockerfile`, which builds the *"server"* alternative.

### Run container

```sh
# First, find all the DRI cards
ls -la /dev/dri
# Then, pick the card that you want to use (here for e.g. /dev/dri/card1).
docker run -it --rm -v "$(pwd):/app:Z" --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card1:/dev/dri/card1 llama-cpp-sycl -m "/app/models/YOUR_MODEL_FILE" -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33
```

*Notes :*
- Docker have been tested succefully on native Linux. WSL support has not been verified yet.
- You may need to install Intel GPU driver on the **host** machine *(Please refer to the [Linux configuration](#linux) for details)*.

## Linux

### I. Setup Environment

1. **Install GPU drivers**

  - **Intel GPU** 

Intel data center GPUs drivers installation guide and download page can be found here : [Get intel dGPU Drivers](https://dgpu-docs.intel.com/driver/installation.html#ubuntu-install-steps).

*Note* : for client GPUs *(iGPU & Arc A-Series)*, please refer to the [client iGPU driver installation](https://dgpu-docs.intel.com/driver/client/overview.html).

Once installed, please add user(s) to group: `video`, `render`.

```sh
sudo usermod -aG render <username>
sudo usermod -aG video <username>
```

*Note* : logout/re-login for the changes to take effect.

Verify installation through `clinfo`:

```sh
sudo apt install clinfo
sudo clinfo -l
```

Sample output:

```sh
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Arc(TM) A770 Graphics

Platform #0: Intel(R) OpenCL HD Graphics
 `-- Device #0: Intel(R) Iris(R) Xe Graphics [0x9a49]
```

- **Nvidia GPU**

In order to target Nvidia GPUs through SYCL, please make sure the CUDA/CUBLAS native requirements *-found [here](README.md#cublas)-* are installed.
Installation can be verified by running the following :
```sh
nvidia-smi
```
Please make sure at least one CUDA device is available, which can be displayed like this *(here an A100-40GB Nvidia GPU)* : 
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-PCIE-40GB          On  | 00000000:8D:00.0 Off |                    0 |
| N/A   36C    P0              57W / 250W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
```


2. **Install Intel® oneAPI Base toolkit**

- **Base installation**

The base toolkit can be obtained from the official [Intel® oneAPI Base Toolkit ](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) page.

Please follow the instructions for downloading and installing the Toolkit for Linux, and preferably keep the default installation values unchanged, notably the installation path *(`/opt/intel/oneapi` by default)*.

Following guidelines/code snippets assume the default installation values. Otherwise, please make sure the necessary changes are reflected where applicable.  

Upon a successful installation, SYCL is enabled for the available intel devices, along with relevant libraries such as oneAPI MKL for intel GPUs.

- **Bringing support to Nvidia GPUs**

**oneAPI** : In order to enable SYCL support on Nvidia GPUs through oneAPI, please install the [Codeplay oneAPI Plugin for Nvidia GPUs](https://developer.codeplay.com/products/oneapi/nvidia/download). User should also make sure the plugin version matches the installed base toolkit one *(previous step)* for a seamless "oneAPI on Nvidia GPU" setup.


**oneMKL** : The current oneMKL releases *(shipped with the oneAPI base-toolkit)* does not contain the cuBLAS backend. A build from source of the upstream [oneMKL](https://github.com/oneapi-src/oneMKL) with the *cuBLAS* backend enabled is thus required to run it on Nvidia GPUs.

```sh
git clone https://github.com/oneapi-src/oneMKL
cd oneMKL
mkdir -p buildWithCublas && cd buildWithCublas
cmake ../ -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DENABLE_MKLGPU_BACKEND=OFF -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_CUBLAS_BACKEND=ON -DTARGET_DOMAINS=blas
make
```


3. **Verify installation and environment** 

In order to check the available SYCL devices on the machine, please use the `sycl-ls` command.
```sh
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

- **Intel GPU**

When targeting an intel GPU, the user should expect one or more level-zero devices among the available SYCL devices. Please make sure that at least one GPU is present, for instance [`ext_oneapi_level_zero:gpu:0`] in the sample output below :

```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.10.0.17_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i7-13700K OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [23.30.26918.50]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26918]
```

- **Nvidia GPU** 

Similarly, user targetting Nvidia GPUs should expect at least one SYCL-CUDA device [`ext_oneapi_cuda:gpu`] as bellow : 
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.12.0.12_195853.xmain-hotfix]
[opencl:cpu:1] Intel(R) OpenCL, Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz OpenCL 3.0 (Build 0) [2023.16.12.0.12_195853.xmain-hotfix]
[ext_oneapi_cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100-PCIE-40GB 8.0 [CUDA 12.2]
```

### II. Build llama.cpp

#### Intel GPU
```sh
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh

# Build LLAMA with MKL BLAS acceleration for intel GPU
mkdir -p build && cd build
cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_SYCL_F16=[OFF|ON]
```

#### Nvidia GPU
```sh
# Export relevant ENV variables
export LD_LIBRARY_PATH=/path/to/oneMKL/buildWithCublas/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/path/to/oneMKL/buildWithCublas/lib:$LIBRARY_PATH
export CPLUS_INCLUDE_DIR=/path/to/oneMKL/buildWithCublas/include:$CPLUS_INCLUDE_DIR
export CPLUS_INCLUDE_DIR=/path/to/oneMKL/include:$CPLUS_INCLUDE_DIR

# Build LLAMA with Nvidia BLAS acceleration through SYCL
mkdir -p build && cd build
cmake .. -DLLAMA_SYCL=ON -DLLAMA_SYCL_TARGET=NVIDIA -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
```

*Notes :*
- The **F32** build is enabled by default, but the **F16** yields better performance for long-prompt inference.

### III. Run the inference

1. Retrieve and prepare model

You can refer to the general [*Prepare and Quantize*](README#prepare-and-quantize) guide for model prepration, or simply download [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf) model as example.

2. Enable oneAPI running environment

```sh
source /opt/intel/oneapi/setvars.sh
```

3. List devices information

Similar to the native `sycl-ls`, available SYCL devices can be queried as follow : 

```sh
./build/bin/ls-sycl-device
```
A example of such log in a system with 1 *intel CPU* and 1 *intel GPU* can look like the following :
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
|compute capability 1.3|Level-zero driver/runtime, recommended |
|compute capability 3.0|OpenCL driver/runtime, slower than level-zero in most cases|

4. Launch inference

For instance, in order to target the SYCL device with *ID*=0 *(log from previous command)*, we simply specify `GGML_SYCL_DEVICE=0`.

```sh
GGML_SYCL_DEVICE=0 ./build/bin/main -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33
```

Otherwise, you can run the script :

```sh
./examples/sycl/run_llama2.sh
```

*Notes :*

- By default, `mmap` is used to read model file. In some cases, it causes runtime hang issues. Please disable it by passing `--no-mmap` to the `/bin/main` if faced with the issue.

## Windows

### I. Setup Environment

1. Install GPU driver

Intel GPU drivers instructions guide and download page can be found here : [Get intel GPU Drivers](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html).

2. Install Visual Studio

If you already have a recent version of Microsoft Visual Studio, you can skip this tep. Otherwise, please refer to the official download page for [Microsoft Visual Studio](https://visualstudio.microsoft.com/).

3. Install Intel® oneAPI Base toolkit

The base toolkit can be obtained from the official [Intel® oneAPI Base Toolkit ](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) page.

Please follow the instructions for downloading and installing the Toolkit for Windows, and preferably keep the default installation values unchanged, notably the installation path *(`C:\Program Files (x86)\Intel\oneAPI` by default)*.

Following guidelines/code snippets assume the default installation values. Otherwise, please make sure the necessary changes are reflected where applicable.  

b. Enable oneAPI running environment:

- Type "oneAPI" in the search bar, then open the `Intel oneAPI command prompt for Intel 64 for Visual Studio 2022` App.

- On the command prompt, enable the runtime environment with the following :
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

c. Verify installation

In the oneAPI command line, run the following to print the available SYCL devices : 

```
sycl-ls
```

There should be one or more *level-zero* GPU devices displayed as **[ext_oneapi_level_zero:gpu]**. Below is example of such output detecting an *intel Iris Xe* GPU as a Level-zero SYCL device : 

Output (example):
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.10.0.17_160000]
[opencl:cpu:1] Intel(R) OpenCL, 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Iris(R) Xe Graphics OpenCL 3.0 NEO  [31.0.101.5186]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Iris(R) Xe Graphics 1.3 [1.3.28044]
```

4. Install build tools

a. Download & install cmake for Windows: https://cmake.org/download/

b. Download & install mingw-w64 make for Windows provided by w64devkit

- Download the latest fortran version of [w64devkit](https://github.com/skeeto/w64devkit/releases).

- Extract `w64devkit` on your pc.

- Add the **bin** folder path in the Windows system PATH environment (for e.g. `C:\xxx\w64devkit\bin\`).

### II. Build llama.cpp

On the oneAPI command line window, step into the llama.cpp main directory and run the following :

```
mkdir -p build
cd build
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force

cmake -G "MinGW Makefiles" ..  -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx  -DCMAKE_BUILD_TYPE=Release -DLLAMA_SYCL_F16=ON

make
```

Otherwise, run the `win-build-sycl.bat` wrapper which encapsulates the former instructions : 
```sh
.\examples\sycl\win-build-sycl.bat
```

*Notes :*

- By default, calling `make` will build all target binary files. In case of a minimal experimental setup, the user can build the inference executable only through `make main`.

### III. Run the inference

1. Retrieve and prepare model

You can refer to the general [*Prepare and Quantize*](README#prepare-and-quantize) guide for model prepration, or simply download [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf) model as example.

2. Enable oneAPI running environment

On the oneAPI command line window, run the following and step into the llama.cpp directory : 
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

3. List devices information

Similar to the native `sycl-ls`, available SYCL devices can be queried as follow :

```
build\bin\ls-sycl-device.exe
```

The output of this command in a system with 1 *intel CPU* and 1 *intel GPU* would look like the following :
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

4. Launch inference

Set device ID=0 with `set GGML_SYCL_DEVICE=0` to target the Level-zero intel GPU and run the main :

```
set GGML_SYCL_DEVICE=0
build\bin\main.exe -m models\llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 33 -s 0
```
Otherwise, run the following wrapper script:

```
.\examples\sycl\win-run-llama2.bat
```

Note:

- By default, `mmap` is used to read model file. In some cases, it causes runtime hang issues. Please disable it by passing `--no-mmap` to the `main.exe` if faced with the issue.


## Environment Variables

#### Build

|Name|Value|Function|
|-|-|-|
|LLAMA_SYCL|ON (mandatory)|Enable build with SYCL code path.|
|LLAMA_SYCL_TARGET | INTEL *(default)* \| NVIDIA|Set the SYCL target device type.|
|LLAMA_SYCL_F16|OFF *(default)* \|ON *(optional)*|Enable FP16 build with SYCL code path.|
|CMAKE_C_COMPILER|icx|Set *icx* compiler for SYCL code path.|
|CMAKE_CXX_COMPILER|icpx *(Linux)*, icx *(Windows)*|Set `icpx/icx` compiler for SYCL code path.|

#### Runtime

|Name|Value|Function|
|-|-|-|
|GGML_SYCL_DEVICE|0 (default) or 1|Set the device id used. Check the device ids by default running output|
|GGML_SYCL_DEBUG|0 (default) or 1|Enable log function by macro: GGML_SYCL_DEBUG|
|ZES_ENABLE_SYSMAN| 0 (default) or 1|Support to get free memory of GPU by sycl::aspect::ext_intel_free_memory.<br>Recommended to use when --split-mode = layer|

## Known Issues

- Hanging during startup

  llama.cpp uses *mmap* as the default mode for reading the model file and copying it to the GPU. In some systems, `memcpy` might behave abnormally and therefore hang.

  - **Solution** : add `--no-mmap` or `--mmap 0` flag to the `main` executable.

- `Split-mode:[row]` is not supported.

## Q&A

- Error:  `error while loading shared libraries: libsycl.so.7: cannot open shared object file: No such file or directory`.

  - Potential cause : Unavailable oneAPI installation or invisible ENV variables.
  - Solution : Install *oneAPI base toolkit* and enable its ENV through: `source /opt/intel/oneapi/setvars.sh`.

- General compiler error : 

  - Remove build folder or try a clean-build.

- I can **not** see `[ext_oneapi_level_zero:gpu]` afer installing the GPU driver on Linux.

  Please double-check with `sudo sycl-ls`.

  If it's present in the list, please add video/render group to your user then **logout/login** or restart your system :

  ```
  sudo usermod -aG render <username>
  sudo usermod -aG video <username>
  ```

  Otherwise, please double-check the installation GPU steps.

## Todo

- Add support to multiple cards.
