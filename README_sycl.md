# llama.cpp for SYCL

[Background](#background)

[OS](#os)

[Intel GPU](#intel-gpu)

[Linux](#linux)

[Environment Variable](#environment-variable)

[Known Issue](#known-issue)

[Todo](#todo)

## Background

SYCL is a higher-level programming model to improve programming productivity on various hardware accelerators—such as CPUs, GPUs, and FPGAs. It is a single-source embedded domain-specific language based on pure C++17.

oneAPI is a specification that is open and standards-based, supporting multiple architecture types including but not limited to GPU, CPU, and FPGA. The spec has both direct programming and API-based programming paradigms.

Intel uses the SYCL as direct programming language to support CPU, GPUs and FPGAs.

To avoid to re-invent the wheel, this code refer other code paths in llama.cpp (like OpenBLAS, cuBLAS, CLBlast). We use a open-source tool [SYCLomatic](https://github.com/oneapi-src/SYCLomatic) (Commercial release [Intel® DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html)) migrate to SYCL.

The llama.cpp for SYCL is used to support Intel GPUs.

For Intel CPU, recommend to use llama.cpp for X86 (Intel MKL building).

## OS

|OS|Status|Verified|
|-|-|-|
|Linux|Support|Ubuntu 22.04|
|Windows|Ongoing| |


## Intel GPU

|Intel GPU| Status | Verified Model|
|-|-|-|
|Intel Data Center Max Series| Support| Max 1550|
|Intel Data Center Flex Series| Support| Flex 170|
|Intel Arc Series| Support| Arc 770|
|Intel built-in Arc GPU| Support| built-in Arc GPU in Meteor Lake|
|Intel iGPU| Support| iGPU in i5-1250P, i7-1165G7|


## Linux

### Setup Environment

1. Install Intel GPU driver.

a. Please install Intel GPU driver by official guide: [Install GPU Drivers](https://dgpu-docs.intel.com/driver/installation.html).

Note: for iGPU, please install the client GPU driver.

b. Add user to group: video, render.

```
sudo usermod -aG render username
sudo usermod -aG video username
```

Note: re-login to enable it.

c. Check

```
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

```
source /opt/intel/oneapi/setvars.sh

sycl-ls
```

There should be one or more level-zero devices. Like **[ext_oneapi_level_zero:gpu:0]**.

Output (example):
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.10.0.17_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i7-13700K OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [23.30.26918.50]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26918]

```

2. Build locally:

```
mkdir -p build
cd build
source /opt/intel/oneapi/setvars.sh

#for FP16
#cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_SYCL_F16=ON # faster for long-prompt inference

#for FP32
cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

#build example/main only
#cmake --build . --config Release --target main

#build all binary
cmake --build . --config Release -v

```

or

```
./examples/sycl/build.sh
```

Note:

- By default, it will build for all binary files. It will take more time. To reduce the time, we recommend to build for **example/main** only.

### Run

1. Put model file to folder **models**

2. Enable oneAPI running environment

```
source /opt/intel/oneapi/setvars.sh
```

3. List device ID

Run without parameter:

```
./build/bin/ls-sycl-device

or

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

```
GGML_SYCL_DEVICE=0 ./build/bin/main -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33
```
or run by script:

```
./examples/sycl/run_llama2.sh
```

Note:

- By default, mmap is used to read model file. In some cases, it leads to the hang issue. Recommend to use parameter **--no-mmap** to disable mmap() to skip this issue.


5. Check the device ID in output

Like：
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
|CMAKE_CXX_COMPILER|icpx|use icpx for SYCL code path|

#### Running


|Name|Value|Function|
|-|-|-|
|GGML_SYCL_DEVICE|0 (default) or 1|Set the device id used. Check the device ids by default running output|
|GGML_SYCL_DEBUG|0 (default) or 1|Enable log function by macro: GGML_SYCL_DEBUG|

## Known Issue

- Error:  `error while loading shared libraries: libsycl.so.7: cannot open shared object file: No such file or directory`.

  Miss to enable oneAPI running environment.

  Install oneAPI base toolkit and enable it by: `source /opt/intel/oneapi/setvars.sh`.


- Hang during startup

  llama.cpp use mmap as default way to read model file and copy to GPU. In some system, memcpy will be abnormal and block.

  Solution: add **--no-mmap**.

## Todo

- Support to build in Windows.

- Support multiple cards.
