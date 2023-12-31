# llama.cpp for SYCL

## Background

SYCL is a higher-level programming model to improve programming productivity on various hardware accelerators—such as CPUs, GPUs, and FPGAs. It is a single-source embedded domain-specific language based on pure C++17.

oneAPI is a specification that is open and standards-based, supporting multiple architecture types including but not limited to GPU, CPU, and FPGA. The spec has both direct programming and API-based programming paradigms.

Intel uses the SYCL as direct programming language to support CPU, GPUs and FPGAs.

This project is migrated the CUDA code to SYCL to support Intel CPU, GPU and FPGA.

But we focus on GPU performance tuning. If you want to run llama.cpp on Intel CPU, please use llama.cpp CPU release.

## llama.cpp for SYCL

We migrate the CUDA code SYCL. So the SYCL code replace the CUDA funcitions in llama.cpp, without function name change.

That's why the code macro and log incudes CUBLAS flags.

## OS

### Linux

In Linux, we reuse the CMAKE system of base. It's same as base llama.cpp.

Except branch "windows", other branches are for Linux.

### Windows

In Windows, we change the C source files to meet the requirement of C++ compilers.

So the code is saved in branch **windows** only.

It will output 1 execute file: **llamap.cpp.sycl.exe**.

If you want to get more binary files, please change the build prject.


## Linux

### Setup Environment

1. Install Intel oneAPI Base toolkit.

2. Build locally:

```
mkdir -p build
cd build
source /opt/intel/oneapi/setvars.sh
cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build . --config Release -v
```

or

```
./setup.sh
```

### Run

#### Check device id

Run without parameter:

```
export GGML_SYCL_LIST_DEVICE=1
./build/bin/main
```

Check the id in startup log, like:

```
SYCL devices:
  Device 0: Intel(R) Arc(TM) A770 Graphics, compute capability 1.3
  Device 1: Intel(R) FPGA Emulation Device, compute capability 1.2
  Device 2: 13th Gen Intel(R) Core(TM) i7-13700K, compute capability 3.0
  Device 3: Intel(R) Arc(TM) A770 Graphics, compute capability 3.0
  Device 4: Intel(R) UHD Graphics 770, compute capability 3.0
  Device 5: Intel(R) UHD Graphics 770, compute capability 1.3
Exit for list devices task. unset GGML_SYCL_LIST_DEVICE to restore LLM work!
```

#### Put model file to folder **models**

#### Modify run.sh

Up run.sh as above info:
```
...
GGML_SYCL_DEVICE=0
./build/bin/main -m models/llama-2-7b.Q4_0.gguf -p "${INPUT2}" -n 400 -e -ngl 33
```

#### Run
```
./run.sh
```


## Windows

### Setup Environment

1. Install MS Visual Studio 2022.

2. Install Intel oneAPI Base toolkit.

a. Recommend to install all components and with **default path**.

b. During installation, please choose option to enable compiler in MS Visual Studio.

3. Code

Swith to branch **windows**.

Open **llama.cpp.sycl.sln** by Visual Studio 2022.

4. Set oneAPI Path (optional)

If you chagne the oneAPI installation target path, please modify the oneAPI path in the Visual Studio.

Else, skip this step.

### Build

Build by visual Studio 2022 with x64 & Release.

There will be execute file: **llama.cpp.sycl.exe**.

It will take long time to build due to enable AOT on all hardware flatforms (CPU, GPU, FPGA) as default.

To short it, change AOT target flatforms to one in Visual Studio 2022: **Specify SYCL offloading targets for AOT compilition**.

#### Run

#### Enable oneAPI Environment

Run the command in command line or powershell.

'C:\Program Files (x86)\Intel\oneAPI\setvars.bat'

##### Check device id


Run without parameter:

```
set GGML_SYCL_LIST_DEVICE=1
.\x64\Release\llama.cpp.sycl.exe
```
Check the id in startup log, like:

```
SYCL devices:
  Device 0: Intel(R) Arc(TM) A770 Graphics, compute capability 1.3
  Device 1: Intel(R) FPGA Emulation Device, compute capability 1.2
  Device 2: 13th Gen Intel(R) Core(TM) i7-13700K, compute capability 3.0
  Device 3: Intel(R) Arc(TM) A770 Graphics, compute capability 3.0
  Device 4: Intel(R) UHD Graphics 770, compute capability 3.0
  Device 5: Intel(R) UHD Graphics 770, compute capability 1.3
Exit for list devices task. unset GGML_SYCL_LIST_DEVICE to restore LLM work!
```

#### Put model file to folder **models**

#### Modify run.sh

Up run.sh as above info:
```
...
set GGML_SYCL_DEVICE=0

.\x64\Release\llama.cpp.sycl.exe -m models/llama-2-7b.Q4_0.gguf -p "${INPUT2}" -n 400 -e -ngl 33
```

#### Run
```
.\run.bat
```

### Environment Variable

#### Build

|Name|Value|Function|
|-|-|-|
|LLAMA_SYCL|ON (mandatory)|Enable build with SYCL code path|
|CMAKE_C_COMPILER|icx|Use icx compiler for SYCL code path|
|CMAKE_CXX_COMPILER|icpx|use icpx for SYCL code path|
|GGML_SYCL_F16|OFF (default) or ON|Enable FP16 in computing|


#### Running


|Name|Value|Function|
|-|-|-|
|GGML_SYCL_DEVICE|0 (default) or 1|Set the device id used. List the device ids by **GGML_SYCL_LIST_DEVICE**|
|GGML_SYCL_DEBUG|0 (default) or 1|Enable log funciton by macro: GGML_SYCL_DEBUG|
|GGML_SYCL_LIST_DEVICE|0 (default) or 1|List the device ids only|

