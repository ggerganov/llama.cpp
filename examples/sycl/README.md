# llama.cpp/example/sycl

This example program provides the tools for llama.cpp for SYCL on Intel GPU.

## Tool

|Tool Name| Function|Status|
|-|-|-|
|ls-sycl-device| List all SYCL devices with ID, compute capability, max work group size, ect.|Support|

### ls-sycl-device

List all SYCL devices with ID, compute capability, max work group size, ect.

1. Build the llama.cpp for SYCL for all targets.

2. Enable oneAPI running environment

```
source /opt/intel/oneapi/setvars.sh
```

3. Execute

```
./build/bin/ls-sycl-device
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
