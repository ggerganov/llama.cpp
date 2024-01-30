
rem  MIT license
rem  Copyright (C) 2024 Intel Corporation
rem  SPDX-License-Identifier: MIT

mkdir -p build
cd build
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force

rem  for FP16
rem  cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_SYCL_F16=ON # faster for long-prompt inference

rem  for FP32
cmake .. -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release

rem  build example/main only
rem  make main

rem  build all binary
make
cd ..