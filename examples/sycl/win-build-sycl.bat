
::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT

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
