
::  MIT license
::  Copyright (C) 2024 Intel Corporation
::  SPDX-License-Identifier: MIT


IF not exist build (mkdir build)
cd build
if %errorlevel% neq 0 goto ERROR

@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force
if %errorlevel% neq 0 goto ERROR

::  for FP16
::  faster for long-prompt inference
::  cmake -G "MinGW Makefiles" ..  -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx  -DCMAKE_BUILD_TYPE=Release -DLLAMA_SYCL_F16=ON

::  for FP32
cmake -G "MinGW Makefiles" ..  -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx  -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 goto ERROR
::  build example/main only
::  make main

::  build all binary
make -j
if %errorlevel% neq 0 goto ERROR

cd ..
exit /B 0

:ERROR
echo comomand error: %errorlevel%
exit /B %errorlevel%

