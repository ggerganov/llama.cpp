@echo off
:: Convert models to 4 bit

set "VENV_DIR=%~dp0%venv"
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo venv is located at: %PYTHON%
goto :convert_ggml

:: Convert the 7B model to ggml FP16 format
:convert_ggml
%PYTHON% "convert-pth-to-ggml.py" models/7B/ 1
if %ERRORLEVEL% == 1 goto :fast_quit
goto :build_cmake

:: Build llama.exe and quantize.exe
:build_cmake
cmake -S . -B build/ -D CMAKE_BUILD_TYPE=Release
cmake --build build/ --config Release
if %ERRORLEVEL% == 1 goto :fast_quit
goto :quantize

:quantize
.\build\Release\quantize.exe .\models\7B\ggml-model-f16.bin .\models\7B\ggml-model-q4_0.bin 2
echo All tasks completed successfully.
pause
exit

:fast_quit
echo Something went wrong!
pause
exit