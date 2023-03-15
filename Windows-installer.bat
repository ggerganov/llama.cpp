@echo off
::Install Llama.cpp dependencies, such as Python and CMake (for future building of llama.exe and quantize.exe)

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

%PYTHON% -V
if %ERRORLEVEL% == 0 goto :create_venv
echo Python is not installed, installing
goto :install_python

:install_python
call bitsadmin /transfer Python-3.10.6 /download /priority FOREGROUND "https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe" "%CD%/python-3.10.6-amd64.exe"
python-3.10.6-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
call refreshenv
goto :create_venv

:: Should probably impliment check for pip before installing dependicies
:create_venv
:: Check if venv already exists
dir "%VENV_DIR%\Scripts\Python.exe" -V
if %ERRORLEVEL% == 0 goto :initiate_venv

:: Otherwise create new venv
echo Creating venv in %VENV_DIR%
%PYTHON% -m venv "%VENV_DIR%"
if %ERRORLEVEL% == 0 goto :initiate_venv
echo Unable to create venv in "%VENV_DIR%"
pause
exit

:: Activate venv
:initiate_venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo venv %PYTHON%
goto :install_dependencies

:install_dependencies
%PYTHON% -m pip install cmake torch numpy sentencepiece %*
echo Llama.cpp dependencies are now installed!
echo Put your LLaMA models into the models folder, and run model_conversion to convert and quantize them.
pause
exit