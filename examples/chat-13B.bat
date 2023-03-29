@setlocal disabledelayedexpansion enableextensions
@echo off

cd /d "%~dp0.."
if not "%errorlevel%"=="0" (
    echo Unable to change directory.
    pause
    exit /b 1
)

if not defined MODEL set "MODEL=models\13B\ggml-model-q4_0.bin"
if not defined USER_NAME set "USER_NAME=User"
if not defined AI_NAME set "AI_NAME=ChatLLaMa"
rem Adjust to the number of CPU cores you want to use.
rem if not defined N_THREAD set "N_THREAD=8"
rem Number of tokens to predict (made it larger than default because we want a long interaction)
if not defined N_PREDICTS set "N_PREDICTS=2048"
if not defined GEN_OPTIONS set "GEN_OPTIONS=--ctx_size 2048 --temp 0.7 --top_k 40 --top_p 0.5 --repeat_last_n 256 --batch_size 1024 --repeat_penalty 1.17647"

rem Default main script paths
set "DEFAULT_MAIN_SCRIPT_PATHS=main.exe build\bin\main.exe"

rem Get main script path from command line arguments
set "MAIN_SCRIPT_PATH=%~1"

rem If the main script path was not specified, try the default paths
if not defined MAIN_SCRIPT_PATH (
    for %%i in (%DEFAULT_MAIN_SCRIPT_PATHS%) do (
        if exist "%%i" set "MAIN_SCRIPT_PATH=%%i"
    )
)

rem If the main script path was not found, tell the user how to specify it
if not defined MAIN_SCRIPT_PATH (
    echo The main script could not be found. Please provide the path to the main script as 1st argument to this script, or place the main script in one of the default locations:
    echo %DEFAULT_MAIN_SCRIPT_PATHS%
    pause
    exit /b 1
)

rem Default context, feel free to edit it
set "PROMPT_TEXT=Text transcript of a never ending dialog, where %USER_NAME% interacts with an AI assistant named %AI_NAME%. %AI_NAME% is helpful, kind, honest, friendly, good at writing and never fails to answer %USER_NAME%'s requests immediately and with details and precision. There are no annotations like (30 seconds passed...) or (to himself), just what %USER_NAME% and %AI_NAME% say aloud to each other. The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long. The transcript only includes text, it does not include markup like HTML and Markdown."

rem Set a temporary variable if N_THREAD is set
if defined N_THREAD (
    set "_N_THREAD=--threads %N_THREAD%"
) else (
    set "_N_THREAD="
)

rem Run the script
echo "%MAIN_SCRIPT_PATH%" %GEN_OPTIONS% %_N_THREAD% ^
  --model "%MODEL%" ^
  --n_predict %N_PREDICTS% ^
  --color --interactive ^
  --reverse-prompt "%USER_NAME%:" ^
  --prompt "%PROMPT_TEXT%"
