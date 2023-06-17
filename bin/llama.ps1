param(
    [switch] $clear,     # Clear console prior to execution
    [switch] $debug,     # Enable debug output
    [switch] $verbose,   # Enable verbose output
    [switch] $build,     # Build the executable
    [switch] $test,      # Test the executable
    [switch] $help       # Display help output from executable
)

#region Env. config

Set-StrictMode -Version 3.0

if ($clear) { Clear-Host }
if ($debug) { $DebugPreference = "Continue" }
if ($verbose) { $VerbosePreference = "Continue" }

# Import environment configuration
. "$PSScriptRoot\env.config.ps1"

#endregion

#region Operations

function ExecuteCommand
{
    param(
        [string] $command,
        [string] $cwd = $PROJ_DIR
    )
    Write-Verbose "Executing command: $command"
    Write-Verbose "Working directory: $cwd"

    $cur_dir = Get-Location
    Set-Location $cwd

    Invoke-Expression $command

    Set-Location $cur_dir
}

function Build
{
    param(
        [string] $build_path = $BUILD_DIR
    )

    $cur_dir = Get-Location

    try
    {
        # Test if the build directory exists and create it if needed
        if (!(Test-Path -Path $build_path -PathType Container)) { New-Item -Path $build_path -ItemType Directory | Out-Null }

        # Set the location to the build directory
        Set-Location "$build_path"

        # Run the build commands
        cmake ..
        cmake --build . --config Release
    }
    catch { Write-Error "An error occurred during the build process: $_" }
    finally { Set-Location $cur_dir }
}

function Validate
{
    # Check that llama.exe exists
    if (!(Test-Path "$LLAMA_EXE")) { throw "Could not find llama.exe at path - '$LLAMA_EXE'"}
}

function GenerateTextFromPrompt
{
    param(
        [string] $model_path,
        [string] $prompt,
        [int] $context_size = 2048,
        [int] $thread_cnt = 4
    )
    Validate
    $arguments = "-m '$model_path' -p '$prompt' -c '$context_size' -t '$thread_cnt' --color"
    ExecuteCommand "$LLAMA_EXE $arguments"
}

function GenerateTextFromFile
{
    param(
        [string] $model_path,
        [string] $file,
        [int] $context_size = 2048,
        [int] $thread_cnt = 4
    )
    Validate
    $arguments = "-m '$model_path' -f '$file' -c '$context_size' -t '$thread_cnt' --color"
    ExecuteCommand "$LLAMA_EXE $arguments"
}

function GenerateInteractiveText
{
    param(
        [string] $model_path,
        [int] $context_size = 2048,
        [int] $thread_cnt = 4
    )
    Validate
    $arguments = "-m '$model_path' --interactive-first -c '$context_size' -t '$thread_cnt' --color"
    ExecuteCommand "$LLAMA_EXE $arguments"  # Wait for input before starting
}

function Help
{
    Validate
    ExecuteCommand "$LLAMA_EXE --help"
}

function Test
{
    GenerateTextFromPrompt "$TEST_MODEL_PATH" "$TEST_PROMPT"
}

#endregion

#region Execution

if ($build) { Build }
if ($test) { Test }
if ($help) { Help }

#endregion