<#
#* Utility powerShell script to invoke the LLAMA.CPP binary
#! This script is used to invoke the LLAMA.CPP binary with configured environment paths and required arguments
#>
param(
    [string] $prompt,    # Prompt text to complete
    [string] $file,      # File path to complete
    [switch] $clear,     # Clear console prior to execution
    [switch] $console,   # Start the text completion console
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

# Import environment configuration if one exists
if ((Test-Path "$PSScriptRoot\env.ps1")) { . "$PSScriptRoot\env.ps1" }
# else { throw "Could not find environment configuration at path - '$PSScriptRoot\env.ps1'" }

#endregion

#region Operations

<# Compile LLAMA.CPP binary #>
function Build
{
    param(
        [string] $build_path = $null
    )
    $cur_dir = Get-Location
    try
    {
        # Set the build path to the default if not specified
        if (!$build_path) { $build_path = "$PSScriptRoot\..\build" }
        # Test if the build directory exists and create it if needed
        if (!(Test-Path -Path $build_path -PathType Container)) { New-Item -Path $build_path -ItemType Directory | Out-Null }
        # Set the location to the build directory
        Set-Location "$build_path"
        # Run the build commands
        cmake ..
        cmake --build . --config Release
    }
    catch { Write-Error "An error occurred during the build process - $_" }
    finally { Set-Location $cur_dir }
}

<# Test that a compiled LLAMA.CPP binary exists. Raise exception if not #>
function Validate { if (!(Test-Path "$env:LLAMA_CPP")) { throw "Could not find llama.exe at path - '$env:LLAMA_CPP'"} }

<# Complete text from input prompt #>
function Complete-Text
{
    param(
        [string] $prompt,
        [string] $model_path = $null,
        [int] $context_size = 2048,
        [int] $thread_cnt = 4
    )
    Validate
    if (!$model_path) { $model_path = $env:LLM_MODEL_PATH }
    $arg_map = @{
        "-m" = $model_path
        "-p" = $prompt
        "-c" = $context_size
        "-t" = $thread_cnt
    }
    $opt_set = @(
        "--color"
    )
    $arguments = $arg_map.GetEnumerator() | ForEach-Object { "$($_.Key) '$($_.Value)'" }
    $options = $opt_set | ForEach-Object { "$($_)" }
    Invoke-Expression "$env:LLAMA_CPP $arguments $options"
}

<# Generate and complete text file #>
function Complete-TextFile
{
    param(
        [string] $file,
        [string] $model_path = $null,
        [int] $context_size = 2048,
        [int] $thread_cnt = 4
    )
    Validate
    if (!$model_path) { $model_path = $env:LLM_MODEL_PATH }
    $arg_map = @{
        "-m" = $model_path
        "-f" = $file
        "-c" = $context_size
        "-t" = $thread_cnt
    }
    $opt_set = @(
        "--color"
    )
    $arguments = $arg_map.GetEnumerator() | ForEach-Object { "$($_.Key) '$($_.Value)'" }
    $options = $opt_set | ForEach-Object { "$($_)" }
    Invoke-Expression "$env:LLAMA_CPP $arguments $options"
}

<# Start text completion console #>
function Start-TextCompletionConsole
{
    param(
        [string] $model_path,
        [int] $context_size = 2048,
        [int] $thread_cnt = 4
    )
    Validate
    if (!$model_path) { $model_path = $env:LLM_MODEL_PATH }
    $arg_map = @{
        "-m" = $model_path
        "-c" = $context_size
        "-t" = $thread_cnt
    }
    $opt_set = @(
        "--interactive-first",
        "--color"
    )
    $arguments = $arg_map.GetEnumerator() | ForEach-Object { "$($_.Key) '$($_.Value)'" }
    $options = $opt_set | ForEach-Object { "$($_)" }
    Invoke-Expression "$env:LLAMA_CPP $arguments $options"
}

<# Invoke help details #>
function Help
{
    Validate
    Invoke-Expression "$env:LLAMA_CPP --help"
}

<# Invoke test function #>
function Test { Complete-Text "The life of a cyberpunk is" }

#endregion

#region Execution

if ($build) { Build }
if ($prompt) { Complete-Text -prompt "$prompt" }
if ($file) { Complete-TextFile -file "$file" }
if ($console) { Start-TextCompletionConsole }
if ($test) { Test }
if ($help) { Help }

#endregion