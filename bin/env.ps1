<#
#* Environment configuration script for LLAMA.CPP
#! Sets environment variables for the current user
#>
Set-StrictMode -Version 3.0
Write-Verbose "Loading environment configuration - $PSScriptRoot\env.ps1"

#region Parameters
# LLAMA.CPP executable path
$llama_cpp_binary_path = "$PSScriptRoot\..\build\bin\Release\main.exe"
# Default model path for environment
$llm_model_path = "Z:\AI\LLM\dolphin-2.6-mixtral-8x7b-GGUF\dolphin-2.6-mixtral-8x7b.Q4_K_M.gguf"
#endregion

#region Set current environment variables
$env:LLAMA_CPP = Resolve-Path "$llama_cpp_binary_path"
$env:LLM_MODEL_PATH = Resolve-Path "$llm_model_path"
#endregion

#Region Persistent environment variables
# [Environment]::SetEnvironmentVariable("LLAMA_CPP", "$env:LLAMA_CPP", "User")
# [Environment]::SetEnvironmentVariable("LLM_MODEL_PATH", "$env:LLM_MODEL_PATH", "User")
#endregion