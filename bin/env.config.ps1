Set-StrictMode -Version 3.0

#region Directories

# Project directory
$PROJ_DIR = Resolve-Path "$PSScriptRoot\.."

# Build directory
$BUILD_DIR = "$PROJ_DIR\build"

#endregion

#region Files

# LLAMA.CPP executable
$LLAMA_EXE = "$BUILD_DIR\bin\Release\main.exe"

#endregion

#region Test data

# Test model
$TEST_MODEL_PATH = "F:\Models\Wizard-Vicuna-13B-Uncensored-GGML\Wizard-Vicuna-13B-Uncensored.ggmlv3.q4_0.bin"

# Test prompt
$TEST_PROMPT = "The life of a cyberpunk is"

#endregion