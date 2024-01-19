#!/bin/bash
#### BEGIN SETUP #####
set -euo pipefail
this=$(realpath -- "$0"); readonly this
cd "$(dirname "$this")"
shellcheck --external-sources "$this"
# shellcheck source=lib.sh
source 'lib.sh'
#### END SETUP ####

# TODO: send model to ctest_model tests using env variable
# GG_CI_CTEST_MODELFILE=<snip>

IsValidTestTarget() {
    case "$1" in
        cmake | ctest_main | model_3b | model_7b | test_cpu | test_cuda | test_metal | ctest_model)
            return $_OK;;
        *)
            return $_ERR;;
    esac
}

declare -a targets
if (( $# > 0 )); then
    targets=("$@")
elif _IsSet GG_CI_TARGETS; then
    read -r -a targets <<< "$GG_CI_TARGETS"
else
    cat >&2 <<'EOF'
usage:
    ci-run.sh [targets...]

config variables:
    GG_CI_TARGETS : Space delimited sequence of targets.
                      Overridden by commandline arguments.
    GG_CI_WORKDIR : Build files and results.
                      Defaults to /tmp.
    GG_CI_DATADIR : Persistent model files and datasets, unchanged between runs.
                      Defaults to ~/.cache/llama.cpp/ci-data.
    GG_CI_TEMPDIR : Scratch directory for quantized model files.
                      Defaults to ~/.cache/llama.cpp/ci-temp

examples:
    # A run on a low-spec VM without a dedicated GPU.
    ci-run.sh cmake ctest_main model_3b test_cpu ctest_model

    # A run on a Mac Studio with a ramdisk at ~/tmp
    GG_CI_WORKDIR=~/tmp/ci-work \
        GG_CI_TEMPDIR=~/tmp/ci-temp \
        ci-run.sh cmake ctest_main model_7b test_cpu test_metal ctest_model

test targets:
    cmake           : Run cmake to produce debug and release builds.
    ctest_main      : Run main ctest tests for debug and release.
    model_3b,
        model_7b    : Download and quantize openllama_3b_v2 and/or openllama_7b_v2.
    test_cpu,
        test_cuda,
        test_metal  : Test CPU inference, perplexity tests, etc.
    ctest_model     : Run ctest tests that require the openllama model.
EOF
    exit $_ERR
fi

for target in "${targets[@]}"; do
    if IsValidTestTarget "$target"; then
        _LogInfo "Received test target: $target"
    else
        _LogFatal "Invalid test target: $target"
    fi
done

cd ..
[[ -d .git && -x .git ]] || _LogFatal 'Could not cd to llama.cpp root direcory'

TargetCmake() {
    echo hello
}

for target in "${targets[@]}"; do
    pascal_target="$(_SnakeToPascalCase "$target")"
    "Target$pascal_target"
done
