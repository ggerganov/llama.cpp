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
# GG_TEST_CTEST_MODEL_MODELFILE=<snip>

is_valid_test_target() {
    case "$1" in
        cmake | ctest_main | model_3b | model_7b | test_cpu | test_cuda | test_metal | ctest_model)
            return 0;;
        *)
            return 1;;
    esac
}

declare -a test_targets
if (( $# > 0 )); then
    test_targets=("$@")
elif _isset GG_TEST_TARGETS; then
    read -r -a test_targets <<< "$GG_TEST_TARGETS"
else
    cat >&2 <<'EOF'
You must specify the test targets either as commandline arguments or using the
GG_TEST_TARGETS environment variable. Test targets will be run sequentially
in-order.

Some test targets depend on other test targets, as described below.

cli usage:
	ci-run.sh test_targets...

ci usage:
	GG_TEST_TARGETS='test_targets...' ci-run.sh

example:
	# This is a typical run for a low-spec host without a dedicated GPU.
	ci-run.sh cmake ctest_main model_3b test_cpu ctest_model

test targets:
	cmake		: run cmake to produce debug and release builds
	ctest_main	: run main ctest tests for debug and release
				(requires: cmake)
	model_3b	: download and quantize openllama_3b_v2
	model_7b	: download and quantize openllama_7b_v2
	test_cpu	: test CPU inference, perplexity tests, etc.
				(requires: model_3b or model_7b)
	test_cuda	: test CUDA ...
				(requires: model_3b or model_7b)
	test_metal	: test Metal ...
	ctest_model	: run ctest tests that require the openllama model
				(requires: model_3b or model_7b)
EOF
    exit 1
fi

for target in "${test_targets[@]}"; do
    if is_valid_test_target "$target"; then
        _log_info "Received test target: $target"
    else
        _log_fatal "Invalid test target: $target"
    fi
done
